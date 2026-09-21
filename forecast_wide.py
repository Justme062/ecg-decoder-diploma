# -*- coding: utf-8 -*-
"""
forecast_wide.py — прогноз приближающейся ЖЭ по ШИРОКОМУ окну наблюдения.

Идея (по итогам forecast_arrhythmia): предвестник — это НАРАСТАЮЩАЯ нестабильность
ритма. Чтобы её увидеть, окно наблюдения расширено до ~48 с (60 ударов) и добавлены
классические маркеры вариабельности ритма (HRV) и динамика полюсов.

Задача: по окну из W последних ударов (до горизонта, без утечки) предсказать, будет
ли ЖЭ в ближайшие H ударов (H=10/30/60 ≈ 8/24/48 с).

Наборы признаков (всё из окна наблюдения):
  RATE  — активность: число и доля ЖЭ, ударов с последней ЖЭ, длина последней серии;
  HRV   — вариабельность ритма: mean RR, SDNN, RMSSD, pNN50, CV, min RR, наклон, coupling;
  POLES — динамика матричных пучков: |z| дом. полюса (сред./std/наклон/послед.),
          сингулярные значения, морфологическая изменчивость по окну;
  ALL   — всё вместе.

Контроли встроены: межпациентная 5-fold CV, разброс по фолдам, разложение
ТРЕЙТ vs СОСТОЯНИЕ (внутрипациентный AUC), версия «из спокойного окна».

Запуск (из папки с mitdb/):
    python forecast_wide.py                 # соберёт признаки (тяжело, ~10-15 мин) и проанализирует
    python forecast_wide.py --reuse         # если forecast_wide.npz уже собран — только анализ
"""
import os, sys, argparse
import numpy as np
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.stats import rankdata
import build_sequences as bs

SD = os.path.dirname(os.path.abspath(__file__))
W_OBS = 60           # ширина окна наблюдения в ударах (~48 с)
STEP = 3             # шаг скользящего окна (снижает объём и автокорреляцию)
HORIZONS = [10, 30, 60]
HMAX = max(HORIZONS)
CACHE = os.path.join(SD, 'forecast_wide.npz')


def _slope(a):
    t = np.arange(len(a)); t = t - t.mean()
    d = (t * t).sum()
    return (t * (a - a.mean())).sum() / d if d > 0 else 0.0


def feats_window(mp_w, rr_w, isV_w):
    zdom = mp_w[:, 10]; sv = mp_w[:, 0:10]
    vcount = float(isV_w.sum()); vrate = vcount / len(isV_w)
    vpos = np.where(isV_w == 1)[0]
    since_last = float(len(isV_w) - 1 - vpos[-1]) if len(vpos) else float(len(isV_w))
    mx = c = 0
    for v in isV_w:
        c = c + 1 if v else 0; mx = max(mx, c)
    d = np.diff(rr_w)
    mean_rr = rr_w.mean(); sdnn = rr_w.std()
    rmssd = float(np.sqrt((d ** 2).mean())) if len(d) else 0.0
    pnn50 = float((np.abs(d) > 0.05).mean()) if len(d) else 0.0
    cv = sdnn / mean_rr if mean_rr > 0 else 0.0
    med = np.median(rr_w); coupling_min = rr_w.min() / med if med > 0 else 1.0
    RATE = [vcount, vrate, since_last, float(mx)]
    HRV = [mean_rr, sdnn, rmssd, pnn50, cv, rr_w.min(), _slope(rr_w), coupling_min]
    POLES = [zdom.mean(), zdom.std(), _slope(zdom), zdom[-1],
             sv.mean(), sv.std(), sv[:, 0].mean(), mp_w.std(0).mean()]
    return RATE + HRV + POLES


NR, NH, NP = 4, 8, 8   # размеры блоков признаков


def build():
    rows, recs = [], []
    ys = {H: [] for H in HORIZONS}
    for ri, rid in enumerate(bs.RECORDS):
        try:
            beats = bs.extract_beats_from_record(rid, bs.DATA_DIR)
        except Exception as e:
            print(f"  [{rid}] пропуск: {e}"); continue
        n = len(beats)
        if n < W_OBS + HMAX + 1:
            continue
        mp = np.array([b[0] for b in beats], dtype=np.float32)
        pos = np.array([b[1] for b in beats])
        isV = np.array([1 if s in bs.V_SYMBOLS else 0 for s in (b[2] for b in beats)])
        rr = np.full(n, np.nan); rr[1:] = (pos[1:] - pos[:-1]) / bs.FS
        rr[0] = np.nanmedian(rr[1:])
        cnt = 0
        for a in range(W_OBS, n - HMAX, STEP):
            ow = slice(a - W_OBS, a)
            rows.append(feats_window(mp[ow], rr[ow], isV[ow]))
            recs.append(rid)
            for H in HORIZONS:
                ys[H].append(int(isV[a:a + H].sum() > 0))
            cnt += 1
        print(f"  [{rid}] окон: {cnt}")
    F = np.array(rows, dtype=np.float32)
    recs = np.array(recs)
    Y = np.column_stack([ys[H] for H in HORIZONS]).astype(np.int8)
    np.savez_compressed(CACHE, F=F, Y=Y, recs=recs, horizons=np.array(HORIZONS))
    print(f"\nСобрано окон: {len(F)}, признаков: {F.shape[1]}  -> {CACHE}")
    return F, Y, recs


# ---------- анализ ----------
def auc(y, s):
    y = np.asarray(y); s = np.asarray(s); npos = int(y.sum()); nneg = len(y) - npos
    if npos == 0 or nneg == 0: return None
    r = rankdata(s); return (r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)

def folds_of(recs, k, seed):
    ru = np.array(sorted(set(recs.tolist())))
    rng = np.random.default_rng(seed); ru = ru[rng.permutation(len(ru))]
    return [set(a.tolist()) for a in np.array_split(ru, k)]

def cv(Fm, y, recs, folds):
    oof = np.full(len(y), np.nan); fa = []
    for te_recs in folds:
        te = np.isin(recs, list(te_recs)); tr = ~te
        if y[tr].sum() in (0, tr.sum()) or y[te].sum() in (0, te.sum()): continue
        sc = StandardScaler().fit(Fm[tr]); cl = LogisticRegression(max_iter=3000, class_weight='balanced')
        cl.fit(sc.transform(Fm[tr]), y[tr]); p = cl.predict_proba(sc.transform(Fm[te]))[:, 1]
        oof[te] = p; fa.append(auc(y[te], p))
    return oof, np.array([x for x in fa if x is not None])

def decompose(y, score, recs, minpos=15):
    ok = ~np.isnan(score); y, score, recs = y[ok], score[ok], recs[ok]
    pooled = auc(y, score); trait = np.empty_like(score)
    for p in set(recs.tolist()): trait[recs == p] = score[recs == p].mean()
    wi = []
    for p in set(recs.tolist()):
        idx = recs == p
        if y[idx].sum() >= minpos and (y[idx] == 0).sum() >= minpos:
            a = auc(y[idx], score[idx])
            if a is not None: wi.append(a)
    return pooled, auc(y, trait), np.array(wi)

BLOCKS = {'RATE': slice(0, NR), 'HRV': slice(NR, NR + NH),
          'POLES': slice(NR + NH, NR + NH + NP), 'ALL': slice(0, NR + NH + NP)}

def analyze(F, Y, recs, folds, hor, subset=None, title=""):
    if subset is not None:
        F, Y, recs = F[subset], Y[subset], recs[subset]
    print(f"\n{'='*82}\n{title}")
    for hi, H in enumerate(hor):
        y = Y[:, hi].astype(int)
        if y.sum() < 30: 
            print(f" H={H}: мало положительных ({int(y.sum())})"); continue
        print(f"\n H={H} (~{H*0.8:.0f} с) | окон {len(y)}, положит. {int(y.sum())} ({y.mean()*100:.1f}%)")
        best = None
        for name in ['RATE', 'HRV', 'POLES', 'ALL']:
            oof, fa = cv(F[:, BLOCKS[name]], y, recs, folds)
            pooled, tr, wi = decompose(y, oof, recs)
            wtxt = f"{wi.mean():.3f}±{wi.std(ddof=1) if len(wi)>1 else 0:.3f}" if len(wi) else "—"
            fm = fa.mean() if len(fa) else float('nan'); fs = fa.std(ddof=1) if len(fa) > 1 else 0
            print(f"   {name:<6} AUC пул {pooled:.3f} | фолды {fm:.3f}±{fs:.3f} | "
                  f"трейт {tr:.3f} | ВНУТРИ пац. {wtxt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reuse', action='store_true'); ap.add_argument('-k', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    if a.reuse and os.path.exists(CACHE):
        d = np.load(CACHE, allow_pickle=True); F, Y, recs = d['F'], d['Y'], d['recs']
        print(f"Загружено из кэша: {F.shape[0]} окон")
    else:
        print("Сбор признаков по широкому окну (это тяжёлая часть)...")
        F, Y, recs = build()
    folds = folds_of(recs, a.k, a.seed)
    print("#" * 82)
    print(f"ПРОГНОЗ ЖЭ ПО ШИРОКОМУ ОКНУ ({W_OBS} ударов ≈ {W_OBS*0.8:.0f} с). Межпац. {a.k}-fold CV.")
    print("#" * 82)
    analyze(F, Y, recs, folds, HORIZONS.tolist() if hasattr(HORIZONS,'tolist') else HORIZONS,
            title="ВСЕ ОКНА (прогноз периода нестабильности)")
    # из спокойного окна: активности нет (vcount==0)
    quiet = F[:, 0] == 0
    analyze(F, Y, recs, folds, HORIZONS,
            subset=quiet, title="ИЗ СПОКОЙНОГО ОКНА (vcount=0): предупреждение ДО начала очереди")
    print("\n" + "#" * 82)
    print("ЧТЕНИЕ: смотрим ВНУТРИ пац. — если >0,5 значимо, это настоящий прогноз времени,")
    print("а не черта пациента. Вклад POLES сверх RATE/HRV = польза матричных пучков.")
    print("#" * 82)


if __name__ == '__main__':
    main()
