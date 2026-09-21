# -*- coding: utf-8 -*-
"""
forecast_arrhythmia.py — ПРОГНОЗ приближающейся аритмии (клиническая постановка).

Цель: по последнему окну ритма (10 ударов, ~8 с) предсказать, возникнет ли
желудочковая экстрасистола (ЖЭ) в БЛИЖАЙШИЕ H ударов (H=30 ≈ 25 с, H=60 ≈ 50 с) —
то есть заблаговременно выделить период, когда «скоро может стать плохо».

Отличие от прежней (проваленной) постановки: раньше искали предвестник в форме
НОРМАЛЬНЫХ ударов и намеренно вычищали недавнюю эктопию как «конфаундер». Для
академического вопроса это было верно, но для КЛИНИЧЕСКОЙ цели недавнее состояние
ритма — не конфаундер, а сам полезный сигнал (эктопия идёт очередями во времени;
так работают реальные мониторы). Здесь недавняя активность используется как
признак — при строгом отсутствии утечки (окно наблюдения целиком ДО горизонта).

Наборы признаков (все — из окна наблюдения, без данных из будущего):
  RATE  — недавняя активность: число ЖЭ в окне (n_ectopic);
  RRDYN — динамика ритма: std/min/наклон/RMSSD/последний RR;
  POLES — динамика матричных пучков: |z| доминирующего полюса и сингулярные (сред./std/наклон);
  ALL   — всё вместе.

Контроли (встроены, чтобы не было повторных раундов):
  • межпациентная 5-fold CV (тест — новые пациенты);
  • разброс по фолдам (честная неопределённость, не «пол» перестановки);
  • ТРЕЙТ vs СОСТОЯНИЕ: раскладываем AUC на межпациентную черту и внутрипациентное время;
  • «ИЗ СПОКОЙНОГО ОКНА»: подвыборка окон без недавней эктопии (n_ectopic=0) —
    самая ценная и трудная версия: предупредить ДО начала очереди.

Запуск (из папки с sequences.npz):
    python forecast_arrhythmia.py
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

SD = os.path.dirname(os.path.abspath(__file__))
W, FPB = 10, 31


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s); npos = int(y.sum()); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return None
    r = rankdata(s); return (r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def build_features(X, n_ect):
    rr = X[:, :, 30]; zdom = X[:, :, 10]; sv = X[:, :, 0:10]
    t = np.arange(W)
    def slope(a):
        a = a - a.mean(1, keepdims=True)
        return (a * (t - t.mean())).sum(1) / ((t - t.mean()) ** 2).sum()
    rmssd = np.sqrt((np.diff(rr, axis=1) ** 2).mean(1))
    RATE = np.column_stack([n_ect])
    RRDYN = np.column_stack([rr.std(1), rr.min(1), slope(rr), rmssd, rr[:, -1], rr.max(1) - rr.min(1)])
    POLES = np.column_stack([zdom.mean(1), zdom.std(1), slope(zdom), zdom[:, -1],
                             sv.mean((1, 2)), sv.std((1, 2)), sv[:, :, 0].mean(1),
                             sv[:, :, 0].std(1)])
    return {'RATE': RATE, 'RRDYN': RRDYN, 'POLES': POLES,
            'ALL': np.column_stack([RATE, RRDYN, POLES])}


def make_folds(recs_u, k, seed):
    rng = np.random.default_rng(seed); r = recs_u[rng.permutation(len(recs_u))]
    return [set(a.tolist()) for a in np.array_split(r, k)]


def cv(featmat, y, rec, folds):
    """Возвращает OOF-скор и список пофолдовых AUC."""
    oof = np.full(len(y), np.nan); fold_aucs = []
    for te_recs in folds:
        te = np.isin(rec, list(te_recs)); tr = ~te
        if y[tr].sum() == 0 or y[tr].sum() == tr.sum() or y[te].sum() == 0 or y[te].sum() == te.sum():
            continue
        sc = StandardScaler().fit(featmat[tr])
        clf = LogisticRegression(max_iter=3000, class_weight='balanced')
        clf.fit(sc.transform(featmat[tr]), y[tr])
        p = clf.predict_proba(sc.transform(featmat[te]))[:, 1]
        oof[te] = p; fold_aucs.append(auc(y[te], p))
    return oof, np.array([a for a in fold_aucs if a is not None])


def decompose(y, score, rec, minpos=10):
    ok = ~np.isnan(score); y, score, rec = y[ok], score[ok], rec[ok]
    pooled = auc(y, score)
    trait = np.empty_like(score)
    for p in set(rec.tolist()):
        idx = rec == p; trait[idx] = score[idx].mean()
    tr_auc = auc(y, trait)
    within = []
    for p in set(rec.tolist()):
        idx = rec == p
        if y[idx].sum() >= minpos and (y[idx] == 0).sum() >= minpos:
            a = auc(y[idx], score[idx])
            if a is not None:
                within.append(a)
    within = np.array(within)
    return pooled, tr_auc, within


def run_horizon(X, n_ect, y, rec, nfut, H, folds, seedline=""):
    m = nfut >= H
    Xh, neh, yh, rh = X[m], n_ect[m], y[m], rec[m]
    feats = build_features(Xh, neh)
    print(f"\n{'='*80}\nГОРИЗОНТ H={H} ударов (~{H*0.8:.0f} с){seedline}"
          f"  |  окон {m.sum()}, положительных {int(yh.sum())} ({yh.mean()*100:.1f}%)")
    print(f"{'набор признаков':<28}{'AUC (пул)':<14}{'AUC по фолдам':<20}")
    print("-" * 62)
    scores = {}
    for name in ['RATE', 'RRDYN', 'POLES', 'ALL']:
        oof, fa = cv(feats[name], yh, rh, folds)
        scores[name] = oof
        pooled = auc(yh[~np.isnan(oof)], oof[~np.isnan(oof)])
        fm = fa.mean() if len(fa) else float('nan')
        fs = fa.std(ddof=1) if len(fa) > 1 else 0.0
        print(f"{name:<28}{pooled:<14.3f}{fm:.3f} ± {fs:.3f}")
    # трейт vs состояние — для ALL и RATE
    print("\n  ТРЕЙТ vs СОСТОЯНИЕ (раскладка AUC):")
    for name in ['ALL', 'RATE']:
        pooled, tr, wi = decompose(yh, scores[name], rh)
        if len(wi):
            print(f"   {name:<6}: пул {pooled:.3f} | только трейт {tr:.3f} | "
                  f"ВНУТРИ пациента {wi.mean():.3f} ± {wi.std(ddof=1) if len(wi)>1 else 0:.3f} "
                  f"(по {len(wi)} пациентам)")
        else:
            print(f"   {name:<6}: пул {pooled:.3f} | только трейт {tr:.3f} | внутрипац. — мало данных")
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-k', type=int, default=5); ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    d = np.load(os.path.join(SD, 'sequences.npz'), allow_pickle=True)
    X = d['X_seq']; rec = d['record_ids']; n_ect = d['n_ectopic'].astype(float)
    y_h = d['y_horizon']; hor = d['horizons'].tolist(); nfut = d['n_future']; alln = d['all_normal']
    recs_u = np.array(sorted(set(rec.tolist()))); folds = make_folds(recs_u, a.k, a.seed)

    print("#" * 80)
    print("ПРОГНОЗ ПРИБЛИЖАЮЩЕЙСЯ ЖЭ — окно наблюдения 10 ударов -> горизонт H")
    print("Межпациентная 5-fold CV. Окно наблюдения целиком ДО горизонта (без утечки).")
    print("#" * 80)

    for H in [30, 60]:
        hi = hor.index(H)
        run_horizon(X, n_ect, y_h[:, hi].astype(int), rec, nfut, H, folds)

    # ── версия «ИЗ СПОКОЙНОГО ОКНА» (n_ectopic=0): предупредить ДО начала очереди ──
    print("\n" + "#" * 80)
    print("ТРУДНАЯ ВЕРСИЯ: предупреждение ИЗ СПОКОЙНОГО ОКНА (в окне нет эктопии)")
    print("Признаки RATE тут неинформативны (=0), работают только RRDYN и POLES.")
    print("#" * 80)
    for H in [30, 60]:
        hi = hor.index(H)
        m = (n_ect == 0) & (nfut >= H)
        Xh, yh, rh = X[m], y_h[m, hi].astype(int), rec[m]
        feats = build_features(Xh, n_ect[m])
        print(f"\nH={H} (~{H*0.8:.0f} с) | спокойных окон {m.sum()}, "
              f"положительных {int(yh.sum())} ({yh.mean()*100:.1f}%)")
        for name in ['RRDYN', 'POLES', 'ALL']:
            oof, fa = cv(feats[name], yh, rh, folds)
            pooled = auc(yh[~np.isnan(oof)], oof[~np.isnan(oof)]) if (~np.isnan(oof)).any() else None
            _, tr, wi = decompose(yh, oof, rh)
            wtxt = f"{wi.mean():.3f}" if len(wi) else "—"
            print(f"   {name:<6}: пул {pooled if pooled else float('nan'):.3f} | "
                  f"трейт {tr:.3f} | внутрипац. {wtxt}")

    print("\n" + "#" * 80)
    print("ЧТЕНИЕ РЕЗУЛЬТАТА:")
    print(" • Высокий AUC RATE и ALL + ВНУТРИ пациента > 0,5 → есть реальный прогноз")
    print("   'скоро период ЖЭ' (предупреждение о продолжении/усилении нестабильности).")
    print(" • Вклад POLES сверх RATE/RRDYN → польза именно матричных пучков.")
    print(" • 'Из спокойного окна' внутрипац. > 0,5 → предупреждение ДО начала очереди")
    print("   (самый ценный исход). Если ≈0,5 → онсет из тишины не предсказуем, но")
    print("   прогноз продолжения нестабильности всё равно работает.")
    print("#" * 80)


if __name__ == '__main__':
    main()
