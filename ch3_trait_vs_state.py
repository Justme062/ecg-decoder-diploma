# -*- coding: utf-8 -*-
"""
ch3_trait_vs_state.py — РЕШАЮЩАЯ проверка для главы 3.

Вопрос (по рецензии): AUC 0,62 полюсов на N-only — это «предупреждение о скором
событии» (СОСТОЯНИЕ ритма перед ЖЭ) или просто «этот удар принадлежит ЖЭ-склонному
пациенту» (статическая ЧЕРТА пациента)? Межпациентная CV этого не различает: трейт
прекрасно переносится на новых склонных пациентов.

Разложение AUC на два вклада (на out-of-fold предсказаниях полюсов, N-only):
  1. ПУЛ (как в работе)                — общий AUC(y, score).
  2. ТОЛЬКО ТРЕЙТ                       — каждому окну присвоен СРЕДНИЙ по его пациенту
     score; AUC такого «пациент-константного» предсказания. Показывает, сколько AUC
     объясняется одним лишь ранжированием ПАЦИЕНТОВ (без внутрипациентного времени).
  3. ВНУТРИ ПАЦИЕНТА (состояние)        — AUC считается ОТДЕЛЬНО внутри каждого пациента
     (у кого есть оба класса), затем усредняется. Показывает, отличает ли модель
     окна-перед-ЖЭ от окон-не-перед-ЖЭ У ОДНОГО И ТОГО ЖЕ пациента.

Как читать:
  • если ТОЛЬКО ТРЕЙТ ≈ ПУЛ, а ВНУТРИ ПАЦИЕНТА ≈ 0,5 → сигнал = ЧЕРТА пациента
    (риск-стратификация), а не раннее предупреждение о событии;
  • если ВНУТРИ ПАЦИЕНТА заметно > 0,5 → есть настоящий временно́й/состоянческий сигнал,
    и рамка «раннее предупреждение» оправдана.

Запуск (после build_sequences.py → sequences.npz):
    python ch3_trait_vs_state.py            # H=1 (следующий удар)
    python ch3_trait_vs_state.py --H 10     # ЖЭ в ближайшие 10 ударов
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WINDOW, FPB = 10, 31
MP_IDX = [j * FPB + k for j in range(WINDOW) for k in range(30)]


def make_folds(recs_unique, k, seed):
    rng = np.random.default_rng(seed)
    r = recs_unique[rng.permutation(len(recs_unique))]
    return [set(a.tolist()) for a in np.array_split(r, k)]


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    npos = int(y.sum()); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return None
    r = rankdata(s)
    return (r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def oof_scores(flat, y, rec, folds, cols):
    oof = np.full(len(y), np.nan)
    for te_recs in folds:
        te = np.isin(rec, list(te_recs)); tr = ~te
        if y[tr].sum() == 0 or y[tr].sum() == tr.sum():
            continue
        sc = StandardScaler().fit(flat[tr][:, cols])
        clf = LogisticRegression(max_iter=2000, class_weight='balanced')
        clf.fit(sc.transform(flat[tr][:, cols]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(flat[te][:, cols]))[:, 1]
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--H', type=int, default=1, help='горизонт: ЖЭ в ближайшие H ударов')
    ap.add_argument('-k', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--minpos', type=int, default=10, help='мин. положительных на пациента для внутрипац. AUC')
    a = ap.parse_args()

    d = np.load(os.path.join(SCRIPT_DIR, 'sequences.npz'), allow_pickle=True)
    X = d['X_seq']; rec_all = d['record_ids']; alln = d['all_normal']
    if a.H == 1:
        y_all = d['y_next'].astype(int); nfut = np.full(len(y_all), 1)
    else:
        hor = d['horizons'].tolist(); hi = hor.index(a.H)
        y_all = d['y_horizon'][:, hi].astype(int); nfut = d['n_future']

    m = (alln == 1) & (nfut >= a.H)
    X = X[m]; y = y_all[m]; rec = rec_all[m]
    flat = X.reshape(len(y), -1)
    recs_u = np.array(sorted(set(rec.tolist())))
    folds = make_folds(recs_u, a.k, a.seed)

    print("=" * 78)
    print(f"ТРЕЙТ ПРОТИВ СОСТОЯНИЯ — глава 3 (N-only, горизонт H={a.H})")
    print(f"окон {len(y)}, положительных {int(y.sum())} ({y.mean()*100:.2f}%), "
          f"пациентов {len(recs_u)}")
    print("=" * 78)

    score = oof_scores(flat, y, rec, folds, MP_IDX)
    ok = ~np.isnan(score)
    y, rec, score = y[ok], rec[ok], score[ok]

    pooled = auc(y, score)
    print(f"\n1) ПУЛ (как в работе)         : AUC = {pooled:.3f}")

    # 2) только трейт — заменяем score средним по пациенту
    trait = np.empty_like(score)
    for p in set(rec.tolist()):
        idx = rec == p
        trait[idx] = score[idx].mean()
    tr_auc = auc(y, trait)
    print(f"2) ТОЛЬКО ТРЕЙТ (сред. по пациенту): AUC = {tr_auc:.3f}   "
          f"({tr_auc/pooled*100:.0f}% от пула)")

    # 3) внутри пациента
    within = []
    for p in set(rec.tolist()):
        idx = rec == p
        if y[idx].sum() >= a.minpos and (y[idx] == 0).sum() >= a.minpos:
            wa = auc(y[idx], score[idx])
            if wa is not None:
                within.append(wa)
    within = np.array(within)
    if len(within):
        m_, s_ = within.mean(), (within.std(ddof=1) if len(within) > 1 else 0.0)
        frac = (within > 0.5).mean() * 100
        print(f"3) ВНУТРИ ПАЦИЕНТА (состояние): AUC = {m_:.3f} ± {s_:.3f}  "
              f"(по {len(within)} пациентам с ≥{a.minpos} положит.; выше 0,5 у {frac:.0f}%)")
    else:
        print(f"3) ВНУТРИ ПАЦИЕНТА: недостаточно пациентов с ≥{a.minpos} положительными")

    print("\n" + "=" * 78)
    print("ВЕРДИКТ:")
    if len(within):
        if m_ < 0.55:
            print(" • ВНУТРИ пациента AUC ≈ 0,5 → сигнал определяется РАЗЛИЧИЕМ МЕЖДУ")
            print("   пациентами (статическая черта). Это РИСК-СТРАТИФИКАЦИЯ пациента,")
            print("   а не раннее предупреждение о конкретном приближающемся событии.")
        else:
            print(" • ВНУТРИ пациента AUC заметно > 0,5 → есть настоящий временно́й/")
            print("   состоянческий сигнал. Рамка «раннее предупреждение» оправдана.")
    if tr_auc >= pooled - 0.02:
        print(" • ТОЛЬКО ТРЕЙТ ≈ ПУЛ → почти весь AUC объясняется ранжированием")
        print("   пациентов; внутрипациентного времени в сигнале почти нет.")
    print("=" * 78)


if __name__ == '__main__':
    main()
