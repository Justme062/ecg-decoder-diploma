# -*- coding: utf-8 -*-
"""
risk_horizon.py — практически осмысленная постановка: не «будет ли ЖЭ в
СЛЕДУЮЩЕМ ударе», а «есть ли ЖЭ в БЛИЖАЙШИХ H ударах» (окно риска, ~секунды–
минута вперёд). Идея: склонность к экстрасистолии — инерционное состояние
ритма, оно предсказуемее на горизонте, чем на один удар.

Честный протокол (как в ch3_controls):
  • только N-only окна (10 нормальных ударов) — контроль кластеризации эктопий;
  • межпациентная K-fold CV (out-of-fold предсказания, без утечки пациентов);
  • перестановочный тест значимости AUC;
  • полюса (МП) против тайминга (RR) — на каждом горизонте.

Запуск (после обновлённого build_sequences.py → sequences.npz с y_horizon):
    python risk_horizon.py            # 5 фолдов, 5000 перестановок
    python risk_horizon.py -k 5 --nperm 5000
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
WINDOW, FEAT_PER_BEAT = 10, 31
RR_IDX = [j * FEAT_PER_BEAT + 30 for j in range(WINDOW)]
MP_IDX = [j * FEAT_PER_BEAT + k for j in range(WINDOW) for k in range(30)]


def make_folds(record_ids, k, seed):
    recs = np.array(sorted(set(record_ids.tolist())))
    rng = np.random.default_rng(seed)
    recs = recs[rng.permutation(len(recs))]
    return [set(a.tolist()) for a in np.array_split(recs, k)]


def oof_predict(flat, y, record_ids, folds, cols):
    oof = np.full(len(y), np.nan)
    for test_recs in folds:
        te = np.isin(record_ids, list(test_recs)); tr = ~te
        if y[tr].sum() == 0 or y[tr].sum() == tr.sum() or te.sum() == 0:
            continue
        sc = StandardScaler().fit(flat[tr][:, cols])
        clf = LogisticRegression(max_iter=2000, class_weight='balanced')
        clf.fit(sc.transform(flat[tr][:, cols]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(flat[te][:, cols]))[:, 1]
    return oof


def auc_and_p(y, scores, n_perm, seed):
    m = ~np.isnan(scores)
    yv = y[m].astype(int); pv = scores[m]
    if yv.sum() == 0 or yv.sum() == len(yv):
        return None
    ranks = rankdata(pv)
    n_pos = int(yv.sum()); n_neg = len(yv) - n_pos
    const = n_pos * (n_pos + 1) / 2.0; denom = n_pos * n_neg
    obs = (ranks[yv == 1].sum() - const) / denom
    rng = np.random.default_rng(seed); ge = 0
    for _ in range(n_perm):
        pos = rng.choice(len(yv), n_pos, replace=False)
        if (ranks[pos].sum() - const) / denom >= obs:
            ge += 1
    return obs, (1 + ge) / (n_perm + 1), len(yv), n_pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-k', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--nperm', type=int, default=5000)
    a = ap.parse_args()

    d = np.load(os.path.join(SCRIPT_DIR, 'sequences.npz'), allow_pickle=True)
    if 'y_horizon' not in d:
        print("ОШИБКА: в sequences.npz нет y_horizon. Перезапустите обновлённый build_sequences.py.")
        return
    X_seq = d['X_seq']; record_ids = d['record_ids']
    all_normal = d['all_normal']; n_future = d['n_future']
    y_hor = d['y_horizon']; horizons = d['horizons']
    folds = make_folds(record_ids, a.k, a.seed)

    approx_sec = {1: "~1 удар", 5: "~4 с", 10: "~8 с", 30: "~25 с", 60: "~50 с"}
    print("=" * 82)
    print(f"ОКНО РИСКА: предсказание ЖЭ в ближайшие H ударов по N-only окну")
    print(f"Межпациентная {a.k}-fold CV, перестановочный тест ({a.nperm}), только нормальные окна")
    print("=" * 82)
    print(f"{'H (гор.)':<10}{'≈время':<9}{'окон':<9}{'полож.':<9}"
          f"{'AUC полюса':<22}{'AUC тайминг':<20}")
    print("-" * 82)

    rows = []
    for hi, H in enumerate(horizons):
        H = int(H)
        mask = (all_normal == 1) & (n_future >= H)
        flat = X_seq[mask].reshape(mask.sum(), -1)
        y = y_hor[mask, hi].astype(int)
        recs = record_ids[mask]
        if y.sum() < 15:
            print(f"{H:<10}{approx_sec.get(H,''):<9}{mask.sum():<9}{int(y.sum()):<9}  мало положительных — пропуск")
            continue
        oof_mp = oof_predict(flat, y, recs, folds, MP_IDX)
        oof_rr = oof_predict(flat, y, recs, folds, RR_IDX)
        r_mp = auc_and_p(y, oof_mp, a.nperm, a.seed)
        r_rr = auc_and_p(y, oof_rr, a.nperm, a.seed)
        def fmt(r):
            if r is None: return "н/д"
            auc, p, _, _ = r
            star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "н.з."))
            return f"{auc:.3f} (p={p:.4g} {star})"
        print(f"{H:<10}{approx_sec.get(H,''):<9}{mask.sum():<9}{int(y.sum()):<9}"
              f"{fmt(r_mp):<22}{fmt(r_rr):<20}")
        rows.append((H, r_mp, r_rr))

    print("-" * 82)
    print("Читать так: если AUC полюсов растёт с горизонтом H и значим (p<0.05) —")
    print("метод заблаговременно (за десятки секунд) выделяет период повышенного")
    print("риска ЖЭ по морфологии спокойных нормальных ударов. Тайминг для контраста.")
    if rows:
        best = max(rows, key=lambda r: r[1][0] if r[1] else 0)
        H, r_mp, _ = best
        print(f"\nЛучший горизонт: H={H} ({approx_sec.get(H,'')}) — AUC полюсов {r_mp[0]:.3f}, p={r_mp[1]:.4g}")
    print("Строку с лучшим горизонтом пришлите — впишу в главу 3 как основной результат.")


if __name__ == '__main__':
    main()
