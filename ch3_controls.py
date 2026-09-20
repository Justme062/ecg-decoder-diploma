# -*- coding: utf-8 -*-
"""
ch3_controls.py — честная перепроверка результатов главы 3 (раннее
предупреждение о V) с устранением конфаундеров, найденных при ревью:

  #4  Кластеризация эктопий. Окна раньше строились из ЛЮБЫХ ударов, поэтому
      окно перед V часто содержит другие эктопии, и «предсказание» может быть
      тривиальным «недавно уже была эктопия». Здесь:
        (а) baseline «только число эктопий в окне» — показывает, сколько AUC
            объясняется одной лишь недавней эктопией;
        (б) контроль на N-only окнах (все 10 ударов = нормальные N): если
            полюса/тайминг всё ещё предсказывают V — сигнал реальный
            (морфология нормальных ударов), а не кластеризация.
  #6  Заниженная неопределённость. Вместо одного фиксированного сплита —
      НАСТОЯЩАЯ межпациентная K-fold кросс-валидация (пациенты в тесте меняются
      от фолда к фолду). Разброс по фолдам отражает главную неопределённость.

Запуск (после обновлённого build_sequences.py, создавшего sequences.npz
с полями n_ectopic и all_normal):
    python ch3_controls.py            # 5 фолдов
    python ch3_controls.py -k 10
"""
import argparse, os, sys
import numpy as np
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WINDOW, FEAT_PER_BEAT = 10, 31
RR_IDX = [j * FEAT_PER_BEAT + 30 for j in range(WINDOW)]
MP_IDX = [j * FEAT_PER_BEAT + k for j in range(WINDOW) for k in range(30)]
ZDOM_COL = 10   # |доминирующего полюса| в векторе признаков одного удара


def make_folds(record_ids, k, seed):
    recs = np.array(sorted(set(record_ids.tolist())))
    rng = np.random.default_rng(seed)
    recs = recs[rng.permutation(len(recs))]
    return [set(a.tolist()) for a in np.array_split(recs, k)]


def cv_auc(flat, y, record_ids, folds, cols):
    """AUC логрега на признаках cols по фолдам; возвращает список AUC."""
    aucs = []
    for test_recs in folds:
        te = np.isin(record_ids, list(test_recs))
        tr = ~te
        if y[tr].sum() == 0 or y[te].sum() == 0 or y[te].sum() == te.sum():
            continue  # в train/test нет одного из классов — AUC не определён
        Xtr, Xte = flat[tr][:, cols], flat[te][:, cols]
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000, class_weight='balanced')
        clf.fit(sc.transform(Xtr), y[tr])
        p = clf.predict_proba(sc.transform(Xte))[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return aucs


def ms(a):
    a = np.array(a)
    if len(a) == 0:
        return "н/д (нет валидных фолдов)"
    sd = a.std(ddof=1) if len(a) > 1 else 0.0
    return f"{a.mean():.4f} ± {sd:.4f}  (n={len(a)} фолдов)"


def run_subset(name, mask, X_seq, y, record_ids, n_ect, folds, with_ectopy):
    flat = X_seq[mask].reshape(mask.sum(), -1)
    ys = y[mask]; recs = record_ids[mask]; ne = n_ect[mask]
    print(f"\n{'='*74}\nПОДВЫБОРКА: {name}")
    print(f"  окон: {mask.sum()},  положительных (следующий=V): {ys.sum()} "
          f"({ys.mean()*100:.2f}%),  записей: {len(set(recs.tolist()))}")
    if ys.sum() < 20:
        print("  ! Слишком мало положительных окон — оценки ненадёжны.")
    if with_ectopy:
        ect_feat = ne.reshape(-1, 1).astype(np.float32)
        # временно подставим ect как единственный столбец через отдельный путь
        aucs = []
        for test_recs in folds:
            te = np.isin(recs, list(test_recs)); tr = ~te
            if ys[tr].sum() == 0 or ys[te].sum() == 0 or ys[te].sum() == te.sum():
                continue
            sc = StandardScaler().fit(ect_feat[tr])
            clf = LogisticRegression(max_iter=2000, class_weight='balanced')
            clf.fit(sc.transform(ect_feat[tr]), ys[tr])
            p = clf.predict_proba(sc.transform(ect_feat[te]))[:, 1]
            aucs.append(roc_auc_score(ys[te], p))
        print(f"  AUC | только ЧИСЛО ЭКТОПИЙ в окне : {ms(aucs)}")
    print(f"  AUC | только тайминг (RR)          : {ms(cv_auc(flat, ys, recs, folds, RR_IDX))}")
    print(f"  AUC | только полюса (МП)           : {ms(cv_auc(flat, ys, recs, folds, MP_IDX))}")
    print(f"  AUC | полюса + RR (без порядка)    : {ms(cv_auc(flat, ys, recs, folds, list(range(flat.shape[1]))))}")
    # механистика: средний |z| доминирующего полюса по окну
    zdom = X_seq[mask][:, :, ZDOM_COL].mean(axis=1)
    if ys.sum() > 0 and (ys == 0).sum() > 0:
        print(f"  |z| дом. полюса (сред. по окну): перед V = {zdom[ys==1].mean():.4f} "
              f"(n={int((ys==1).sum())})  vs  перед не-V = {zdom[ys==0].mean():.4f} "
              f"(n={int((ys==0).sum())})  |  разница = {(zdom[ys==1].mean()-zdom[ys==0].mean()):+.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-k', type=int, default=5, help='число фолдов межпациентной CV')
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()

    npz = os.path.join(SCRIPT_DIR, 'sequences.npz')
    d = np.load(npz, allow_pickle=True)
    X_seq, y = d['X_seq'], d['y_next']
    record_ids = d['record_ids']
    if 'n_ectopic' not in d or 'all_normal' not in d:
        print("ОШИБКА: в sequences.npz нет полей n_ectopic/all_normal.\n"
              "Сначала перезапустите обновлённый build_sequences.py.")
        return
    n_ect, all_normal = d['n_ectopic'], d['all_normal']

    folds = make_folds(record_ids, a.k, a.seed)
    print("="*74)
    print(f"КОНТРОЛЬНАЯ ПРОВЕРКА ГЛАВЫ 3  |  межпациентная {a.k}-fold CV (seed={a.seed})")
    print(f"Тестовые пациенты меняются от фолда к фолду (в отличие от прежнего")
    print(f"единственного фиксированного сплита).")
    print("="*74)
    print(f"Всего окон: {len(y)},  из них N-only (все 10 ударов нормальные): "
          f"{(all_normal==1).sum()} ({(all_normal==1).mean()*100:.1f}%)")

    # ПОЛНАЯ выборка (как в дипломе) — с baseline по числу эктопий
    run_subset("ПОЛНАЯ (как в текущей главе 3)",
               np.ones(len(y), bool), X_seq, y, record_ids, n_ect, folds, with_ectopy=True)

    # N-ONLY: окно целиком из нормальных ударов (ключевой контроль #4)
    run_subset("N-ONLY: все 10 ударов окна = N (контроль кластеризации эктопий)",
               all_normal == 1, X_seq, y, record_ids, n_ect, folds, with_ectopy=False)

    print("\n" + "="*74)
    print("КАК ЧИТАТЬ:")
    print(" • Если на ПОЛНОЙ выборке 'только число эктопий' даёт AUC, близкий к")
    print("   'полюсам', — значит превосходство полюсов во многом объясняется")
    print("   недавней эктопией, а не тонкой морфологией.")
    print(" • Если на N-ONLY полюса/тайминг всё ещё дают AUC заметно выше 0.5 —")
    print("   сигнал реальный (морфология нормальных ударов). Если ~0.5 —")
    print("   headline-вывод главы 3 держался на кластеризации эктопий.")
    print("="*74)


if __name__ == '__main__':
    main()
