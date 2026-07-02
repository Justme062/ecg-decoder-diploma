# -*- coding: utf-8 -*-
"""
statistical_significance.py

Проверка статистической значимости прироста гибридной модели (CNN + Matrix
Pencil) над базовой CNN. Работает на уже сохранённых предсказаниях одного
train/test split (pred_base.npy, pred_hybrid.npy, true_labels.npy) —
переобучение моделей не требуется.

Два теста:
  1. McNemar's test — сравнивает характер ошибок двух моделей на ОДНИХ И ТЕХ
     ЖЕ примерах (парный тест). Проверяет гипотезу H0: обе модели ошибаются
     одинаково часто на разных примерах (нет систематического различия).
  2. Bootstrap-доверительный интервал для разницы Macro F1 (N,S,V,F, без Q —
     класс Q исключён как статистически вырожденный, support=1 на тесте).

Класс Q (индекс 4 в разметке AAMI: N=0,S=1,V=2,F=3,Q=4) исключается из
анализа полностью — как из McNemar (примеры этого класса), так и из
bootstrap (Macro F1 считается только по N,S,V,F).

Запуск:
    python statistical_significance.py
(в той же папке, где лежат pred_base.npy, pred_hybrid.npy, true_labels.npy)
"""

import numpy as np
from scipy.stats import chi2
from sklearn.metrics import f1_score

RELEVANT_CLASSES = [0, 1, 2, 3]  # N, S, V, F — класс Q (4) исключён
N_BOOTSTRAP = 5000
RANDOM_SEED = 42


def mcnemar_test(y_true, pred_a, pred_b, relevant_classes=None):
    """
    McNemar's test с поправкой на непрерывность для сравнения двух
    классификаторов на одном и том же тестовом наборе.

    b = число примеров, где модель A верна, а модель B ошиблась
    c = число примеров, где модель A ошиблась, а модель B верна

    H0: b и c распределены одинаково (модели ошибаются в одних и тех же
    местах с равной вероятностью в обе стороны) -> различие моделей случайно.
    Малое p-value отвергает H0 -> различие моделей статистически значимо.
    """
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)

    if relevant_classes is not None:
        mask = np.isin(y_true, relevant_classes)
        y_true, pred_a, pred_b = y_true[mask], pred_a[mask], pred_b[mask]

    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)

    b = int(np.sum(correct_a & ~correct_b))   # A верна, B ошиблась
    c = int(np.sum(~correct_a & correct_b))   # A ошиблась, B верна

    n = b + c
    if n == 0:
        return {'b': b, 'c': c, 'statistic': 0.0, 'p_value': 1.0, 'n_discordant': 0}

    statistic = (abs(b - c) - 1) ** 2 / n   # с поправкой на непрерывность
    p_value = 1 - chi2.cdf(statistic, df=1)

    return {'b': b, 'c': c, 'statistic': statistic, 'p_value': p_value, 'n_discordant': n}


def bootstrap_macro_f1_diff(y_true, pred_a, pred_b, relevant_classes,
                             n_bootstrap=N_BOOTSTRAP, seed=RANDOM_SEED):
    """
    Bootstrap-оценка разницы Macro F1 (модель B минус модель A) на классах
    relevant_classes. Пересемплирует индексы тестовой выборки с возвращением
    n_bootstrap раз, считает разницу F1 на каждой копии -> получаем
    эмпирическое распределение разницы, из него 95% ДИ и приближённый p-value.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)

    mask = np.isin(y_true, relevant_classes)
    y_true, pred_a, pred_b = y_true[mask], pred_a[mask], pred_b[mask]
    n = len(y_true)

    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        f1_a = f1_score(y_true[idx], pred_a[idx], labels=relevant_classes, average='macro', zero_division=0)
        f1_b = f1_score(y_true[idx], pred_b[idx], labels=relevant_classes, average='macro', zero_division=0)
        diffs[i] = f1_b - f1_a

    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    mean_diff = diffs.mean()
    # доля бутстрэп-повторов, где разница <= 0 (гибрид не лучше базовой)
    p_approx = np.mean(diffs <= 0)

    return {
        'mean_diff': mean_diff,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'p_approx_le_zero': p_approx,
        'diffs': diffs,
    }


def main():
    y_true = np.load('true_labels.npy')
    pred_base = np.load('pred_base.npy')
    pred_hybrid = np.load('pred_hybrid.npy')

    print("=" * 72)
    print("СТАТИСТИЧЕСКАЯ ЗНАЧИМОСТЬ ПРИРОСТА ГИБРИДНОЙ МОДЕЛИ НАД БАЗОВОЙ CNN")
    print("=" * 72)
    print(f"Размер тестовой выборки: {len(y_true)}")
    print("Классы в анализе: N, S, V, F (класс Q исключён — support=1, вырожден)\n")

    # ───── 1. McNemar's test ─────
    mc = mcnemar_test(y_true, pred_base, pred_hybrid, relevant_classes=RELEVANT_CLASSES)
    print("-" * 72)
    print("1. McNemar's test (парное сравнение ошибок на одних и тех же примерах)")
    print("-" * 72)
    print(f"   Примеров, где база верна, гибрид ошибся (b): {mc['b']}")
    print(f"   Примеров, где база ошиблась, гибрид верен (c): {mc['c']}")
    print(f"   Несовпадающих примеров всего (b+c): {mc['n_discordant']}")
    print(f"   Статистика хи-квадрат (с поправкой на непрерывность): {mc['statistic']:.4f}")
    print(f"   p-value: {mc['p_value']:.6f}")
    if mc['p_value'] < 0.05:
        direction = "гибрид лучше" if mc['c'] > mc['b'] else "база лучше"
        print(f"   => Значимо на уровне 0.05 (различие моделей неслучайно, {direction})")
    else:
        print("   => Не значимо на уровне 0.05 (недостаточно доказательств различия)")

    # ───── 2. Bootstrap CI для разницы Macro F1 ─────
    print("\n" + "-" * 72)
    print(f"2. Bootstrap-доверительный интервал для разницы Macro F1 ({N_BOOTSTRAP} повторов)")
    print("-" * 72)
    bs = bootstrap_macro_f1_diff(y_true, pred_base, pred_hybrid, RELEVANT_CLASSES)
    print(f"   Средняя разница (гибрид - база): {bs['mean_diff']*100:+.3f} п.п.")
    print(f"   95% доверительный интервал: [{bs['ci_low']*100:+.3f}, {bs['ci_high']*100:+.3f}] п.п.")
    print(f"   Доля бутстрэп-повторов, где гибрид НЕ лучше базы: {bs['p_approx_le_zero']*100:.2f}%")
    if bs['ci_low'] > 0:
        print("   => 0 НЕ входит в 95% ДИ и интервал целиком положительный:")
        print("      прирост гибридной модели статистически значим на уровне 0.05")
    elif bs['ci_high'] < 0:
        print("   => 0 НЕ входит в 95% ДИ, интервал целиком отрицательный:")
        print("      базовая модель значимо лучше гибридной")
    else:
        print("   => 0 входит в 95% ДИ: различие статистически не значимо")

    print("\n" + "=" * 72)
    print("ИТОГ ДЛЯ ГЛАВЫ 2 ДИПЛОМА")
    print("=" * 72)
    print(f"Прирост Macro F1 (N,S,V,F): {bs['mean_diff']*100:+.2f} п.п., "
          f"95% ДИ [{bs['ci_low']*100:+.2f}; {bs['ci_high']*100:+.2f}] п.п.")
    print(f"McNemar's test: p = {mc['p_value']:.4f}")


if __name__ == '__main__':
    main()
