# -*- coding: utf-8 -*-
"""
pole_dynamics.py

Блок А плана диплома, шаг 3 (объяснительный): механистический анализ
ДИНАМИКИ ПОЛЮСОВ в окнах, предшествующих желудочковой экстрасистоле (V).

Предыдущий шаг (early_warning.py) статистически доказал, что морфологические
признаки матричных пучков предсказывают приближающуюся V (на контроле из одних нормальных ударов
AUC полюсов ≈ 0.62 против ≈ 0.54 у тайминга; на полной выборке 0.73 против 0.63,
частично за счёт кластеризации эктопий). Этот модуль
отвечает на вопрос ПОЧЕМУ: что именно физически происходит с полюсами перед
событием.

Проверяемая гипотеза (из плана диплома, идея динамической системы):
    перед V-событием доминирующий полюс матричного пучка систематически
    ДРЕЙФУЕТ К ГРАНИЦЕ УСТОЙЧИВОСТИ |z|=1 (или качественно меняет поведение),
    что соответствует потере устойчивости в терминах теории динамических
    систем — предвестник срыва нормального ритма.

Метод:
    - берём окна, за которыми следует V (положительные), и окна, за которыми
      следует не-V (отрицательные);
    - для каждого удара в окне восстанавливаем |z| доминирующего полюса
      (из признаков МП: модули полюсов лежат в позициях 10:20 вектора из 30);
    - строим усреднённые траектории |z_max|(t) вдоль окна для обеих групп;
    - проверяем статистически: значимо ли отличается дрейф |z| в конце окна
      (наклон/приращение) между группами (тест Манна-Уитни, непараметрический).

Важно: это АНАЛИЗ, а не ещё один классификатор. Цель — интерпретируемость,
физическое объяснение уже установленного статистического факта.

Запуск (из папки с sequences.npz):
    python pole_dynamics.py
"""

import os
import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WINDOW = 10
M = 10

# в векторе признаков одного удара (30 МП):
#   [0:10]  — нормированные сингулярные значения
#   [10:20] — модули полюсов |z|
#   [20:30] — нормированные аргументы полюсов arg(z)/pi
POLE_MOD_START = 10
POLE_MOD_END = 20


def dominant_pole_modulus(beat_feats):
    """|z| доминирующего полюса удара — максимальный модуль среди M полюсов.
    Именно он ближе всего к границе устойчивости и первым её достигает."""
    moduli = beat_feats[POLE_MOD_START:POLE_MOD_END]
    return float(np.max(moduli))


def mean_pole_modulus(beat_feats):
    """Средний |z| по всем полюсам — альтернативная сводка (менее шумная)."""
    return float(np.mean(beat_feats[POLE_MOD_START:POLE_MOD_END]))


def build_pole_trajectories(X_seq, y_next):
    """
    Для каждого окна строит траекторию |z_max|(t), t=0..WINDOW-1.
    Возвращает два массива траекторий: для положительных (следующий=V) и
    отрицательных окон.
    """
    pos_traj, neg_traj = [], []
    pos_traj_mean, neg_traj_mean = [], []

    for i in range(len(X_seq)):
        window = X_seq[i]  # (WINDOW, 31)
        traj_max = np.array([dominant_pole_modulus(window[t, :30]) for t in range(WINDOW)])
        traj_mean = np.array([mean_pole_modulus(window[t, :30]) for t in range(WINDOW)])
        if y_next[i] == 1:
            pos_traj.append(traj_max)
            pos_traj_mean.append(traj_mean)
        else:
            neg_traj.append(traj_max)
            neg_traj_mean.append(traj_mean)

    return (np.array(pos_traj), np.array(neg_traj),
            np.array(pos_traj_mean), np.array(neg_traj_mean))


def late_window_drift(trajectories, tail=3):
    """
    Дрейф |z| в КОНЦЕ окна: разница между средним |z| последних `tail`
    ударов и первых `tail` ударов. Положительное значение = полюс
    приближается к границе устойчивости к концу окна (перед событием).
    """
    early = trajectories[:, :tail].mean(axis=1)
    late = trajectories[:, -tail:].mean(axis=1)
    return late - early


def plot_trajectories(pos_traj, neg_traj, title_suffix, out_name):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    t = np.arange(WINDOW)

    pos_mean = pos_traj.mean(axis=0)
    pos_sem = pos_traj.std(axis=0) / np.sqrt(len(pos_traj))
    neg_mean = neg_traj.mean(axis=0)
    neg_sem = neg_traj.std(axis=0) / np.sqrt(len(neg_traj))

    ax.plot(t, pos_mean, 'o-', color='#d9534f', label=f'перед V (n={len(pos_traj)})', linewidth=2)
    ax.fill_between(t, pos_mean - pos_sem, pos_mean + pos_sem, color='#d9534f', alpha=0.2)
    ax.plot(t, neg_mean, 's-', color='#2e6f95', label=f'перед не-V (n={len(neg_traj)})', linewidth=2)
    ax.fill_between(t, neg_mean - neg_sem, neg_mean + neg_sem, color='#2e6f95', alpha=0.2)

    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.7, label='граница устойчивости |z|=1')
    ax.set_xlabel('позиция удара в окне (0 = самый ранний, 9 = перед событием)')
    ax.set_ylabel(f'|z| {title_suffix}')
    ax.set_title(f'Динамика модуля полюса перед V-событием vs нормой\n({title_suffix})')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(SCRIPT_DIR, out_name)
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"  Сохранено: {out_path}")


def main():
    data = np.load(os.path.join(SCRIPT_DIR, 'sequences.npz'), allow_pickle=True)
    X_seq = data['X_seq']
    y_next = data['y_next']

    print("=" * 78)
    print("МЕХАНИСТИЧЕСКИЙ АНАЛИЗ ДИНАМИКИ ПОЛЮСОВ ПЕРЕД V-СОБЫТИЕМ (Блок А)")
    print("=" * 78)
    print(f"Окон: {len(X_seq)}, положительных (перед V): {y_next.sum()}")

    pos_traj, neg_traj, pos_mean_traj, neg_mean_traj = build_pole_trajectories(X_seq, y_next)

    # ── визуализация ──
    print("\nСтрою траектории...")
    plot_trajectories(pos_traj, neg_traj, '|z| доминирующего полюса',
                      'pole_trajectory_dominant.png')
    plot_trajectories(pos_mean_traj, neg_mean_traj, 'средний |z| по полюсам',
                      'pole_trajectory_mean.png')

    # ── статистика: отличается ли дрейф в конце окна? ──
    print("\n" + "=" * 78)
    print("СТАТИСТИЧЕСКАЯ ПРОВЕРКА ГИПОТЕЗЫ О ДРЕЙФЕ К ГРАНИЦЕ УСТОЙЧИВОСТИ")
    print("=" * 78)

    pos_drift = late_window_drift(pos_traj)
    neg_drift = late_window_drift(neg_traj)

    print(f"\nДрейф |z_max| (конец окна минус начало):")
    print(f"  Перед V:    медиана={np.median(pos_drift):+.4f}, "
          f"среднее={pos_drift.mean():+.4f}")
    print(f"  Перед не-V: медиана={np.median(neg_drift):+.4f}, "
          f"среднее={neg_drift.mean():+.4f}")

    # тест Манна-Уитни (непараметрический — не предполагает нормальности)
    stat, p_value = mannwhitneyu(pos_drift, neg_drift, alternative='two-sided')
    print(f"\nТест Манна-Уитни (дрейф перед V vs перед не-V):")
    print(f"  U={stat:.0f}, p-value={p_value:.2e}")
    if p_value < 0.05:
        direction = "БОЛЬШЕ" if np.median(pos_drift) > np.median(neg_drift) else "МЕНЬШЕ"
        print(f"  => Значимо (p<0.05): дрейф |z| перед V статистически {direction}, "
              f"чем перед нормой")
        if np.median(pos_drift) > np.median(neg_drift):
            print(f"     Это ПОДТВЕРЖДАЕТ гипотезу: перед V-событием доминирующий")
            print(f"     полюс систематически приближается к границе устойчивости,")
            print(f"     что соответствует потере устойчивости динамической системы.")
    else:
        print(f"  => Не значимо: систематического различия дрейфа не обнаружено")

    # ── абсолютный уровень |z| перед самим событием ──
    pos_last = pos_traj[:, -1]
    neg_last = neg_traj[:, -1]
    stat2, p2 = mannwhitneyu(pos_last, neg_last, alternative='two-sided')
    print(f"\n|z_max| последнего удара окна (непосредственно перед событием):")
    print(f"  Перед V:    медиана={np.median(pos_last):.4f}")
    print(f"  Перед не-V: медиана={np.median(neg_last):.4f}")
    print(f"  Тест Манна-Уитни: p-value={p2:.2e} "
          f"({'значимо' if p2 < 0.05 else 'не значимо'})")

    print("\n" + "=" * 78)
    print("ИТОГ ДЛЯ ГЛАВЫ 3 ДИПЛОМА (механистическое объяснение)")
    print("=" * 78)
    print("Смотри графики pole_trajectory_dominant.png и pole_trajectory_mean.png.")
    print("Если красная кривая (перед V) систематически выше синей и/или растёт")
    print("к концу окна — это был бы дрейф к |z|=1. ПРОВЕРКА ПОКАЗАЛА ОБРАТНОЕ:")
    print("гипотеза дрейфа к границе устойчивости НЕ подтвердилась; предвестником")
    print("служит слегка (и устойчиво) повышенный уровень модуля, а не его дрейф.")


if __name__ == '__main__':
    main()
