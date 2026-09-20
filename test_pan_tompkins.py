# -*- coding: utf-8 -*-
"""
test_pan_tompkins.py

Валидация детектора R-пиков (pan_tompkins.py) на синтетическом сигнале
с ТОЧНО известными позициями пиков — это стандартная практика при разработке
детектора QRS до прогона на реальной базе данных.

Логика: строим сигнал из последовательности QRS-подобных всплесков
(гауссова форма с крутым фронтом) в позициях, которые мы сами задаём,
добавляем реалистичный дрейф изолинии + гауссовский шум разного уровня,
затем прогоняем detect_r_peaks() и сравниваем найденные позиции с истинными.

Реальная проверка на MIT-BIH (с разметкой .atr как эталоном) выполняется
отдельно, на компьютере с загруженной базой данных — см. инструкцию в конце файла.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pan_tompkins import detect_r_peaks, evaluate_detection

FS = 360  # частота дискретизации MIT-BIH


def make_synthetic_ecg(duration_s=60, fs=FS, hr_bpm=75, hr_jitter=0.08,
                        noise_std=0.0, baseline_wander=True, seed=42):
    """
    Генерирует синтетический ЭКГ-подобный сигнал с известными R-пиками.

    QRS моделируется как узкий асимметричный всплеск (сумма двух гауссиан
    разного знака и ширины — грубая, но достаточная для тестирования
    детектора имитация комплекса QRS).
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    sig = np.zeros(n)

    mean_rr = 60.0 / hr_bpm  # средний интервал в секундах
    true_peaks = []
    pos_s = 0.5  # первый удар не в самом начале
    while pos_s < duration_s - 0.5:
        rr = mean_rr * (1 + rng.normal(0, hr_jitter))
        rr = max(rr, 0.25)  # физиологический предел ЧСС ~240 уд/мин
        pos_s += rr
        if pos_s >= duration_s - 0.5:
            break
        true_peaks.append(pos_s)

    for peak_t in true_peaks:
        idx = int(peak_t * fs)
        # QRS: узкий острый пик + более широкая небольшая P/T волна вокруг
        window = np.arange(-int(0.08 * fs), int(0.08 * fs))
        valid = (idx + window >= 0) & (idx + window < n)
        w = window[valid]
        qrs = 1.0 * np.exp(-(w ** 2) / (2 * (0.008 * fs) ** 2))          # R-зубец
        qrs -= 0.25 * np.exp(-((w - 0.03 * fs) ** 2) / (2 * (0.015 * fs) ** 2))  # S-зубец
        sig[idx + w] += qrs

        # T-волна (широкая, положительная, после QRS)
        t_window = np.arange(int(0.1 * fs), int(0.35 * fs))
        valid_t = (idx + t_window >= 0) & (idx + t_window < n)
        wt = t_window[valid_t]
        sig[idx + wt] += 0.2 * np.exp(-((wt - 0.2 * fs) ** 2) / (2 * (0.04 * fs) ** 2))

    if baseline_wander:
        sig += 0.05 * np.sin(2 * np.pi * 0.3 * t + rng.uniform(0, 2 * np.pi))

    if noise_std > 0:
        sig += rng.normal(0, noise_std, size=n)

    true_peak_idx = np.array([int(p * fs) for p in true_peaks], dtype=np.int64)
    return sig.astype(np.float32), true_peak_idx


def run_experiment(noise_levels=(0.0, 0.02, 0.05, 0.1, 0.2), duration_s=120):
    print("=" * 70)
    print("ВАЛИДАЦИЯ ДЕТЕКТОРА R-ПИКОВ (Пан-Томпкинс) НА СИНТЕТИЧЕСКОМ СИГНАЛЕ")
    print("=" * 70)
    print(f"Длительность записи: {duration_s} с, fs={FS} Гц, ЧСС~75 уд/мин\n")

    results = []
    for noise_std in noise_levels:
        sig, true_peaks = make_synthetic_ecg(duration_s=duration_s, noise_std=noise_std)
        detected = detect_r_peaks(sig, FS)
        metrics = evaluate_detection(true_peaks, detected, FS, tolerance_ms=75)
        results.append((noise_std, metrics))
        print(f"noise_std={noise_std:.2f} | "
              f"истинных={metrics['n_true']:3d}  найдено={metrics['n_detected']:3d}  "
              f"TP={metrics['tp']:3d} FP={metrics['fp']:2d} FN={metrics['fn']:2d} | "
              f"Se={metrics['sensitivity']*100:6.2f}%  PPV={metrics['ppv']*100:6.2f}%")

    # Визуализация одного фрагмента сигнала с найденными и истинными пиками
    sig, true_peaks = make_synthetic_ecg(duration_s=8, noise_std=0.05, seed=1)
    detected = detect_r_peaks(sig, FS)
    t = np.arange(len(sig)) / FS

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, sig, color='#2e6f95', linewidth=1, label='синтетический сигнал (noise_std=0.05)')
    ax.scatter(true_peaks / FS, sig[true_peaks], color='green', marker='o',
               s=60, zorder=5, label='истинные R-пики')
    ax.scatter(detected / FS, sig[detected], color='red', marker='x',
               s=60, zorder=6, label='найдено детектором')
    ax.set_xlabel('время, с')
    ax.set_ylabel('амплитуда')
    ax.set_title('Детекция R-пиков алгоритмом Пан-Томпкинса на синтетическом сигнале')
    ax.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    _out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pan_tompkins_validation.png')
    plt.savefig(_out, dpi=130)
    print("\nГрафик сохранён: pan_tompkins_validation.png")

    return results


if __name__ == '__main__':
    run_experiment()

    print("\n" + "=" * 70)
    print("СЛЕДУЮЩИЙ ШАГ (выполняется на компьютере с загруженной MIT-BIH):")
    print("=" * 70)
    print("""
В load_dataset() (main.py) замени:

    r_peaks  = ann.sample
    sym_list = ann.symbol

на:

    from pan_tompkins import detect_r_peaks, evaluate_detection

    detected = detect_r_peaks(sig, FS)
    metrics = evaluate_detection(ann.sample, detected, FS, tolerance_ms=75)
    print(f"  {rec_id}: Se={metrics['sensitivity']*100:.2f}%  PPV={metrics['ppv']*100:.2f}%")

    # Дальше нужно сопоставить каждому найденному r_peak ближайшую метку
    # из ann.symbol (в пределах допуска tolerance_ms), т.к. detect_r_peaks
    # не знает классов - это отдельная функция map_labels_to_detected_peaks()

Накопи метрики Se/PPV по всем 48 записям -> получишь таблицу для главы 2
диплома ("средняя чувствительность детектора X%, точность Y%"), которая
заменит нынешнее допущение "детекция на уровне разметки экспертов".
""")
