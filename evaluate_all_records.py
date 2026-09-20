# -*- coding: utf-8 -*-
"""
evaluate_all_records.py

Прогоняет детектор Пан-Томпкинса (pan_tompkins.py) по всем 48 записям
MIT-BIH Arrhythmia Database, сравнивает найденные R-пики с эталонной
разметкой (.atr) и собирает итоговую таблицу Sensitivity/PPV — готовый
материал для главы 2 диплома.

ЗАПУСКАЕТСЯ У ТЕБЯ ЛОКАЛЬНО, там где уже лежит скачанная MIT-BIH
(в той же папке, где сейчас лежит main.py и куда указывает data_dir).

Использование:
    python evaluate_all_records.py

Результат:
    - r_peak_detection_results.csv   — таблица по каждой записи
    - r_peak_detection_summary.md    — сводная markdown-таблица для диплома
    - вывод в консоль по каждой записи и итоговое среднее
"""

import os
import numpy as np
import wfdb
import pandas as pd

from pan_tompkins import detect_r_peaks, evaluate_detection

# ───── параметры (совпадают с main.py курсовой) ─────
FS = 360
DATA_DIR = 'mitdb'   # записи MIT-BIH из get_data.py (поправь, если лежат в другом месте)

RECORDS = [
    '100', '101', '102', '103', '104', '105', '106', '107',
    '108', '109', '111', '112', '113', '114', '115', '116',
    '117', '118', '119', '121', '122', '123', '124', '200',
    '201', '202', '203', '205', '207', '208', '209', '210',
    '212', '213', '214', '215', '217', '219', '220', '221',
    '222', '223', '228', '230', '231', '232', '233', '234'
]

# импортируем ту же фильтрацию, что и в main.py курсовой, чтобы условия
# были идентичны основному пайплайну (детектор рассчитан на уже
# отфильтрованный 0.5-40 Гц сигнал)
from scipy.signal import butter, filtfilt


def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=FS, order=4):
    nyq = fs / 2
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal)


def main():
    rows = []
    print("=" * 78)
    print("ОЦЕНКА ДЕТЕКТОРА R-ПИКОВ (ПАН-ТОМПКИНС) НА ВСЕХ 48 ЗАПИСЯХ MIT-BIH")
    print("=" * 78)

    for rec_id in RECORDS:
        path = os.path.join(DATA_DIR, rec_id)
        try:
            record = wfdb.rdrecord(path)
            ann = wfdb.rdann(path, 'atr')
        except Exception as e:
            print(f"  [{rec_id}] пропущена: {e}")
            continue

        raw = record.p_signal[:, 0].astype(np.float32)
        sig = bandpass_filter(raw).astype(np.float32)

        # эталон: позиции ВСЕХ аннотаций (включая не-ударные метки типа '+'
        # можно отфильтровать, но для честной оценки детектора как есть —
        # оставляем все точки разметки .atr, это и есть Se/PPV детекции ударов)
        true_peaks = ann.sample

        detected = detect_r_peaks(sig, FS)
        metrics = evaluate_detection(true_peaks, detected, FS, tolerance_ms=75)

        rows.append({
            'record': rec_id,
            'n_true': metrics['n_true'],
            'n_detected': metrics['n_detected'],
            'tp': metrics['tp'],
            'fp': metrics['fp'],
            'fn': metrics['fn'],
            'sensitivity_%': round(metrics['sensitivity'] * 100, 3),
            'ppv_%': round(metrics['ppv'] * 100, 3),
        })

        print(f"  [{rec_id}] истинных={metrics['n_true']:4d}  найдено={metrics['n_detected']:4d}  "
              f"Se={metrics['sensitivity']*100:6.2f}%  PPV={metrics['ppv']*100:6.2f}%")

    if not rows:
        print("\nНичего не обработано — проверь DATA_DIR (путь к MIT-BIH).")
        return

    df = pd.DataFrame(rows)
    df.to_csv('r_peak_detection_results.csv', index=False)

    # ───── агрегированные показатели по всей базе ─────
    total_tp = df['tp'].sum()
    total_fp = df['fp'].sum()
    total_fn = df['fn'].sum()
    overall_se = total_tp / (total_tp + total_fn) * 100 if (total_tp + total_fn) else 0
    overall_ppv = total_tp / (total_tp + total_fp) * 100 if (total_tp + total_fp) else 0

    mean_se = df['sensitivity_%'].mean()
    mean_ppv = df['ppv_%'].mean()
    worst_se_row = df.loc[df['sensitivity_%'].idxmin()]
    worst_ppv_row = df.loc[df['ppv_%'].idxmin()]

    print("\n" + "=" * 78)
    print("ИТОГО ПО ВСЕЙ БАЗЕ (48 записей)")
    print("=" * 78)
    print(f"Суммарно: TP={total_tp}  FP={total_fp}  FN={total_fn}")
    print(f"Overall Sensitivity (микро-усреднение) = {overall_se:.3f}%")
    print(f"Overall PPV         (микро-усреднение) = {overall_ppv:.3f}%")
    print(f"Mean Sensitivity    (по записям)       = {mean_se:.3f}%")
    print(f"Mean PPV            (по записям)       = {mean_ppv:.3f}%")
    print(f"Худшая запись по Se:  {worst_se_row['record']} ({worst_se_row['sensitivity_%']:.2f}%)")
    print(f"Худшая запись по PPV: {worst_ppv_row['record']} ({worst_ppv_row['ppv_%']:.2f}%)")

    # ───── markdown-таблица для вставки в текст диплома ─────
    with open('r_peak_detection_summary.md', 'w', encoding='utf-8') as f:
        f.write("# Результаты детекции R-пиков алгоритмом Пан-Томпкинса на MIT-BIH\n\n")
        f.write(f"Обработано записей: {len(df)} из 48\n\n")
        f.write("## Сводные показатели\n\n")
        f.write("| Показатель | Значение |\n|---|---|\n")
        f.write(f"| Overall Sensitivity (микро) | {overall_se:.3f}% |\n")
        f.write(f"| Overall PPV (микро) | {overall_ppv:.3f}% |\n")
        f.write(f"| Средняя Sensitivity по записям | {mean_se:.3f}% |\n")
        f.write(f"| Средняя PPV по записям | {mean_ppv:.3f}% |\n")
        f.write(f"| Суммарно TP / FP / FN | {total_tp} / {total_fp} / {total_fn} |\n\n")
        f.write("## Результаты по каждой записи\n\n")
        f.write("| Запись | Истинных | Найдено | TP | FP | FN | Se, % | PPV, % |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for _, r in df.iterrows():
            f.write(f"| {r['record']} | {r['n_true']} | {r['n_detected']} | {r['tp']} | "
                     f"{r['fp']} | {r['fn']} | {r['sensitivity_%']:.2f} | {r['ppv_%']:.2f} |\n")

    print("\nСохранено:")
    print("  r_peak_detection_results.csv   (сырые данные по записям)")
    print("  r_peak_detection_summary.md    (готовая markdown-таблица для диплома)")


if __name__ == '__main__':
    main()
