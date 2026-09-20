# -*- coding: utf-8 -*-
"""
diagnose_domain_shift.py

Диагностика причины резкой просадки качества на INCART (Macro F1 упал
с 0.87-0.89 на MIT-BIH до 0.45-0.46 на INCART — намного больше, чем
ожидаемый умеренный domain gap). Строит визуальное сравнение сегментов
N/S/V-классов из MIT-BIH и INCART бок о бок, плюс базовую статистику
(среднее, знак доминирующего отклонения) — чтобы увидеть, не искажена
ли морфология сигнала на этапе препроцессинга INCART (инверсия
полярности отведения, артефакты ресемплинга и т.п.).

Запуск (из той же папки, где лежит X.npy/y.npy от MIT-BIH и где доступен
INCART_DIR с записями):
    python diagnose_domain_shift.py
"""

import os
import numpy as np
import wfdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import resample_poly

from pan_tompkins import detect_r_peaks, map_labels_to_detected_peaks, correct_polarity
from cross_dataset_incart import (
    bandpass_filter, pick_lead_ii, AAMI, FS_INCART, FS_TARGET, BEFORE, AFTER
)

# папка скрипта — используем как базу для путей, чтобы результат не зависел
# от того, откуда фактически запущен python (рабочая директория может
# отличаться от папки со скриптом, особенно при запуске из VS Code)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# по умолчанию ищем INCART в подпапке incart-data/files рядом со скриптом;
# если у тебя данные лежат в другом месте — просто впиши сюда абсолютный путь
INCART_DIR = os.path.join(SCRIPT_DIR, 'incart-data', 'files')
INCART_PROBE_RECORDS = ['I01', 'I02', 'I03']


def get_incart_segments(data_dir, records, max_per_class=6):
    """Быстрая обработка нескольких записей INCART для диагностики
    (не всех 75 — только для визуального сравнения)."""
    by_class = {0: [], 1: [], 2: []}  # N, S, V

    for rec_id in records:
        path = os.path.join(data_dir, rec_id)
        try:
            record = wfdb.rdrecord(path)
            ann = wfdb.rdann(path, 'atr')
        except Exception as e:
            print(f"  Пропускаю {rec_id}: {e}")
            continue

        lead_idx = pick_lead_ii(record)
        raw = np.nan_to_num(record.p_signal[:, lead_idx].astype(np.float64))
        filtered = bandpass_filter(raw, fs=FS_INCART)
        filtered, was_flipped = correct_polarity(filtered, ann.sample, symbols=ann.symbol, fs=FS_INCART)
        if was_flipped:
            print(f"      [{rec_id}] обнаружена инверсия полярности -> исправлено")
        sig = resample_poly(filtered, FS_TARGET, FS_INCART).astype(np.float32)
        ann_sample_resampled = (np.asarray(ann.sample) * FS_TARGET / FS_INCART).astype(np.int64)

        detected = detect_r_peaks(sig, FS_TARGET)
        matched = map_labels_to_detected_peaks(detected, ann_sample_resampled, ann.symbol, FS_TARGET, tolerance_ms=75)

        for peak, sym in matched:
            if sym not in AAMI:
                continue
            cls = AAMI[sym]
            if cls not in by_class or len(by_class[cls]) >= max_per_class:
                continue
            start, end = peak - BEFORE, peak + AFTER
            if start < 0 or end > len(sig):
                continue
            raw_seg = sig[start:end].copy()          # ДО нормализации — для анализа знака/масштаба
            seg = raw_seg.copy()
            rng = seg.max() - seg.min()
            if rng > 1e-6:
                seg = (seg - seg.min()) / rng
            by_class[cls].append((raw_seg, seg))

        if all(len(v) >= max_per_class for v in by_class.values()):
            break

    return by_class


def get_mitbih_segments(x_path='X.npy', y_path='y.npy', max_per_class=6):
    X = np.load(x_path)
    y = np.load(y_path)
    by_class = {0: [], 1: [], 2: []}
    for cls in by_class:
        idx = np.where(y == cls)[0][:max_per_class]
        for i in idx:
            seg = X[i]  # уже нормализован в [0,1] в исходном пайплайне
            by_class[cls].append((seg, seg))
    return by_class


def plot_comparison(mit_segs, incart_segs, cls_name, cls_idx, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax = axes[0]
    for _, seg in mit_segs.get(cls_idx, [])[:6]:
        ax.plot(seg, alpha=0.6)
    ax.set_title(f'MIT-BIH — класс {cls_name} (нормализовано в [0,1])')
    ax.set_ylabel('амплитуда')
    ax.grid(alpha=0.3)

    ax = axes[1]
    for _, seg in incart_segs.get(cls_idx, [])[:6]:
        ax.plot(seg, alpha=0.6)
    ax.set_title(f'INCART — класс {cls_name} (нормализовано в [0,1])')
    ax.set_xlabel('отсчёты (0-180, 360 Гц)')
    ax.set_ylabel('амплитуда')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"  Сохранено: {out_path}")


def print_polarity_stats(segs_dict, name):
    print(f"\n--- Статистика знака доминирующего отклонения ({name}) ---")
    for cls, items in segs_dict.items():
        if not items:
            continue
        signs = []
        for raw_seg, _ in items:
            centered = raw_seg - np.median(raw_seg)
            # знак отсчёта с максимальным по модулю отклонением от медианы
            dominant_idx = np.argmax(np.abs(centered))
            signs.append(np.sign(centered[dominant_idx]))
        pct_positive = 100 * np.mean(np.array(signs) > 0)
        print(f"  Класс {cls}: доля сегментов с ПОЛОЖИТЕЛЬНЫМ доминирующим "
              f"зубцом = {pct_positive:.0f}% (из {len(items)} примеров)")


def main():
    print("Обрабатываю несколько записей INCART для диагностики...")
    incart_segs = get_incart_segments(INCART_DIR, INCART_PROBE_RECORDS)

    print("Загружаю сегменты MIT-BIH из X.npy/y.npy для сравнения...")
    mit_segs = get_mitbih_segments()

    class_names = {0: 'N', 1: 'S', 2: 'V'}
    for cls_idx, cls_name in class_names.items():
        out_path = os.path.join(SCRIPT_DIR, f'compare_{cls_name}.png')
        plot_comparison(mit_segs, incart_segs, cls_name, cls_idx, out_path)

    print_polarity_stats(incart_segs, 'INCART')
    print_polarity_stats(mit_segs, 'MIT-BIH')

    print("\nГотово. Посмотри на compare_N.png, compare_S.png, compare_V.png —")
    print("если форма QRS в INCART выглядит перевёрнутой (пик направлен вниз")
    print("вместо вверх) или сильно \"дребезжащей\"/зашумлённой по сравнению")
    print("с MIT-BIH — это и есть причина просадки качества.")


if __name__ == '__main__':
    main()
