# -*- coding: utf-8 -*-
"""
main.py — базовый слой конвейера (ДИПЛОМНАЯ версия).

Отличие от курсового main.py: R-пики находятся РЕАЛЬНЫМ алгоритмом
Пан–Томпкинса (модуль pan_tompkins.py), а не берутся из готовой разметки .atr.
Разметка используется только для присвоения меток классов найденным пикам.

Запуск из папки с записями MIT-BIH:
    python main.py           # -> X.npy, y.npy
"""
import os
import numpy as np
import wfdb
from scipy.signal import butter, filtfilt

from pan_tompkins import detect_r_peaks, map_labels_to_detected_peaks

# ───── параметры ─────
FS = 360          # частота дискретизации MIT-BIH
BEFORE = 90       # отсчётов до R-пика
AFTER = 90        # отсчётов после R-пика
SEG_LEN = BEFORE + AFTER  # 180 отсчётов = 0.5 с

# Маппинг меток MIT-BIH → 5 классов AAMI
AAMI = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,        # Normal
    'A': 1, 'a': 1, 'J': 1, 'S': 1,                # Supraventricular
    'V': 2, 'E': 2,                                # Ventricular
    'F': 3,                                        # Fusion
    '/': 4, 'f': 4, 'Q': 4,                        # Unknown
}

# 44 записи (исключены 102,104,107,217 — навязанный ритм, по рекомендации AAMI)
RECORDS = [
    '100', '101', '103', '105', '106', '108', '109', '111',
    '112', '113', '114', '115', '116', '117', '118', '119',
    '121', '122', '123', '124', '200', '201', '202', '203',
    '205', '207', '208', '209', '210', '212', '213', '214',
    '215', '219', '220', '221', '222', '223', '228', '230',
    '231', '232', '233', '234'
]


def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=FS, order=4):
    """Полосовой фильтр 0.5–40 Гц (нулевой фазовый сдвиг) — дрейф изолинии + ВЧ-шум."""
    nyq = fs / 2
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal)


def load_dataset(data_dir='.'):
    """
    Загружает все записи, фильтрует, ДЕТЕКТИРУЕТ R-пики алгоритмом Пан–Томпкинса
    (не из разметки), нарезает сегменты вокруг найденных пиков, сопоставляет им
    метки классов из .atr по ближайшей позиции (допуск 75 мс).
    """
    segments, labels, groups = [], [], []
    skipped_no_label = 0
    total_true = total_detected = total_tp = 0

    for rec_id in RECORDS:
        path = os.path.join(data_dir, rec_id)
        try:
            record = wfdb.rdrecord(path)
            ann = wfdb.rdann(path, 'atr')
        except Exception as e:
            print(f"  Пропускаю {rec_id}: {e}")
            continue

        raw = record.p_signal[:, 0].astype(np.float32)   # канал MLII
        sig = bandpass_filter(raw).astype(np.float32)

        # ───── реальная детекция + сопоставление меток ─────
        detected_peaks = detect_r_peaks(sig, FS)
        matched = map_labels_to_detected_peaks(
            detected_peaks, ann.sample, ann.symbol, FS, tolerance_ms=75
        )

        total_true += len(ann.sample)
        total_detected += len(detected_peaks)
        total_tp += len(matched)

        for peak, sym in matched:
            if sym not in AAMI:
                skipped_no_label += 1
                continue
            start, end = peak - BEFORE, peak + AFTER
            if start < 0 or end > len(sig):
                continue
            seg = sig[start:end]
            rng = seg.max() - seg.min()
            if rng > 1e-6:
                seg = (seg - seg.min()) / rng
            segments.append(seg)
            labels.append(AAMI[sym])
            groups.append(rec_id)

    X = np.array(segments, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    g = np.array(groups)   # номер записи (пациента) для каждого сегмента

    print(f"\nГотово! Сегментов: {len(X)}")
    print(f"Детекция по всей базе: истинных={total_true}, найдено={total_detected}, "
          f"совпало с разметкой={total_tp}")
    print(f"Пропущено (неизвестные метки): {skipped_no_label}")
    print(f"Форма X: {X.shape}")

    names = ['N (норма)', 'S (суправентр.)', 'V (желудочк.)', 'F (сливные)', 'Q (прочие)']
    for i, name in enumerate(names):
        print(f"  Класс {i} {name}: {(y == i).sum()} примеров")

    return X, y, g


if __name__ == '__main__':
    X, y, g = load_dataset(data_dir='mitdb')
    np.save('X.npy', X)
    np.save('y.npy', y)
    np.save('groups.npy', g)
    print("\nСохранено: X.npy, y.npy, groups.npy")
