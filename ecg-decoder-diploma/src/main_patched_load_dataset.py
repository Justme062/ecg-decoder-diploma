# -*- coding: utf-8 -*-
"""
main_patched_load_dataset.py

Это НЕ отдельный модуль для импорта, а справочный файл: показывает,
как именно меняется функция load_dataset() в main.py курсовой работы,
чтобы использовать реальную детекцию R-пиков вместо готовой разметки .atr.

Просто скопируй тело этой функции поверх старой load_dataset() в main.py
и добавь импорт в начало файла.
"""

# ───── добавить в начало main.py, рядом с остальными import'ами ─────
#
# from pan_tompkins import detect_r_peaks, map_labels_to_detected_peaks


def load_dataset(data_dir='.'):
    """
    Загружает все записи, фильтрует, детектирует R-пики АЛГОРИТМОМ
    (не из разметки), нарезает сегменты вокруг найденных пиков,
    сопоставляет им метки классов из .atr по ближайшей позиции.
    """
    import wfdb
    import numpy as np
    import os
    from pan_tompkins import detect_r_peaks, map_labels_to_detected_peaks

    segments, labels = [], []
    skipped_no_label = 0     # найденный пик не попал ни в одну метку разметки
    total_true = 0
    total_detected = 0
    total_tp = 0

    for rec_id in RECORDS:
        path = os.path.join(data_dir, rec_id)
        try:
            record = wfdb.rdrecord(path)
            ann = wfdb.rdann(path, 'atr')
        except Exception as e:
            print(f"  Пропускаю {rec_id}: {e}")
            continue

        raw = record.p_signal[:, 0].astype(np.float32)
        sig = bandpass_filter(raw).astype(np.float32)

        # ───── БЫЛО: r_peaks = ann.sample; sym_list = ann.symbol ─────
        # ───── СТАЛО: реальная детекция + сопоставление меток ─────
        detected_peaks = detect_r_peaks(sig, FS)
        matched = map_labels_to_detected_peaks(
            detected_peaks, ann.sample, ann.symbol, FS, tolerance_ms=75
        )
        # matched — список (позиция_детектора, символ_метки)

        total_true += len(ann.sample)
        total_detected += len(detected_peaks)
        total_tp += len(matched)

        for peak, sym in matched:
            if sym not in AAMI:
                skipped_no_label += 1
                continue
            start = peak - BEFORE
            end = peak + AFTER
            if start < 0 or end > len(sig):
                continue

            seg = sig[start:end]
            rng = seg.max() - seg.min()
            if rng > 1e-6:
                seg = (seg - seg.min()) / rng

            segments.append(seg)
            labels.append(AAMI[sym])

    X = np.array(segments, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)

    print(f"\nГотово! Сегментов: {len(X)}")
    print(f"Детекция по всей базе: истинных={total_true}, найдено={total_detected}, "
          f"совпало с разметкой={total_tp}")
    print(f"Пропущено (не-ударные/неизвестные метки): {skipped_no_label}")
    print(f"Форма X: {X.shape}")

    names = ['N (норма)', 'S (суправентр.)', 'V (желудочк.)', 'F (сливные)', 'Q (прочие)']
    for i, name in enumerate(names):
        print(f"  Класс {i} {name}: {(y == i).sum()} примеров")

    return X, y
