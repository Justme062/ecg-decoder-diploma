# -*- coding: utf-8 -*-
"""
cross_dataset_incart.py

Кросс-датасетная валидация: модели (BaseCNN, HybridCNN), обученные на
MIT-BIH, применяются БЕЗ ДООБУЧЕНИЯ к St. Petersburg INCART 12-lead
Arrhythmia Database. Цель — проверить, что качество классификации не
является переобучением под один источник данных (MIT-BIH), как того
требует раздел 3.1 плана диплома.

Как скачать INCART (делается один раз, вручную):
    https://physionet.org/content/incartdb/1.0.0/
    -> кнопка "Download the ZIP file" внизу страницы
    распаковать в отдельную папку, например incart-data/

Требования к окружению: те же, что и для MIT-BIH пайплайна
(wfdb, numpy, scipy, torch, pandas, scikit-learn).

Запуск:
    python cross_dataset_incart.py
(положить рядом: pan_tompkins.py, base_cnn.pth, hybrid_cnn.pth,
 и поправить INCART_DIR ниже)
"""

import os
import numpy as np
import wfdb
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt, resample_poly
from sklearn.metrics import classification_report, f1_score

from pan_tompkins import detect_r_peaks, map_labels_to_detected_peaks, evaluate_detection, correct_polarity

# ───── параметры ─────
# папка скрипта — используем как базу для путей по умолчанию, чтобы не
# зависеть от рабочей директории (которая может отличаться от папки со
# скриптом, особенно при запуске из VS Code)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INCART_DIR = os.path.join(SCRIPT_DIR, 'incart-data', 'files')   # ⚠️ поменяй, если данные лежат в другом месте
FS_INCART = 257
FS_TARGET = 360               # частота, на которой обучены модели (MIT-BIH)
BEFORE = 90
AFTER = 90
SEG_LEN = BEFORE + AFTER

# тот же маппинг, что и в основном пайплайне на MIT-BIH — символы разметки
# в PhysioBank-формате стандартны и совпадают между базами
AAMI = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,
    'A': 1, 'a': 1, 'J': 1, 'S': 1,
    'V': 2, 'E': 2,
    'F': 3,
    '/': 4, 'f': 4, 'Q': 4,
}
CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']
RELEVANT_CLASSES = [0, 1, 2, 3]  # без Q, как и в основном анализе

# 75 записей INCART: I01..I75
INCART_RECORDS = [f'I{i:02d}' for i in range(1, 76)]


def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=FS_INCART, order=4):
    nyq = fs / 2
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal)


def pick_lead_ii(record):
    """Находит индекс отведения II в записи INCART (12 отведений)."""
    names = [s.strip() for s in record.sig_name]
    if 'II' in names:
        return names.index('II')
    # запасной вариант: некоторые записи могут маркировать иначе
    for alt in ('ii', 'MLII', 'Lead II'):
        if alt in names:
            return names.index(alt)
    raise ValueError(f"Отведение II не найдено среди {names}")


def matrix_pencil_features(segment, M=10, L=None):
    from numpy.linalg import svd, lstsq
    N = len(segment)
    if L is None:
        L = N // 2
    rows, cols = N - L, L + 1
    Y = np.array([segment[i:i + cols] for i in range(rows)], dtype=np.float64)
    U, s, Vh = svd(Y, full_matrices=False)
    s_norm = s[:M] / (s[0] + 1e-10)
    Vm = Vh[:M, :]
    Y1, Y2 = Vm[:, :-1], Vm[:, 1:]
    Z, _, _, _ = lstsq(Y1.T, Y2.T, rcond=None)
    Z = Z.T
    eigenvalues = np.linalg.eigvals(Z)
    idx = np.argsort(np.abs(eigenvalues))[::-1][:M]
    poles = eigenvalues[idx]
    features = np.concatenate([s_norm, np.abs(poles), np.angle(poles) / np.pi])
    return features.astype(np.float32)


def load_incart_dataset(data_dir):
    segments, labels = [], []
    total_true = total_detected = total_tp = 0

    for rec_id in INCART_RECORDS:
        path = os.path.join(data_dir, rec_id)
        try:
            record = wfdb.rdrecord(path)
            ann = wfdb.rdann(path, 'atr')
        except Exception as e:
            print(f"  Пропускаю {rec_id}: {e}")
            continue

        lead_idx = pick_lead_ii(record)
        raw = record.p_signal[:, lead_idx].astype(np.float64)
        raw = np.nan_to_num(raw)  # на случай пропусков в сигнале

        # 1. фильтруем на РОДНОЙ частоте 257 Гц
        filtered = bandpass_filter(raw, fs=FS_INCART)

        # 1.5. коррекция полярности отведения (INCART lead II может быть
        #      инвертирован относительно MIT-BIH MLII — критично для CNN,
        #      обученной на конкретной форме QRS)
        filtered, was_flipped = correct_polarity(filtered, ann.sample, symbols=ann.symbol, fs=FS_INCART)
        if was_flipped:
            print(f"      [{rec_id}] обнаружена инверсия полярности -> исправлено")

        # 2. ресемплим сигнал к 360 Гц, ЧТОБЫ длительность сегмента в
        #    отсчётах (180) соответствовала той же длительности по времени
        #    (0.5 с), на которой обучены модели
        n_target = int(len(filtered) * FS_TARGET / FS_INCART)
        sig = resample_poly(filtered, FS_TARGET, FS_INCART).astype(np.float32)
        # пересчитываем и позиции разметки в новую шкалу отсчётов
        ann_sample_resampled = (np.asarray(ann.sample) * FS_TARGET / FS_INCART).astype(np.int64)

        # 3. детекция R-пиков ТЕМ ЖЕ алгоритмом, что и на MIT-BIH
        detected = detect_r_peaks(sig, FS_TARGET)
        metrics = evaluate_detection(ann_sample_resampled, detected, FS_TARGET, tolerance_ms=75)
        total_true += metrics['n_true']
        total_detected += metrics['n_detected']
        total_tp += metrics['tp']

        matched = map_labels_to_detected_peaks(
            detected, ann_sample_resampled, ann.symbol, FS_TARGET, tolerance_ms=75
        )

        for peak, sym in matched:
            if sym not in AAMI:
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

        print(f"  [{rec_id}] обработана, накоплено сегментов: {len(segments)}")

    print(f"\nДетекция R-пиков на INCART: истинных={total_true}, "
          f"найдено={total_detected}, совпало={total_tp}, "
          f"Se={total_tp/total_true*100:.2f}%  PPV={total_tp/total_detected*100:.2f}%"
          if total_true and total_detected else "")

    X = np.array(segments, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    return X, y


# ───── архитектуры — должны буквально совпадать с train.py, иначе
#        state_dict не загрузится ─────
class BaseCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(8),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, f=None):
        return self.fc(self.conv(x))


class HybridCNN(nn.Module):
    def __init__(self, feat_dim=30, num_classes=5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(8),
        )
        self.feat_proj = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Linear(feat_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 8 + 32, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, f):
        cnn_flat = self.conv(x).view(x.size(0), -1)
        feat_out = self.feat_proj(f)
        return self.fc(torch.cat([cnn_flat, feat_out], dim=1))


def evaluate_on_incart(X, y, F):
    device = torch.device('cpu')

    base_model = BaseCNN()
    base_model.load_state_dict(torch.load(os.path.join(SCRIPT_DIR, 'base_cnn.pth'), map_location=device))
    base_model.eval()

    hybrid_model = HybridCNN(feat_dim=F.shape[1])
    hybrid_model.load_state_dict(torch.load(os.path.join(SCRIPT_DIR, 'hybrid_cnn.pth'), map_location=device))
    hybrid_model.eval()

    Xt = torch.tensor(X[:, None, :])
    Ft = torch.tensor(F)

    with torch.no_grad():
        pred_base = base_model(Xt).argmax(1).numpy()
        pred_hybrid = hybrid_model(Xt, Ft).argmax(1).numpy()

    print("\n" + "=" * 72)
    print("РЕЗУЛЬТАТЫ НА INCART (модели обучены ТОЛЬКО на MIT-BIH, без дообучения)")
    print("=" * 72)

    print("\n--- Базовая CNN ---")
    print(classification_report(y, pred_base, labels=[0, 1, 2, 3, 4],
                                 target_names=CLASS_NAMES, digits=3, zero_division=0))
    f1_base_full = f1_score(y, pred_base, average='macro', zero_division=0)
    f1_base_4 = f1_score(y, pred_base, labels=RELEVANT_CLASSES, average='macro', zero_division=0)
    print(f"Macro F1 (5 классов): {f1_base_full:.4f}   Macro F1 (N,S,V,F): {f1_base_4:.4f}")

    print("\n--- Гибридная CNN + Matrix Pencil ---")
    print(classification_report(y, pred_hybrid, labels=[0, 1, 2, 3, 4],
                                 target_names=CLASS_NAMES, digits=3, zero_division=0))
    f1_hybrid_full = f1_score(y, pred_hybrid, average='macro', zero_division=0)
    f1_hybrid_4 = f1_score(y, pred_hybrid, labels=RELEVANT_CLASSES, average='macro', zero_division=0)
    print(f"Macro F1 (5 классов): {f1_hybrid_full:.4f}   Macro F1 (N,S,V,F): {f1_hybrid_4:.4f}")

    print("\n" + "=" * 72)
    print("СРАВНЕНИЕ С РЕЗУЛЬТАТАМИ НА MIT-BIH (внутри-датасетный тест)")
    print("=" * 72)
    print("Заполни вручную из вывода train.py для сравнения:")
    print("  MIT-BIH  Базовая:  Macro F1 (N,S,V,F) = 0.8731")
    print("  MIT-BIH  Гибрид:   Macro F1 (N,S,V,F) = 0.8929")
    print(f"  INCART   Базовая:  Macro F1 (N,S,V,F) = {f1_base_4:.4f}")
    print(f"  INCART   Гибрид:   Macro F1 (N,S,V,F) = {f1_hybrid_4:.4f}")

    np.save(os.path.join(SCRIPT_DIR, 'incart_true.npy'), y)
    np.save(os.path.join(SCRIPT_DIR, 'incart_pred_base.npy'), pred_base)
    np.save(os.path.join(SCRIPT_DIR, 'incart_pred_hybrid.npy'), pred_hybrid)
    print("\nСохранено: incart_true.npy, incart_pred_base.npy, incart_pred_hybrid.npy")


def main():
    print("Загружаю и обрабатываю INCART...")
    X, y = load_incart_dataset(INCART_DIR)
    print(f"\nГотово! Сегментов: {len(X)}")
    if len(X) == 0:
        print("Ничего не обработано — проверь INCART_DIR.")
        return

    for i, name in enumerate(CLASS_NAMES):
        print(f"  Класс {i} {name}: {(y == i).sum()} примеров")

    print("\nИзвлекаю МП-признаки (M=10)...")
    F = np.array([matrix_pencil_features(seg, M=10) for seg in X], dtype=np.float32)
    print(f"Форма F: {F.shape}")

    evaluate_on_incart(X, y, F)


if __name__ == '__main__':
    main()
