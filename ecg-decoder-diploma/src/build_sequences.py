# -*- coding: utf-8 -*-
"""
build_sequences.py

Блок А плана диплома, шаг 1: построение датасета ПОСЛЕДОВАТЕЛЬНОСТЕЙ ударов
для задачи РАННЕГО ПРЕДУПРЕЖДЕНИЯ о желудочковой экстрасистоле (PVC/класс V).

Единица анализа — скользящее окно из N последовательных ударов одного
пациента. Для каждого удара в окне вычисляются:
    - 30 признаков матричных пучков (морфология комплекса)
    - RR-интервал до предыдущего удара, нормированный (тайминг)
Итого последовательность формы (N, 31).

Целевая переменная: класс удара, идущего СРАЗУ ПОСЛЕ окна (предсказание
вперёд, а не классификация текущего удара). Бинарно: 1 = следующий удар
это V, 0 = следующий удар не V.

Почему именно так (обосновано литературой по предсказанию PVC):
    - чистая морфология одного удара слабо предсказывает будущую PVC;
    - предиктивный сигнал в основном в ДИНАМИКЕ интервалов (coupling
      interval variability, паттерны short-long-short перед событием)
      и в эволюции параметров во времени — поэтому RR-интервалы обязательны
      наряду с полюсами.

КРИТИЧНО ДЛЯ ЧЕСТНОСТИ:
    - окно НЕ включает целевой удар и ничего после него (нет утечки будущего);
    - разбиение train/test выполняется ПО ПАЦИЕНТАМ (в build_sequences
      сохраняется record_id каждого окна), чтобы удары одного пациента не
      попали и в train, и в test одновременно;
    - RR-интервалы нормируются по СОБСТВЕННОЙ статистике окна (медиана окна),
      а не по глобальной, — это устраняет зависимость от индивидуальной ЧСС
      пациента и работает в реальном времени без знания всей записи.

Запуск (из папки с записями MIT-BIH):
    python build_sequences.py
Результат: sequences.npz (X_seq, y_next, record_ids)
"""

import os
import numpy as np
import wfdb
from scipy.signal import butter, filtfilt
from numpy.linalg import svd, lstsq

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = SCRIPT_DIR   # ⚠️ поменяй, если записи MIT-BIH лежат в другом месте

FS = 360
BEFORE = 90
AFTER = 90
SEG_LEN = BEFORE + AFTER
WINDOW = 10            # число ударов в окне
M = 10                # число полюсов в методе матричных пучков
FEAT_PER_BEAT = 3 * M + 1   # 30 МП-признаков + 1 RR-интервал = 31

# записи MIT-BIH без paced-ритма (102,104,107,217 исключены по AAMI)
RECORDS = [
    '100', '101', '103', '105', '106',
    '108', '109', '111', '112', '113', '114', '115', '116',
    '117', '118', '119', '121', '122', '123', '124', '200',
    '201', '202', '203', '205', '208', '209', '210',
    '212', '213', '214', '215', '219', '220', '221',
    '222', '223', '228', '230', '231', '232', '233', '234'
]

# маппинг символов разметки в бинарную цель "является ли удар V"
V_SYMBOLS = {'V', 'E'}   # желудочковые: PVC и желудочковый escape
# символы, которые считаем "нормальными/не-V" ударами для окна и цели
VALID_BEAT_SYMBOLS = {'N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F'}


def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=FS, order=4):
    nyq = fs / 2
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal)


def matrix_pencil_features(segment, M=M, L=None):
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


def extract_beats_from_record(record_id, data_dir):
    """
    Возвращает для одной записи упорядоченный по времени список ударов:
        beats = [(мп-признаки[30], r_peak_pos, symbol), ...]
    Использует РАЗМЕТКУ .atr для позиций (в Блоке А фокус на динамике, не на
    детекции — детекция отдельно проверена в Блоке 3.1). Позиции идут по
    возрастанию, что и нужно для построения последовательностей.
    """
    path = os.path.join(data_dir, record_id)
    record = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, 'atr')

    raw = record.p_signal[:, 0].astype(np.float64)
    sig = bandpass_filter(raw).astype(np.float32)

    beats = []
    for pos, sym in zip(ann.sample, ann.symbol):
        if sym not in VALID_BEAT_SYMBOLS:
            continue
        start, end = pos - BEFORE, pos + AFTER
        if start < 0 or end > len(sig):
            continue
        seg = sig[start:end]
        rng = seg.max() - seg.min()
        if rng > 1e-6:
            seg = (seg - seg.min()) / rng
        feats = matrix_pencil_features(seg)
        beats.append((feats, int(pos), sym))

    return beats


def build_sequences_for_record(beats):
    """
    Из упорядоченного списка ударов одной записи строит окна.

    Для окна ударов [i, i+WINDOW) целевой удар — (i+WINDOW)-й.
    Признак каждого удара в окне: [30 МП-признаков, нормированный RR-интервал].
    RR-интервал удара j = (pos_j - pos_{j-1}) / fs (в секундах), затем внутри
    окна нормируется на медиану RR этого окна — устраняет зависимость от
    индивидуальной ЧСС и не требует знания всей записи (работает онлайн).
    """
    X_list, y_list = [], []
    n = len(beats)
    if n < WINDOW + 1:
        return X_list, y_list

    # RR-интервалы: для первого удара RR неизвестен, ставим NaN и позже
    # окна, начинающиеся с самого первого удара записи, будут содержать один
    # такой NaN; чтобы не терять данные, заменяем его на медиану окна
    positions = np.array([b[1] for b in beats])
    rr = np.full(n, np.nan)
    rr[1:] = (positions[1:] - positions[:-1]) / FS  # в секундах

    for i in range(n - WINDOW):
        window_beats = beats[i:i + WINDOW]
        target_beat = beats[i + WINDOW]

        # признаки окна: [WINDOW, 31]
        window_feats = np.zeros((WINDOW, FEAT_PER_BEAT), dtype=np.float32)
        window_rr = rr[i:i + WINDOW].copy()
        # медиана RR окна (игнорируя возможный NaN первого удара записи)
        valid_rr = window_rr[~np.isnan(window_rr)]
        rr_median = np.median(valid_rr) if len(valid_rr) > 0 else 1.0
        if rr_median <= 0:
            rr_median = 1.0
        window_rr = np.where(np.isnan(window_rr), rr_median, window_rr)
        window_rr_norm = window_rr / rr_median

        for j, (feats, pos, sym) in enumerate(window_beats):
            window_feats[j, :30] = feats
            window_feats[j, 30] = window_rr_norm[j]

        # цель: является ли СЛЕДУЮЩИЙ удар желудочковым
        target_is_v = 1 if target_beat[2] in V_SYMBOLS else 0

        X_list.append(window_feats)
        y_list.append(target_is_v)

    return X_list, y_list


def main():
    all_X, all_y, all_records = [], [], []

    print("=" * 72)
    print("ПОСТРОЕНИЕ ДАТАСЕТА ОКОН ДЛЯ РАННЕГО ПРЕДУПРЕЖДЕНИЯ О V (Блок А)")
    print("=" * 72)

    for rec_id in RECORDS:
        try:
            beats = extract_beats_from_record(rec_id, DATA_DIR)
        except Exception as e:
            print(f"  Пропускаю {rec_id}: {e}")
            continue

        X_list, y_list = build_sequences_for_record(beats)
        all_X.extend(X_list)
        all_y.extend(y_list)
        all_records.extend([rec_id] * len(X_list))

        n_v = sum(y_list)
        print(f"  [{rec_id}] ударов={len(beats):5d}  окон={len(X_list):5d}  "
              f"из них 'следующий = V': {n_v:4d} ({n_v/max(1,len(X_list))*100:.1f}%)")

    X_seq = np.array(all_X, dtype=np.float32)
    y_next = np.array(all_y, dtype=np.int64)
    record_ids = np.array(all_records)

    print("\n" + "=" * 72)
    print(f"Итого окон: {len(X_seq)}, форма X: {X_seq.shape}")
    print(f"Положительных (следующий удар = V): {y_next.sum()} "
          f"({y_next.mean()*100:.2f}%)")
    print(f"Уникальных записей: {len(set(record_ids))}")

    out_path = os.path.join(SCRIPT_DIR, 'sequences.npz')
    np.savez_compressed(out_path, X_seq=X_seq, y_next=y_next, record_ids=record_ids)
    print(f"Сохранено: {out_path}")


if __name__ == '__main__':
    main()
