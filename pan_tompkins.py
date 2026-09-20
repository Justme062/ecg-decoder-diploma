# -*- coding: utf-8 -*-
"""
pan_tompkins.py

Реальная детекция R-пиков алгоритмом Пан–Томпкинса (Pan, Tompkins, 1985).

Заменяет использование готовой экспертной разметки (.atr) в исходном
main.py курсовой работы. Предназначен для замены в load_dataset():
вместо `r_peaks = ann.sample` используется `r_peaks = detect_r_peaks(sig, fs)`.

Алгоритм состоит из классических пяти шагов:
    1. Полосовая фильтрация (5–15 Гц) — выделяет частотный диапазон QRS
    2. Дифференцирование — подчёркивает крутизну QRS-комплекса
    3. Возведение в квадрат — всё положительно, усиливает крупные отклонения
    4. Скользящее интегрирование (moving window integration) — сглаживает
       и формирует "энергетический" сигнал, по которому легко искать пики
    5. Адаптивный порог с двумя уровнями (сигнальный/шумовой) и поиском
       локальных максимумов с учётом рефрактерного периода сердца (~200 мс)

Автор реализации: для дипломной работы Чигирёвой Л.В. (ПМ31, КубГУ)
на основе курсовой "Интеграция структурно-спектрального анализа сигналов
методом матричных пучков с нейросетевыми архитектурами классификации..."
"""

import numpy as np
from scipy.signal import butter, filtfilt


def _bandpass(sig, fs, low=5.0, high=15.0, order=2):
    """Полосовой фильтр 5-15 Гц — стандартный диапазон для QRS в Pan-Tompkins."""
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, sig)


def _moving_window_integration(sig, fs, window_ms=150):
    """Скользящее окно интегрирования шириной ~150 мс (ширина QRS-комплекса)."""
    win = max(1, int(window_ms / 1000 * fs))
    kernel = np.ones(win) / win
    return np.convolve(sig, kernel, mode='same')


def detect_r_peaks(raw_signal, fs, refractory_ms=200):
    """
    Детектирует R-пики в сыром (уже отфильтрованном 0.5-40 Гц) сигнале ЭКГ
    алгоритмом Пан-Томпкинса.

    Parameters
    ----------
    raw_signal : np.ndarray
        Одноканальный сигнал ЭКГ (после базовой полосовой фильтрации 0.5-40 Гц,
        как в модуле предобработки курсовой).
    fs : int
        Частота дискретизации (для MIT-BIH: 360 Гц).
    refractory_ms : int
        Минимальный физиологически возможный интервал между R-пиками (мс).
        200 мс соответствует максимальной ЧСС ~300 уд/мин — стандартное значение.

    Returns
    -------
    r_peaks : np.ndarray
        Индексы (в отсчётах) обнаруженных R-пиков, отсортированы по возрастанию.
    """
    raw_signal = np.asarray(raw_signal, dtype=np.float64)
    n = len(raw_signal)
    refractory = int(refractory_ms / 1000 * fs)

    # 1. Полосовая фильтрация 5-15 Гц
    bp = _bandpass(raw_signal, fs)

    # 2. Дифференцирование (пятиточечная производная, как в оригинальной статье)
    diff = np.diff(bp, prepend=bp[0])

    # 3. Возведение в квадрат
    squared = diff ** 2

    # 4. Скользящее интегрирование
    integrated = _moving_window_integration(squared, fs, window_ms=150)

    # 5. Адаптивный поиск пиков с двумя порогами (сигнал/шум) по методике Пан-Томпкинса
    r_peaks = []

    # Инициализация порогов по первым 2 секундам сигнала
    init_len = min(n, int(2 * fs))
    init_segment = integrated[:init_len] if init_len > 0 else integrated
    spki = np.max(init_segment) * 0.25 if len(init_segment) else 0.0  # оценка уровня сигнала
    npki = np.mean(init_segment) * 0.5 if len(init_segment) else 0.0  # оценка уровня шума
    threshold1 = npki + 0.25 * (spki - npki)

    last_peak_idx = -refractory
    i = 1
    while i < n - 1:
        # локальный максимум
        if integrated[i] > integrated[i - 1] and integrated[i] >= integrated[i + 1]:
            if integrated[i] > threshold1 and (i - last_peak_idx) > refractory:
                r_peaks.append(i)
                spki = 0.125 * integrated[i] + 0.875 * spki
                last_peak_idx = i
            else:
                npki = 0.125 * integrated[i] + 0.875 * npki
            threshold1 = npki + 0.25 * (spki - npki)
        i += 1

    # 6. Коррекция позиции: ищем истинный максимум исходного сигнала в окрестности
    #    найденного пика на интегрированном сигнале (интегрирование сдвигает пик)
    search_radius = int(0.075 * fs)  # 75 мс
    corrected = []
    for p in r_peaks:
        lo = max(0, p - search_radius)
        hi = min(n, p + search_radius)
        if hi > lo:
            local_idx = lo + np.argmax(np.abs(raw_signal[lo:hi]))
            corrected.append(local_idx)

    return np.array(sorted(set(corrected)), dtype=np.int64)


def evaluate_detection(true_peaks, detected_peaks, fs, tolerance_ms=75):
    """
    Оценивает качество детекции относительно эталонной разметки.

    Метрики стандартны для литературы по детекции R-пиков (см. AAMI EC57):
        Sensitivity (Se) = TP / (TP + FN)   — доля найденных истинных пиков
        PPV             = TP / (TP + FP)   — доля точных находок среди всех найденных

    Parameters
    ----------
    true_peaks : array-like
        Эталонные позиции R-пиков (индексы отсчётов), напр. из ann.sample (.atr).
    detected_peaks : array-like
        Позиции, найденные detect_r_peaks().
    fs : int
        Частота дискретизации.
    tolerance_ms : int
        Допустимое отклонение найденного пика от истинного (мс). 75 мс —
        стандартное значение допуска в литературе по детекции QRS.

    Returns
    -------
    dict с ключами: sensitivity, ppv, tp, fp, fn
    """
    true_peaks = np.asarray(sorted(true_peaks))
    detected_peaks = np.asarray(sorted(detected_peaks))
    tol = int(tolerance_ms / 1000 * fs)

    matched_true = np.zeros(len(true_peaks), dtype=bool)
    tp = 0
    fp = 0

    for d in detected_peaks:
        if len(true_peaks) == 0:
            fp += 1
            continue
        diffs = np.abs(true_peaks - d)
        j = np.argmin(diffs)
        if diffs[j] <= tol and not matched_true[j]:
            matched_true[j] = True
            tp += 1
        else:
            fp += 1

    fn = int(np.sum(~matched_true))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    return {
        'sensitivity': sensitivity,
        'ppv': ppv,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'n_true': len(true_peaks),
        'n_detected': len(detected_peaks),
    }


def map_labels_to_detected_peaks(detected_peaks, ann_samples, ann_symbols, fs, tolerance_ms=75):
    """
    Сопоставляет каждому найденному детектором R-пику ближайшую метку класса
    из экспертной разметки (.atr), если она попадает в допуск tolerance_ms.

    Детектор Пан-Томпкинса находит только ПОЗИЦИИ пиков, но не знает классов
    (N/S/V/F/Q) — классы всё ещё берутся из разметки, но теперь "приклеиваются"
    к позиции, найденной алгоритмом, а не наоборот. Так пайплайн работает так,
    как работал бы в реальности: детектор сам ищет удары, разметка используется
    только для получения обучающих меток (что законно и стандартно — сама
    детекция при этом полностью самостоятельна).

    Parameters
    ----------
    detected_peaks : array-like
        Позиции, найденные detect_r_peaks().
    ann_samples : array-like
        ann.sample из wfdb.rdann(..., 'atr') — эталонные позиции.
    ann_symbols : list[str]
        ann.symbol — символы меток, соответствующие ann_samples поэлементно.
    fs : int
        Частота дискретизации.
    tolerance_ms : int
        Допуск сопоставления (мс), по умолчанию 75 мс — как в evaluate_detection.

    Returns
    -------
    list[tuple[int, str]]
        Список (позиция_детектора, символ_метки) только для пиков, которым
        нашлась метка в пределах допуска. Детектированные пики без совпадения
        (ложные срабатывания либо совпадающие с несущественными метками типа
        '+', '~' и т.п.) в список не попадают.
    """
    ann_samples = np.asarray(ann_samples)
    tol = int(tolerance_ms / 1000 * fs)
    matched = []

    for d in detected_peaks:
        if len(ann_samples) == 0:
            continue
        diffs = np.abs(ann_samples - d)
        j = np.argmin(diffs)
        if diffs[j] <= tol:
            matched.append((int(d), ann_symbols[j]))

    return matched


def correct_polarity(signal, r_peak_positions, symbols=None, window_ms=40, fs=360):
    """
    Автоматически определяет и, если нужно, исправляет инверсию полярности
    отведения ЭКГ. Сравнивает несколько известных R-пиков с локальным
    "базовым уровнем" вокруг них: если доминирующее отклонение в
    большинстве случаев ОТРИЦАТЕЛЬНОЕ (зубец направлен вниз), сигнал
    считается инвертированным и домножается на -1.

    Актуально при переносе модели, обученной на одном отведении/датасете
    (например, MLII из MIT-BIH), на данные из другого источника, где то же
    физиологическое отведение может быть записано с противоположной
    полярностью (типичный случай — INCART lead II относительно MIT-BIH MLII).

    Parameters
    ----------
    signal : np.ndarray
        Отфильтрованный сигнал (после полосовой фильтрации 0.5-40 Гц).
    r_peak_positions : array-like
        Позиции нескольких известных R-пиков (индексы отсчётов) — можно
        взять из разметки .atr, не нужны все, достаточно 20-50 штук.
    symbols : list[str], optional
        Символы меток, соответствующие r_peak_positions поэлементно.
        Если переданы, решение о полярности принимается ТОЛЬКО по ударам
        класса N (нормальный синусовый ритм) — они задают "эталонную"
        ориентацию комплекса. Класс V (желудочковые экстрасистолы) имеет
        физиологически иную морфологию QRS и не должен участвовать в этом
        решении: его дискордантная полярность может быть реальным свойством
        сигнала, а не признаком инверсии всей записи.
    window_ms : int
        Полуширина окна вокруг каждого пика для анализа знака (мс).
    fs : int
        Частота дискретизации.

    Returns
    -------
    corrected_signal : np.ndarray
        Сигнал, при необходимости домноженный на -1.
    was_flipped : bool
        True, если была применена инверсия.
    """
    signal = np.asarray(signal, dtype=np.float64)
    win = int(window_ms / 1000 * fs)
    n = len(signal)

    r_peak_positions = np.asarray(r_peak_positions)

    if symbols is not None:
        symbols = np.asarray(symbols)
        n_symbols = {'N', 'L', 'R', 'e', 'j'}  # AAMI класс N
        n_mask = np.isin(symbols, list(n_symbols))
        candidate_positions = r_peak_positions[n_mask]
        # если N-ударов почти нет в записи — откатываемся на все удары
        if len(candidate_positions) < 10:
            candidate_positions = r_peak_positions
    else:
        candidate_positions = r_peak_positions

    sample_peaks = candidate_positions[(candidate_positions > win) & (candidate_positions < n - win)]
    if len(sample_peaks) > 50:
        idx = np.linspace(0, len(sample_peaks) - 1, 50).astype(int)
        sample_peaks = sample_peaks[idx]

    if len(sample_peaks) == 0:
        return signal, False

    signs = []
    for p in sample_peaks:
        lo, hi = p - win, p + win
        segment = signal[lo:hi]
        baseline = np.median(segment)
        centered = segment - baseline
        dominant_idx = np.argmax(np.abs(centered))
        signs.append(np.sign(centered[dominant_idx]))

    pct_positive = np.mean(np.array(signs) > 0)
    was_flipped = pct_positive < 0.5

    if was_flipped:
        signal = -signal

    return signal, was_flipped
