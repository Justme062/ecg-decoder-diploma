# -*- coding: utf-8 -*-
"""
repro.py — единая фиксация случайности для воспроизводимости.

set_seed(seed) фиксирует random, numpy, torch (CPU и CUDA) и включает
детерминированный режим cuDNN. Обучающие скрипты читают seed из переменной
окружения SEED (по умолчанию 42): канонический прогон не меняется, а
run_multiseed.py прогоняет эксперимент на нескольких сидах.

ВАЖНО: фиксация seed делает число ПОВТОРЯЕМЫМ, но не «истинным». Чтобы число
было устойчиво к смене машины/сида, в тексте приводите СРЕДНЕЕ ± СКО по
нескольким сидам (run_multiseed.py) и формулируйте выводы через интервал —
тогда любой перезапуск попадает внутрь заявленного диапазона.
"""
import os, random
import numpy as np


def get_seed(default=42):
    try:
        return int(os.environ.get('SEED', default))
    except (TypeError, ValueError):
        return default


def set_seed(seed=None):
    if seed is None:
        seed = get_seed()
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    return seed
