# ECG Arrhythmia Diagnosis — гибридная система (матричные пучки + нейросети)

Дипломная работа, развивающая курсовой проект по классификации аритмий на базе
MIT-BIH методом матричных пучков (МП) и нейросетевыми архитектурами.

Работа развивает курсовую в трёх направлениях:

1. **Методологическое усиление базового конвейера** — реальная детекция R-пиков
   (Пан–Томпкинс) вместо разметки, статистическая значимость превосходства
   гибридной модели, кросс-датасетная валидация на INCART.
2. **Генеративное моделирование (Блок Б)** — Conditional VAE в пространстве
   МП-признаков для аугментации редких классов (итог: +12.61 п.п. Macro F1).
3. **Раннее предупреждение (Блок А)** — предсказание желудочковой экстрасистолы
   по окну из 10 предыдущих ударов через Transformer (итог: AUC 0.825,
   +14 п.п. над таймингом) + механистический анализ динамики полюсов.

## Ключевые результаты

| Направление | Результат |
|---|---|
| Детекция R-пиков (Пан–Томпкинс) | Se=92.75%, PPV=98.9% (MIT-BIH); Se=91.93%, PPV=98.78% (INCART) |
| Прирост гибридной модели | +1.98 п.п. Macro F1, значим (McNemar p<0.001, bootstrap ДИ [+1.13;+2.86]) |
| Раннее предупреждение о V | AUC 0.825, recall 0.80 (patient-wise split) |
| VAE-аугментация | +12.61 п.п. Macro F1 (+25 на классе F, +18 на S) |

## Структура

```
src/
├── pan_tompkins.py              # детектор R-пиков + коррекция полярности + сопоставление меток
├── main_patched_load_dataset.py # патч load_dataset() для реальной детекции
├── evaluate_all_records.py      # оценка детектора по всем записям (Se/PPV)
├── statistical_significance.py  # McNemar + bootstrap CI
├── cross_dataset_incart.py      # кросс-датасетная валидация на INCART
├── diagnose_domain_shift.py     # диагностика домен-сдвига (морфология)
├── vae_pole_features.py         # Conditional VAE + TSTR (логрег/MLP) + PCA-диагностика
├── latent_interpolation.py      # интерполяция латента (морфинг классов)
├── augmented_training.py        # интеграция VAE-аугментации в нейросеть
├── build_sequences.py           # датасет окон для раннего предупреждения
├── early_warning.py             # Transformer + иерархия baseline
├── pole_dynamics.py             # механистический анализ динамики полюсов
└── test_pan_tompkins.py         # синтетический тест детектора
```

## Порядок запуска

Скрипты запускаются из папки с записями MIT-BIH (данные скачиваются с
[physionet.org](https://physionet.org/content/mitdb/1.0.0/); INCART —
[здесь](https://physionet.org/content/incartdb/1.0.0/)).

```bash
# базовый слой
python main.py                    # → X.npy, y.npy (реальная детекция R-пиков)
python refeatures.py              # → F.npy (МП-признаки, M=10, размерность 30)
python train.py                   # → base_cnn.pth, hybrid_cnn.pth, pred_*.npy
python evaluate_all_records.py    # таблица Se/PPV детектора
python statistical_significance.py
python cross_dataset_incart.py    # требует incart-data/

# Блок Б (генеративка)
python vae_pole_features.py       # → pole_vae.pth
python latent_interpolation.py
python augmented_training.py

# Блок А (раннее предупреждение)
python build_sequences.py         # → sequences.npz
python early_warning.py
python pole_dynamics.py
```

## Окружение

```
python >= 3.11
torch
numpy
scipy
scikit-learn
wfdb
pandas
matplotlib
```

## Методологические замечания

- Класс Q после исключения записей с навязанным ритмом (102, 104, 107, 217 по AAMI)
  вырожден (support=1 на тесте), поэтому Macro F1 приводится по 4 классам (N, S, V, F).
- Разбиение train/test в задаче раннего предупреждения — строго **по пациентам**
  (V-события концентрируются у части пациентов).
- Батч-нормализация на входе ветви МП-признаков критична для работы гибридной модели.
- Весь код проверялся на синтетических данных с известным ответом до прогона на
  реальных данных.

## Базовый (курсовой) проект

Исходный курсовой проект: [github.com/JustMe062/ecg-decoder](https://github.com/JustMe062/ecg-decoder)
