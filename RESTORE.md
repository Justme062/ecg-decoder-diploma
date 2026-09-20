# Восстановление проекта на новом компьютере

Полностью собранный рабочий проект: базовый код курсовой + дипломные скрипты +
рисунки + скрипты восстановления данных. Данные и веса моделей в git не хранятся
(они большие/публичные и воспроизводимы) — они докачиваются и пересоздаются.

## БЫСТРЫЙ СТАРТ (3 команды)

Открой терминал (PowerShell) в папке `ecg_project` и выполни по очереди:

```powershell
pip install -r requirements.txt      # 1. поставить пакеты (Python 3.11+)
python get_data.py                   # 2. скачать MIT-BIH и INCART в нужные папки
python check_setup.py                # 3. убедиться, что всё на месте
```

Как открыть терминал прямо в папке: в Проводнике, находясь в `ecg_project`,
набери в адресной строке `powershell` и нажми Enter (или Shift+ПКМ →
«Открыть окно PowerShell здесь»).

После этого данные лежат так, как ждут скрипты — **перекладывать ничего не надо**:

```
ecg_project/
├── mitdb/               ← MIT-BIH (100.dat, 100.hea, 100.atr, ...)
├── incart-data/files/   ← INCART (I01.dat, ...)
├── main.py, refeatures.py, train.py, ...
└── figures/
```

Дальше — прогон пайплайна по порядку (см. ниже). `python check_setup.py` можно
запускать в любой момент: он покажет, что уже готово, а чего не хватает.

> INCART большой (~1.3 ГБ) и нужен только для 2 скриптов
> (`cross_dataset_incart.py`, `diagnose_domain_shift.py`). Если он пока не нужен —
> качай только MIT-BIH: `python get_data.py mitdb` (~100 МБ).

---

## Порядок запуска (с нуля до всех результатов)

```powershell
# базовый слой (курсовой код)
python main.py            # -> X.npy, y.npy   (реальная детекция R-пиков, папка mitdb/)
python refeatures.py      # -> F.npy          (МП-признаки, M=10, dim=30)
python train.py           # -> base_cnn.pth, hybrid_cnn.pth, pred_*.npy, true_labels.npy

# глава 2 — методологическое усиление
python evaluate_all_records.py       # табл. 1 (Se/PPV детектора)
python statistical_significance.py   # табл. 2 (McNemar + bootstrap CI)
python cross_dataset_incart.py       # табл. 3 (INCART) — нужен INCART
python diagnose_domain_shift.py      # рис. 2 — нужен INCART

# глава 4 — генеративка (Блок Б)
python vae_pole_features.py          # -> pole_vae.pth, TSTR, рис. 4
python latent_interpolation.py       # рис. 5, 6
python augmented_training.py         # табл. 6

# глава 3 — раннее предупреждение (Блок А)
python build_sequences.py            # -> sequences.npz (окна по 10 ударов, папка mitdb/)
python early_warning.py              # табл. 4 -> early_warning_results.npz
python pole_dynamics.py              # табл. 5, рис. 3
```

Промежуточные файлы (`X.npy`, `F.npy`, `*.pth`, `sequences.npz`) сохраняются прямо
в `ecg_project/` и подхватываются следующими скриптами автоматически.

---

## Откуда что взялось (два репозитория)

| Репозиторий | Что содержит |
|---|---|
| `Justme062/ecg-decoder` (курсовой) | базовый конвейер: `main.py`, `refeatures.py`, `train.py` |
| `Justme062/ecg-decoder-diploma` (диплом) | 13 дипломных скриптов + рисунки |

Диплом — надстройка: его скрипты грузят файлы, которые производит базовый код.
Здесь собрано и то, и другое.

## Что было изменено при сборке

- **`main.py`** — дипломный вариант с реальной детекцией Пан–Томпкинса (в курсовом
  детекция бралась из разметки; для диплома заменено, п. 2.2). Читает MIT-BIH из `mitdb/`.
- **`evaluate_all_records.py`, `build_sequences.py`** — путь к MIT-BIH переведён на `mitdb/`.
- **`get_data.py`** — качает данные сразу в `mitdb/` и `incart-data/files/`.
- **`test_pan_tompkins.py`** — убран жёсткий путь со старого компа.
- добавлен **`check_setup.py`** — проверка готовности.

## ⚠️ Единственная точка неопределённости: 44 или 48 записей

Работа исключает 4 записи с кардиостимулятором (102, 104, 107, 217) из анализа
(п. 2.2), поэтому в `main.py` датасет строится по **44** записям.
`evaluate_all_records.py` намеренно прогоняет **все 48**, чтобы показать деградацию
детектора на навязанном ритме (тоже п. 2.2). Если исходный `main.py` строил `X.npy`
по всем 48, точные Macro F1 в табл. 2 сдвинутся в третьем знаке (другой split при
seed=42). Это единственное, что нельзя на 100% восстановить из гита.

## Соответствие рисунков и файлов (папка figures/)

| В работе | Файл |
|---|---|
| Рис. 1 — F1 по классам | `f1_comparison.png` |
| Рис. 2 — морфология MIT-BIH/INCART | `compare_N.png` (и `compare_S/V.png`) |
| Рис. 3 — динамика доминирующего полюса | `pole_trajectory_dominant.png` |
| Рис. 4 — реальные vs синтетические F | `real_vs_synthetic_F.png` |
| Рис. 5 — интерполяция N→V (карта) | `interp_heatmap_N_to_V.png` |
| Рис. 6 — гладкость перехода N→V | `interp_smoothness_N_to_V.png` |
