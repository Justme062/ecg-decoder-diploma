# -*- coding: utf-8 -*-
"""
run_multiseed.py — прогоняет стохастичный эксперимент на нескольких сидах и
печатает СРЕДНЕЕ ± СКО в готовом для вставки в текст виде.

Зачем: метрики, зависящие от обучения нейросети (прирост классификации,
AUC прогноза, эффект аугментации), меняются от запуска к запуску. Одно число
— это один случайный прогон. Устойчивый результат = среднее по K сидам с
разбросом; любой перезапуск попадёт внутрь интервала, и переписывать текст
больше не придётся.

Запуск (из папки с данными X.npy / F.npy / y.npy / sequences.npz):

    python run_multiseed.py class    -k 5     # прирост классификации (train.py)
    python run_multiseed.py predict  -k 5     # AUC прогноза (early_warning.py)
    python run_multiseed.py augment  -k 5     # эффект VAE-аугментации
    python run_multiseed.py interp   -k 5     # монотонность латента (VAE+интерп.)
    python run_multiseed.py all      -k 5     # всё подряд

-k     число сидов (по умолчанию 5). Сиды берутся 0,1,...,k-1.
--base базовый сид (по умолчанию 0), сиды = base..base+k-1.

Каждый скрипт печатает служебную строку «MULTISEED tag=... key=value ...» —
её и парсит обёртка. Человекочитаемый вывод скриптов не трогается.
"""
import argparse, os, re, subprocess, sys, math

PY = sys.executable
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# target -> (список скриптов на один сид, человекочитаемые метрики для отчёта)
# report: (ключ, подпись, единица, знаков после запятой)
TARGETS = {
    'class': {
        'scripts': ['train.py'], 'tag': 'train',
        'report': [
            ('gap4_pp', 'Прирост Macro F1 (N,S,V,F), гибрид − база', 'п.п.', 2),
            ('f1_base4', 'Macro F1 базовой (4 класса)', '', 4),
            ('f1_hybrid4', 'Macro F1 гибридной (4 класса)', '', 4),
        ]},
    'predict': {
        'scripts': ['early_warning.py'], 'tag': 'early',
        'report': [
            ('auc_tf', 'AUC Transformer', '', 4),
            ('recall', 'Полнота Transformer', '', 3),
            ('prec_tf', 'Точность Transformer (порог 0,5)', '', 3),
            ('gap_vs_rr_pp', 'AUC Transformer − AUC тайминга', 'п.п.', 2),
            ('auc_poles', 'AUC (только полюса)', '', 4),
        ]},
    'augment': {
        # VAE переобучается на каждом сиде (иначе тест виден генератору → утечка)
        'scripts': ['vae_pole_features.py', 'augmented_training.py'], 'tag': 'aug',
        'report': [
            ('gap_pp', 'Прирост Macro F1 от аугментации', 'п.п.', 2),
            ('dS', 'Прирост на классе S', 'п.п.', 2),
            ('dF', 'Прирост на классе F', 'п.п.', 2),
        ]},
    'crossds': {
        'scripts': ['train.py', 'cross_dataset_incart.py'], 'tag': 'crossds',
        'report': [
            ('incart_gap_pp', 'Гибрид − база на INCART (перенос)', 'п.п.', 2),
            ('incart_base', 'Macro F1 базовой на INCART', '', 4),
            ('incart_hybrid', 'Macro F1 гибридной на INCART', '', 4),
        ]},
    'interp': {
        # монотонность зависит от обученного VAE — обучаем VAE и интерполируем на одном сиде
        'scripts': ['vae_pole_features.py', 'latent_interpolation.py'], 'tag': 'interp',
        'report': [
            ('mono_mean', 'Средняя доля монотонных признаков', '%', 1),
            ('mono_min', 'Минимум по переходам', '%', 1),
            ('mono_max', 'Максимум по переходам', '%', 1),
        ]},
}


def parse_multiseed(text, tag):
    """Находит строку 'MULTISEED tag=<tag> k=v ...' и возвращает dict{str:float}."""
    for line in text.splitlines():
        if line.strip().startswith('MULTISEED') and f'tag={tag}' in line:
            out = {}
            for k, v in re.findall(r'(\w+)=(-?\d+\.?\d*)', line):
                if k in ('tag', 'seed'):
                    continue
                try:
                    out[k] = float(v)
                except ValueError:
                    pass
            return out
    return None


def run_one(scripts, seed):
    env = dict(os.environ)
    env['SEED'] = str(seed)
    env['PYTHONHASHSEED'] = str(seed)
    env['PYTHONUTF8'] = '1'            # заставляем дочерний Python писать в UTF-8
    env['PYTHONIOENCODING'] = 'utf-8'  # (иначе Windows-консоль cp1251 роняет Δ, −, ≈)
    combined = ''
    for sc in scripts:
        if not os.path.exists(sc):
            print(f"  ! нет файла {sc} — пропускаю сид {seed}")
            return ''
        print(f"  [seed={seed}] {sc} ...", flush=True)
        r = subprocess.run([PY, sc], env=env, capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
        combined += r.stdout + '\n' + r.stderr + '\n'
        if r.returncode != 0:
            print(f"    ошибка (код {r.returncode}); последние строки stderr:")
            print('\n'.join(r.stderr.strip().splitlines()[-5:]))
    return combined


def mean_std(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)   # выборочное СКО (ddof=1)
    return m, math.sqrt(var)


def run_target(name, seeds):
    spec = TARGETS[name]
    print(f"\n{'='*70}\nЦЕЛЬ: {name}  |  скрипты: {', '.join(spec['scripts'])}  |  сиды: {list(seeds)}\n{'='*70}")
    collected = {}
    ok = 0
    for s in seeds:
        out = run_one(spec['scripts'], s)
        d = parse_multiseed(out, spec['tag'])
        if not d:
            print(f"  [seed={s}] не найдена строка MULTISEED tag={spec['tag']} — пропуск")
            continue
        ok += 1
        for k, v in d.items():
            collected.setdefault(k, []).append(v)
    if ok == 0:
        print("  Ни один сид не дал результата. Проверьте, что данные (.npy/.npz) на месте.")
        return
    print(f"\n--- РЕЗУЛЬТАТ ПО {ok} СИДАМ (среднее ± СКО) ---")
    lines = []
    for key, label, unit, nd in spec['report']:
        if key not in collected:
            continue
        m, sd = mean_std(collected[key])
        u = (' ' + unit) if unit else ''
        val = f"{m:+.{nd}f} ± {sd:.{nd}f}{u}" if unit == 'п.п.' else f"{m:.{nd}f} ± {sd:.{nd}f}{u}"
        print(f"  {label:<48}: {val}   (n={len(collected[key])})")
        lines.append((label, m, sd, unit, nd))
    print("\n--- ГОТОВО ДЛЯ ВСТАВКИ В ТЕКСТ ---")
    for label, m, sd, unit, nd in lines:
        u = (' ' + unit) if unit else ''
        if unit == 'п.п.':
            print(f"  {label}: {m:+.{nd}f} ± {sd:.{nd}f}{u} (по {ok} сидам)")
        else:
            print(f"  {label}: {m:.{nd}f} ± {sd:.{nd}f}{u} (по {ok} сидам)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('target', choices=list(TARGETS) + ['all'])
    ap.add_argument('-k', type=int, default=5, help='число сидов')
    ap.add_argument('--base', type=int, default=0, help='базовый сид')
    a = ap.parse_args()
    seeds = range(a.base, a.base + a.k)
    targets = list(TARGETS) if a.target == 'all' else [a.target]
    for t in targets:
        run_target(t, seeds)


if __name__ == '__main__':
    main()
