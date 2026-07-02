# -*- coding: utf-8 -*-
"""
latent_interpolation.py

Блок Б плана диплома, финальный штрих: интерполяция между классами в
латентном пространстве обученного Conditional VAE (pole_vae.pth).

Идея: взять реальные примеры двух классов, закодировать в латент, пройти
по прямой между их латентными представлениями, декодировать промежуточные
точки — и увидеть, как 30-мерный вектор признаков матричных пучков ПЛАВНО
"перетекает" из одного класса в другой ("морфинг" N -> V).

Зачем это в дипломе:
    - подтверждает, что латентное пространство СОДЕРЖАТЕЛЬНО и НЕПРЕРЫВНО
      (промежуточные точки осмысленны, а не мусор) — прямое следствие того,
      что мы победили posterior collapse;
    - показывает, что признаки матричных пучков образуют гладкое
      многообразие, где классы связаны непрерывными траекториями, а не
      разбросаны изолированными кластерами;
    - даёт наглядную визуализацию для защиты.

Два режима интерполяции:
    A) по УСЛОВИЮ класса (conditional): фиксируем латент z, меняем метку
       класса c от N к V — показывает "чистый" вклад условия класса в
       генерацию при неизменном стиле z;
    B) по ЛАТЕНТУ конкретных примеров: кодируем реальный N-удар и реальный
       V-удар, интерполируем их z — показывает переход между конкретными
       реальными точками.

Запуск (из папки с pole_vae.pth, F.npy, y.npy):
    python latent_interpolation.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEAT_DIM = 30
NUM_CLASSES = 5
LATENT_DIM = 8
CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ─── та же архитектура, что в vae_pole_features.py (иначе не загрузится) ───
class ConditionalVAE(nn.Module):
    def __init__(self, feat_dim=FEAT_DIM, num_classes=NUM_CLASSES, latent_dim=LATENT_DIM, hidden=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.encoder = nn.Sequential(
            nn.Linear(feat_dim + num_classes, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, feat_dim),
        )

    def encode(self, x, c):
        h = self.encoder(torch.cat([x, c], dim=1))
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z, c):
        return self.decoder(torch.cat([z, c], dim=1))


def one_hot(idx, num_classes):
    v = torch.zeros(num_classes)
    v[idx] = 1.0
    return v


# ─── названия групп признаков для интерпретации графиков ───
def feature_group_labels():
    labels = []
    for i in range(10):
        labels.append(f'σ{i+1}')       # сингулярные значения
    for i in range(10):
        labels.append(f'|z{i+1}|')     # модули полюсов
    for i in range(10):
        labels.append(f'arg{i+1}')     # аргументы полюсов
    return labels


def conditional_interpolation(model, class_from, class_to, n_steps=9, seed=0):
    """
    Режим A: фиксируем случайный латент z, плавно меняем условие класса
    от class_from к class_to (линейная интерполяция one-hot векторов).
    Возвращает массив (n_steps, 30) декодированных признаков.
    """
    rng = torch.Generator().manual_seed(seed)
    z = torch.randn(1, model.latent_dim, generator=rng)

    c_from = one_hot(class_from, NUM_CLASSES).unsqueeze(0)
    c_to = one_hot(class_to, NUM_CLASSES).unsqueeze(0)

    outputs = []
    alphas = np.linspace(0, 1, n_steps)
    with torch.no_grad():
        for a in alphas:
            c = (1 - a) * c_from + a * c_to
            out = model.decode(z, c)
            outputs.append(out.squeeze(0).numpy())
    return np.array(outputs), alphas


def latent_interpolation(model, x_from, x_to, class_from, class_to, n_steps=9):
    """
    Режим B: кодируем два реальных примера в латент, интерполируем z по
    прямой, условие класса тоже интерполируем. Показывает переход между
    конкретными реальными точками.
    """
    xf = torch.tensor(x_from, dtype=torch.float32).unsqueeze(0)
    xt = torch.tensor(x_to, dtype=torch.float32).unsqueeze(0)
    cf = one_hot(class_from, NUM_CLASSES).unsqueeze(0)
    ct = one_hot(class_to, NUM_CLASSES).unsqueeze(0)

    with torch.no_grad():
        mu_f, _ = model.encode(xf, cf)
        mu_t, _ = model.encode(xt, ct)

    outputs = []
    alphas = np.linspace(0, 1, n_steps)
    with torch.no_grad():
        for a in alphas:
            z = (1 - a) * mu_f + a * mu_t
            c = (1 - a) * cf + a * ct
            out = model.decode(z, c)
            outputs.append(out.squeeze(0).numpy())
    return np.array(outputs), alphas


def plot_interpolation_heatmap(features_seq, alphas, class_from, class_to, mode_name, out_name):
    """Тепловая карта: строки — шаги интерполяции, столбцы — 30 признаков.
    Показывает, как каждый признак плавно меняется вдоль перехода."""
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(features_seq, aspect='auto', cmap='RdBu_r',
                   interpolation='nearest')
    ax.set_yticks(range(len(alphas)))
    ax.set_yticklabels([f'{CLASS_NAMES[class_from]}'
                        if i == 0 else (f'{CLASS_NAMES[class_to]}'
                        if i == len(alphas) - 1 else f'{a:.2f}')
                        for i, a in enumerate(alphas)])
    labels = feature_group_labels()
    ax.set_xticks(range(30))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel(f'шаг интерполяции {CLASS_NAMES[class_from]} → {CLASS_NAMES[class_to]}')
    ax.set_xlabel('признаки матричных пучков (σ — сингулярные, |z| — модули, arg — аргументы полюсов)')
    ax.set_title(f'Интерполяция признаков в латентном пространстве VAE\n'
                 f'{CLASS_NAMES[class_from]} → {CLASS_NAMES[class_to]} ({mode_name})')
    # разделители групп признаков
    for x in [9.5, 19.5]:
        ax.axvline(x, color='black', linewidth=1.2)
    plt.colorbar(im, ax=ax, label='значение признака')
    plt.tight_layout()
    out_path = os.path.join(SCRIPT_DIR, out_name)
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"  Сохранено: {out_path}")


def plot_smoothness(features_seq, alphas, class_from, class_to, out_name):
    """График гладкости: показывает, что переход монотонный/плавный, а не
    скачкообразный. По оси Y — среднее по модулям полюсов (|z|), ключевая
    группа признаков; по X — шаг интерполяции."""
    mod_mean = features_seq[:, 10:20].mean(axis=1)   # средний |z|
    sv_mean = features_seq[:, 0:10].mean(axis=1)      # среднее сингулярное
    arg_mean = features_seq[:, 20:30].mean(axis=1)    # средний аргумент

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, mod_mean, 'o-', color='#d9534f', label='средний |z| (модули полюсов)')
    ax.plot(alphas, sv_mean, 's-', color='#2e6f95', label='среднее σ (сингулярные значения)')
    ax.plot(alphas, arg_mean, '^-', color='#5cb85c', label='средний arg (аргументы полюсов)')
    ax.set_xlabel(f'шаг интерполяции: 0 = {CLASS_NAMES[class_from]}, 1 = {CLASS_NAMES[class_to]}')
    ax.set_ylabel('среднее значение группы признаков')
    ax.set_title(f'Гладкость перехода {CLASS_NAMES[class_from]} → {CLASS_NAMES[class_to]}\n'
                 f'в латентном пространстве VAE')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(SCRIPT_DIR, out_name)
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"  Сохранено: {out_path}")


def monotonicity_score(features_seq):
    """Доля признаков, меняющихся МОНОТОННО вдоль интерполяции (без резких
    разворотов). Высокая доля = гладкое, содержательное латентное
    пространство. Считаем по знаку разностей между соседними шагами."""
    diffs = np.diff(features_seq, axis=0)          # (n_steps-1, 30)
    # признак монотонен, если все разности одного знака (или близки к нулю)
    signs = np.sign(diffs)
    monotone = []
    for j in range(features_seq.shape[1]):
        s = signs[:, j]
        s_nonzero = s[np.abs(diffs[:, j]) > 1e-4]
        if len(s_nonzero) == 0:
            monotone.append(True)  # признак почти не менялся — считаем гладким
        else:
            monotone.append(np.all(s_nonzero == s_nonzero[0]))
    return np.mean(monotone)


def main():
    device = torch.device('cpu')

    vae = ConditionalVAE()
    vae.load_state_dict(torch.load(os.path.join(SCRIPT_DIR, 'pole_vae.pth'),
                                   map_location=device))
    vae.eval()

    F_data = np.load(os.path.join(SCRIPT_DIR, 'F.npy'))
    y_data = np.load(os.path.join(SCRIPT_DIR, 'y.npy'))

    print("=" * 74)
    print("ИНТЕРПОЛЯЦИЯ МЕЖДУ КЛАССАМИ В ЛАТЕНТНОМ ПРОСТРАНСТВЕ VAE (Блок Б)")
    print("=" * 74)

    # интересующие переходы: N->V (норма -> желудочковая), N->S (норма -> суправентр.)
    transitions = [(0, 2), (0, 1)]   # (N,V), (N,S)

    for class_from, class_to in transitions:
        name = f"{CLASS_NAMES[class_from]}_to_{CLASS_NAMES[class_to]}"
        print(f"\n--- Переход {CLASS_NAMES[class_from]} → {CLASS_NAMES[class_to]} ---")

        # режим A: conditional interpolation
        feats_cond, alphas = conditional_interpolation(vae, class_from, class_to)
        plot_interpolation_heatmap(feats_cond, alphas, class_from, class_to,
                                   'по условию класса', f'interp_heatmap_{name}.png')
        plot_smoothness(feats_cond, alphas, class_from, class_to,
                        f'interp_smoothness_{name}.png')
        mono = monotonicity_score(feats_cond)
        print(f"  Доля монотонно меняющихся признаков (гладкость): {mono*100:.1f}%")

        # режим B: интерполяция реальных примеров
        idx_from = np.where(y_data == class_from)[0]
        idx_to = np.where(y_data == class_to)[0]
        if len(idx_from) > 0 and len(idx_to) > 0:
            x_from = F_data[idx_from[0]]
            x_to = F_data[idx_to[0]]
            feats_lat, _ = latent_interpolation(vae, x_from, x_to, class_from, class_to)
            mono_lat = monotonicity_score(feats_lat)
            print(f"  (режим латентной интерполяции реальных примеров: "
                  f"гладкость {mono_lat*100:.1f}%)")

    print("\n" + "=" * 74)
    print("ИТОГ ДЛЯ ГЛАВЫ 4 ДИПЛОМА")
    print("=" * 74)
    print("Тепловые карты (interp_heatmap_*.png) показывают плавное перетекание")
    print("30 признаков МП вдоль перехода между классами. Графики гладкости")
    print("(interp_smoothness_*.png) — как меняются группы признаков (модули")
    print("полюсов, сингулярные значения, аргументы). Высокая доля монотонных")
    print("признаков подтверждает, что латентное пространство содержательно и")
    print("непрерывно — классы связаны гладкими траекториями, а не изолированы.")


if __name__ == '__main__':
    main()
