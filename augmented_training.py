# -*- coding: utf-8 -*-
"""
augmented_training.py

Блок Б плана диплома, финальная интеграция: проверка, поднимает ли
VAE-аугментация метрики НАСТОЯЩЕГО нейросетевого классификатора (а не
вспомогательного линейного/MLP-зонда, как в TSTR), на полной 5-классовой
задаче.

Методологическая тонкость: полная гибридная CNN принимает И сырой сигнал
(180 отсчётов), И МП-признаки (30). VAE генерирует ТОЛЬКО МП-признаки —
сырого сигнала для синтетических примеров нет. Поэтому честно
аугментировать можно именно МП-ветвь. Здесь обучается нейросетевой
классификатор на 30 МП-признаках (та же архитектура feat_proj +
классификатор, что в гибридной CNN, но без сверточной части) — и
сравнивается обучение С синтетической аугментацией редких классов и БЕЗ неё.

Это замыкает логику Блока Б: VAE не просто "генерирует правдоподобные
векторы" (что показал TSTR), а реально влияет на качество нейросетевого
классификатора того же семейства, что основная модель.

Сравниваются:
    A) Baseline: нейросеть на реальных МП-признаках (со взвешенным
       сэмплером против дисбаланса, как в основном train.py).
    B) Augmented: то же + синтетические примеры редких классов (S, F) из VAE.

Метрики: per-class F1 и Macro F1 (по классам N, S, V, F — Q исключён как
вырожденный, см. основной пайплайн).

Запуск (из папки с pole_vae.pth, F.npy, y.npy):
    python augmented_training.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEAT_DIM = 30
NUM_CLASSES = 5
LATENT_DIM = 8
CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']
RARE_CLASSES = [1, 3]        # S, F — пополняем синтетикой
RELEVANT_CLASSES = [0, 1, 2, 3]  # N,S,V,F — Q исключён из Macro F1
import sys as _sys
try:
    _sys.stdout.reconfigure(encoding='utf-8')
    _sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass
RANDOM_SEED = int(os.environ.get('SEED', 42))

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
try:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
except Exception:
    pass


# ─── VAE (для генерации; архитектура как в vae_pole_features.py) ───
class ConditionalVAE(nn.Module):
    def __init__(self, feat_dim=FEAT_DIM, num_classes=NUM_CLASSES, latent_dim=LATENT_DIM, hidden=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.encoder = nn.Sequential(
            nn.Linear(feat_dim + num_classes, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, feat_dim))

    def decode(self, z, c):
        return self.decoder(torch.cat([z, c], dim=1))

    @torch.no_grad()
    def sample(self, class_idx, n):
        z = torch.randn(n, self.latent_dim)
        c = torch.zeros(n, self.num_classes)
        c[:, class_idx] = 1.0
        return self.decode(z, c).numpy()


# ─── классификатор на МП-признаках (ветвь гибридной CNN без свёрток) ───
class MPClassifier(nn.Module):
    """Повторяет структуру feat-ветви и головы гибридной CNN, но работает
    только на 30 МП-признаках. BatchNorm на входе — та же критичная деталь,
    что и в гибридной модели (признаки разного масштаба)."""
    def __init__(self, feat_dim=FEAT_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Linear(feat_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def make_loader(X, y, batch=256, use_sampler=True, drop_last=True):
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    ds = TensorDataset(Xt, yt)
    if use_sampler:
        counts = np.bincount(y, minlength=NUM_CLASSES).astype(np.float64)
        counts[counts == 0] = 1
        w = 1.0 / counts[y]
        sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double),
                                        num_samples=len(y), replacement=True)
        return DataLoader(ds, batch_size=batch, sampler=sampler, drop_last=drop_last)
    return DataLoader(ds, batch_size=batch, shuffle=True, drop_last=drop_last)


def train_classifier(X_train, y_train, X_test, y_test, epochs=30, device='cpu',
                     tag="", n_runs=3):
    """Обучает классификатор n_runs раз с разными зёрнами и усредняет
    метрики — разница между A и B на редких классах может быть шумной при
    одном запуске, усреднение делает сравнение устойчивым."""
    f1_relevant_runs = []
    per_class_runs = []
    for run in range(n_runs):
        torch.manual_seed(RANDOM_SEED + run)
        model = MPClassifier().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        loader = make_loader(X_train, y_train)

        for epoch in range(1, epochs + 1):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(X_test, dtype=torch.float32).to(device)).argmax(1).cpu().numpy()
        f1_relevant_runs.append(
            f1_score(y_test, pred, labels=RELEVANT_CLASSES, average='macro', zero_division=0))
        per_class_runs.append(
            f1_score(y_test, pred, labels=[0, 1, 2, 3, 4], average=None, zero_division=0))

    f1_relevant = np.mean(f1_relevant_runs)
    per_class = np.mean(per_class_runs, axis=0)
    return f1_relevant, per_class, None


def main():
    device = torch.device('cpu')

    F_data = np.load(os.path.join(SCRIPT_DIR, 'F.npy'))
    y_data = np.load(os.path.join(SCRIPT_DIR, 'y.npy'))

    print("=" * 74)
    print("ИНТЕГРАЦИЯ VAE-АУГМЕНТАЦИИ В НЕЙРОСЕТЕВОЙ КЛАССИФИКАТОР (Блок Б)")
    print("=" * 74)
    for i, name in enumerate(CLASS_NAMES):
        print(f"  Класс {i} {name}: {(y_data == i).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        F_data, y_data, test_size=0.2, random_state=RANDOM_SEED, stratify=y_data)

    # ── A) baseline без аугментации ──
    print("\nОбучаю A) baseline (реальные данные, взвешенный сэмплер)...")
    f1_a, per_a, _ = train_classifier(X_train, y_train, X_test, y_test, device=device)

    # ── B) с VAE-аугментацией редких классов ──
    print("Генерирую синтетику и обучаю B) с аугментацией...")
    vae = ConditionalVAE()
    vae.load_state_dict(torch.load(os.path.join(SCRIPT_DIR, 'pole_vae.pth'),
                                   map_location=device))
    vae.eval()

    # сколько добавить: доводим редкие классы до ~медианной численности
    counts = np.bincount(y_train, minlength=NUM_CLASSES)
    target = int(np.median(counts[counts > 0]))
    synth_X, synth_y = [], []
    for cls in RARE_CLASSES:
        need = max(0, target - counts[cls])
        if need > 0:
            synth_X.append(vae.sample(cls, need))
            synth_y.append(np.full(need, cls))
            print(f"  класс {CLASS_NAMES[cls]}: +{need} синтетических (до {target})")
    if synth_X:
        X_aug = np.concatenate([X_train] + synth_X, axis=0).astype(np.float32)
        y_aug = np.concatenate([y_train] + synth_y, axis=0).astype(np.int64)
    else:
        X_aug, y_aug = X_train, y_train

    f1_b, per_b, _ = train_classifier(X_aug, y_aug, X_test, y_test, device=device)

    # ── сравнение ──
    print("\n" + "=" * 74)
    print("РЕЗУЛЬТАТЫ (per-class F1 на реальном тесте, среднее по 3 запускам)")
    print("=" * 74)
    print(f"{'Класс':<8}{'A) baseline':<15}{'B) +VAE-аугм.':<16}{'Δ (B−A)':<10}")
    print("-" * 49)
    for i in RELEVANT_CLASSES:
        delta = (per_b[i] - per_a[i]) * 100
        print(f"{CLASS_NAMES[i]:<8}{per_a[i]:<15.4f}{per_b[i]:<16.4f}{delta:<+10.2f}")
    print("-" * 49)
    print(f"{'Macro':<8}{f1_a:<15.4f}{f1_b:<16.4f}{(f1_b-f1_a)*100:<+10.2f}")
    print(f"MULTISEED tag=aug seed={RANDOM_SEED} macro_a={f1_a:.4f} macro_b={f1_b:.4f} gap_pp={(f1_b-f1_a)*100:.2f} dN={(per_b[0]-per_a[0])*100:.2f} dS={(per_b[1]-per_a[1])*100:.2f} dV={(per_b[2]-per_a[2])*100:.2f} dF={(per_b[3]-per_a[3])*100:.2f}")

    print("\nВывод для главы 4:")
    if f1_b > f1_a:
        print(f"  VAE-аугментация УЛУЧШАЕТ Macro F1 нейросетевого классификатора "
              f"на {(f1_b-f1_a)*100:+.2f} п.п.")
        print("  Это подтверждает практическую ценность генеративной модели не только")
        print("  на вспомогательных зондах (TSTR), но и на классификаторе того же")
        print("  нейросетевого семейства, что основная гибридная модель.")
    else:
        print(f"  VAE-аугментация не улучшает Macro F1 ({(f1_b-f1_a)*100:+.2f} п.п.) —")
        print("  на данном классификаторе эффект нейтрален; синтетика полезна для")
        print("  слабых зондов, но сильный нейросетевой классификатор с взвешенным")
        print("  сэмплером уже извлекает достаточно из реальных данных.")


if __name__ == '__main__':
    main()
