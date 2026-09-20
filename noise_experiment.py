import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

# ───── Загрузка данных ─────
X_clean = np.load('X.npy')
y       = np.load('y.npy')
F_clean = np.load('F.npy')  # признаки МП от чистого сигнала

def add_noise(X, snr_db):
    """Добавляет гауссовский белый шум с заданным SNR в дБ."""
    if snr_db is None:
        return X.copy()
    snr_linear = 10 ** (snr_db / 10)
    signal_power = np.mean(X ** 2, axis=1, keepdims=True)
    noise_power  = signal_power / snr_linear
    noise = np.random.randn(*X.shape).astype(np.float32)
    noise *= np.sqrt(noise_power)
    return X + noise

# ───── Архитектуры ─────
class BaseCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1,32,7,padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32,64,5,padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64,128,3,padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(8),
        )
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(1024,128), nn.ReLU(), nn.Dropout(0.4), nn.Linear(128,5)
        )
    def forward(self, x, f=None): return self.fc(self.conv(x))

class HybridCNN(nn.Module):
    def __init__(self, feat_dim=30, num_classes=5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1,32,7,padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32,64,5,padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64,128,3,padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(8),
        )
        self.feat_proj = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Linear(feat_dim,64), nn.ReLU(),
            nn.Linear(64,32), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(1024+32,256), nn.ReLU(), nn.Dropout(0.4), nn.Linear(256,5)
        )
    def forward(self, x, f):
        return self.fc(torch.cat([self.conv(x).view(x.size(0),-1), self.feat_proj(f)], dim=1))

def run_experiment(snr_db, epochs=15):
    label = f"SNR={snr_db} дБ" if snr_db is not None else "Чистый сигнал"
    print(f"\n{'='*55}")
    print(f"  Эксперимент: {label}")
    print(f"{'='*55}")

    # Зашумляем сигналы
    X_noisy = add_noise(X_clean, snr_db)

    # Разбиение
    X_tr, X_te, F_tr, F_te, y_tr, y_te = train_test_split(
        X_noisy, F_clean, y, test_size=0.2, random_state=42, stratify=y
    )

    # Балансировка
    counts  = np.bincount(y_tr)
    weights = 1.0 / counts[y_tr]
    sampler = WeightedRandomSampler(weights, len(y_tr), replacement=True)

    def make_loader(Xs, Fs, ys, sampler=None, shuffle=False):
        ds = TensorDataset(
            torch.tensor(Xs[:,None,:]),
            torch.tensor(Fs),
            torch.tensor(ys, dtype=torch.long)
        )
        return DataLoader(ds, batch_size=128, sampler=sampler, shuffle=shuffle)

    train_loader = make_loader(X_tr, F_tr, y_tr, sampler=sampler)
    test_loader  = make_loader(X_te, F_te, y_te, shuffle=False)

    device = torch.device('cpu')

    def train_and_eval(model, name):
        model = model.to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
        sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        crit  = nn.CrossEntropyLoss()

        for epoch in range(1, epochs+1):
            model.train()
            for xb, fb, yb in train_loader:
                xb, fb, yb = xb.to(device), fb.to(device), yb.to(device)
                opt.zero_grad()
                loss = crit(model(xb, fb), yb)
                loss.backward()
                opt.step()
            sch.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, fb, yb in test_loader:
                preds.extend(model(xb.to(device), fb.to(device)).argmax(1).cpu().numpy())
                trues.extend(yb.numpy())

        f1 = f1_score(trues, preds, average='macro')
        f1_per = f1_score(trues, preds, average=None)
        print(f"  {name:35s} Macro F1={f1:.4f}  "
              f"[N={f1_per[0]:.3f} S={f1_per[1]:.3f} "
              f"V={f1_per[2]:.3f} F={f1_per[3]:.3f} Q={f1_per[4]:.3f}]")
        return f1

    f1_base   = train_and_eval(BaseCNN(),   "Базовая CNN")
    f1_hybrid = train_and_eval(HybridCNN(), "Гибридная CNN+МП")
    delta = (f1_hybrid - f1_base) * 100
    print(f"  {'Прирост':35s} {delta:+.2f} п.п.")
    return f1_base, f1_hybrid, delta

# ───── Эксперименты ─────
snr_levels = [None, 20, 10, 5]  # None = чистый сигнал
results = []

for snr in snr_levels:
    f1b, f1h, delta = run_experiment(snr, epochs=15)
    results.append((snr, f1b, f1h, delta))

# ───── Итоговая таблица ─────
print(f"\n{'='*65}")
print(f"{'ИТОГОВАЯ ТАБЛИЦА':^65}")
print(f"{'='*65}")
print(f"{'Условие':<20} {'Baseline F1':>12} {'Hybrid F1':>12} {'Прирост':>12}")
print(f"{'-'*65}")
for snr, f1b, f1h, delta in results:
    label = "Чистый сигнал" if snr is None else f"SNR = {snr} дБ"
    print(f"{label:<20} {f1b:>12.4f} {f1h:>12.4f} {delta:>+11.2f} п.п.")
print(f"{'='*65}")

# Сохраняем для графика
np.save('noise_results.npy', np.array(results, dtype=object))
print("\nСохранено: noise_results.npy")