import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import numpy as np
import os
from repro import set_seed

import sys as _sys
try:
    _sys.stdout.reconfigure(encoding='utf-8')
    _sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass
SEED = set_seed()  # seed из окружения (env SEED), по умолчанию 42

# Загружаем готовые файлы
X = np.load('X.npy')
y = np.load('y.npy')
F = np.load('F.npy')
print(f"Загружено: X={X.shape}, F={F.shape}, y={y.shape}")

X_tr, X_te, F_tr, F_te, y_tr, y_te = train_test_split(
    X, F, y, test_size=0.2, random_state=SEED, stratify=y
)

class_counts = np.bincount(y_tr)
weights = 1.0 / class_counts[y_tr]
sampler = WeightedRandomSampler(weights, num_samples=len(y_tr), replacement=True)

def make_loader(Xs, Fs, ys, batch=128, sampler=None, shuffle=False):
    Xt = torch.tensor(Xs[:, None, :])
    Ft = torch.tensor(Fs)
    yt = torch.tensor(ys, dtype=torch.long)
    ds = TensorDataset(Xt, Ft, yt)
    return DataLoader(ds, batch_size=batch, sampler=sampler, shuffle=shuffle)

train_loader = make_loader(X_tr, F_tr, y_tr, sampler=sampler)
test_loader  = make_loader(X_te, F_te, y_te, shuffle=False)

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
            nn.Linear(128*8, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
    def forward(self, x, f=None):
        return self.fc(self.conv(x))

class HybridCNN(nn.Module):
    def __init__(self, feat_dim=30, num_classes=5):  # было 18, стало 30
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(8),
        )
        self.feat_proj = nn.Sequential(
            nn.BatchNorm1d(feat_dim),        # нормализуем признаки МП
            nn.Linear(feat_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(128*8 + 32, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
    def forward(self, x, f):
        cnn_flat = self.conv(x).view(x.size(0), -1)
        feat_out = self.feat_proj(f)
        return self.fc(torch.cat([cnn_flat, feat_out], dim=1))

def train_model(model, name, epochs=20, lr=3e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*50}\nОбучаю: {name} | устройство: {device}\n{'='*50}")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # Плавное снижение lr
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_f1, best_state = 0, None

    for epoch in range(1, epochs+1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for xb, fb, yb in train_loader:
            xb, fb, yb = xb.to(device), fb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb, fb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
            correct += (out.argmax(1) == yb).sum().item()
            total += len(yb)
        scheduler.step()
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:2d} | loss={total_loss/total:.4f} | acc={correct/total:.4f}")

    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, fb, yb in test_loader:
            xb, fb = xb.to(device), fb.to(device)
            all_preds.extend(model(xb, fb).argmax(1).cpu().numpy())
            all_true.extend(yb.numpy())

    print(f"\nРезультаты {name}:")
    print(classification_report(all_true, all_preds,
          target_names=['N','S','V','F','Q'], digits=3))
    macro_f1 = f1_score(all_true, all_preds, average='macro')
    print(f"Macro F1: {macro_f1:.4f}")
    return macro_f1, all_true, all_preds, model


f1_base, true_base, pred_base, model_base = train_model(BaseCNN(), "Базовая CNN")
torch.save(model_base.state_dict(), 'base_cnn.pth')

f1_hybrid, true_hybrid, pred_hybrid, model_hybrid = train_model(HybridCNN(), "Гибридная CNN + Matrix Pencil")
torch.save(model_hybrid.state_dict(), 'hybrid_cnn.pth')

print(f"\n{'='*50}")
print(f"ИТОГ:")
print(f"  Базовая CNN:  Macro F1 = {f1_base:.4f}")
print(f"  Гибридная:    Macro F1 = {f1_hybrid:.4f}")
print(f"  Прирост:      {(f1_hybrid - f1_base)*100:+.2f} п.п.")

# 4-классовый Macro F1 (N,S,V,F; Q исключён) — именно он цитируется в главе 2
f1_base4 = f1_score(true_base, pred_base, labels=[0,1,2,3], average="macro")
f1_hybrid4 = f1_score(true_hybrid, pred_hybrid, labels=[0,1,2,3], average="macro")
_pc_b = f1_score(true_base, pred_base, labels=[0,1,2,3], average=None)
_pc_h = f1_score(true_hybrid, pred_hybrid, labels=[0,1,2,3], average=None)
print("PERCLASS base   N/S/V/F: " + " ".join(f"{v:.4f}" for v in _pc_b))
print("PERCLASS hybrid N/S/V/F: " + " ".join(f"{v:.4f}" for v in _pc_h))
print(f"MULTISEED tag=train seed={SEED} f1_base4={f1_base4:.4f} f1_hybrid4={f1_hybrid4:.4f} gap4_pp={(f1_hybrid4-f1_base4)*100:.2f} f1_base5={f1_base:.4f} f1_hybrid5={f1_hybrid:.4f}")

# Сохраняем результаты для отчёта
np.save('pred_base.npy',   np.array(pred_base))
np.save('pred_hybrid.npy', np.array(pred_hybrid))
np.save('true_labels.npy', np.array(true_base))
print("\nСохранено: pred_base.npy, pred_hybrid.npy, true_labels.npy")