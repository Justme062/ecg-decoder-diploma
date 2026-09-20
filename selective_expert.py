import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

# ───── Загрузка данных ─────
X = np.load('X.npy')
y = np.load('y.npy')
F = np.load('F.npy')  # (109468, 30)

_, X_te, _, F_te, _, y_te = train_test_split(
    X, F, y, test_size=0.2, random_state=42, stratify=y
)

def make_loader(Xs, Fs, ys, batch=128):
    Xt = torch.tensor(Xs[:, None, :])
    Ft = torch.tensor(Fs)
    yt = torch.tensor(ys, dtype=torch.long)
    return DataLoader(TensorDataset(Xt, Ft, yt), batch_size=batch)

test_loader = make_loader(X_te, F_te, y_te)

# ───── Те же архитектуры ─────
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
    def __init__(self, feat_dim=30, num_classes=5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(8),
        )
        self.feat_proj = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
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

# ───── Загружаем обученные веса ─────
device = torch.device('cpu')
base_model   = BaseCNN().to(device)
hybrid_model = HybridCNN().to(device)
base_model.load_state_dict(torch.load('base_cnn.pth',   map_location=device))
hybrid_model.load_state_dict(torch.load('hybrid_cnn.pth', map_location=device))
base_model.eval()
hybrid_model.eval()

# ───── Selective Expert ─────
THRESHOLD = 0.90  # порог уверенности

all_preds, all_true = [], []
expert_calls = 0

with torch.no_grad():
    for xb, fb, yb in test_loader:
        xb, fb = xb.to(device), fb.to(device)

        # Шаг 1: базовая CNN + softmax
        logits_base = base_model(xb)
        probs = torch.softmax(logits_base, dim=1)
        confidence, pred_base = probs.max(dim=1)

        # Шаг 2: для неуверенных — гибридная модель
        pred_final = pred_base.clone()
        uncertain = confidence < THRESHOLD
        if uncertain.any():
            expert_calls += uncertain.sum().item()
            logits_hybrid = hybrid_model(xb[uncertain], fb[uncertain])
            pred_final[uncertain] = logits_hybrid.argmax(dim=1)

        all_preds.extend(pred_final.cpu().numpy())
        all_true.extend(yb.numpy())

total = len(all_true)
print(f"\nSelective Expert (порог={THRESHOLD}):")
print(f"  Вызовов эксперта: {expert_calls}/{total} ({100*expert_calls/total:.1f}%)")
print(classification_report(all_true, all_preds,
      target_names=['N','S','V','F','Q'], digits=3))
print(f"Macro F1: {f1_score(all_true, all_preds, average='macro'):.4f}")