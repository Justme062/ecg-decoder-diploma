import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Загружаем результаты
true_labels  = np.load('true_labels.npy')
pred_base    = np.load('pred_base.npy')
pred_hybrid  = np.load('pred_hybrid.npy')

CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']

# ── График 1: Confusion Matrix для базовой CNN ──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, preds, title in zip(axes,
    [pred_base, pred_hybrid],
    ['Базовая CNN', 'Гибридная CNN + МП']):
    cm = confusion_matrix(true_labels, preds, normalize='true')
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, colorbar=False, cmap='Blues', values_format='.2f')
    ax.set_title(title, fontsize=14, fontweight='bold')

plt.suptitle('Нормализованные матрицы ошибок', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("Сохранено: confusion_matrices.png")

# ── График 2: F1 по классам — сравнение трёх моделей ──
from sklearn.metrics import f1_score

# Считаем F1 по каждому классу для всех трёх моделей
pred_expert = np.load('pred_hybrid.npy')  # заглушка, заменим ниже

# Пересчитываем selective expert предсказания
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

X = np.load('X.npy')
y = np.load('y.npy')
F = np.load('F.npy')
_, X_te, _, F_te, _, y_te = train_test_split(
    X, F, y, test_size=0.2, random_state=42, stratify=y)

class BaseCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1,32,7,padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32,64,5,padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64,128,3,padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(8),
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(1024,128), nn.ReLU(), nn.Dropout(0.4), nn.Linear(128,5))
    def forward(self, x, f=None): return self.fc(self.conv(x))

class HybridCNN(nn.Module):
    def __init__(self, feat_dim=30, num_classes=5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1,32,7,padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32,64,5,padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64,128,3,padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(8),
        )
        self.feat_proj = nn.Sequential(nn.BatchNorm1d(feat_dim), nn.Linear(feat_dim,64), nn.ReLU(), nn.Linear(64,32), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(1024+32,256), nn.ReLU(), nn.Dropout(0.4), nn.Linear(256,5))
    def forward(self, x, f):
        return self.fc(torch.cat([self.conv(x).view(x.size(0),-1), self.feat_proj(f)], dim=1))

base_m   = BaseCNN();   base_m.load_state_dict(torch.load('base_cnn.pth',   map_location='cpu')); base_m.eval()
hybrid_m = HybridCNN(); hybrid_m.load_state_dict(torch.load('hybrid_cnn.pth', map_location='cpu')); hybrid_m.eval()

Xt = torch.tensor(X_te[:,None,:]); Ft = torch.tensor(F_te); yt = torch.tensor(y_te, dtype=torch.long)
loader = DataLoader(TensorDataset(Xt, Ft, yt), batch_size=128)

pred_expert = []
with torch.no_grad():
    for xb, fb, yb in loader:
        logits = base_m(xb)
        probs  = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        uncertain = conf < 0.90
        if uncertain.any():
            pred[uncertain] = hybrid_m(xb[uncertain], fb[uncertain]).argmax(dim=1)
        pred_expert.extend(pred.numpy())
pred_expert = np.array(pred_expert)

# F1 по классам
f1_base_cls   = f1_score(true_labels, pred_base,    average=None)
f1_hybrid_cls = f1_score(true_labels, pred_hybrid,  average=None)
f1_expert_cls = f1_score(y_te,        pred_expert,  average=None)

x = np.arange(len(CLASS_NAMES))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width, f1_base_cls,   width, label='Базовая CNN',        color='#4C72B0')
ax.bar(x,         f1_hybrid_cls, width, label='Гибридная CNN+МП',   color='#DD8452')
ax.bar(x + width, f1_expert_cls, width, label='Selective Expert',   color='#55A868')

ax.set_xlabel('Класс аритмии', fontsize=13)
ax.set_ylabel('F1-мера', fontsize=13)
ax.set_title('F1-мера по классам: сравнение трёх моделей', fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'{n}\n({c})' for n, c in zip(
    CLASS_NAMES, ['Норма','Суправентр.','Желудочк.','Сливные','Прочие'])], fontsize=11)
ax.set_ylim(0.5, 1.02)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

for bars in [ax.containers[0], ax.containers[1], ax.containers[2]]:
    ax.bar_label(bars, fmt='%.3f', fontsize=8, padding=2)

plt.tight_layout()
plt.savefig('f1_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Сохранено: f1_comparison.png")
print("\nГотово! Все графики сохранены.")