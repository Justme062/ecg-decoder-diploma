# -*- coding: utf-8 -*-
"""
train_patientwise.py — ЧЕСТНАЯ оценка главы 2 с разбиением ПО ПАЦИЕНТАМ.

Основной train.py делит удары случайно (intra-patient): удары одного пациента
попадают и в train, и в test, поэтому модель частично запоминает морфологию
конкретных людей, а метрики завышены. Здесь используется межпациентная
(inter-patient) 5-фолдовая кросс-валидация: ни один пациент не встречается
одновременно в обучении и тесте — как это делает де Шазал [7] и как ты сама
поступаешь в главе 3.

Архитектуры BaseCNN / HybridCNN дословно совпадают с train.py.
Результат: Macro F1 (N,S,V,F) базовой и гибридной моделей как среднее ± СКО
по 5 фолдам, прирост гибрида и McNemar на out-of-fold предсказаниях.

Требует: X.npy, F.npy, y.npy, groups.npy (пересоздай main.py — он теперь
сохраняет groups.npy).

Запуск:
    python train_patientwise.py
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, classification_report

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FOLDS = 5          # число фолдов межпациентной кросс-валидации (можно снизить до 3 — быстрее)
EPOCHS = 20
LR = 3e-4
RELEVANT = [0, 1, 2, 3]           # N, S, V, F (Q исключён как вырожденный)
CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']


# ── архитектуры: ДОСЛОВНО как в train.py ──
class BaseCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(8),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, f=None):
        return self.fc(self.conv(x))


class HybridCNN(nn.Module):
    def __init__(self, feat_dim=30, num_classes=5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(8),
        )
        self.feat_proj = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Linear(feat_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 8 + 32, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, f):
        cnn_flat = self.conv(x).view(x.size(0), -1)
        return self.fc(torch.cat([cnn_flat, self.feat_proj(f)], dim=1))


def make_loader(Xs, Fs, ys, batch=128, sampler=None, shuffle=False):
    ds = TensorDataset(torch.tensor(Xs[:, None, :]), torch.tensor(Fs),
                       torch.tensor(ys, dtype=torch.long))
    return DataLoader(ds, batch_size=batch, sampler=sampler, shuffle=shuffle)


def train_eval(model, train_loader, test_loader, device):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss()
    for _ in range(EPOCHS):
        model.train()
        for xb, fb, yb in train_loader:
            xb, fb, yb = xb.to(device), fb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb, fb), yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for xb, fb, yb in test_loader:
            preds.extend(model(xb.to(device), fb.to(device)).argmax(1).cpu().numpy())
            true.extend(yb.numpy())
    return np.array(true), np.array(preds)


def mcnemar(true, pred_a, pred_b):
    """McNemar на дискордантных парах (a=база, b=гибрид)."""
    from scipy.stats import chi2
    a_ok = pred_a == true
    b_ok = pred_b == true
    b = int(np.sum(a_ok & ~b_ok))   # база верна, гибрид ошибся
    c = int(np.sum(~a_ok & b_ok))   # база ошиблась, гибрид верен
    if b + c == 0:
        return b, c, float('nan'), 1.0
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p = 1 - chi2.cdf(stat, df=1)
    return b, c, stat, p


def main():
    device = torch.device('cpu')
    X = np.load(os.path.join(SCRIPT_DIR, 'X.npy'))
    F = np.load(os.path.join(SCRIPT_DIR, 'F.npy'))
    y = np.load(os.path.join(SCRIPT_DIR, 'y.npy'))
    gpath = os.path.join(SCRIPT_DIR, 'groups.npy')
    if not os.path.exists(gpath):
        raise SystemExit("Нет groups.npy — сначала перезапусти обновлённый main.py "
                         "(он теперь сохраняет groups.npy).")
    groups = np.load(gpath, allow_pickle=True)

    print("=" * 74)
    print("МЕЖПАЦИЕНТНАЯ 5-ФОЛДОВАЯ ОЦЕНКА (глава 2, честное разбиение по пациентам)")
    print("=" * 74)
    print(f"Сегментов: {len(X)}, уникальных пациентов: {len(np.unique(groups))}, фолдов: {FOLDS}")

    sgkf = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=42)

    base_f1s, hyb_f1s = [], []
    oof_true, oof_base, oof_hybrid = [], [], []

    for fold, (tr, te) in enumerate(sgkf.split(X, y, groups), 1):
        n_pat_te = len(np.unique(groups[te]))
        print(f"\n── Фолд {fold}/{FOLDS} | test: {len(te)} ударов, {n_pat_te} пациентов ──")

        # взвешенный сэмплер против дисбаланса (как в train.py)
        cc = np.bincount(y[tr], minlength=5).astype(np.float64)
        cc[cc == 0] = 1
        w = 1.0 / cc[y[tr]]
        sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double),
                                        num_samples=len(tr), replacement=True)

        tl = make_loader(X[tr], F[tr], y[tr], sampler=sampler)
        vl = make_loader(X[te], F[te], y[te], shuffle=False)

        torch.manual_seed(42 + fold); np.random.seed(42 + fold)
        t, pb = train_eval(BaseCNN(), tl, vl, device)
        torch.manual_seed(42 + fold); np.random.seed(42 + fold)
        _, ph = train_eval(HybridCNN(), tl, vl, device)

        fb = f1_score(t, pb, labels=RELEVANT, average='macro', zero_division=0)
        fh = f1_score(t, ph, labels=RELEVANT, average='macro', zero_division=0)
        base_f1s.append(fb); hyb_f1s.append(fh)
        oof_true.extend(t); oof_base.extend(pb); oof_hybrid.extend(ph)
        print(f"   Macro F1 (N,S,V,F): база={fb:.4f}  гибрид={fh:.4f}  Δ={100*(fh-fb):+.2f} п.п.")

    base_f1s, hyb_f1s = np.array(base_f1s), np.array(hyb_f1s)
    gap = hyb_f1s - base_f1s

    print("\n" + "=" * 74)
    print("ИТОГ (межпациентная CV, среднее ± СКО по фолдам) — для главы 2")
    print("=" * 74)
    print(f"  Базовая CNN:  Macro F1 = {base_f1s.mean():.4f} ± {base_f1s.std():.4f}")
    print(f"  Гибридная:    Macro F1 = {hyb_f1s.mean():.4f} ± {hyb_f1s.std():.4f}")
    print(f"  Прирост:      {100*gap.mean():+.2f} ± {100*gap.std():.2f} п.п.  "
          f"(по фолдам: {', '.join(f'{100*g:+.2f}' for g in gap)})")

    oof_true = np.array(oof_true); oof_base = np.array(oof_base); oof_hybrid = np.array(oof_hybrid)
    b, c, stat, p = mcnemar(oof_true, oof_base, oof_hybrid)
    print(f"\n  McNemar на out-of-fold предсказаниях: b(база✓,гибрид✗)={b}, "
          f"c(база✗,гибрид✓)={c}, χ²={stat:.3f}, p={p:.4f}")
    print(f"  => {'значимо' if p < 0.05 else 'НЕ значимо'} на уровне 0.05")

    print("\n  Полный отчёт по out-of-fold предсказаниям (гибрид):")
    print(classification_report(oof_true, oof_hybrid,
          target_names=CLASS_NAMES, digits=3, zero_division=0))

    np.save(os.path.join(SCRIPT_DIR, 'pw_true.npy'), oof_true)
    np.save(os.path.join(SCRIPT_DIR, 'pw_pred_base.npy'), oof_base)
    np.save(os.path.join(SCRIPT_DIR, 'pw_pred_hybrid.npy'), oof_hybrid)
    print("\nСохранено: pw_true.npy, pw_pred_base.npy, pw_pred_hybrid.npy")


if __name__ == '__main__':
    main()
