# -*- coding: utf-8 -*-
"""
early_warning.py

Блок А плана диплома, шаг 2: обучение и сравнение моделей раннего
предупреждения о желудочковой экстрасистоле (V) по окну из 10 предыдущих
ударов (данные из build_sequences.py -> sequences.npz).

Ключевая идея эксперимента — не просто "обучить Transformer", а честно
выяснить, ДОБАВЛЯЕТ ли морфологическая динамика полюсов прогностическую
силу СВЕРХ простого тайминга (RR-интервалов). Для этого строится иерархия
моделей по нарастающей сложности:

    Baseline 0 (тривиальный):  всегда предсказывать "не V"
                                — доказывает, что accuracy обманчива.
    Baseline 1 (только RR):     логрегрессия на RR-интервалах окна
                                — решается ли задача ОДНИМ таймингом?
    Baseline 2 (только полюса): логрегрессия на МП-признаках без RR
                                — вклад морфологии в изоляции.
    Transformer (полное):       полюса + RR, последовательность с attention
                                — основная модель.

Если Transformer бьёт Baseline 1 — морфологическая динамика полюсов реально
информативна сверх тайминга. Если не бьёт — честный вывод, что задача
решается таймингом (тоже результат, достойный диплома).

КРИТИЧНО: разбиение train/test ПО ПАЦИЕНТАМ (V-события концентрируются у
части пациентов; без patient-wise split модель "узнаёт пациента", а не
предсказывает событие -> утечка и фиктивно высокие метрики).

Запуск (из папки с sequences.npz):
    python early_warning.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, roc_auc_score, precision_score,
                             recall_score, average_precision_score)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RANDOM_SEED = 42
WINDOW = 10
FEAT_PER_BEAT = 31   # 30 МП + 1 RR
TEST_FRACTION = 0.25  # доля записей (не окон!) в тест

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ───────────────────────── разбиение по пациентам ─────────────────────────
def patient_wise_split(record_ids, y, test_fraction=TEST_FRACTION, seed=RANDOM_SEED):
    """
    Делит НАБОР ЗАПИСЕЙ (не окон) на train/test так, чтобы:
      - ни одна запись не попала в обе части (нет утечки пациента);
      - доля положительных окон в train и test была сопоставима
        (стратификация на уровне записей по их "V-насыщенности").
    """
    rng = np.random.default_rng(seed)
    unique_records = np.array(sorted(set(record_ids)))

    # доля V-окон в каждой записи — для грубой стратификации
    pos_rate = {}
    for rec in unique_records:
        mask = (record_ids == rec)
        pos_rate[rec] = y[mask].mean()

    # сортируем записи по насыщенности V и раскидываем через одну,
    # чтобы и train, и test получили и "богатые", и "бедные" записи
    sorted_recs = sorted(unique_records, key=lambda r: pos_rate[r])
    test_recs, train_recs = [], []
    # берём каждую k-ю запись в тест так, чтобы набрать ~test_fraction
    step = int(round(1 / test_fraction))
    for i, rec in enumerate(sorted_recs):
        if i % step == 0:
            test_recs.append(rec)
        else:
            train_recs.append(rec)

    train_mask = np.isin(record_ids, train_recs)
    test_mask = np.isin(record_ids, test_recs)
    return train_mask, test_mask, train_recs, test_recs


# ───────────────────────── Transformer ─────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=WINDOW):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class EarlyWarningTransformer(nn.Module):
    """
    Лёгкий Transformer-encoder над последовательностью из WINDOW ударов.
    Каждый удар — вектор из 31 признака (30 МП + RR). Позиционное кодирование
    задаёт порядок ударов. После энкодера — усреднение по времени и линейный
    классификатор в 2 класса (следующий удар V / не V).
    """
    def __init__(self, feat_dim=FEAT_PER_BEAT, d_model=64, nhead=4,
                 num_layers=2, dim_ff=128, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h)
        h = h.mean(dim=1)  # усреднение по времени
        return self.classifier(h)


def train_transformer(X_train, y_train, X_test, y_test, epochs=40,
                      batch_size=256, lr=1e-3, device='cpu',
                      X_val=None, y_val=None):
    model = EarlyWarningTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # взвешивание классов против дисбаланса (~1:13)
    pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    n = len(Xt)

    use_val = X_val is not None and y_val is not None and len(set(y_val)) > 1
    best_val_auc, best_state, patience, bad_epochs = -1.0, None, 6, 0

    print(f"\nОбучаю Transformer на {n} окнах, до {epochs} эпох "
          f"(pos_weight={pos_weight:.1f}"
          f"{', early stopping по val AUC' if use_val else ''})...")
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = Xt[idx].to(device)
            yb = yt[idx].to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)

        # early stopping по валидационному AUC (если валидация передана)
        if use_val:
            model.eval()
            with torch.no_grad():
                Xval_t = torch.tensor(X_val, dtype=torch.float32).to(device)
                val_prob = torch.softmax(model(Xval_t), dim=1)[:, 1].cpu().numpy()
            val_auc = roc_auc_score(y_val, val_prob)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:2d} | loss={total_loss/n:.4f} | val_AUC={val_auc:.4f}")
            if bad_epochs >= patience:
                print(f"  Ранняя остановка на эпохе {epoch} (лучший val_AUC={best_val_auc:.4f})")
                break
        else:
            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:2d} | loss={total_loss/n:.4f}")

    if use_val and best_state is not None:
        model.load_state_dict(best_state)

    # предсказания на тесте (вероятность класса V)
    model.eval()
    Xtest_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(Xtest_t)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return model, probs


# ───────────────────────── метрики ─────────────────────────
def evaluate(name, y_true, y_prob, threshold=0.5):
    """Единый расчёт метрик по вероятностям. Для baseline 0 (константа)
    y_prob — это массив нулей."""
    y_pred = (y_prob >= threshold).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    # AUC определён только если в y_prob есть вариация и оба класса в y_true
    try:
        auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float('nan')
        ap = average_precision_score(y_true, y_prob) if len(set(y_true)) > 1 else float('nan')
    except Exception:
        auc = ap = float('nan')
    print(f"  {name:<32} F1={f1:.4f}  P={prec:.4f}  R={rec:.4f}  "
          f"AUC={auc:.4f}  AP={ap:.4f}")
    return {'name': name, 'f1': f1, 'precision': prec, 'recall': rec,
            'auc': auc, 'ap': ap}


def main():
    device = torch.device('cpu')

    data = np.load(os.path.join(SCRIPT_DIR, 'sequences.npz'), allow_pickle=True)
    X_seq = data['X_seq']          # (N, WINDOW, 31)
    y_next = data['y_next']        # (N,)
    record_ids = data['record_ids']

    print("=" * 78)
    print("РАННЕЕ ПРЕДУПРЕЖДЕНИЕ О V: СРАВНЕНИЕ МОДЕЛЕЙ (Блок А)")
    print("=" * 78)
    print(f"Всего окон: {len(X_seq)}, положительных: {y_next.sum()} "
          f"({y_next.mean()*100:.2f}%)")

    train_mask, test_mask, train_recs, test_recs = patient_wise_split(
        record_ids, y_next
    )
    print(f"\nРазбиение ПО ПАЦИЕНТАМ:")
    print(f"  train: {train_mask.sum()} окон из {len(train_recs)} записей, "
          f"положительных {y_next[train_mask].mean()*100:.2f}%")
    print(f"  test:  {test_mask.sum()} окон из {len(test_recs)} записей, "
          f"положительных {y_next[test_mask].mean()*100:.2f}%")

    X_train, y_train = X_seq[train_mask], y_next[train_mask]
    X_test, y_test = X_seq[test_mask], y_next[test_mask]

    # выделяем валидацию ИЗ TRAIN тоже по пациентам (для early stopping
    # Transformer'а) — берём ~20% обучающих записей, снова без утечки
    train_record_ids = record_ids[train_mask]
    val_train_mask, val_mask, _, val_recs = patient_wise_split(
        train_record_ids, y_train, test_fraction=0.2, seed=RANDOM_SEED + 1
    )
    X_tr_core, y_tr_core = X_train[val_train_mask], y_train[val_train_mask]
    X_val, y_val = X_train[val_mask], y_train[val_mask]
    print(f"  (из train выделено {val_mask.sum()} окон валидации "
          f"из {len(val_recs)} записей для early stopping)")

    # плоские представления для baseline-логрегрессий
    X_train_flat = X_train.reshape(len(X_train), -1)   # (N, WINDOW*31)
    X_test_flat = X_test.reshape(len(X_test), -1)

    # индексы RR-признака и МП-признаков в плоском векторе
    rr_idx = [j * FEAT_PER_BEAT + 30 for j in range(WINDOW)]          # только RR
    mp_idx = [j * FEAT_PER_BEAT + k for j in range(WINDOW) for k in range(30)]  # только МП

    print("\n" + "=" * 78)
    print("РЕЗУЛЬТАТЫ (все метрики на тестовых пациентах, ранее не виденных)")
    print("=" * 78)

    results = []

    # ── Baseline 0: всегда "не V" ──
    results.append(evaluate("Baseline 0 (всегда не-V)", y_test,
                            np.zeros(len(y_test))))

    # ── Baseline 1: только RR-интервалы ──
    scaler_rr = StandardScaler().fit(X_train_flat[:, rr_idx])
    clf_rr = LogisticRegression(max_iter=2000, class_weight='balanced')
    clf_rr.fit(scaler_rr.transform(X_train_flat[:, rr_idx]), y_train)
    prob_rr = clf_rr.predict_proba(scaler_rr.transform(X_test_flat[:, rr_idx]))[:, 1]
    results.append(evaluate("Baseline 1 (только RR, логрег)", y_test, prob_rr))

    # ── Baseline 2: только МП-признаки ──
    scaler_mp = StandardScaler().fit(X_train_flat[:, mp_idx])
    clf_mp = LogisticRegression(max_iter=2000, class_weight='balanced')
    clf_mp.fit(scaler_mp.transform(X_train_flat[:, mp_idx]), y_train)
    prob_mp = clf_mp.predict_proba(scaler_mp.transform(X_test_flat[:, mp_idx]))[:, 1]
    results.append(evaluate("Baseline 2 (только полюса, логрег)", y_test, prob_mp))

    # ── Baseline 3: RR + полюса, но без учёта порядка (логрег на всём плоском) ──
    scaler_all = StandardScaler().fit(X_train_flat)
    clf_all = LogisticRegression(max_iter=2000, class_weight='balanced')
    clf_all.fit(scaler_all.transform(X_train_flat), y_train)
    prob_all = clf_all.predict_proba(scaler_all.transform(X_test_flat))[:, 1]
    results.append(evaluate("Baseline 3 (RR+полюса, без порядка)", y_test, prob_all))

    # ── Transformer: RR + полюса + порядок (attention) ──
    _, prob_tf = train_transformer(X_tr_core, y_tr_core, X_test, y_test,
                                   X_val=X_val, y_val=y_val, device=device)
    results.append(evaluate("Transformer (RR+полюса+порядок)", y_test, prob_tf))

    # ── итоговая сводка ──
    print("\n" + "=" * 78)
    print("ИТОГ ДЛЯ ГЛАВЫ 3 ДИПЛОМА")
    print("=" * 78)
    print(f"{'Модель':<38}{'F1':<9}{'AUC':<9}{'AP':<9}")
    print("-" * 65)
    for r in results:
        print(f"{r['name']:<38}{r['f1']:<9.4f}{r['auc']:<9.4f}{r['ap']:<9.4f}")

    print("\nКлючевые вопросы:")
    b1 = next(r for r in results if 'Baseline 1' in r['name'])
    tf = next(r for r in results if 'Transformer' in r['name'])
    b2 = next(r for r in results if 'Baseline 2' in r['name'])
    if not np.isnan(tf['auc']) and not np.isnan(b1['auc']):
        delta = (tf['auc'] - b1['auc']) * 100
        print(f"  Transformer vs только-RR (Baseline 1) по AUC: {delta:+.2f} п.п.")
        if delta > 1:
            print("  => морфологическая динамика полюсов ДОБАВЛЯЕТ сигнал сверх тайминга")
        else:
            print("  => задача решается преимущественно таймингом (RR); "
                  "полюса дают мало сверх него")
    if not np.isnan(b2['auc']):
        print(f"  Только полюса (Baseline 2) AUC={b2['auc']:.4f} — "
              f"насколько морфология предсказывает V без тайминга")

    np.savez(os.path.join(SCRIPT_DIR, 'early_warning_results.npz'),
             y_test=y_test, prob_transformer=prob_tf, prob_rr=prob_rr,
             prob_mp=prob_mp, test_recs=np.array(test_recs))
    print("\nСохранено: early_warning_results.npz")


if __name__ == '__main__':
    main()
