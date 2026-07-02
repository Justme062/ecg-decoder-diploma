# -*- coding: utf-8 -*-
"""
vae_pole_features.py

Блок Б плана диплома: генеративная модель (Conditional VAE) в 30-мерном
пространстве признаков матричных пучков (МП). Задача — синтетическое
пополнение редких классов (прежде всего F, затем S), чтобы бороться с
дисбалансом, который мы измерили экспериментально: на INCART класс F имеет
precision=0.012 при поддержке всего 185 примеров из 161 835.

Почему VAE в ПРОСТРАНСТВЕ ПРИЗНАКОВ, а не на сырых сегментах (180 отсчётов):
    - 30-мерное пространство МП кратно компактнее и физически осмысленно
      (сингулярные значения + модули/аргументы полюсов), генерация в нём
      устойчивее, чем GAN/VAE на сыром сигнале.
    - Не нужно синтезировать реалистичную форму QRS с нуля — работаем с
      уже извлечённой структурной информацией.

Валидация по протоколу TSTR (Train on Synthetic, Test on Real):
    классификатор обучается ТОЛЬКО на синтетике, тестируется на реальных,
    ранее не виденных примерах того же класса. Если качество сопоставимо
    с классификатором на реальных данных — синтетика несёт полезный сигнал,
    а не шум.

Запуск (из папки с X.npy, F.npy, y.npy):
    python vae_pole_features.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_nn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, classification_report
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


# ───────────────────────── Conditional VAE ─────────────────────────
class ConditionalVAE(nn.Module):
    """
    VAE, обусловленный на класс (one-hot вектор конкатенируется на входе
    энкодера и декодера) — так одна модель умеет генерировать образцы
    любого из 5 классов, а не только того, на котором обучалась отдельно.
    """
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

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        return self.decoder(torch.cat([z, c], dim=1))

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, c)
        return recon, mu, logvar

    @torch.no_grad()
    def sample(self, class_idx, n_samples, device='cpu'):
        z = torch.randn(n_samples, self.latent_dim, device=device)
        c = torch.zeros(n_samples, self.num_classes, device=device)
        c[:, class_idx] = 1.0
        return self.decode(z, c)


def vae_loss(recon_x, x, mu, logvar, beta=0.3, free_bits=0.05):
    """Реконструкция (MSE) + KL-дивергенция с весом beta и FREE BITS.

    Free bits (Kingma et al., 2016) — стандартная защита от posterior
    collapse, более надёжная, чем один только KL annealing. Идея: для
    каждого измерения латента считаем его собственный вклад в KL и
    "подрезаем" снизу порогом free_bits (в нац/измерение). Пока KL по
    измерению НИЖЕ порога, градиент от KL-члена по этому измерению равен
    нулю — то есть модели незачем схлопывать его ЕЩЁ сильнее, регуляризация
    "не давит" дальше вниз. Без этого механизма оптимизатор при малейшей
    возможности выбирает самое дешёвое решение — обнулить KL полностью
    и игнорировать z, что мы наблюдали на реальных МП-признаках
    (0 активных измерений из 8).

    free_bits=0.05 нат на измерение — умеренное значение: достаточно,
    чтобы не дать latent полностью схлопнуться, но не настолько большое,
    чтобы задушить реконструкцию бессмысленным шумом в z.
    """
    recon_loss = F_nn.mse_loss(recon_x, x, reduction='mean')

    # KL по каждому измерению отдельно (не усредняем сразу, чтобы применить free bits поэлементно)
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (batch, latent_dim)
    kl_per_dim_clamped = torch.clamp(kl_per_dim, min=free_bits)
    kl = kl_per_dim_clamped.mean(dim=0).sum()  # суммируем по измерениям, усредняем по батчу

    # для логирования — "настоящий" KL без подрезки, чтобы видеть реальный коллапс/не-коллапс
    true_kl = kl_per_dim.mean()

    return recon_loss + beta * kl, recon_loss.item(), true_kl.item()


def one_hot(labels, num_classes):
    return torch.eye(num_classes)[labels]


def train_vae(F_data, y_data, epochs=300, batch_size=256, lr=1e-3, device='cpu',
              beta_max=0.1, warmup_epochs=150, free_bits=0.05, max_class_weight_ratio=10.0):
    """Обучает Conditional VAE на ВСЕХ классах сразу (условность на класс
    позволяет одной модели обслуживать все классы). Использует ВЗВЕШЕННОЕ
    сэмплирование по классам вместо равномерного перемешивания: при сильном
    дисбалансе (класс F — 0.7% данных) редкий класс почти не появляется в
    батчах за эпоху при обычном shuffle, и декодер получает недостаточно
    градиентного сигнала, чтобы точно выучить его условное распределение
    (а не просто "не путать с другими классами", как для классификации —
    для ГЕНЕРАЦИИ нужна точная форма распределения, это более требовательная
    задача к количеству сигнала на класс).

    Веса семплирования обратно пропорциональны частоте класса, но ОГРАНИЧЕНЫ
    сверху (max_class_weight_ratio) — иначе редчайший класс (Q, если он есть)
    забьёт батчи почти одними своими повторами и собьёт обучение остальных
    классов. Разумный компромисс, а не полное выравнивание частот.

    KL annealing + free bits: см. vae_loss — защита от posterior collapse,
    без которой декодер обучается игнорировать z."""
    model = ConditionalVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    Ft = torch.tensor(F_data, dtype=torch.float32)
    yt = torch.tensor(y_data, dtype=torch.long)
    n = len(Ft)

    class_counts = np.bincount(y_data, minlength=NUM_CLASSES).astype(np.float64)
    class_counts[class_counts == 0] = 1  # защита от деления на 0 для отсутствующих классов
    inv_freq = class_counts.sum() / class_counts
    inv_freq = np.clip(inv_freq, None, max_class_weight_ratio * (class_counts.sum() / class_counts.max()))
    sample_weights = inv_freq[y_data]
    sampler_weights_t = torch.tensor(sample_weights, dtype=torch.double)

    print(f"Обучаю Conditional VAE на {n} примерах, {epochs} эпох "
          f"(KL annealing: 0 -> {beta_max} за {warmup_epochs} эпох, "
          f"взвешенное сэмплирование редких классов)...")
    for epoch in range(1, epochs + 1):
        beta = beta_max * min(1.0, epoch / warmup_epochs)
        # взвешенный сэмплинг вместо torch.randperm: индексы редких классов
        # получают пропорционально больше шансов попасть в батч за эпоху
        perm = torch.multinomial(sampler_weights_t, n, replacement=True)
        total_loss = total_recon = total_kl = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = Ft[idx].to(device)
            cb = one_hot(yt[idx], NUM_CLASSES).to(device)

            optimizer.zero_grad()
            recon, mu, logvar = model(xb, cb)
            loss, recon_l, kl_l = vae_loss(recon, xb, mu, logvar, beta=beta, free_bits=free_bits)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(idx)
            total_recon += recon_l * len(idx)
            total_kl += kl_l * len(idx)

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | beta={beta:.3f} | loss={total_loss/n:.4f} | "
                  f"recon={total_recon/n:.4f} | KL={total_kl/n:.4f}")

    return model


# ───────────────────────── TSTR-валидация ─────────────────────────
def make_logreg():
    return LogisticRegression(max_iter=2000, class_weight='balanced')


def make_mlp():
    """Небольшой двухслойный MLP как нелинейный зонд. МП-признаки, как мы
    выяснили, плохо разделимы линейно (логрегрессия даёт F1~0.11), но их
    дискриминативная сила проявляется при нелинейном классификаторе —
    поэтому MLP должен и поднять baseline, и ярче показать эффект
    аугментации. class_weight у MLPClassifier нет, дисбаланс компенсируется
    тем, что при аугментации/TSTR редкий класс представлен обильно."""
    return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                          early_stopping=True, random_state=RANDOM_SEED)


def tstr_experiment(model, F_train, y_train, F_test, y_test, target_class,
                    scaler, clf_factory, clf_name, n_synthetic=2000, device='cpu'):
    """
    Train on Synthetic, Test on Real — для ОДНОГО целевого редкого класса
    против остальных (бинарная постановка: target_class vs всё остальное).

    Сравниваются три режима обучения классификатора (классификатор создаётся
    через clf_factory — логрегрессия или MLP):
      A) обучен на РЕАЛЬНЫХ обучающих данных (baseline "как есть")
      B) реальные + СИНТЕТИЧЕСКАЯ аугментация целевого класса
      C) строгий TSTR: без единого реального примера target_class в обучении

    Тест всегда на РЕАЛЬНЫХ отложенных данных. Все данные масштабируются
    заранее обученным scaler (см. main).
    """
    print(f"\n{'='*72}")
    print(f"TSTR-эксперимент для класса {CLASS_NAMES[target_class]} "
          f"(target_class={target_class}) | классификатор: {clf_name}")
    print(f"{'='*72}")

    y_train_bin = (y_train == target_class).astype(int)
    y_test_bin = (y_test == target_class).astype(int)

    n_real_target = int((y_train == target_class).sum())
    print(f"Реальных примеров класса {CLASS_NAMES[target_class]} в трейне: {n_real_target}")
    print(f"Реальных примеров класса {CLASS_NAMES[target_class]} в тесте: {int((y_test==target_class).sum())}")

    F_train_sc = scaler.transform(F_train)
    F_test_sc = scaler.transform(F_test)

    # A) классификатор на реальных данных как есть (масштабированных)
    clf_a = clf_factory()
    clf_a.fit(F_train_sc, y_train_bin)
    f1_a = f1_score(y_test_bin, clf_a.predict(F_test_sc))

    # генерируем синтетику для целевого класса (в ИСХОДНОМ масштабе — так
    # обучался VAE), масштабируем ТЕМ ЖЕ scaler перед подачей в классификатор
    synthetic = model.sample(target_class, n_synthetic, device=device).cpu().numpy()
    synthetic_sc = scaler.transform(synthetic)

    # B) реальные (все классы) + синтетическая аугментация целевого класса
    F_aug_sc = np.concatenate([F_train_sc, synthetic_sc], axis=0)
    y_aug_bin = np.concatenate([y_train_bin, np.ones(n_synthetic, dtype=int)])
    clf_b = clf_factory()
    clf_b.fit(F_aug_sc, y_aug_bin)
    f1_b = f1_score(y_test_bin, clf_b.predict(F_test_sc))

    # C) строгий TSTR: реальные данные БЕЗ target_class + синтетика вместо него
    mask_not_target = (y_train != target_class)
    F_others_sc = F_train_sc[mask_not_target]
    y_others_bin = np.zeros(mask_not_target.sum(), dtype=int)
    F_tstr_sc = np.concatenate([F_others_sc, synthetic_sc], axis=0)
    y_tstr = np.concatenate([y_others_bin, np.ones(n_synthetic, dtype=int)])
    clf_c = clf_factory()
    clf_c.fit(F_tstr_sc, y_tstr)
    f1_c = f1_score(y_test_bin, clf_c.predict(F_test_sc))

    print(f"\nF1 (класс {CLASS_NAMES[target_class]} vs остальные), тест — реальные данные:")
    print(f"  A) Baseline (реальные данные как есть):              F1 = {f1_a:.4f}")
    print(f"  B) Реальные + синтетическая аугментация:              F1 = {f1_b:.4f}  ({(f1_b-f1_a)*100:+.2f} п.п. к A)")
    print(f"  C) Строгий TSTR (0 реальных примеров класса в трейне): F1 = {f1_c:.4f}")
    if f1_b > f1_a:
        print(f"  => Аугментация улучшает baseline на {(f1_b-f1_a)*100:.2f} п.п.")
    else:
        print(f"  => Аугментация не улучшает baseline ({(f1_b-f1_a)*100:+.2f} п.п.)")

    return {'f1_baseline': f1_a, 'f1_augmented': f1_b, 'f1_tstr_strict': f1_c}


def plot_real_vs_synthetic(model, F_train, y_train, scaler, target_class, n_synthetic=1000, device='cpu'):
    """
    Визуальная диагностика распределений: PCA-проекция (в 2D) реальных
    примеров целевого класса и синтетики от VAE, наложенные друг на друга.

    Это не заменяет количественный TSTR, а дополняет его: строгий TSTR
    показывает ЧИСЛО, отражающее разрыв между распределениями, а этот график
    показывает КАК ИМЕННО они расходятся — компактнее ли синтетика, чем
    реальные данные, смещена ли в сторону, покрывает ли форму реального
    облака точек или коллапсирует в одну точку. PCA обучается на реальных
    данных ВСЕХ классов (для содержательной проекции), затем в неё
    проецируются и реальные, и синтетические точки целевого класса.
    """
    real_mask = (y_train == target_class)
    F_real_target = F_train[real_mask]

    synthetic = model.sample(target_class, n_synthetic, device=device).cpu().numpy()

    # PCA обучаем на масштабированных реальных данных всего трейна —
    # так проекция отражает содержательную структуру признакового
    # пространства, а не подгоняется под один класс
    F_train_sc = scaler.transform(F_train)
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    pca.fit(F_train_sc)

    real_2d = pca.transform(scaler.transform(F_real_target))
    synth_2d = pca.transform(scaler.transform(synthetic))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(real_2d[:, 0], real_2d[:, 1], alpha=0.35, s=12,
               label=f'реальные (n={len(real_2d)})', color='#2e6f95')
    ax.scatter(synth_2d[:, 0], synth_2d[:, 1], alpha=0.35, s=12,
               label=f'синтетические (n={len(synth_2d)})', color='#d9534f')
    ax.set_title(f'PCA: реальные vs синтетические — класс {CLASS_NAMES[target_class]}\n'
                 f'(объяснённая дисперсия PC1+PC2 = {pca.explained_variance_ratio_[:2].sum()*100:.1f}%)')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(SCRIPT_DIR, f'real_vs_synthetic_{CLASS_NAMES[target_class]}.png')
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"  Сохранено: {out_path}")

    # количественная мера расхождения центров облаков (простая, но наглядная)
    real_centroid = real_2d.mean(axis=0)
    synth_centroid = synth_2d.mean(axis=0)
    centroid_dist = np.linalg.norm(real_centroid - synth_centroid)
    real_spread = real_2d.std(axis=0).mean()
    synth_spread = synth_2d.std(axis=0).mean()
    print(f"  Расстояние между центрами облаков (в PCA-координатах): {centroid_dist:.3f}")
    print(f"  Разброс (std) реальных точек: {real_spread:.3f}, синтетических: {synth_spread:.3f} "
          f"(если синтетический разброс << реального — генератор слишком консервативен/коллапсирует к среднему)")


def main():
    device = torch.device('cpu')

    F_data = np.load(os.path.join(SCRIPT_DIR, 'F.npy'))
    y_data = np.load(os.path.join(SCRIPT_DIR, 'y.npy'))
    print(f"Загружено: F={F_data.shape}, y={y_data.shape}")

    for i, name in enumerate(CLASS_NAMES):
        print(f"  Класс {i} {name}: {(y_data == i).sum()} примеров")

    F_train, F_test, y_train, y_test = train_test_split(
        F_data, y_data, test_size=0.2, random_state=RANDOM_SEED, stratify=y_data
    )

    # обучаем VAE ТОЛЬКО на трейне, чтобы тест оставался полностью честным
    vae = train_vae(F_train, y_train, epochs=300, device=device)
    torch.save(vae.state_dict(), os.path.join(SCRIPT_DIR, 'pole_vae.pth'))
    print(f"\nМодель сохранена: pole_vae.pth")

    # диагностика: не схлопнулось ли латентное пространство (posterior collapse)
    with torch.no_grad():
        Ft_check = torch.tensor(F_train[:2000], dtype=torch.float32)
        ct_check = one_hot(torch.tensor(y_train[:2000], dtype=torch.long), NUM_CLASSES)
        mu, logvar = vae.encode(Ft_check, ct_check)
        avg_kl_per_dim = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean(dim=0)
        active_dims = (avg_kl_per_dim > 0.01).sum().item()
    print(f"Диагностика латента: активных измерений из {LATENT_DIM} = {active_dims} "
          f"(если 0 — латент схлопнулся, интерполяция между классами будет бессмысленной)")

    # StandardScaler для TSTR-классификаторов (не для VAE — та обучается на
    # исходном масштабе МП-признаков без проблем, стандартизация нужна
    # именно линейной логистической регрессии, которая иначе доминируется
    # признаками с наибольшим абсолютным масштабом). Обучаем ТОЛЬКО на
    # реальном train, чтобы не было утечки информации из теста.
    scaler = StandardScaler()
    scaler.fit(F_train)

    # визуальная диагностика: насколько синтетика похожа на реальные данные
    # (см. plot_real_vs_synthetic) — до количественного TSTR, чтобы иметь
    # наглядную картину независимо от того, что покажут метрики
    print(f"\n{'='*72}")
    print("ВИЗУАЛЬНАЯ ДИАГНОСТИКА: РЕАЛЬНЫЕ VS СИНТЕТИЧЕСКИЕ РАСПРЕДЕЛЕНИЯ")
    print(f"{'='*72}")
    for target_class in [1, 3]:  # S, F
        if (y_train == target_class).sum() < 5:
            continue
        print(f"\nКласс {CLASS_NAMES[target_class]}:")
        plot_real_vs_synthetic(vae, F_train, y_train, scaler, target_class, device=device)

    # TSTR для каждого редкого класса, ДВУМЯ классификаторами для сравнения:
    # линейный (логрегрессия) vs нелинейный (MLP). МП-признаки плохо
    # разделимы линейно, поэтому нелинейный зонд должен и поднять baseline,
    # и ярче показать эффект аугментации.
    classifiers = [('LogReg', make_logreg), ('MLP', make_mlp)]
    results = {}  # results[clf_name][target_class] = {...}

    for clf_name, clf_factory in classifiers:
        results[clf_name] = {}
        for target_class in [1, 3]:  # S, F
            if (y_train == target_class).sum() < 5:
                print(f"\nПропускаю класс {CLASS_NAMES[target_class]} — слишком мало примеров")
                continue
            results[clf_name][target_class] = tstr_experiment(
                vae, F_train, y_train, F_test, y_test, target_class,
                scaler=scaler, clf_factory=clf_factory, clf_name=clf_name, device=device
            )

    print(f"\n{'='*72}")
    print("ИТОГО ДЛЯ ГЛАВЫ 4 ДИПЛОМА (сравнение линейного и нелинейного зонда)")
    print(f"{'='*72}")
    print(f"{'Классиф.':<10}{'Класс':<7}{'Baseline':<11}{'Аугмент.':<11}{'СтрогийTSTR':<13}{'Δ(B-A)':<10}")
    print("-" * 62)
    for clf_name in results:
        for cls, res in results[clf_name].items():
            delta = (res['f1_augmented'] - res['f1_baseline']) * 100
            print(f"{clf_name:<10}{CLASS_NAMES[cls]:<7}"
                  f"{res['f1_baseline']:<11.4f}{res['f1_augmented']:<11.4f}"
                  f"{res['f1_tstr_strict']:<13.4f}{delta:<+10.2f}")


if __name__ == '__main__':
    main()
