# -*- coding: utf-8 -*-
"""
augment_vs_imbalance.py — усиление тезиса 3 (генеративное моделирование).

Вопрос: растёт ли польза VAE-аугментации по мере УСИЛЕНИЯ дефицита класса?
Практический смысл: если да — синтетика помогает тем сильнее, чем острее
проблема (мало реальных примеров редкого класса), то есть именно там, где
она нужнее всего.

Честный дизайн:
  • тестовая выборка — реальная и ФИКСИРОВАННАЯ на всех уровнях (сравнимость);
  • редкий класс (по умолчанию F) в ОБУЧЕНИИ урезается до доли p;
  • VAE ПЕРЕОБУЧАЕТСЯ на урезанном обучении (few-shot генерация — генератор
    видит ровно столько же реальных примеров класса, сколько и классификатор,
    без утечки);
  • сравнивается F1 редкого класса без аугментации и с добавлением синтетики
    до медианной численности; усреднение по нескольким сидам.

Запуск (из папки с F.npy, y.npy):
    python augment_vs_imbalance.py                     # F, доли 1.0/0.5/0.25/0.1, 3 сида
    python augment_vs_imbalance.py --cls 1             # класс S
    python augment_vs_imbalance.py --quick             # быстрее (2 сида, короче VAE)
"""
import os, sys, argparse
import numpy as np
import torch
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
from sklearn.model_selection import train_test_split

import vae_pole_features as vpf
import augmented_training as aug

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']


def reduce_class(X, y, cls, frac, rng):
    """Оставляет только долю frac реальных примеров класса cls (прочие классы целы)."""
    idx = np.where(y == cls)[0]
    keep_n = max(5, int(round(len(idx) * frac)))
    keep = set(rng.choice(idx, keep_n, replace=False).tolist())
    mask = np.array([(yy != cls) or (i in keep) for i, yy in enumerate(y)])
    return X[mask], y[mask], keep_n


def one_level(frac, seed, X_train, y_train, X_test, y_test, cls, vae_epochs, device='cpu'):
    np.random.seed(seed); torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    Xr, yr, nkeep = reduce_class(X_train, y_train, cls, frac, rng)

    # VAE few-shot на урезанном обучении
    vae = vpf.train_vae(Xr, yr, epochs=vae_epochs, warmup_epochs=max(1, vae_epochs // 2),
                        device=device)
    vae.eval()

    aug.RANDOM_SEED = seed  # синхронизируем сиды внутренних прогонов классификатора

    # A) без аугментации
    _, per_a, _ = aug.train_classifier(Xr, yr, X_test, y_test, device=device)

    # B) + синтетика класса cls до медианной численности
    counts = np.bincount(yr, minlength=5)
    target = int(np.median(counts[counts > 0]))
    need = max(0, target - counts[cls])
    if need > 0:
        synth = vae.sample(cls, need).detach().cpu().numpy().astype(np.float32)
        Xa = np.concatenate([Xr, synth]).astype(np.float32)
        ya = np.concatenate([yr, np.full(need, cls)]).astype(np.int64)
    else:
        Xa, ya = Xr, yr
    _, per_b, _ = aug.train_classifier(Xa, ya, X_test, y_test, device=device)

    return nkeep, float(per_a[cls]), float(per_b[cls])


def ms(a):
    a = np.array(a); sd = a.std(ddof=1) if len(a) > 1 else 0.0
    return a.mean(), sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cls', type=int, default=3, help='класс для урезания (3=F, 1=S)')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=200, help='эпох обучения VAE на уровень')
    ap.add_argument('--fracs', type=str, default='1.0,0.5,0.25,0.1')
    ap.add_argument('--quick', action='store_true', help='2 сида, короткий VAE (150 эпох)')
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.epochs = 2, 150
    fracs = [float(x) for x in a.fracs.split(',')]
    cls = a.cls; cname = CLASS_NAMES[cls]

    F = np.load(os.path.join(SCRIPT_DIR, 'F.npy'))
    y = np.load(os.path.join(SCRIPT_DIR, 'y.npy'))
    X_train, X_test, y_train, y_test = train_test_split(
        F, y, test_size=0.2, random_state=42, stratify=y)   # реальный ФИКСИРОВАННЫЙ тест

    print("=" * 78)
    print(f"ТЕЗИС 3: выигрыш VAE-аугментации vs дефицит класса {cname}")
    print(f"Реальных примеров {cname} в обучении (полном): {(y_train==cls).sum()}; "
          f"в тесте (реальном, фикс.): {(y_test==cls).sum()}")
    print(f"Сидов: {a.seeds}, эпох VAE/уровень: {a.epochs}")
    print("=" * 78)

    results = []
    for frac in fracs:
        base, augv, gain, nkeep = [], [], [], None
        for s in range(a.seeds):
            nk, ba, bb = one_level(frac, 100 + s, X_train, y_train, X_test, y_test,
                                   cls, a.epochs)
            nkeep = nk; base.append(ba); augv.append(bb); gain.append((bb - ba) * 100)
        bm, bs = ms(base); am, as_ = ms(augv); gm, gs = ms(gain)
        results.append((frac, nkeep, bm, am, gm, gs))
        print(f"\n[доля {frac:>4} | реальных {cname}={nkeep:4d}]  "
              f"F1({cname}) без аугм. = {bm:.3f} ± {bs:.3f}  |  с аугм. = {am:.3f} ± {as_:.3f}  "
              f"|  выигрыш = {gm:+.2f} ± {gs:.2f} п.п.")

    print("\n" + "=" * 78)
    print(f"СВОДКА — выигрыш аугментации на классе {cname} по мере роста дефицита:")
    print(f"{'реальных '+cname:<16}{'F1 без':<12}{'F1 с аугм.':<14}{'выигрыш, п.п.':<16}")
    print("-" * 58)
    for frac, nkeep, bm, am, gm, gs in results:
        print(f"{nkeep:<16}{bm:<12.3f}{am:<14.3f}{gm:+.2f} ± {gs:.2f}")
    print("=" * 78)
    print("Ожидаемый практический вывод: чем меньше реальных примеров редкого")
    print("класса, тем БОЛЬШЕ выигрыш от синтетики — то есть генеративная")
    print("аугментация помогает именно там, где данных остро не хватает.")
    print("Строку 'ГОТОВО ДЛЯ ВСТАВКИ' пришлите — впишу кривую в главу 4.")
    print("MULTISEED_TABLE " + "; ".join(
        f"{cname}={nk}:gain={gm:.2f}±{gs:.2f}" for _, nk, _, _, gm, gs in results))


if __name__ == '__main__':
    main()
