# -*- coding: utf-8 -*-
"""
check_setup.py — проверяет, что всё готово к запуску пайплайна.
Запусти ПЕРЕД прогоном скриптов:  python check_setup.py
"""
import os
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))


def check_pkg(name, import_name=None):
    ok = importlib.util.find_spec(import_name or name) is not None
    print(f"  [{'V' if ok else 'X'}] пакет {name}")
    return ok


def check_dir_records(path, ext=".hea"):
    full = os.path.join(HERE, path)
    if not os.path.isdir(full):
        print(f"  [X] {path}/  — папки нет")
        return False
    n = len([f for f in os.listdir(full) if f.endswith(ext)])
    print(f"  [{'V' if n else 'X'}] {path}/  — записей: {n}")
    return n > 0


print("=" * 60)
print("1) Python-пакеты")
pkgs = all([
    check_pkg("numpy"), check_pkg("scipy"), check_pkg("wfdb"),
    check_pkg("scikit-learn", "sklearn"), check_pkg("pandas"),
    check_pkg("matplotlib"), check_pkg("torch"),
])

print("\n2) Данные (после get_data.py)")
mit = check_dir_records("mitdb")
inc = check_dir_records(os.path.join("incart-data", "files"))

print("\n3) Промежуточные артефакты (появятся после прогона)")
for f in ["X.npy", "y.npy", "F.npy", "base_cnn.pth", "hybrid_cnn.pth"]:
    exists = os.path.exists(os.path.join(HERE, f))
    print(f"  [{'V' if exists else '.'}] {f}")

print("\n" + "=" * 60)
if pkgs and mit:
    print("Готово к запуску MIT-BIH-части: python main.py")
    if inc:
        print("INCART тоже на месте — кросс-датасетные скрипты запустятся.")
    else:
        print("INCART не скачан — только cross_dataset_incart.py и diagnose_domain_shift.py")
        print("не запустятся. Остальное работает. Докачать: python get_data.py incartdb")
else:
    if not pkgs:
        print("Установи пакеты:  pip install -r requirements.txt")
    if not mit:
        print("Скачай данные:    python get_data.py")
