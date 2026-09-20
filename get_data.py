# -*- coding: utf-8 -*-
"""
get_data.py — скачивает базы MIT-BIH и INCART с PhysioNet СРАЗУ туда,
где их ждут скрипты. После запуска ничего перекладывать вручную не нужно.

    python get_data.py            # обе базы
    python get_data.py mitdb      # только MIT-BIH  (~100 МБ)
    python get_data.py incartdb   # только INCART   (~1.3 ГБ, качается дольше)

Куда что ложится:
    MIT-BIH  ->  ./mitdb/                 (100.dat, 100.hea, 100.atr, ...)
    INCART   ->  ./incart-data/files/     (I01.dat, I01.hea, I01.atr, ...)

Именно эти пути прописаны в main.py / evaluate_all_records.py /
build_sequences.py (mitdb) и в cross_dataset_incart.py /
diagnose_domain_shift.py (incart-data/files).
"""
import os
import sys

try:
    import wfdb
except ImportError:
    sys.exit("Сначала установи пакеты:  pip install -r requirements.txt")

# каждой базе — свой каталог назначения
TARGETS = {
    "mitdb":    "mitdb",
    "incartdb": os.path.join("incart-data", "files"),
}


def download(db):
    out = TARGETS[db]
    os.makedirs(out, exist_ok=True)
    print(f"\nСкачиваю '{db}' в ./{out}/ ...")
    print("  (идёт загрузка с physionet.org, это может занять несколько минут)")
    wfdb.dl_database(db, dl_dir=out)
    n = len([f for f in os.listdir(out) if f.endswith(".hea")])
    print(f"Готово: ./{out}/  (записей: {n})")


if __name__ == "__main__":
    args = sys.argv[1:] or ["mitdb", "incartdb"]
    for db in args:
        if db not in TARGETS:
            print(f"Неизвестная база: {db}  (доступны: mitdb, incartdb)")
            continue
        try:
            download(db)
        except Exception as e:
            print(f"\nНе удалось скачать '{db}': {e}")
            print("Проверь интернет или скачай вручную:")
            print("  MIT-BIH:  https://physionet.org/content/mitdb/1.0.0/")
            print("  INCART :  https://physionet.org/content/incartdb/1.0.0/")

    print("\nПроверка структуры:")
    for db, path in TARGETS.items():
        ok = os.path.isdir(path) and any(f.endswith(".hea") for f in os.listdir(path)) if os.path.isdir(path) else False
        print(f"  {'V' if ok else '-'}  ./{path}/")
