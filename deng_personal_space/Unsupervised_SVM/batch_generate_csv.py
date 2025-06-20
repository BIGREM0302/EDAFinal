#!/usr/bin/env python3
import os, subprocess, pathlib, sys

BIN = "./v2csv"  # 剛才編譯出的執行檔
SRC_DIR = "./trojan"  # design*.v 所在目錄
DST_DIR = "./temp"  # 產生 *.csv 目錄
os.makedirs(DST_DIR, exist_ok=True)

for i in range(21):  # 0..20 共 21 個
    vin = f"{SRC_DIR}/design{i}.v"
    vout = f"{DST_DIR}/{i}.csv"
    if not pathlib.Path(vin).exists():
        print(f"[skip] {vin} 不存在")
        continue
    print(f"[run] {vin} → {vout}")
    ret = subprocess.run([BIN, vin, vout])
    if ret.returncode != 0:
        sys.exit(f"Error when processing {vin}")
print("✅ All done.")
