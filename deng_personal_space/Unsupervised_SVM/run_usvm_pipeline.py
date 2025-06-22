#!/usr/bin/env python3
"""
run_usvm_pipeline.py  —  一鍵完成  V → CSV → OC-SVM → F1 → 彙總

預設目錄
────────
./trojan/design0.v … design19.v
./trojan/result0.txt … result19.txt
./temp/  存放中繼與輸出
最終成果： ./Final_USVM.csv

可調旗標（全部可選）
──────────────────
--nu 0.01           OC-SVM ν
--gamma scale       RBF γ（float / scale / auto）
--cols LGFi Pi FFo  特徵欄位（空白分隔）
--design-dir ./trojan
--gt-dir     ./trojan
--temp-dir   ./temp
--begin 0    處理 designN.v 起始編號
--end   19   結束編號（含）
--compile-only        只編譯 v2csv，不跑流程
--skip-compile        不管 v2csv.cpp 是否較新，直接用現有 ./v2csv
"""
import argparse
import csv
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime


def compile_v2csv(force: bool):
    src = pathlib.Path("v2csv.cpp")
    bin = pathlib.Path("v2csv")
    if force or not bin.exists() or src.stat().st_mtime > bin.stat().st_mtime:
        compiler = shutil.which("g++") or shutil.which("clang++")
        if not compiler:
            sys.exit("❌ 找不到 g++ / clang++ 編譯器")
        print(f"[{datetime.now():%H:%M:%S}] compile {src} → {bin}")
        cmd = [compiler, "-std=c++17", str(src), "-o", str(bin)]
        subprocess.check_call(cmd)
    else:
        print(f"[{datetime.now():%H:%M:%S}] v2csv 已是最新，跳過編譯")
    return "./v2csv"


def run(cmd: list[str]):
    res = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if res.returncode != 0:
        print(res.stdout)
        sys.exit(f"❌ 指令失敗：{' '.join(cmd)}")
    return res.stdout


def parse_f1(calc_output: str) -> float:
    """從 calc_f1.py 輸出文字抓 F1-score = 0.xxxx"""
    m = re.search(r"F1-score\s*=\s*([0-9.]+)", calc_output)
    return float(m.group(1)) if m else 0.0


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Batch USVM pipeline",
    )
    ap.add_argument("--nu", default="0.01")
    ap.add_argument("--gamma", default="scale")
    ap.add_argument("--cols", nargs="+", default=["LGFi", "Pi", "FFo"])
    ap.add_argument("--design-dir", default="./trojan")
    ap.add_argument("--gt-dir", default="./trojan")
    ap.add_argument("--temp-dir", default="./temp")
    ap.add_argument("--begin", type=int, default=0)
    ap.add_argument("--end", type=int, default=19)
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--skip-compile", action="store_true")
    args = ap.parse_args()

    # 1. 編譯 v2csv
    if not args.skip_compile:
        compile_v2csv(force=False)
    if args.compile_only:
        return

    os.makedirs(args.temp_dir, exist_ok=True)
    f1_records = []

    for i in range(args.begin, args.end + 1):
        design_v = f"{args.design_dir}/design{i}.v"
        if not pathlib.Path(design_v).exists():
            print(f"[warn] 缺少 {design_v}，跳過")
            continue
        csv_path = f"{args.temp_dir}/{i}.csv"
        usvm_csv = f"{args.temp_dir}/{i}_USVM.csv"
        gt_path = f"{args.gt_dir}/result{i}.txt"

        print(f"[{datetime.now():%H:%M:%S}] ▶ design{i}.v → CSV")
        run(["./v2csv", design_v, csv_path])

        print(f"[{datetime.now():%H:%M:%S}] ▶ OC-SVM on {csv_path}")
        ocsvm_cmd = [
            "python",
            "ocsvm.py",
            "--csv",
            csv_path,
            "--out",
            usvm_csv,  # ← 指定輸出到 temp 資料夾
            "--nu",
            args.nu,
            "--gamma",
            args.gamma,
            "--cols",
            *args.cols,
        ]
        run(ocsvm_cmd)
        if not pathlib.Path(usvm_csv).exists():
            sys.exit(f"❌ 未找到 {usvm_csv}")

        if not pathlib.Path(gt_path).exists():
            print(f"[warn] 缺少 {gt_path}，F1 用 0")
            f1 = 0.0
        else:
            print(f"[{datetime.now():%H:%M:%S}] ▶ F1 {usvm_csv} vs {gt_path}")
            out = run(["python", "calc_f1.py", usvm_csv, gt_path])
            f1 = parse_f1(out)
            print(out.strip().splitlines()[-1])  # 顯示 F1 行

        f1_records.append((i, f1))

    # ★ NEW ────────────────────────────────────────────────
    # 2. 計算所有 design 的平均 F1
    if f1_records:  # 避免分母為 0
        avg_f1 = sum(f for _, f in f1_records) / len(f1_records)
    else:
        avg_f1 = 0.0
    # ───────────────────────────────────────────────────────

    # 3. 建立安全字串
    safe_nu = str(args.nu).replace(".", "p")
    safe_gamma = str(args.gamma).replace(".", "p")
    safe_cols = "-".join(args.cols)

    final_csv = f"Final_USVM_nu{safe_nu}_g{safe_gamma}_c{safe_cols}.csv"

    with open(final_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["design", "f1 score", "nu", "gamma", "cols"])
        for d, f1 in f1_records:
            writer.writerow([d, f"{f1:.6f}", args.nu, args.gamma, "|".join(args.cols)])
        # 最後再補上一列「平均值」
        writer.writerow(
            ["average", f"{avg_f1:.6f}", args.nu, args.gamma, "|".join(args.cols)]
        )
    print(f"\n✅ Pipeline 完成，平均 F1 = {avg_f1:.6f}，結果寫入 {final_csv}")


if __name__ == "__main__":
    main()
