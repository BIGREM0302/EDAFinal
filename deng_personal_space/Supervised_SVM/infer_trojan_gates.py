#!/usr/bin/env python3
"""
infer_trojan_gates.py — Predict Trojan gates using a saved SVM model+scaler,
                        and filter out primary inputs/outputs before saving.

Example
-------
python infer_trojan_gates.py \
    --csv new_GNNfeature.csv \
    --cols 2 3 4 \
    --model trojan_svm \
    --out predicted_trojans.txt
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

INT_MAX = 2147483647


def parse_args():
    p = argparse.ArgumentParser(
        description="Predict Trojan gates using a saved SVM model+scaler",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv", required=True, help="feature CSV")
    p.add_argument(
        "--cols", type=int, nargs="+", default=[2, 3, 4], help="0-based feature columns"
    )
    p.add_argument(
        "--model",
        required=True,
        help="model prefix (loads '<prefix>_model.joblib' & scaler)",
    )
    p.add_argument("--out", required=True, help="output file for Trojan gate names")
    return p.parse_args()


def is_gate(name: str) -> bool:
    """Heuristic: real gate names start with 'g' and不含 '['."""
    return name.lower().startswith("g") and "[" not in name


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")

    # 1. 讀 CSV
    df = pd.read_csv(csv_path, header=None)

    # 2. INT_MAX → 欄內均值
    for c in args.cols:
        col_mean = df.loc[df[c] != INT_MAX, c].mean()
        df.loc[df[c] == INT_MAX, c] = col_mean

    # 3. 特徵前處理
    X = np.log1p(df[args.cols].values)

    # 4. 載入 scaler & model
    scaler = joblib.load(f"{args.model}_scaler.joblib")
    clf = joblib.load(f"{args.model}_model.joblib")

    X_scaled = scaler.transform(X)
    preds = clf.predict(X_scaled)

    # 5. 取出被判為 Trojan 的名字
    trojan_names = df.loc[preds == 1, 1].astype(str)

    # 6. 過濾 PI/PO，只留 gate
    trojan_gates = [n for n in trojan_names if is_gate(n)]

    # 7. 輸出
    out_path = Path(args.out)
    with out_path.open("w") as f:
        for g in trojan_gates:
            f.write(g + "\n")

    print(
        f"Detected {len(trojan_gates)} Trojan **gates** "
        f"(filtered from {len(trojan_names)} raw positives) → {out_path}"
    )


if __name__ == "__main__":
    main()
