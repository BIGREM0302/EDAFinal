#!/usr/bin/env python3
"""
ocsvm.py  —  One-Class SVM for CSV (id, name, features)

功能概要
────────
1. 讀取含「id、name、特徵」的 CSV（預設逗號分隔）
2. 僅對 --cols 指定的特徵欄做：
   • INT_MAX → 欄內其餘值平均
   • log1p 壓縮 → StandardScaler → RBF One-Class SVM 訓練
3. 推論後，把異常列 (y = –1) 輸出到 --out（含 id、name、全部特徵）
4. 另存模型與 scaler 到 <prefix>.joblib

使用範例
────────
python ocsvm.py --csv GNNfeature.csv --cols 2 3 4 --out test.out
python ocsvm.py --csv data.csv --cols 2 3 4 5 6 --nu 0.03 --gamma 0.2
"""

import argparse
import pathlib

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import OneClassSVM


# ───────────────────────── argparse ─────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OC-SVM on selected feature columns")
    p.add_argument("--csv", required=True,
                   help="Input CSV path (id, name, feat1..featN)")
    p.add_argument("--cols", "--column", nargs="+", type=int, required=True,
                   help="0-based feature column indices (>=2). Ex: --cols 2 3 4")
    p.add_argument("--intmax", type=int,
                   default=np.iinfo(np.int32).max,
                   help="Sentinel value representing invalid data (default INT_MAX)")
    p.add_argument("--nu", type=float, default=0.05,
                   help="Upper-bound on training outlier fraction (default 0.05)")
    p.add_argument("--gamma", default="scale",
                   help="'scale' | 'auto' | float — RBF bandwidth")
    p.add_argument("--out", default="anomaly.out",
                   help="Output file for abnormal rows (default anomaly.out)")
    p.add_argument("--save", default="ocsvm",
                   help="Prefix for saved model bundle (default ocsvm.joblib)")
    p.add_argument("--sep", default=",",
                   help="Field separator (default ','); "
                        "regex allowed, e.g. '\\s+' for whitespace")
    return p.parse_args()


# ────────────────────────── helpers ─────────────────────────
def replace_intmax_with_col_mean(df: pd.DataFrame,
                                 cols: list[int],
                                 sentinel: int) -> None:
    """In-place: replace sentinel in given columns with column mean."""
    for c in cols:
        col = df[c].astype(float)
        mask = col == sentinel
        if mask.any():
            mean_good = col[~mask].mean()
            if np.isnan(mean_good):          # 該欄全部都是 sentinel
                mean_good = 0.0
            df.loc[mask, c] = mean_good


# ─────────────────────────── main ───────────────────────────
def main() -> None:
    args = parse_args()
    csv_path = pathlib.Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    # 1. 讀檔
    df = pd.read_csv(csv_path,
                     header=None,
                     sep=args.sep,
                     engine="python" if args.sep != "," else "c")

    # 2. 驗證欄位
    if any(c < 2 or c >= df.shape[1] for c in args.cols):
        raise ValueError(f"Feature columns must be between 2 and {df.shape[1]-1}")

    # 3. 將 INT_MAX 轉為欄平均
    replace_intmax_with_col_mean(df, args.cols, args.intmax)

    # 4. 拆分 id/name 與特徵
    id_name = df.iloc[:, :2]
    X_raw   = df.iloc[:, args.cols].astype(float).values

    # 5. 前處理 → OC-SVM
    log_scaler = FunctionTransformer(np.log1p, validate=False)
    std_scaler = StandardScaler()
    X_std = std_scaler.fit_transform(log_scaler.fit_transform(X_raw))

    ocsvm = OneClassSVM(kernel="rbf",
                        gamma=float(args.gamma) if args.gamma not in {"scale", "auto"}
                        else args.gamma,
                        nu=args.nu)
    ocsvm.fit(X_std)

    # 6. 推論 & 輸出異常
    y_pred = ocsvm.predict(X_std)           # +1 normal, −1 anomaly
    anomalies = df[y_pred == -1]
    if anomalies.empty:
        print("[Info] No anomalies detected.")
    else:
        # 輸出分隔符沿用輸入: 若 args.sep 是 regex，改用空白輸出最安全
        out_sep = " " if len(args.sep) > 1 or args.sep.strip() == r"\s+" else args.sep
        anomalies.to_csv(args.out, sep=out_sep, index=False, header=False)
        print(f"[Saved] {args.out}  (rows: {len(anomalies)})")

    # 7. 儲存模型 + scaler
    dump({
        "model": ocsvm,
        "log_scaler": log_scaler,
        "std_scaler": std_scaler,
        "feature_cols": args.cols
    }, f"{args.save}.joblib")
    print(f"[Saved] {args.save}.joblib")


if __name__ == "__main__":
    main()
