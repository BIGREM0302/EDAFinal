#!/usr/bin/env python3
# ocsvm_5d_idname.py
"""
Train / infer RBF One-Class SVM on CSV with:
(id) (name) (feat1) (feat2) (feat3) (feat4) (feat5)

1. 忽略 id、name 做訓練
2. 只輸出異常樣本到 .out 檔，內容含完整 7 欄
"""

import argparse
import pathlib

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import OneClassSVM


# ──────────────────────────────────────────────────────────────────────────
# Command-line Args
# ──────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="5-D OC-SVM (id+name version)")
    p.add_argument(
        "--csv", required=True, help="Input CSV: id, name, feat1..feat5 (7 columns)"
    )
    p.add_argument(
        "--nu",
        type=float,
        default=0.05,
        help="Upper-bound on training outlier fraction (default=0.05)",
    )
    p.add_argument(
        "--gamma", default="scale", help="'scale' | 'auto' | float – RBF bandwidth"
    )
    p.add_argument(
        "--save", default="ocsvm_5d", help="Prefix for saved model (.joblib)"
    )
    p.add_argument(
        "--out",
        default="anomaly.out",
        help="Output file for abnormal rows (default=anomaly.out)",
    )
    p.add_argument(
        "--cols",
        nargs="+",
        type=int,
        default=[2, 3, 4, 5, 6],
        help="0-based column indices of features (default=2..6)",
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    csv_path = pathlib.Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path, header=None, delim_whitespace=True)

    id_name = df.iloc[:, :2]
    feature_cols = args.cols
    if any(c < 2 or c >= df.shape[1] for c in feature_cols):
        raise ValueError(f"Feature columns must be between 2 and {df.shape[1]-1}")

    X_raw = df.iloc[:, feature_cols].astype(float).values
    # 1. 讀檔

# ---------- 取代 INT_MAX（僅第 2,3,4 欄） ----------
    INT_MAX = np.iinfo(np.int32).max          # 2147483647
    cols_to_fix = [2, 3, 4]                   # 欄 index：LGFi, FFi, FFo

    for col in cols_to_fix:
        col_series = df[col].astype(float)
        mask_bad   = col_series == INT_MAX

        if mask_bad.any():
            mean_good = col_series[~mask_bad].mean()   # 只算正常值的平均
            if np.isnan(mean_good):                    # 該欄全是 INT_MAX
                mean_good = 0.0                       # 你也可換成全檔平均等策略
            df.loc[mask_bad, col] = mean_good
# ---------- 取代 INT_MAX（僅第 2,3,4 欄） ----------

INT_MAX = np.iinfo(np.int32).max          # 2147483647

    # 2. 前處理
    log_scaler = FunctionTransformer(np.log1p, validate=False)
    std_scaler = StandardScaler()

    X_log = log_scaler.fit_transform(X_raw)
    X_std = std_scaler.fit_transform(X_log)

    # 3. 訓練 OC-SVM
    ocsvm = OneClassSVM(
        kernel="rbf",
        gamma=float(args.gamma) if args.gamma not in {"scale", "auto"} else args.gamma,
        nu=args.nu,
    )
    ocsvm.fit(X_std)

    # 4. 推論
    y_pred = ocsvm.predict(X_std)  # +1=normal, -1=anomaly

    # 5. 匯出異常行
    anomalies = df[y_pred == -1]  # 仍保留 7 欄
    if anomalies.empty:
        print("\n[Info] No anomalies detected.")
    else:
        anomalies.to_csv(args.out, sep=" ", index=False, header=False)
        print(f"\n[Saved] {args.out}  (rows: {len(anomalies)})")

    # 6. 存模型+Scaler
    dump(
        {"model": ocsvm, "log_scaler": log_scaler, "std_scaler": std_scaler},
        f"{args.save}.joblib",
    )
    print(f"[Saved] {args.save}.joblib")


if __name__ == "__main__":
    main()
