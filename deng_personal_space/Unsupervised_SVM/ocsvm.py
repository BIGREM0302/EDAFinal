#!/usr/bin/env python3
"""
ocsvm.py — One-Class SVM for CSV (with header)
若不指定 --out，預設輸出 <input>_USVM.csv

功能概要
────────
1. 讀取含 header 的 CSV
2. 對 --cols 指定欄位做：
   • INT_MAX → 欄內其餘值平均
   • log1p → StandardScaler → RBF One-Class SVM 訓練
3. 推論後，把異常列 (y = –1)  
   • 輸出到 <out>（預設 <input>_USVM.csv）  
   • 儲存模型與前處理器到 <out_prefix>_USVM.joblib
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import OneClassSVM


def gamma_type(x: str):
    try:
        return float(x)
    except ValueError:
        return x


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train / infer One-Class SVM from CSV",
    )
    p.add_argument("--csv", required=True, help="輸入 CSV 檔路徑")
    p.add_argument(
        "--cols",
        nargs="+",
        required=True,
        help="特徵欄位：欄位名稱或 0-based 索引，可混用",
    )
    p.add_argument(
        "--out",
        default=None,
        help="輸出異常列的 CSV 檔名；若省略，預設 <input>_USVM.csv",
    )
    p.add_argument("--sep", default=",", help="CSV 欄位分隔符")
    p.add_argument("--nu", type=float, default=0.01, help="One-Class SVM 的 ν 參數")
    p.add_argument(
        "--gamma",
        type=gamma_type,
        default="scale",
        help="RBF γ（float 或 'scale' / 'auto'）",
    )
    p.add_argument(
        "--int-max", type=int, default=2**31 - 1, help="視為遺失值的整數常數"
    )
    return p.parse_args()


def main():
    args = parse_args()

    # 建立預設輸出檔名前綴
    prefix, _ = os.path.splitext(os.path.basename(args.csv))
    if args.out:
        out_csv = args.out
    else:
        out_csv = f"{prefix}_USVM.csv"
    out_prefix = os.path.splitext(out_csv)[0]

    # 1. 讀入 CSV（header=0）
    df = pd.read_csv(
        args.csv, header=0, sep=args.sep, engine="python" if args.sep != "," else "c"
    )

    # 2. 解析 --cols（名稱或索引皆可）
    feature_cols = []
    for c in args.cols:
        if c.isdigit():
            idx = int(c)
            if idx >= len(df.columns):
                raise ValueError(f"索引 {idx} 超出範圍；CSV 只有 {len(df.columns)} 欄")
            feature_cols.append(df.columns[idx])
        else:
            if c not in df.columns:
                raise ValueError(f"找不到欄位 {c!r}")
            feature_cols.append(c)
    X = df[feature_cols].copy()

    # 3. INT_MAX → 欄內平均值
    for col in feature_cols:
        mask = X[col] == args.int_max
        if mask.any():
            mean_val = X.loc[~mask, col].mean()
            X.loc[mask, col] = mean_val

    # 4. 前處理與訓練
    log_tf = FunctionTransformer(np.log1p, validate=False)
    scaler = StandardScaler()
    X_proc = scaler.fit_transform(log_tf.fit_transform(X))

    clf = OneClassSVM(kernel="rbf", nu=args.nu, gamma=args.gamma)
    clf.fit(X_proc)

    # 5. 推論並輸出異常列
    preds = clf.predict(X_proc)  # 1=正常, -1=異常
    anomalies = df[preds == -1]
    anomalies.to_csv(out_csv, index=False)
    print(f"異常列已輸出至 {out_csv!r}")

    # 6. 儲存模型與前處理器
    bundle = {"clf": clf, "scaler": scaler, "log_tf": log_tf, "features": feature_cols}
    joblib.dump(bundle, f"{out_prefix}_USVM.joblib")
    print(f"模型及前處理器已儲存為 {out_prefix}_USVM.joblib")


if __name__ == "__main__":
    main()
