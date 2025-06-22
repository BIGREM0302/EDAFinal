#!/usr/bin/env python3
"""
infer_all_svm.py
────────────────────────────────────────────────────────────────────────
• 使用 ./result/trojan_svm.joblib（或指定模型路徑）
• 0–19 design 都讀入 inference
• per-file 0–1 normalization + --mp {avg|nochange}
• 計算 TP, FP, FN, TN, F1（zero_division=0）
• CLI 印出：
    ======Score Result <id> ======
    TP: xx, FP: xx
    FN: xx, TN: xx
    F1 score =<f1>
• 最後將所有 design 的 F1 按 design id 順序，輸出到
  ./result/f1_table.csv（兩欄：design_id, f1_score）
"""

import argparse, joblib, warnings, re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

warnings.filterwarnings("ignore", category=UserWarning)

FEATURES = ["LGFi", "FFi", "FFo", "Pi", "Po"]
LABEL = "Trojan_gate"


def per_file_normalize(df):
    df2 = df.copy()
    for c in FEATURES:
        mask = df2[c] != -1
        if mask.any():
            mx = df2.loc[mask, c].max()
            if mx != 0:
                df2.loc[mask, c] = df2.loc[mask, c] / mx
    return df2


def load_csv(path, means=None, mode="avg"):
    df = pd.read_csv(path)
    df[FEATURES] = df[FEATURES].astype(float)
    df = per_file_normalize(df)
    if mode == "avg" and means is not None:
        for c, m in means.items():
            df.loc[df[c] == -1, c] = m
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model", default="result/trojan_svm.joblib", help="訓練好的 SVM 模型路徑"
    )
    p.add_argument("--data_dir", default="training_data_for_svm")
    p.add_argument(
        "--mp", choices=["avg", "nochange"], default="avg", help="缺值(-1)策略"
    )
    args = p.parse_args()

    # 載模型
    ens = joblib.load(args.model)

    # 如果 avg，從訓練資料讀 means
    means = None
    if args.mp == "avg":
        # 假設我們已存訓練時 means 在模型裏
        means = ens.get("means", None)
        if means is None:
            # fallback: 重新從 0–9 算
            dfs0 = []
            for i in range(10):
                df = pd.read_csv(Path(args.data_dir) / f"GNNfeature{i}.csv")
                df[FEATURES] = df[FEATURES].astype(float)
                dfs0.append(df)
            concat = pd.concat(dfs0, ignore_index=True)
            means = {c: concat.loc[concat[c] != -1, c].mean() for c in FEATURES}

    results = []
    RE_N_IN = re.compile(r"^n\[\d+\]$")
    for i in range(20):
        path = Path(args.data_dir) / f"GNNfeature{i}.csv"
        df = load_csv(path, means, mode=args.mp)

        # 填 -1（avg）
        if args.mp == "avg":
            for c, m in means.items():
                df.loc[df[c] == -1, c] = m

        # 預測
        X = df[FEATURES].values
        y_true = df[LABEL].astype(int).values
        y_pred = (
            ens["models"][0].predict(X)
            if isinstance(ens, dict) and "models" in ens
            else ens.predict(X)
        )

        # confusion
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        # f1
        f1 = f1_score(y_true, y_pred, zero_division=0)
        print(f"======Score Result {i} ======")
        print(f"TP: {tp}, FP: {fp}")
        print(f"FN: {fn}, TN: {tn}")
        print(f"F1 score ={f1:.5f}\n")

        results.append({"design_id": i, "f1_score": f1})

    # 存 f1_table.csv
    out_df = pd.DataFrame(results)
    out_df.to_csv(Path("result") / "f1_table.csv", index=False)
    print("F1 table saved to ./result/f1_table.csv")


if __name__ == "__main__":
    main()
