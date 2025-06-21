#!/usr/bin/env python3
"""
train_eval_svm.py  (updated)
──────────────────────────────────────────────────────────────────────────
• 讀取 GNNfeature0.csv–GNNfeature19.csv
• 0–9 作訓練、10–19 作測試
• -1 缺值策略由 --mp {avg|nochange} 控制
• GridSearch + GroupKFold 交叉驗證 (f1_macro)
• 產出：
    ./result/trojan_svm.joblib
    ./result/f1_score.txt
    ./result/GNNfeatureXX_prediction.csv
    ./result/GNNfeatureXX_SVM.csv   (只含 name；pred=1；排除 n[ ] )
"""

import argparse
import glob
import os
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------- 固定常數 ---------------------------------------
FEATURES = ["LGFi", "FFi", "FFo", "Pi", "Po"]
LABEL = "Trojan_gate"  # ←← 這裡已改
TEST_IDS = list(range(0, 20))
TRAIN_IDS = list(range(0, 10))
ALL_IDS = list(range(0,20))
RE_N_INPUT = re.compile(r"^n\[\d+\]$")


# -------------------------- 讀檔與前處理 --------------------------------------
def read_split(data_dir: Path):
    train_dfs, test_dfs = {}, {}
    for idx in ALL_IDS:
        f = data_dir / f"GNNfeature{idx}.csv"
        df = pd.read_csv(f)

        # 先把 5 個特徵轉成 float，避免之後補平均值報 dtype 警告
        df[FEATURES] = df[FEATURES].astype(float)
        if idx in TRAIN_IDS:
            train_dfs[idx] = df
        if idx in TEST_IDS:
            test_dfs[idx] = df
    return train_dfs, test_dfs


def handle_missing(dfs: dict, policy: str, col_means=None):
    if policy == "nochange":
        return dfs, col_means

    if col_means is None:  # 第一次：用 training set 算均值
        vals = pd.concat(dfs.values(), ignore_index=True)
        col_means = {c: vals.loc[vals[c] != -1, c].mean() for c in FEATURES}

    fixed = {}
    for k, df in dfs.items():
        df2 = df.copy()
        for c, mean in col_means.items():
            df2.loc[df2[c] == -1, c] = mean
        fixed[k] = df2
    return fixed, col_means


# ----------------------- SVM 訓練與評估 ---------------------------------------
def train_and_eval(train_dfs, test_dfs):
    train_df = pd.concat(
        [df.assign(group=f"{idx}") for idx, df in train_dfs.items()], ignore_index=True
    )
    X_train = train_df[FEATURES].values
    y_train = train_df[LABEL].astype(int).values
    groups = train_df["group"].values

    pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced"))

    param_grid = {"svc__C": [0.1, 1, 10, 100], "svc__gamma": ["scale", 0.01, 0.1, 1]}
    cv = GroupKFold(n_splits=5)
    search = GridSearchCV(
        pipe, param_grid, scoring="f1_macro", cv=cv, n_jobs=-1, verbose=1
    )
    search.fit(X_train, y_train, groups=groups)

    best_model = search.best_estimator_
    print("➜ Best params :", search.best_params_)

    # ---- 測試集 ----
    test_concat = pd.concat(test_dfs.values(), ignore_index=True)

    y_pred = best_model.predict(test_concat[FEATURES].values)
    
    mask_n = test_concat["name"].str.startswith("n")
    y_pred = np.where(mask_n, 0, y_pred)

    f1 = f1_score(test_concat[LABEL].astype(int).values, y_pred)
    print(f"➜ Test F1     : {f1:.4f}")

    return best_model, f1, y_pred


# ------------------------------ 輸出 ------------------------------------------
def save_outputs(result_dir: Path, model, f1, test_dfs, y_pred):
    result_dir.mkdir(parents=True, exist_ok=True)

    (result_dir / "f1_score.txt").write_text(f"{f1:.6f}\n")
    joblib.dump(model, result_dir / "trojan_svm.joblib")

    offset = 0
    for idx in TEST_IDS:
        df = test_dfs[idx].copy()
        n = len(df)
        df["pred"] = y_pred[offset : offset + n]
        offset += n

        df.to_csv(result_dir / f"GNNfeature{idx}_prediction.csv", index=False)

        trojan_df = df[(df["pred"] == 1) & ~df["name"].str.match(RE_N_INPUT)]
        trojan_df[["name"]].to_csv(
            result_dir / f"GNNfeature{idx}_SVM.csv", header=False, index=False
        )

# ------------------------------ 主程式 ----------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="training_data_for_svm")
    ap.add_argument(
        "--mp", choices=["avg", "nochange"], default="avg", help="缺值(-1)處理方式"
    )
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    result_dir = Path("./result")

    train_dfs, test_dfs = read_split(data_dir)
    train_dfs, col_means = handle_missing(train_dfs, args.mp)
    test_dfs, _ = handle_missing(test_dfs, args.mp, col_means)

    model, f1, y_pred = train_and_eval(train_dfs, test_dfs)
    save_outputs(result_dir, model, f1, test_dfs, y_pred)

    print(f"所有結果已存至 {result_dir.resolve()}")


if __name__ == "__main__":
    main()
