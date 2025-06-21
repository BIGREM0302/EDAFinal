#!/usr/bin/env python3
"""
ensemble_train_eval_svm.py  (updated)
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
TEST_IDS = list(range(10, 20))
TRAIN_IDS = list(range(0, 10))
RE_N_INPUT = re.compile(r"^n\[\d+\]$")


# -------------------------- 讀檔與前處理 --------------------------------------
# --------- read_split (加入 per-file column-wise normalization) ---------------
def read_split(data_dir: Path):
    """
    1. 先把 TRAIN_IDS ∪ TEST_IDS 指定的檔案全部讀進 all_dfs
    2. 對 FEATURES 欄做 0–1 normalization（以本檔非 -1 的最大值為基準）
    3. 切出 train_dfs、test_dfs 兩個字典
    """
    all_dfs = {}
    for idx in set(TRAIN_IDS).union(TEST_IDS):
        f = data_dir / f"GNNfeature{idx}.csv"
        df = pd.read_csv(f)

        # ---- 轉 float，避免後續補平均值型別衝突 ----
        df[FEATURES] = df[FEATURES].astype(float)

        # ---- 每檔內自行 0–1 normalization ----
        for c in FEATURES:
            col_mask = df[c] != -1  # 忽略 -1 (缺值)
            if col_mask.any():
                col_max = df.loc[col_mask, c].max()
                if col_max != 0:
                    df.loc[col_mask, c] = df.loc[col_mask, c] / col_max

        all_dfs[idx] = df

    # ---- 分派到 train / test dict ----
    train_dfs = {idx: all_dfs[idx] for idx in TRAIN_IDS}
    test_dfs = {idx: all_dfs[idx] for idx in TEST_IDS}
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
