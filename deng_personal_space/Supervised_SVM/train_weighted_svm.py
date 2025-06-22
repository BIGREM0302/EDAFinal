#!/usr/bin/env python3
"""
train_eval_svm.py  (with per-design sample weights)
──────────────────────────────────────────────────────────────────────────
• 讀取 GNNfeature0–9.csv 做 training, GNNfeature10–19.csv 做 test
• per-file 0–1 normalization（忽略 -1）
• 缺值策略 --mp {avg|nochange}
• 合併 0–9 為 train_df，並 based on group 計算 sample_weight:
    weight[i] = 1.0 / (gate_count of that design)
• GridSearchCV + GroupKFold (5 折)，scoring=f1_macro
  - 傳入 fit_params {'svc__sample_weight': sample_weights}
• refit 時同樣帶 sample_weight
• 輸出模型 ./result/trojan_svm.joblib
  以及 test F1、prediction.csv、_SVM.csv
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

# -------- 常數 --------
FEATURES = ["LGFi", "FFi", "FFo", "Pi", "Po"]
LABEL = "Trojan_gate"
TRAIN_IDS = list(range(0, 10))
TEST_IDS = list(range(10, 20))
GRID = {"svc__C": [0.1, 1, 10, 100], "svc__gamma": ["scale", 0.01, 0.1, 1]}
RE_N_INPUT = re.compile(r"^n\[\d+\]$")


# ---- per-file 0–1 normalizaton ----
def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    for c in FEATURES:
        mask = df2[c] != -1
        if mask.any():
            mx = df2.loc[mask, c].max()
            if mx != 0:
                df2.loc[mask, c] = df2.loc[mask, c] / mx
    return df2


def read_split(data_dir: Path):
    train_dfs, test_dfs = {}, {}
    for idx in TRAIN_IDS + TEST_IDS:
        df = pd.read_csv(data_dir / f"GNNfeature{idx}.csv")
        df[FEATURES] = df[FEATURES].astype(float)
        df = normalize_df(df)
        if idx in TRAIN_IDS:
            train_dfs[str(idx)] = df
        else:
            test_dfs[str(idx)] = df
    return train_dfs, test_dfs


# ---- handle -1 ----
def handle_missing(dfs: dict, policy: str, means=None):
    if policy == "nochange":
        return dfs, means
    # compute means once on train
    if means is None:
        all_df = pd.concat(dfs.values(), ignore_index=True)
        means = {c: all_df.loc[all_df[c] != -1, c].mean() for c in FEATURES}
    fixed = {}
    for k, df in dfs.items():
        df2 = df.copy()
        for c, m in means.items():
            df2.loc[df2[c] == -1, c] = m
        fixed[k] = df2
    return fixed, means


# ---- train & eval with sample_weight ----
def train_and_eval(train_dfs, test_dfs):
    # build train_df
    records = []
    groups = []
    for gid, df in train_dfs.items():
        records.append(df.assign(group=gid))
        groups.extend([gid] * len(df))
    train_df = pd.concat(records, ignore_index=True)
    X_train = train_df[FEATURES].values
    y_train = train_df[LABEL].astype(int).values

    # compute sample_weight per sample
    counts = {gid: len(df) for gid, df in train_dfs.items()}
    sample_weight = np.array([1.0 / counts[g] for g in train_df["group"]])

    # pipeline & gridsearch
    pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced"))
    cv = GroupKFold(n_splits=5)
    search = GridSearchCV(
        pipe, GRID, scoring="f1_macro", cv=cv, n_jobs=-1, verbose=1, refit=True
    )
    # pass sample_weight to both CV and refit
    search.fit(
        X_train, y_train, groups=np.array(groups), svc__sample_weight=sample_weight
    )

    best_model = search.best_estimator_
    print("➜ Best params :", search.best_params_)

    # test eval
    test_df = pd.concat(test_dfs.values(), ignore_index=True)
    X_test = test_df[FEATURES].values
    y_test = test_df[LABEL].astype(int).values
    y_pred = best_model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    print(f"➜ Test F1     : {f1:.4f}")

    return best_model, f1, y_pred


# ---- save outputs ----
def save_outputs(result_dir: Path, model, f1, test_dfs, y_pred):
    result_dir.mkdir(parents=True, exist_ok=True)
    # F1
    (result_dir / "f1_score.txt").write_text(f"{f1:.6f}\n")
    # model
    joblib.dump(model, result_dir / "trojan_svm.joblib")

    # per-file outputs
    offset = 0
    for idx in TEST_IDS:
        df = test_dfs[str(idx)].copy()
        n = len(df)
        df["pred"] = y_pred[offset : offset + n]
        offset += n

        df.to_csv(result_dir / f"GNNfeature{idx}_prediction.csv", index=False)
        trojan_df = df[(df["pred"] == 1) & ~df["name"].str.match(RE_N_INPUT)]
        trojan_df[["name"]].to_csv(
            result_dir / f"GNNfeature{idx}_SVM.csv", header=False, index=False
        )


# ---- main ----
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="training_data_for_svm")
    p.add_argument("--mp", choices=["avg", "nochange"], default="avg")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    result_dir = Path("result")

    train_dfs, test_dfs = read_split(data_dir)
    train_dfs, means = handle_missing(train_dfs, args.mp)
    test_dfs, _ = handle_missing(test_dfs, args.mp, means)

    model, f1, y_pred = train_and_eval(train_dfs, test_dfs)
    save_outputs(result_dir, model, f1, test_dfs, y_pred)
    print(f"所有結果已存至 {result_dir.resolve()}")


if __name__ == "__main__":
    main()
