#!/usr/bin/env python3
"""
train_ensemble_svm.py
────────────────────────────────────────────────────────────────────────
• 讀取 training_data_for_svm/GNNfeature0.csv … 9.csv
• per-file column-wise 0–1 normalization（忽略 -1）
• 缺值策略: --mp {avg|nochange} (default=avg)
• 每檔訓練 1 顆 RBF-SVM (C∈{1,10}, γ∈{0.1,0.01})
  - probability=True, class_weight=balanced
  - 5-fold CV，cv_f1 作權重
• 存成 ./model/ensemble_svm.joblib：
  dict{models, weights, mp, means(dict or None)}
"""

import argparse
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=UserWarning)

TRAIN_IDS = list(range(0, 10))
FEATURES = ["LGFi", "FFi", "FFo", "Pi", "Po"]
LABEL = "Trojan_gate"
GRID = {"svc__C": [1, 10], "svc__gamma": [0.1, 0.01]}


# --------------------- utils ---------------------------
def per_file_normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in FEATURES:
        mask = out[col] != -1
        if mask.any():
            mx = out.loc[mask, col].max()
            if mx != 0:
                out.loc[mask, col] = out.loc[mask, col] / mx
    return out


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[FEATURES] = df[FEATURES].astype(float)
    return per_file_normalize(df)


def train_single(idx: int, df: pd.DataFrame):
    X, y = df[FEATURES].values, df[LABEL].values
    pipe = make_pipeline(
        StandardScaler(), SVC(kernel="rbf", probability=True, class_weight="balanced")
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=idx)
    gs = GridSearchCV(pipe, GRID, scoring="f1_macro", cv=cv, n_jobs=-1).fit(X, y)
    return gs.best_estimator_, gs.best_score_


# --------------------- main ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="training_data_for_svm")
    ap.add_argument(
        "--mp", choices=["avg", "nochange"], default="avg", help="缺值 -1 處理策略"
    )
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)

    # 1. 讀檔與 normalize
    dfs = {i: load_csv(data_dir / f"GNNfeature{i}.csv") for i in TRAIN_IDS}

    # 2. 平均值填補（若 mp=avg）
    means = None
    if args.mp == "avg":
        concat = pd.concat(dfs.values(), ignore_index=True)
        means = {c: concat.loc[concat[c] != -1, c].mean() for c in FEATURES}
        for i, df in dfs.items():
            for c, m in means.items():
                df.loc[df[c] == -1, c] = m

    # 3. 個別訓練
    estimators, weights = [], []
    print("Training 10 SVMs …")
    for idx in TRAIN_IDS:
        mdl, cv_f1 = train_single(idx, dfs[idx])
        estimators.append(mdl)
        weights.append(cv_f1)
        print(f"  svm_{idx}: CV-F1 = {cv_f1:.3f}")
    weights = np.array(weights, float)
    weights /= weights.sum()

    # 4. 存模型 (dict)
    ensemble = {"models": estimators, "weights": weights, "mp": args.mp, "means": means}
    joblib.dump(ensemble, model_dir / "ensemble_svm.joblib")
    print("✓ Saved to ./model/ensemble_svm.joblib")

    # 5. 報訓練內 F1（僅供參考）
    X_all = pd.concat([dfs[i][FEATURES] for i in TRAIN_IDS]).values
    y_all = pd.concat([dfs[i][LABEL] for i in TRAIN_IDS]).values
    preds = weighted_vote_predict(ensemble, X_all)
    print(f"Training-set F1 = {f1_score(y_all, preds):.4f}")


def weighted_vote_predict(ens, X):
    probs = np.zeros((X.shape[0], 2))
    for m, w in zip(ens["models"], ens["weights"]):
        probs += m.predict_proba(X) * w
    return probs.argmax(axis=1)


if __name__ == "__main__":
    main()
