#!/usr/bin/env python3
"""
train_model.py
──────────────
--mode {old,n,w,nw}
    old : 無 normalize、無 sample_weight
    n   : per-file 0–1 normalize
    w   : 1/gate_count sample_weight
    nw  : normalize + sample_weight
--mp {avg,nochange}   缺值 -1 策略 (default=avg)
"""

import argparse, re, warnings
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold, GridSearchCV

warnings.filterwarnings("ignore", category=UserWarning)

FEATURES = ["LGFi", "FFi", "FFo", "Pi", "Po"]
LABEL = "Trojan_gate"
TRAIN_IDS = range(0, 10)
GRID = {"svc__C": [0.1, 1, 10, 100], "svc__gamma": ["scale", 0.01, 0.1, 1]}


def normalize(df):
    out = df.copy()
    for c in FEATURES:
        m = out[c] != -1
        if m.any():
            mx = out.loc[m, c].max()
            if mx != 0:
                out.loc[m, c] = out.loc[m, c] / mx
    return out


def load_all(data_dir):
    dfs = {}
    for i in TRAIN_IDS:
        df = pd.read_csv(data_dir / f"GNNfeature{i}.csv")
        df[FEATURES] = df[FEATURES].astype(float)
        dfs[i] = df
    return dfs


def fill_missing(dfs, mode, means=None):
    if mode == "nochange":
        return dfs, means
    if means is None:
        cat = pd.concat(dfs.values())
        means = {c: cat.loc[cat[c] != -1, c].mean() for c in FEATURES}
    out = {}
    for k, df in dfs.items():
        d = df.copy()
        for c, m in means.items():
            d.loc[d[c] == -1, c] = m
        out[k] = d
    return out, means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["old", "n", "w", "nw"], required=True)
    ap.add_argument("--data_dir", default="training_data_for_svm")
    ap.add_argument("--mp", choices=["avg", "nochange"], default="avg")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)

    dfs = load_all(data_dir)

    if args.mode in ("n", "nw"):  # normalize if needed
        dfs = {k: normalize(df) for k, df in dfs.items()}

    dfs, means = fill_missing(dfs, args.mp)

    # build train_df
    train_df = pd.concat(
        [df.assign(group=str(i)) for i, df in dfs.items()], ignore_index=True
    )
    X = train_df[FEATURES].values
    y = train_df[LABEL].astype(int).values
    groups = train_df["group"].values

    sample_w = None
    if args.mode in ("w", "nw"):
        counts = {str(i): len(df) for i, df in dfs.items()}
        sample_w = np.array([1.0 / counts[g] for g in groups])

    pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced"))
    cv = GroupKFold(n_splits=5)
    gs = GridSearchCV(pipe, GRID, scoring="f1_macro", cv=cv, n_jobs=-1, verbose=1)
    fit_kwargs = {}
    if sample_w is not None:
        fit_kwargs["svc__sample_weight"] = sample_w
    gs.fit(X, y, groups=groups, **fit_kwargs)
    best = gs.best_estimator_
    print("Best params:", gs.best_params_)

    fname = {
        "old": "old_svm.joblib",
        "n": "n_svm.joblib",
        "w": "w_svm.joblib",
        "nw": "nw_svm.joblib",
    }[args.mode]
    joblib.dump(best, model_dir / fname)
    print(f"✓ saved model/{fname}")


if __name__ == "__main__":
    main()
