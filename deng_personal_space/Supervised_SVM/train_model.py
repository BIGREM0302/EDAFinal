#!/usr/bin/env python3
"""
train_model.py
───────────────────────────────────────────────────────────────
--mode {origin,n,w,nw}
    origin : 無 normalize、無 sample_weight
    n      : per-file 0–1 normalize
    w      : per-design 1/size sample_weight
    nw     : normalize + sample_weight
--mp {avg,nochange}  缺值策略 (default=avg)
輸出：model/<mode>_svm.joblib
"""

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=UserWarning)

FEATS = ["LGFi", "FFi", "FFo", "Pi", "Po"]
LABEL = "Trojan_gate"
TRAIN_ID = range(0, 10)
GRID = {"svc__C": [0.1, 1, 10, 100], "svc__gamma": ["scale", 0.01, 0.1, 1]}


def normalize(df):
    out = df.copy()
    for c in FEATS:
        m = out[c] != -1
        if m.any():
            mx = out.loc[m, c].max()
            if mx != 0:
                out.loc[m, c] = out.loc[m, c] / mx
    return out


def load_all(data_dir):
    dfs = {}
    for i in TRAIN_ID:
        df = pd.read_csv(data_dir / f"GNNfeature{i}.csv")
        df[FEATS] = df[FEATS].astype(float)
        dfs[i] = df
    return dfs


def fill_missing(dfs, mode, means=None):
    if mode == "nochange":
        return dfs, means
    if means is None:
        cat = pd.concat(dfs.values())
        means = {c: cat.loc[cat[c] != -1, c].mean() for c in FEATS}
    fixed = {}
    for k, df in dfs.items():
        d = df.copy()
        for c, m in means.items():
            d.loc[d[c] == -1, c] = m
        fixed[k] = d
    return fixed, means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["origin", "n", "w", "nw"], required=True)
    ap.add_argument("--data_dir", default="training_data_for_svm")
    ap.add_argument("--mp", choices=["avg", "nochange"], default="avg")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)

    dfs = load_all(data_dir)
    if args.mode in ("n", "nw"):
        dfs = {k: normalize(df) for k, df in dfs.items()}
    dfs, _ = fill_missing(dfs, args.mp)

    train_df = pd.concat(
        [df.assign(group=str(i)) for i, df in dfs.items()], ignore_index=True
    )
    X = train_df[FEATS].values
    y = train_df[LABEL].astype(int).values
    groups = train_df["group"].values

    fit_kwargs = {}
    if args.mode in ("w", "nw"):
        counts = {str(i): len(df) for i, df in dfs.items()}
        sw = np.array([1.0 / counts[g] for g in groups])
        fit_kwargs["svc__sample_weight"] = sw

    pipe = make_pipeline(
        StandardScaler(), SVC(kernel="rbf", class_weight="balanced", probability=False)
    )
    gs = GridSearchCV(
        pipe, GRID, scoring="f1_macro", cv=GroupKFold(n_splits=5), n_jobs=-1, verbose=1
    )
    gs.fit(X, y, groups=groups, **fit_kwargs)
    print("Best params:", gs.best_params_)

    fname = f"{args.mode}_svm.joblib"
    joblib.dump(gs.best_estimator_, model_dir / fname)
    print(f"✓ saved model/{fname}")


if __name__ == "__main__":
    main()
