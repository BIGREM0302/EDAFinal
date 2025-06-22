#!/usr/bin/env python3
"""
train_eval_svm_weighted.py
────────────────────────────────────────────────────────────────────────
• 訓練：GNNfeature0–9.csv
• 測試：GNNfeature10–19.csv
• -1 缺值策略  --mp {avg|nochange}
• 依 design gate 數反比設定 sample_weight:
      w_design = 1 / gate_count(design)
• 單一 RBF-SVM，GridSearchCV 找最佳 (C, γ)
• 輸出：
    ./result/f1_score.txt
    ./result/trojan_svm.joblib
    ./result/GNNfeatureXX_prediction.csv
    ./result/GNNfeatureXX_SVM.csv   (name only；排除 n[ ])
"""

import argparse, re, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore", category=UserWarning)

# ---------- 常數 ----------
FEATURES = ["LGFi", "FFi", "FFo", "Pi", "Po"]
LABEL = "Trojan_gate"
TRAIN_IDS = list(range(0, 10))
TEST_IDS = list(range(10, 20))
GRID = {"svc__C": [0.1, 1, 10, 100], "svc__gamma": ["scale", 0.01, 0.1, 1]}
RE_N_IN = re.compile(r"^n\\[\\d+\\]$")


# ---------- 讀取 ----------
def read_data(data_dir: Path):
    tr, te = {}, {}
    for idx in TRAIN_IDS + TEST_IDS:
        df = pd.read_csv(data_dir / f"GNNfeature{idx}.csv")
        df[FEATURES] = df[FEATURES].astype(float)
        (tr if idx in TRAIN_IDS else te)[idx] = df
    return tr, te


def fill_missing(dfs, mode, means=None):
    if mode == "nochange":
        return dfs, means
    if means is None:
        cat = pd.concat(dfs.values(), ignore_index=True)
        means = {c: cat.loc[cat[c] != -1, c].mean() for c in FEATURES}
    fixed = {}
    for k, df in dfs.items():
        d = df.copy()
        for c, m in means.items():
            d.loc[d[c] == -1, c] = m
        fixed[k] = d
    return fixed, means


# ---------- 主程式 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="training_data_for_svm")
    ap.add_argument("--mp", choices=["avg", "nochange"], default="avg")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    result_dir = Path("result")
    result_dir.mkdir(exist_ok=True)

    tr, te = read_data(data_dir)
    tr, means = fill_missing(tr, args.mp)
    te, _ = fill_missing(te, args.mp, means)

    # --- 構建 train_df 與 sample_weight ---
    train_df = pd.concat(
        [df.assign(group=str(i)) for i, df in tr.items()], ignore_index=True
    )
    X_tr = train_df[FEATURES].values
    y_tr = train_df[LABEL].astype(int).values
    groups = train_df["group"].values
    counts = {str(i): len(df) for i, df in tr.items()}
    sample_w = np.array([1.0 / counts[g] for g in groups])

    # --- GridSearch ---
    pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced"))
    cv = GroupKFold(n_splits=5)
    search = GridSearchCV(pipe, GRID, scoring="f1_macro", cv=cv, n_jobs=-1, verbose=1)
    search.fit(X_tr, y_tr, groups=groups, svc__sample_weight=sample_w)
    clf = search.best_estimator_
    print("Best params:", search.best_params_)

    # --- 測試 ---
    test_df = pd.concat(te.values(), ignore_index=True)
    X_te, y_te = test_df[FEATURES].values, test_df[LABEL].astype(int).values
    y_pred = clf.predict(X_te)
    f1 = f1_score(y_te, y_pred)
    print(f"Test F1 = {f1:.4f}")

    # --- 儲存 ---
    joblib.dump(clf, result_dir / "trojan_svm.joblib")
    (result_dir / "f1_score.txt").write_text(f"{f1:.6f}\n")

    off = 0
    for idx in TEST_IDS:
        df = te[idx].copy()
        n = len(df)
        df["pred"] = y_pred[off : off + n]
        off += n
        df.to_csv(result_dir / f"GNNfeature{idx}_prediction.csv", index=False)

        trojan = df[(df["pred"] == 1) & ~df["name"].str.match(RE_N_IN)]
        trojan[["name"]].to_csv(
            result_dir / f"GNNfeature{idx}_SVM.csv", header=False, index=False
        )

    print("Outputs saved to", result_dir.resolve())


if __name__ == "__main__":
    main()
