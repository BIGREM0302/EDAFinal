#!/usr/bin/env python3
"""
train_eval_svm_small.py
────────────────────────────────────────────────────────────────────────
• 讀取 GNNfeature0–9.csv，僅保留 gate 數 ≤ 3000 的 design 做訓練
• GNNfeature10–19.csv 為測試集
• per-file 0–1 normalization（忽略 -1）
• 缺值策略 --mp {avg|nochange}
• 反比 gate 數 sample_weight: w = 1.0 / gate_count_design
• 5-fold GroupKFold + GridSearch (C∈{0.1,1,10,100}, γ∈{scale,0.01,0.1,1})
• 輸出：
    ./result/    → *_prediction.csv, *_SVM.csv, f1_score.txt
    ./result/trojan_svm.joblib
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

FEATURES  = ["LGFi", "FFi", "FFo", "Pi", "Po"]
LABEL     = "Trojan_gate"
TRAIN_IDS = list(range(0, 10))
TEST_IDS  = list(range(10, 20))
GATE_CAP  = 3000
GRID      = {"svc__C":[0.1,1,10,100],
             "svc__gamma":["scale",0.01,0.1,1]}
RE_N_IN   = re.compile(r"^n\\[\\d+\\]$")

# ---------- utils -----------------------------------------------------
def normalize(df):
    out = df.copy()
    for c in FEATURES:
        m = out[c] != -1
        if m.any():
            mx = out.loc[m, c].max()
            if mx != 0:
                out.loc[m, c] = out.loc[m, c] / mx
    return out

def read_data(data_dir: Path):
    tr, te = {}, {}
    for idx in TRAIN_IDS + TEST_IDS:
        df = pd.read_csv(data_dir/f"GNNfeature{idx}.csv")
        df[FEATURES] = df[FEATURES].astype(float)
        df = normalize(df)
        (tr if idx in TRAIN_IDS else te)[idx] = df
    return tr, te

def fill_missing(dfs, mode, means=None):
    if mode=="nochange": return dfs, means
    if means is None:
        concat = pd.concat(dfs.values(), ignore_index=True)
        means = {c: concat.loc[concat[c]!=-1, c].mean() for c in FEATURES}
    fixed = {}
    for k, df in dfs.items():
        d = df.copy()
        for c,m in means.items():
            d.loc[d[c]==-1, c] = m
        fixed[k]=d
    return fixed, means

# ---------- main ------------------------------------------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_dir",default="training_data_for_svm")
    ap.add_argument("--mp",choices=["avg","nochange"],default="avg")
    args=ap.parse_args()

    data_dir = Path(args.data_dir)
    result_dir = Path("result"); result_dir.mkdir(exist_ok=True)

    tr, te = read_data(data_dir)
    # ---- 只留小模型 ----
    small_ids = [i for i,df in tr.items() if len(df)<=GATE_CAP]
    if not small_ids:
        raise RuntimeError("0–9 中沒有 gate<=3000 的 design！")
    print("Using SMALL designs:", small_ids)

    tr_small = {i:tr[i] for i in small_ids}

    tr_small, means = fill_missing(tr_small, args.mp)
    te, _           = fill_missing(te, args.mp, means)

    # ---- 構建訓練資料 ----
    train_df = pd.concat([df.assign(group=str(i)) for i,df in tr_small.items()],
                         ignore_index=True)
    X_tr = train_df[FEATURES].values
    y_tr = train_df[LABEL].astype(int).values
    groups = train_df["group"].values
    counts = {str(i):len(df) for i,df in tr_small.items()}
    sample_w = np.array([1.0/counts[g] for g in groups])

    # ---- SVM ----
    pipe = make_pipeline(StandardScaler(),
                         SVC(kernel="rbf", class_weight="balanced"))
    cv = GroupKFold(n_splits=5)
    search = GridSearchCV(pipe, GRID, scoring="f1_macro",
                          cv=cv, n_jobs=-1, verbose=1)
    search.fit(X_tr, y_tr,
               groups=groups,
               svc__sample_weight=sample_w)
    clf = search.best_estimator_
    print("Best params:", search.best_params_)

    # ---- TEST ----
    test_df = pd.concat(te.values(), ignore_index=True)
    y_pred = clf.predict(test_df[FEATURES].values)
    f1 = f1_score(test_df[LABEL].astype(int).values, y_pred)
    print(f"Test F1 = {f1:.4f}")

    # ---- save ----
    joblib.dump(clf, result_dir/"trojan_svm.joblib")
    (result_dir/"f1_score.txt").write_text(f"{f1:.6f}\n")

    off=0
    for idx in TEST_IDS:
        df = te[idx].copy()
        n=len(df)
        df["pred"]=y_pred[off:off+n]; off+=n
        df.to_csv(result_dir/f"GNNfeature{idx}_prediction.csv",index=False)
        trojan=df[(df["pred"]==1)&~df["name"].str.match(RE_N_IN)]
        trojan[["name"]].to_csv(result_dir/f"GNNfeature{idx}_SVM.csv",
                                header=False,index=False)
    print("Outputs saved to", result_dir.resolve())

if __name__=="__main__":
    main()
