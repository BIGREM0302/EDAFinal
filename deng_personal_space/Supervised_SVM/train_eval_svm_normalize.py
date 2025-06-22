#!/usr/bin/env python3
"""
train_eval_svm_normalize.py   (normalize + per-design F1 table)
────────────────────────────────────────────────────────────────
• 訓練集：GNNfeature0–9.csv
• 測試集：GNNfeature10–19.csv   ← 整體 F1 寫入 f1_score.txt
• 推論與輸出覆蓋 0–19 全部 design：
    result/GNNfeature<ID>_prediction.csv
    result/GNNfeature<ID>_SVM.csv
• 每 design 的 F1 → f1_table.csv
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

# ───────── 固定參數 ───────────────────────────────────────────
FEATURES = ["LGFi", "FFi", "FFo", "Pi", "Po"]
LABEL = "Trojan_gate"
TRAIN_IDS = list(range(0, 10))
TEST_IDS = list(range(10, 20))
ALL_IDS = TRAIN_IDS + TEST_IDS
GRID = {"svc__C": [0.1, 1, 10, 100], "svc__gamma": ["scale", 0.01, 0.1, 1]}
RE_N_IN = re.compile(r"^n\\[\\d+\\]$")


# ───────── 工具函式 ───────────────────────────────────────────
def normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in FEATURES:
        mask = out[col] != -1
        if mask.any():
            mx = out.loc[mask, col].max()
            if mx != 0:
                out.loc[mask, col] = out.loc[mask, col] / mx
    return out


def read_corpus(data_dir: Path):
    dfs = {}
    for idx in ALL_IDS:
        df = pd.read_csv(data_dir / f"GNNfeature{idx}.csv")
        df[FEATURES] = df[FEATURES].astype(float)
        dfs[idx] = normalize(df)
    return dfs


def fill_missing(dfs, mode: str, means=None):
    if mode == "nochange":
        return dfs, means
    if means is None:
        cat = pd.concat(dfs[i] for i in TRAIN_IDS)  # 只用 train 算均值
        means = {c: cat.loc[cat[c] != -1, c].mean() for c in FEATURES}
    fixed = {}
    for k, df in dfs.items():
        d = df.copy()
        for c, m in means.items():
            d.loc[d[c] == -1, c] = m
        fixed[k] = d
    return fixed, means


# ───────── 主程式 ────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="training_data_for_svm")
    ap.add_argument("--mp", choices=["avg", "nochange"], default="avg")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    result_dir = Path("result")
    result_dir.mkdir(exist_ok=True)

    # 1. 讀檔 + normalize + 缺值處理
    dfs = read_corpus(data_dir)
    dfs, means = fill_missing(dfs, args.mp)

    # 2. 組訓練資料
    train_df = pd.concat(
        [dfs[i].assign(group=str(i)) for i in TRAIN_IDS], ignore_index=True
    )
    X_tr = train_df[FEATURES].values
    y_tr = train_df[LABEL].astype(int).values
    groups = train_df["group"].values

    # 3. 單一 SVM + GridSearch
    pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced"))
    cv = GroupKFold(n_splits=5)
    gs = GridSearchCV(pipe, GRID, scoring="f1_macro", cv=cv, n_jobs=-1, verbose=1)
    gs.fit(X_tr, y_tr, groups=groups)
    clf = gs.best_estimator_
    print("Best params:", gs.best_params_)

    # 4. 針對 0–19 全部 design 推論
    f1_records = []
    all_test_preds = []
    for idx in ALL_IDS:
        df = dfs[idx].copy()
        preds = clf.predict(df[FEATURES].values)
        df["pred"] = preds
        # prediction.csv
        df.to_csv(result_dir / f"GNNfeature{idx}_prediction.csv", index=False)
        # _SVM.csv (name only)
        trojan = df[(df["pred"] == 1) & ~df["name"].str.match(RE_N_IN)]
        trojan[["name"]].to_csv(
            result_dir / f"GNNfeature{idx}_SVM.csv", header=False, index=False
        )
        # per-design F1
        if LABEL in df.columns:
            f1 = f1_score(df[LABEL].values, preds)
            f1_records.append({"design_id": idx, "f1_score": f1})
            if idx in TEST_IDS:  # 收集外部測試用
                all_test_preds.append((df[LABEL].values, preds))

    # 5. 整體外部測試 F1 (10–19)
    if all_test_preds:
        y_true = np.concatenate([t for t, _ in all_test_preds])
        y_pred = np.concatenate([p for _, p in all_test_preds])
        test_f1 = f1_score(y_true, y_pred)
        (result_dir / "f1_score.txt").write_text(f"{test_f1:.6f}\n")
        print(f"Test F1 (10–19) = {test_f1:.4f}")

    # 6. per-design F1 table
    pd.DataFrame(f1_records).to_csv("f1_table.csv", index=False)
    print("Per-design F1 saved to f1_table.csv")

    # 7. 存模型
    joblib.dump(clf, result_dir / "trojan_svm.joblib")
    print("All outputs in", result_dir.resolve())


if __name__ == "__main__":
    main()
