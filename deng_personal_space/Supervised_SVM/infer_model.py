#!/usr/bin/env python3
"""
infer_model.py
──────────────
--mode {old,n,w,nw}   指定用哪個模型推論
--threshold T         avg_prob > T 視為 trojan (default 0.5)
--data_dir PATH       CSV 目錄 (default=training_data_for_svm)
產物：
  result/<mode>/GNNfeature<ID>_prediction.csv
  result/<mode>/GNNfeature<ID>_SVM.csv
  f1_<mode>.csv       (design_id,f1_score)
"""

import argparse, re, warnings, joblib
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore", category=UserWarning)

FEATURES = ["LGFi", "FFi", "FFo", "Pi", "Po"]
LABEL = "Trojan_gate"
ALL_IDS = list(range(0, 20))
RE_N_IN = re.compile(r"^n\\[\\d+\\]$")


def normalize(df):
    out = df.copy()
    for c in FEATURES:
        m = out[c] != -1
        if m.any():
            mx = out.loc[m, c].max()
            if mx != 0:
                out.loc[m, c] = out.loc[m, c] / mx
    return out


def load_csv(path, need_norm):
    df = pd.read_csv(path)
    df[FEATURES] = df[FEATURES].astype(float)
    return normalize(df) if need_norm else df


def weighted_proba(ens, X):
    probs = np.zeros((X.shape[0], 2))
    for m, w in zip(ens["models"], ens["weights"]):
        probs += m.predict_proba(X) * w
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["old", "n", "w", "nw"], required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--data_dir", default="training_data_for_svm")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    mode = args.mode
    need_norm = mode in ("n", "nw")
    model_path = {
        "old": "old_svm.joblib",
        "n": "n_svm.joblib",
        "w": "w_svm.joblib",
        "nw": "nw_svm.joblib",
    }[mode]
    model = joblib.load(Path("model") / model_path)

    # 若是 ensemble dict (nw/w/n) 情況?  這裡單一 SVM -> 直接 predict_proba
    is_pipe = hasattr(model, "predict_proba")

    res_dir = Path(f"result/{mode}")
    res_dir.mkdir(parents=True, exist_ok=True)
    f1_records = []
    for idx in ALL_IDS:
        df = load_csv(data_dir / f"GNNfeature{idx}.csv", need_norm)
        probs = (
            model.predict_proba(df[FEATURES].values)
            if is_pipe
            else model.predict_proba(df[FEATURES].values)
        )
        trojan_prob = probs[:, 1]
        pred = (trojan_prob > args.threshold).astype(int)
        df["pred"] = pred

        # prediction csv
        df.to_csv(res_dir / f"GNNfeature{idx}_prediction.csv", index=False)
        trojan = df[(df["pred"] == 1) & ~df["name"].str.match(RE_N_IN)]
        trojan[["name"]].to_csv(
            res_dir / f"GNNfeature{idx}_SVM.csv", header=False, index=False
        )

        if LABEL in df.columns:
            f1_records.append({"design_id": idx, "f1_score": f1_score(df[LABEL], pred)})

    pd.DataFrame(f1_records).to_csv(f"f1_{mode}.csv", index=False)
    print(f"✓ wrote f1_{mode}.csv and outputs in {res_dir}")


if __name__ == "__main__":
    main()
