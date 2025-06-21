#!/usr/bin/env python3
"""
infer_ensemble_svm.py  (with soft‐vote threshold)
────────────────────────────────────────────────────────────────────────
• 使用 ./model/ensemble_svm.joblib
• 透過 --csv_list 10 11 12 … 指定欲推論的檔案 id
  (實際檔名為 <data_dir>/GNNfeature<ID>.csv)
• 重複 train 端的 normalize 與 -1 處理
• Soft‐vote: 如果加權平均的 Trojan 機率 > threshold，就判 1
• 產生
    ./result/GNNfeature<ID>_prediction.csv
    ./result/GNNfeature<ID>_SVM.csv  (僅 name；pred=1；排除 n[數字])
• 若檔案含 Trojan_gate 欄，計算整體 F1 並寫 f1_score.txt
"""

import argparse
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore", category=UserWarning)

FEATURES = ["LGFi", "FFi", "FFo", "Pi", "Po"]
LABEL = "Trojan_gate"
RE_N_IN = re.compile(r"^n\[\d+\]$")


# -------------------- utils ----------------------------------
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


def weighted_vote_proba(ens, X: np.ndarray) -> np.ndarray:
    """回傳 shape (n_samples, 2) 的加權平均機率矩陣"""
    probs = np.zeros((X.shape[0], 2))
    for m, w in zip(ens["models"], ens["weights"]):
        probs += m.predict_proba(X) * w
    return probs


# -------------------- main -----------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv_list", nargs="+", required=True, help="僅輸入檔案 id，例如 10 11 12"
    )
    ap.add_argument("--data_dir", default="training_data_for_svm")
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Soft‐vote 閾值，平均 Trojan 機率 > threshold 才判為 1",
    )
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    result_dir = Path("result")
    result_dir.mkdir(exist_ok=True)

    # 1. 載模型
    ensemble = joblib.load(Path("model/ensemble_svm.joblib"))

    # 2. 推論
    all_has_label, y_true_all, y_pred_all = [], [], []
    for sid in args.csv_list:
        csv_path = data_dir / f"GNNfeature{sid}.csv"
        df = load_csv(csv_path)

        # 填補 -1（若 mp=avg）
        if ensemble["mp"] == "avg":
            for c, m in ensemble["means"].items():
                df.loc[df[c] == -1, c] = m

        # Soft vote 機率
        probs = weighted_vote_proba(ensemble, df[FEATURES].values)
        trojan_prob = probs[:, 1]
        preds = (trojan_prob > args.threshold).astype(int)
        df["pred"] = preds

        # 輸出 prediction.csv
        out_pred = result_dir / f"GNNfeature{sid}_prediction.csv"
        df.to_csv(out_pred, index=False)

        # 輸出 _SVM.csv
        trojan_only = df[(df["pred"] == 1) & ~df["name"].str.match(RE_N_IN)]
        trojan_only[["name"]].to_csv(
            result_dir / f"GNNfeature{sid}_SVM.csv", header=False, index=False
        )

        # 收集 F1
        if LABEL in df.columns:
            all_has_label.append(True)
            y_true_all.append(df[LABEL].values)
            y_pred_all.append(preds)
        else:
            all_has_label.append(False)

        print(f"✓ Done {csv_path.name} (threshold={args.threshold})")

    # 3. 整體 F1
    if all(all_has_label):
        y_true = np.concatenate(y_true_all)
        y_pred = np.concatenate(y_pred_all)
        f1 = f1_score(y_true, y_pred)
        Path("f1_score.txt").write_text(f"Overall_F1={f1:.6f}\n")
        print(f"Overall F1 = {f1:.4f} (寫入 f1_score.txt)")
    else:
        print("部分檔案無標籤，未計算 Overall F1。")


if __name__ == "__main__":
    main()
