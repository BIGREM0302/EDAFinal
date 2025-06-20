#!/usr/bin/env python3
"""
calc_f1.py — Evaluate F1 score for Trojan-gate prediction

用法
────
python calc_f1.py 0_USVM.csv result0.txt
# 或
python calc_f1.py --pred 0_USVM.csv --gt result0.txt

規則
────
* 只把預測檔中第 2 欄 (name) 以 g 開頭的字串視為「預測為 Trojan」。
* ground-truth 取自 resultX.txt 中 TROJAN_GATES 與 END_TROJAN_GATES 之間的行。
"""

import argparse
import csv


def load_predicted(csv_path: str) -> set[str]:
    """讀 0_USVM.csv → 回傳以 'g' 開頭的 gate 名稱集合"""
    gates: set[str] = set()
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            name = row[1].strip()
            # 略過 header 與非 gate 名稱
            if name.lower() == "name":
                continue
            if name.startswith("g"):
                gates.add(name)
    return gates


def load_ground_truth(txt_path: str) -> set[str]:
    """讀 resultX.txt → 回傳正確 Trojan gate 名稱集合"""
    gates: set[str] = set()
    with open(txt_path) as f:
        inside = False
        for line in f:
            l = line.strip()
            if l == "TROJAN_GATES":
                inside = True
                continue
            if l == "END_TROJAN_GATES":
                break
            if inside and l:
                gates.add(l)
    return gates


def f1_score(tp: int, fp: int, fn: int) -> float:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    reca = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * prec * reca / (prec + reca) if (prec + reca) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred", nargs="?", help="預測 CSV，例如 0_USVM.csv")
    ap.add_argument("gt", nargs="?", help="ground truth TXT，例如 result0.txt")
    ap.add_argument("--pred", dest="pred_kw")
    ap.add_argument("--gt", dest="gt_kw")
    args = ap.parse_args()

    pred_path = args.pred or args.pred_kw
    gt_path = args.gt or args.gt_kw
    if not (pred_path and gt_path):
        ap.error("必須提供預測 CSV 與 ground-truth TXT 路徑")

    pred = load_predicted(pred_path)
    truth = load_ground_truth(gt_path)

    tp = len(pred & truth)
    fp = len(pred - truth)
    fn = len(truth - pred)
    f1 = f1_score(tp, fp, fn)

    print(f"TP={tp}  FP={fp}  FN={fn}")
    print(f"Precision={tp / (tp+fp):.4f}" if tp + fp else "Precision=0.0000")
    print(f"Recall   ={tp / (tp+fn):.4f}" if tp + fn else "Recall   =0.0000")
    print(f"F1-score ={f1:.4f}")


if __name__ == "__main__":
    main()
