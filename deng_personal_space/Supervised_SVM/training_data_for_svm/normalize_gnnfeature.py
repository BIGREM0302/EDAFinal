#!/usr/bin/env python3
"""
normalize_gnnfeature.py
────────────────────────────────────────────────────────────
• 讀取 GNNfeature0.csv ~ GNNfeature19.csv
• 對   LGFi, Pi, Po, FFi, FFo   做 0–1 normalization
  → 採「全檔案」最大值為分母
• 依 Trojan gate (=0/1) 分別計算五項特徵的平均
• 產出 booktabs 風格的 LaTeX 表格
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd

# === 可自行調整的參數 ===============================
FEATURES = ["LGFi", "Pi", "Po", "FFi", "FFo"]
FILE_PATTERN = "GNNfeature*.csv"  # 讀檔樣式
TROJAN_COL = "Trojan_gate"  # 0 = normal, 1 = trojan
# =====================================================


def main() -> None:
    csv_files = sorted(Path(".").glob(FILE_PATTERN))
    if not csv_files:
        raise FileNotFoundError("找不到任何符合 GNNfeature*.csv 的檔案")

    # 1. 讀檔後串成一張大表
    df_all = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    # 2. INT_MAX → NaN（若有缺值佔位）
    int_max = np.iinfo(np.int32).max
    df_all[FEATURES] = df_all[FEATURES].replace(int_max, np.nan)

    # 3. 以「所有檔案」最大值做 0–1 normalization
    max_vals = df_all[FEATURES].max()
    df_all_norm = df_all.copy()
    df_all_norm[FEATURES] = df_all_norm[FEATURES].div(max_vals)

    # 4. 依 Trojan_gate 分組平均
    group_mean = (
        df_all_norm.groupby(TROJAN_COL, dropna=False)[FEATURES]
        .mean()
        .round(4)  # 取四位小數，視需求調整
    )

    # 5. 輸出 LaTeX 表格
    latex_lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r" & " + " & ".join(FEATURES) + r" \\",
        r"\midrule",
        "Normal gate & " + " & ".join(group_mean.loc[0].astype(str)) + r" \\",
        "Trojan gate & " + " & ".join(group_mean.loc[1].astype(str)) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    print("\n".join(latex_lines))


if __name__ == "__main__":
    main()
