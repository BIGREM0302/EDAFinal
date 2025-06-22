#!/usr/bin/env python3
"""
big_case_normalize_gnnfeature.py
────────────────────────────────────────────────────────────
• 只納入 gate 數量 (CSV 列數) > 3000 的設計
• 對   LGFi, Pi, Po, FFi, FFo   做 0–1 normalization
  → 採「所有符合條件之檔案」最大值為分母
• 依 Trojan gate (=0/1) 分別計算五項特徵的平均
• 產出 big_case_feature_table.tex（booktabs 風格）
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd

# === 可自行調整的參數 ===============================
FEATURES = ["LGFi", "Pi", "Po", "FFi", "FFo"]
FILE_PATTERN = "GNNfeature*.csv"
TROJAN_COL = "Trojan_gate"
BIG_THRESHOLD = 3000  # gate > 3000 才算 big case
OUT_TEX_FILE = "big_case_feature_table.tex"
# =====================================================


def main() -> None:
    big_dfs = []

    # 1. 篩選 gate > 3000 的檔案
    for csv_path in sorted(Path(".").glob(FILE_PATTERN)):
        df = pd.read_csv(csv_path)
        if len(df) > BIG_THRESHOLD:
            big_dfs.append(df)

    if not big_dfs:
        raise RuntimeError(f"沒有任何檔案的 gate 數 > {BIG_THRESHOLD}")

    # 2. 串成一張大表
    df_all = pd.concat(big_dfs, ignore_index=True)

    # 3. INT_MAX → NaN（若用來佔位）
    int_max = np.iinfo(np.int32).max
    df_all[FEATURES] = df_all[FEATURES].replace(int_max, np.nan)

    # 4. 0–1 normalization（以符合條件檔案的最大值為準）
    max_vals = df_all[FEATURES].max()
    df_norm = df_all.copy()
    df_norm[FEATURES] = df_norm[FEATURES].div(max_vals)

    # 5. 依 Trojan_gate 分組平均
    mean_tbl = df_norm.groupby(TROJAN_COL, dropna=False)[FEATURES].mean().round(4)

    # 6. 組 LaTeX 表格
    latex_lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r" & " + " & ".join(FEATURES) + r" \\",
        r"\midrule",
        "Normal gate & " + " & ".join(mean_tbl.loc[0].astype(str)) + r" \\",
        "Trojan gate & " + " & ".join(mean_tbl.loc[1].astype(str)) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ]

    # 7. 寫檔
    Path(OUT_TEX_FILE).write_text("\n".join(latex_lines), encoding="utf-8")
    print(f"LaTeX 表格已輸出到 {OUT_TEX_FILE}")


if __name__ == "__main__":
    main()
