import argparse
import pandas as pd
import joblib
import re
from pathlib import Path

# 固定參數
FEATURES = ["LGFi", "FFi", "FFo", "Pi", "Po"]
#RE_N_INPUT = re.compile(r"^n\[\d+\]$")

def run_inference(model_path, csv_path, output_path):
    model = joblib.load(model_path)
    df = pd.read_csv(csv_path)

    # use not -1 as mean value
    col_means = {
        c: df.loc[df[c] != -1, c].astype(float).mean()
        for c in FEATURES
    }

    # 將 -1 col_means[c]
    for c in FEATURES:
        print(f"feature:{c}, its mean value:{col_means[c]}")
        df[c] = df[c].astype(float)
        df.loc[df[c] == -1, c] = col_means[c]

    y_pred = model.predict(df[FEATURES].values)
    df["pred"] = y_pred

    # 輸出標記為 Trojan 的 gate（且排除 n[]）
    trojan_df = df[(df["pred"] == 1) & ~df["name"].str.startswith("n")]
    trojan_df[["name"]].to_csv(output_path, index=False, header=False)
    print(f"推論完成，結果寫入 {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/trojan_svm.joblib")
    parser.add_argument("--input_csv", default="parser_result/GNNfeature.csv")
    parser.add_argument("--output_txt", default="picked_gates.txt")

    args = parser.parse_args()
    run_inference(args.model, args.input_csv, args.output_txt)
