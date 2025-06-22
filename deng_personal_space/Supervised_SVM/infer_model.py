#!/usr/bin/env python3
"""
infer_model.py
───────────────────────────────────────────────────────────────
--mode {origin,n,w,nw}
--data_dir PATH   CSV 資料夾 (default=training_data_for_svm)
輸出：
  result/<mode>/*_prediction.csv
  result/<mode>/*_SVM.csv
  f1_<mode>.csv   (design_id,f1_score)
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

FEATS=["LGFi","FFi","FFo","Pi","Po"]
LABEL="Trojan_gate"
ALL_ID=range(0,20)
RE_N  = re.compile(r"^n\\[\\d+\\]$")

def normalize(df):
    out=df.copy()
    for c in FEATS:
        m=out[c]!=-1
        if m.any():
            mx=out.loc[m,c].max()
            if mx!=0:
                out.loc[m,c]=out.loc[m,c]/mx
    return out

def load_csv(path,norm):
    df=pd.read_csv(path)
    df[FEATS]=df[FEATS].astype(float)
    return normalize(df) if norm else df

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["origin","n","w","nw"],required=True)
    ap.add_argument("--data_dir",default="training_data_for_svm")
    args=ap.parse_args()

    need_norm = args.mode in ("n","nw")
    model=joblib.load(Path("model")/f"{args.mode}_svm.joblib")
    res_dir=Path(f"result/{args.mode}"); res_dir.mkdir(parents=True,exist_ok=True)

    f1_rows=[]
    for idx in ALL_ID:
        df=load_csv(Path(args.data_dir)/f"GNNfeature{idx}.csv",need_norm)
        preds=model.predict(df[FEATS].values)
        df["pred"]=preds
        # prediction
        df.to_csv(res_dir/f"GNNfeature{idx}_prediction.csv",index=False)
        # SVM.csv
        trojan=df[(df["pred"]==1)&~df["name"].str.match(RE_N)]
        trojan[["name"]].to_csv(res_dir/f"GNNfeature{idx}_SVM.csv",
                                header=False,index=False)
        # F1
        if LABEL in df.columns:
            f1_rows.append({"design_id":idx,
                            "f1_score":f1_score(df[LABEL],preds)})

    pd.DataFrame(f1_rows).to_csv(f"f1_{args.mode}.csv",index=False)
    print(f"✓ wrote f1_{args.mode}.csv & files in {res_dir}")

if __name__=="__main__":
    main()
