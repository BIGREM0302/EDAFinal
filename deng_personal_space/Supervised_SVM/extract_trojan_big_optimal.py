#!/usr/bin/env python3
"""
train_eval_svm_gt3000.py
──────────────────────────────────────────────────────────────
• 僅選 gate 行數 > 3000 的 design
    TRAIN  : GNNfeature0–9 之中符合條件者
    TEST   : GNNfeature10–19 之中符合條件者
• 無 normalize、無 sample_weight
• SVC(kernel='rbf', probability=False)
• 產出:
    result_gt3000/<ID>_prediction.csv
    result_gt3000/<ID>_SVM.csv
    result_gt3000/f1_score.txt       (test 整體 F1)
    f1_table.csv                     (design_id,f1_score)
"""

import argparse, re, warnings
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore", category=UserWarning)

FEATS = ["LGFi","FFi","FFo","Pi","Po"]
LABEL = "Trojan_gate"
GRID  = {"svc__C":[0.1,1,10,100],
         "svc__gamma":["scale",0.01,0.1,1]}
RE_N  = re.compile(r"^n\\[\\d+\\]$")

def load_csv(path):
    df = pd.read_csv(path)
    df[FEATS] = df[FEATS].astype(float)
    return df

def fill_missing(dfs, mode, means=None):
    if mode=="nochange": return dfs, means
    if means is None:
        concat = pd.concat(dfs.values())
        means  = {c: concat.loc[concat[c]!=-1,c].mean() for c in FEATS}
    fixed={}
    for k,df in dfs.items():
        d=df.copy()
        for c,m in means.items():
            d.loc[d[c]==-1,c]=m
        fixed[k]=d
    return fixed, means

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data_dir",default="training_data_for_svm")
    p.add_argument("--mp",choices=["avg","nochange"],default="avg")
    args=p.parse_args()

    data_dir=Path(args.data_dir)
    result_dir=Path("result_gt3000"); result_dir.mkdir(exist_ok=True)

    # 1. 讀檔並篩選 gate>3000
    train_dfs, test_dfs = {}, {}
    TRAIN_SRC = range(0,10)
    TEST_SRC  = range(10,20)

    for idx in TRAIN_SRC:
        df=load_csv(data_dir/f"GNNfeature{idx}.csv")
        if len(df)>3000:
            train_dfs[idx]=df
    for idx in TEST_SRC:
        df=load_csv(data_dir/f"GNNfeature{idx}.csv")
        if len(df)>3000:
            test_dfs[idx]=df

    if not train_dfs or not test_dfs:
        raise RuntimeError("過濾後沒有足夠的 train 或 test design (>3000 rows)。")

    print("TRAIN_IDS:", list(train_dfs.keys()))
    print("TEST_IDS :", list(test_dfs.keys()))

    # 2. 填補 -1
    train_dfs, means = fill_missing(train_dfs, args.mp)
    test_dfs , _     = fill_missing(test_dfs , args.mp, means)

    # 3. 組訓練資料
    tr_df = pd.concat([df.assign(group=str(i)) for i,df in train_dfs.items()],
                      ignore_index=True)
    X_tr = tr_df[FEATS].values
    y_tr = tr_df[LABEL].astype(int).values
    groups = tr_df["group"].values

    # 4. GridSearch
    pipe = make_pipeline(StandardScaler(),
                         SVC(kernel="rbf",class_weight="balanced",probability=False))
    cv = GroupKFold(n_splits=5)
    gs = GridSearchCV(pipe, GRID, scoring="f1_macro",
                      cv=cv, n_jobs=-1, verbose=1)
    gs.fit(X_tr, y_tr, groups=groups)
    clf = gs.best_estimator_
    print("Best params:", gs.best_params_)

    # 5. 推論 & 輸出
    f1_records=[]
    for idx,df in {**train_dfs,**test_dfs}.items():
        preds = clf.predict(df[FEATS].values)
        df["pred"]=preds
        # prediction.csv
        df.to_csv(result_dir/f"GNNfeature{idx}_prediction.csv",index=False)
        # _SVM.csv
        trojan=df[(df["pred"]==1)&~df["name"].str.match(RE_N)]
        trojan[["name"]].to_csv(result_dir/f"GNNfeature{idx}_SVM.csv",
                                header=False,index=False)
        # F1
        f1_records.append({"design_id":idx,
                           "f1_score":f1_score(df[LABEL],preds)})

    # 整體 test F1
    y_true=np.concatenate([test_dfs[i][LABEL] for i in test_dfs])
    y_pred=np.concatenate([clf.predict(test_dfs[i][FEATS].values)
                           for i in test_dfs])
    test_f1=f1_score(y_true,y_pred)
    (result_dir/"f1_score.txt").write_text(f"{test_f1:.6f}\n")
    print(f"Test F1 (gate>3000) = {test_f1:.4f}")

    # per design F1 table
    pd.DataFrame(f1_records).to_csv("f1_table.csv",index=False)
    print("f1_table.csv saved.  All outputs in", result_dir.resolve())

    # 6. save model
    joblib.dump(clf, result_dir/"trojan_svm.joblib")

if __name__=="__main__":
    main()
