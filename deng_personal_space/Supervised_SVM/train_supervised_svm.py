#!/usr/bin/env python3
"""
train_supervised_svm.py — Train a 3-feature RBF SVM to detect Trojan gates.

Example
-------
python train_supervised_svm.py --csv GNNfeature.csv --result result0.txt --cols 2 3 4 --model trojan_svm --plot

Options
-------
--csv     Path to the feature CSV (required)
--result  Path to result*.txt that lists Trojan gate names (required)
--cols    Integer indices (0-based) of feature columns to use (default: 2 3 4)
--test    Test-set ratio (default: 0.3)
--C       SVM C parameter (default: 1.0)
--gamma   SVM γ parameter (default: "scale")
--model   Prefix to save trained model + scaler as <prefix>_model.joblib, <prefix>_scaler.joblib
--plot    Add this flag to draw a 3-D scatter (requires GUI backend)
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

INT_MAX = 2147483647


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a supervised 3-feature RBF SVM to detect Trojan gates",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", required=True, help="CSV file with features")
    parser.add_argument(
        "--result", required=True, help="result*.txt file listing Trojan gate IDs/names"
    )
    parser.add_argument(
        "--cols",
        type=int,
        nargs="+",
        default=[2, 3, 4],
        help="0-based feature column indices",
    )
    parser.add_argument("--test", type=float, default=0.3, help="test-set ratio")
    parser.add_argument("--C", type=float, default=1.0, help="SVM C parameter")
    parser.add_argument(
        "--gamma",
        default="scale",
        help="SVM gamma parameter ('scale', 'auto', or float)",
    )
    parser.add_argument("--model", help="prefix for saving model and scaler")
    parser.add_argument("--plot", action="store_true", help="display 3-D scatter plot")
    return parser.parse_args()


def load_labels(path: Path):
    trojan = set()
    recording = False
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line.upper() == "TROJAN_GATES":
                recording = True
                continue
            if line.upper() == "END_TROJAN_GATES":
                break
            if recording and line:
                trojan.add(line)
    if not trojan:
        sys.exit("ERROR: No TROJAN_GATES block found in result file")
    return trojan


def main():
    args = parse_args()

    csv_path = Path(args.csv).expanduser()
    result_path = Path(args.result).expanduser()

    if not csv_path.exists():
        sys.exit(f"ERROR: CSV file not found: {csv_path}")
    if not result_path.exists():
        sys.exit(f"ERROR: Result file not found: {result_path}")

    # Load data
    df = pd.read_csv(csv_path, header=None)

    # Replace INT_MAX placeholders with column mean
    for c in args.cols:
        mean_val = df.loc[df[c] != INT_MAX, c].mean()
        df.loc[df[c] == INT_MAX, c] = mean_val

    # Load labels
    trojan_set = load_labels(result_path)
    df["label"] = df[1].apply(lambda x: 1 if x in trojan_set else 0)

    # Prepare features and labels
    X = np.log1p(df[args.cols].values)
    y = df["label"].values

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=args.test, stratify=y, random_state=42
    )

    # Train SVM
    clf = SVC(
        kernel="rbf",
        C=args.C,
        gamma=args.gamma,
        class_weight="balanced",
        probability=True,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    print("\nClassification report (test set):")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"AUROC: {roc_auc_score(y_test, y_proba):.4f}")

    # Optional 3D scatter
    if args.plot:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(
            X_scaled[:, 0],
            X_scaled[:, 1],
            X_scaled[:, 2],
            c=y,
            s=20,
            cmap="coolwarm",
            depthshade=False,
        )
        ax.set_xlabel(f"log1p(col{args.cols[0]}) [std]")
        ax.set_ylabel(f"log1p(col{args.cols[1]}) [std]")
        ax.set_zlabel(f"log1p(col{args.cols[2]}) [std]")
        ax.set_title("Trojan (1) vs Normal (0)")
        plt.colorbar(sc, label="Label")
        plt.tight_layout()
        plt.show()

    # Save model and scaler
    if args.model:
        joblib.dump(clf, f"{args.model}_model.joblib")
        joblib.dump(scaler, f"{args.model}_scaler.joblib")
        print(f"\nSaved model to {args.model}_model.joblib")
        print(f"Saved scaler to {args.model}_scaler.joblib")


if __name__ == "__main__":
    main()
