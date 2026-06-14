"""
Fast Random Forest training on temporal NGSIM dataset.

Usage:
    python src/models/train_ngsim_temporal_rf.py
    python src/models/train_ngsim_temporal_rf.py --sample 100000
"""

import os
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)

DATA_PATH = "data/processed/platoon_data_ngsim_temporal.csv"
MODEL_PATH = "models/ngsim_temporal_rf_model.joblib"
REPORT_PATH = "outputs/reports/ngsim_temporal_rf_report.txt"
CONFUSION_MATRIX_PATH = "outputs/plots/ngsim_temporal_rf_confusion_matrix.png"

LABEL_NAMES = ["BRAKE", "FOLLOW", "CHANGE_LANE_LEFT", "CHANGE_LANE_RIGHT"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=120000)
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)

    print("Loading temporal NGSIM dataset...")

    df = pd.read_csv(DATA_PATH).dropna()

    print(f"Original dataset shape: {df.shape}")

    if args.sample is not None and args.sample < len(df):
        df = (
            df.groupby("decision", group_keys=False)
            .apply(lambda x: x.sample(
                min(len(x), max(1, int(args.sample * len(x) / len(df)))),
                random_state=42
            ))
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )
        print(f"Using stratified sample shape: {df.shape}")

    print("\nDecision distribution:")
    print(df["decision"].value_counts().sort_index())

    X = df.drop("decision", axis=1)
    y = df["decision"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("\nTraining fast temporal Random Forest model...")

    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=20,
        min_samples_split=8,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )

    model.fit(X_train, y_train)

    print("\nEvaluating temporal RF model...")

    y_pred = model.predict(X_test)

    accuracy = (y_pred == y_test).mean() * 100

    report = classification_report(
        y_test,
        y_pred,
        target_names=LABEL_NAMES,
        digits=3,
        zero_division=0,
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        labels=[0, 1, 2, 3],
        zero_division=0,
    )

    brake_recall = recall[0] * 100
    brake_f1 = f1[0] * 100
    macro_f1 = f1.mean() * 100

    print("\n--- NGSIM TEMPORAL RF RESULTS ---")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"BRAKE Recall: {brake_recall:.2f}%")
    print(f"BRAKE F1: {brake_f1:.2f}%")
    print(f"Macro F1: {macro_f1:.2f}%")
    print()
    print(report)

    with open(REPORT_PATH, "w") as file:
        file.write("NGSIM TEMPORAL RANDOM FOREST REPORT\n\n")
        file.write(f"Accuracy: {accuracy:.2f}%\n")
        file.write(f"BRAKE Recall: {brake_recall:.2f}%\n")
        file.write(f"BRAKE F1: {brake_f1:.2f}%\n")
        file.write(f"Macro F1: {macro_f1:.2f}%\n\n")
        file.write(report)

    joblib.dump(model, MODEL_PATH)

    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=LABEL_NAMES,
    )

    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("NGSIM Temporal RF Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close()

    print("\nSaved files:")
    print(f"- {MODEL_PATH}")
    print(f"- {REPORT_PATH}")
    print(f"- {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()