"""
Temporal model using platoon_data_enriched.csv (t0-t5 window).

Trains and compares:
  - Temporal MLP: flattened t0-t5 sensor window
  - LSTM: sequence-aware t0-t5 sensor window

Usage:
    python src\models\train_temporal_model.py
    python src\models\train_temporal_model.py --epochs 10
"""

import os
import argparse
import warnings

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, precision_recall_fscore_support

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

TIMESTEPS = 6

LABEL_NAMES = ["BRAKE", "FOLLOW", "LANE_LEFT", "LANE_RIGHT"]

SENSOR_COLS = [
    "ego_vel",
    "front_dist",
    "front_rel_vel",
    "time_gap",
    "left_front_dist",
    "left_rear_dist",
    "right_front_dist",
    "right_rear_dist",
]


def load_enriched(path):
    df = pd.read_csv(path).dropna()
    print(f"Loaded enriched dataset: {len(df):,} rows, {len(df.columns)} columns")
    return df


def build_flat_features(df):
    """
    Builds flattened temporal features:
    [ego_vel_t0, front_dist_t0, ..., ego_vel_t5, front_dist_t5, ...]
    """
    cols = []

    for t in range(TIMESTEPS):
        for feat in SENSOR_COLS:
            col = f"{feat}_t{t}"
            if col in df.columns:
                cols.append(col)

    if not cols:
        raise ValueError("No temporal feature columns found. Check platoon_data_enriched.csv.")

    X = df[cols].values.astype(np.float32)
    y = df["decision"].values

    return X, y, cols


def build_sequence_features(df):
    """
    Builds sequence features for LSTM:
    shape = (samples, timesteps, features)
    """
    n_samples = len(df)
    n_features = len(SENSOR_COLS)

    X_seq = np.zeros((n_samples, TIMESTEPS, n_features), dtype=np.float32)

    for t in range(TIMESTEPS):
        for f_idx, feat in enumerate(SENSOR_COLS):
            col = f"{feat}_t{t}"
            if col in df.columns:
                X_seq[:, t, f_idx] = df[col].values

    y = df["decision"].values

    return X_seq, y


def get_class_weights(y):
    classes = np.unique(y)
    weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y,
    )
    return dict(zip(classes, weights))


def build_flat_mlp(input_dim, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])

    return model


def build_lstm(timesteps, n_features, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(timesteps, n_features)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])

    return model


def train_and_eval(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    class_weights,
    model_name,
    epochs,
):
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    print(f"\n--- Training {model_name} ---")

    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=64,
        validation_split=0.1,
        class_weight=class_weights,
        callbacks=[early_stopping],
        verbose=1,
    )

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    accuracy = (y_pred == y_test).mean() * 100

    print(f"\n{model_name} - Test accuracy: {accuracy:.2f}%")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=LABEL_NAMES,
            digits=3,
            zero_division=0,
        )
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        labels=[0, 1, 2, 3],
        zero_division=0,
    )

    result = {
        "model": model_name,
        "accuracy": round(accuracy, 2),
        "brake_recall": round(recall[0] * 100, 1),
        "brake_f1": round(f1[0] * 100, 1),
        "macro_f1": round(f1.mean() * 100, 1),
    }

    return result, model


def print_comparison_table(results):
    print(f"\n{'=' * 70}")
    print("  TEMPORAL MODEL COMPARISON TABLE")
    print(f"{'=' * 70}")

    header = f"{'Metric':<25}"

    for result in results:
        header += f" {result['model']:>22}"

    print(header)
    print("-" * 70)

    metrics = [
        ("Overall accuracy %", "accuracy"),
        ("BRAKE recall %", "brake_recall"),
        ("BRAKE F1 %", "brake_f1"),
        ("Macro F1 %", "macro_f1"),
    ]

    for label, key in metrics:
        row = f"  {label:<23}"

        for result in results:
            row += f" {str(result[key]):>22}"

        print(row)

    print("=" * 70)
    print("\nBRAKE recall is the safety-critical metric.")
    print("Low BRAKE recall means the model misses emergency braking cases.\n")


def save_report(results, output_path):
    lines = []
    lines.append("TEMPORAL MODEL COMPARISON REPORT\n")

    for result in results:
        lines.append(f"Model: {result['model']}")
        lines.append(f"Overall accuracy %: {result['accuracy']}")
        lines.append(f"BRAKE recall %: {result['brake_recall']}")
        lines.append(f"BRAKE F1 %: {result['brake_f1']}")
        lines.append(f"Macro F1 %: {result['macro_f1']}")
        lines.append("")

    with open(output_path, "w") as file:
        file.write("\n".join(lines))

    print(f"Saved report to: {output_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="data/processed/platoon_data_enriched.csv",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--out_lstm",
        default="models/lstm_model.h5",
    )
    parser.add_argument(
        "--out_mlp",
        default="models/temporal_mlp_model.h5",
    )

    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    df = load_enriched(args.data)

    num_classes = len(np.unique(df["decision"].values))

    # ---------------------------
    # Temporal MLP
    # ---------------------------
    X_flat, y, feature_cols = build_flat_features(df)

    print(f"\nTemporal MLP feature count: {len(feature_cols)}")

    scaler_flat = StandardScaler()
    X_flat_scaled = scaler_flat.fit_transform(X_flat).astype(np.float32)

    class_weights = get_class_weights(y)

    X_train_flat, X_test_flat, y_train_flat, y_test_flat = train_test_split(
        X_flat_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    temporal_mlp = build_flat_mlp(
        input_dim=X_flat_scaled.shape[1],
        num_classes=num_classes,
    )

    result_mlp, trained_mlp = train_and_eval(
        temporal_mlp,
        X_train_flat,
        y_train_flat,
        X_test_flat,
        y_test_flat,
        class_weights,
        "Temporal MLP",
        args.epochs,
    )

    # ---------------------------
    # LSTM
    # ---------------------------
    X_seq, y_seq = build_sequence_features(df)

    scaler_seq = StandardScaler()

    X_seq_2d = X_seq.reshape(-1, len(SENSOR_COLS))
    X_seq_2d_scaled = scaler_seq.fit_transform(X_seq_2d).astype(np.float32)
    X_seq_scaled = X_seq_2d_scaled.reshape(-1, TIMESTEPS, len(SENSOR_COLS))

    X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(
        X_seq_scaled,
        y_seq,
        test_size=0.2,
        random_state=42,
        stratify=y_seq,
    )

    lstm_model = build_lstm(
        timesteps=TIMESTEPS,
        n_features=len(SENSOR_COLS),
        num_classes=num_classes,
    )

    result_lstm, trained_lstm = train_and_eval(
        lstm_model,
        X_train_seq,
        y_train_seq,
        X_test_seq,
        y_test_seq,
        class_weights,
        "LSTM",
        args.epochs,
    )

    results = [result_mlp, result_lstm]

    print_comparison_table(results)

    trained_mlp.save(args.out_mlp)
    trained_lstm.save(args.out_lstm)

    joblib.dump(scaler_flat, "models/scaler_temporal_mlp.joblib")
    joblib.dump(scaler_seq, "models/scaler_lstm.joblib")

    save_report(
        results,
        "outputs/reports/temporal_model_comparison.txt",
    )

    print("\nSaved files:")
    print(f"- {args.out_mlp}")
    print(f"- {args.out_lstm}")
    print("- models/scaler_temporal_mlp.joblib")
    print("- models/scaler_lstm.joblib")
    print("- outputs/reports/temporal_model_comparison.txt")


if __name__ == "__main__":
    main()