"""
Full adversarial robustness evaluation.
Attacks N BRAKE samples, reports attack success rate for both
the vulnerable baseline and the safety-aware model side by side.

Usage:
    python adversarial_evaluation.py
    python adversarial_evaluation.py --epsilon 0.02 --n_samples 500
"""

import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import warnings
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)
tf.get_logger().setLevel("ERROR")

DECISION_BRAKE     = 0
DECISION_LANE_LEFT = 2   # most dangerous confusion: should brake, changes lane instead

LABEL_NAMES = {0: "BRAKE", 1: "FOLLOW", 2: "LANE_LEFT", 3: "LANE_RIGHT"}


def load_artifacts(data_path, baseline_path, safe_path, scaler_path):
    df            = pd.read_csv(data_path).dropna()
    baseline      = tf.keras.models.load_model(baseline_path)
    safe_model    = tf.keras.models.load_model(safe_path)
    scaler        = joblib.load(scaler_path)
    return df, baseline, safe_model, scaler


def get_brake_samples(df, scaler, n_samples, time_gap_thresh=1.5):
    """Pull genuine BRAKE samples under the time-gap threshold."""
    brake_df = df[
        (df["decision"] == DECISION_BRAKE) &
        (df["time_gap"] < time_gap_thresh)
    ]
    if len(brake_df) == 0:
        raise ValueError(
            f"No BRAKE samples found with time_gap < {time_gap_thresh}. "
            "Lower the threshold or check your dataset."
        )
    n = min(n_samples, len(brake_df))
    sample = brake_df.sample(n, random_state=42)
    X_raw  = sample.drop("decision", axis=1)
    X_sc   = scaler.transform(X_raw).astype(np.float32)
    y_true = sample["decision"].to_numpy()
    return X_sc, y_true, n


def fgsm_targeted_attack(model, X_batch, target_class, epsilon):
    """
    Targeted FGSM: nudge inputs toward making model predict target_class.
    For safety testing: target = LANE_LEFT (worst confusion for a BRAKE event).
    Returns perturbed X, clipped to [-3, 3] (StandardScaler range).
    """
    loss_fn  = tf.keras.losses.SparseCategoricalCrossentropy()
    X_tensor = tf.Variable(X_batch, dtype=tf.float32)
    targets  = tf.fill([len(X_batch)], target_class)

    with tf.GradientTape() as tape:
        tape.watch(X_tensor)
        preds = model(X_tensor, training=False)
        loss  = loss_fn(targets, preds)

    grads      = tape.gradient(loss, X_tensor)
    # Subtract gradient direction → move toward target class
    X_attacked = X_tensor - epsilon * tf.sign(grads)
    X_attacked = tf.clip_by_value(X_attacked, -3.0, 3.0)
    return X_attacked.numpy()


def evaluate_robustness(model, X_clean, X_attacked, y_true, model_name, epsilon):
    """
    For each sample:
      - Was the clean prediction correct?  (model must get it right first)
      - Did the attack flip it?            (attack success)
    """
    pred_clean   = np.argmax(model.predict(X_clean,   verbose=0), axis=1)
    pred_attacked= np.argmax(model.predict(X_attacked, verbose=0), axis=1)

    # Only count samples the model got right on clean input
    correct_mask      = (pred_clean == y_true)
    n_correct         = correct_mask.sum()

    # Attack flipped a correct prediction to something wrong
    flipped_mask      = correct_mask & (pred_attacked != y_true)
    n_flipped         = flipped_mask.sum()

    # Specifically flipped to the dangerous LANE_LEFT class
    dangerous_mask    = correct_mask & (pred_attacked == DECISION_LANE_LEFT)
    n_dangerous       = dangerous_mask.sum()

    attack_success    = n_flipped   / n_correct * 100 if n_correct > 0 else 0
    dangerous_rate    = n_dangerous / n_correct * 100 if n_correct > 0 else 0

    print(f"\n{'─'*50}")
    print(f"  Model : {model_name}")
    print(f"  ε     : {epsilon}")
    print(f"{'─'*50}")
    print(f"  Samples tested (clean correct) : {n_correct}")
    print(f"  Predictions flipped by attack  : {n_flipped}  ({attack_success:.1f}%)")
    print(f"  Flipped to LANE_LEFT (danger)  : {n_dangerous}  ({dangerous_rate:.1f}%)")

    # Confusion breakdown of attacked predictions
    print(f"\n  Post-attack prediction breakdown:")
    for cls, name in LABEL_NAMES.items():
        count = (pred_attacked[correct_mask] == cls).sum()
        print(f"    {name:15s}: {count}")

    return {
        "model":           model_name,
        "epsilon":         epsilon,
        "n_tested":        n_correct,
        "n_flipped":       n_flipped,
        "attack_success%": round(attack_success, 1),
        "n_dangerous":     n_dangerous,
        "dangerous_rate%": round(dangerous_rate, 1),
    }


def print_comparison_table(results):
    """The table that belongs in your README."""
    print(f"\n{'='*60}")
    print("  ROBUSTNESS COMPARISON TABLE")
    print(f"{'='*60}")
    header = f"{'Metric':<30} {'Baseline':>12} {'Safe Model':>12}"
    print(header)
    print("─" * 56)

    metrics = [
        ("ε (epsilon)",         "epsilon"),
        ("Samples tested",      "n_tested"),
        ("Attack success %",    "attack_success%"),
        ("Dangerous flip %",    "dangerous_rate%"),
    ]

    for label, key in metrics:
        vals = [str(r[key]) for r in results]
        print(f"  {label:<28} {vals[0]:>12} {vals[1]:>12}")

    print('='*60)

    improvement = results[0]["attack_success%"] - results[1]["attack_success%"]
    print(f"\n  Robustness improvement: {improvement:.1f} percentage points")
    print("  (lower attack success = more robust)\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",      default="data/processed/platoon_data.csv")
    parser.add_argument("--baseline",  default="models/good_vulnerable_model.h5")
    parser.add_argument("--safe",      default="models/FINAL_OPTIMIZED_MODEL.h5")
    parser.add_argument("--scaler",    default="models/scaler.joblib")
    parser.add_argument("--epsilon",   type=float, default=0.01)
    parser.add_argument("--n_samples", type=int,   default=300)
    parser.add_argument("--out",       default="outputs/reports/robustness_report.txt")
    args = parser.parse_args()

    print("Loading models, scaler, and data...")
    df, baseline, safe_model, scaler = load_artifacts(
        args.data, args.baseline, args.safe, args.scaler
    )

    print(f"\nSampling {args.n_samples} BRAKE events (time_gap < 1.5s)...")
    X_clean, y_true, n = get_brake_samples(df, scaler, args.n_samples)
    print(f"Found {n} qualifying BRAKE samples.")

    # Generate one set of adversarial examples per model
    # (attack is model-specific — uses that model's gradients)
    print(f"\nGenerating adversarial examples (ε={args.epsilon})...")
    X_attacked_baseline = fgsm_targeted_attack(
        baseline, X_clean, DECISION_LANE_LEFT, args.epsilon
    )
    X_attacked_safe = fgsm_targeted_attack(
        safe_model, X_clean, DECISION_LANE_LEFT, args.epsilon
    )

    print("\n" + "="*50)
    print("  ADVERSARIAL ROBUSTNESS EVALUATION")
    print("="*50)

    results = []
    results.append(evaluate_robustness(
        baseline, X_clean, X_attacked_baseline, y_true,
        "Baseline (vulnerable)", args.epsilon
    ))
    results.append(evaluate_robustness(
        safe_model, X_clean, X_attacked_safe, y_true,
        "Safe model (FGSM-trained)", args.epsilon
    ))

    print_comparison_table(results)

    # Save report
    import io, sys
    report_lines = []
    report_lines.append("ADVERSARIAL ROBUSTNESS REPORT\n")
    for r in results:
        for k, v in r.items():
            report_lines.append(f"{k}: {v}")
        report_lines.append("")

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(report_lines))
    print(f"Report saved to: {args.out}")


if __name__ == "__main__":
    main()