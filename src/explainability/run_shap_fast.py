"""
Fast SHAP explanation for the safety-aware model.

Generates:
  - outputs/plots/shap_bar_global.png
  - outputs/plots/shap_beeswarm.png
  - outputs/plots/shap_force_brake.png

Usage:
    python src/explainability/run_shap_fast.py
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

DECISION_BRAKE = 0

FEATURE_NAMES = [
    "ego_vel",
    "front_dist",
    "front_rel_vel",
    "time_gap",
    "left_front_dist",
    "left_rear_dist",
    "right_front_dist",
    "right_rear_dist",
]

LABEL_NAMES = ["BRAKE", "FOLLOW", "LANE_LEFT", "LANE_RIGHT"]


def load_artifacts():
    model = tf.keras.models.load_model("models/FINAL_OPTIMIZED_MODEL.h5")
    scaler = joblib.load("models/scaler.joblib")
    df = pd.read_csv("data/processed/platoon_data.csv").dropna()
    return model, scaler, df


def normalize_shap_values(shap_values, num_classes):
    """
    Converts SHAP output into a list:
    [
        class_0_values,
        class_1_values,
        class_2_values,
        class_3_values
    ]

    Each item should have shape:
    (n_samples, n_features)
    """
    if isinstance(shap_values, list):
        normalized = []

        for class_values in shap_values:
            class_values = np.array(class_values)

            if class_values.ndim == 3:
                class_values = class_values[:, :, 0]

            normalized.append(class_values)

        return normalized

    shap_values = np.array(shap_values)

    if shap_values.ndim == 3:
        return [
            shap_values[:, :, class_idx]
            for class_idx in range(num_classes)
        ]

    raise ValueError(f"Unexpected SHAP values shape: {shap_values.shape}")


def main():
    os.makedirs("outputs/plots", exist_ok=True)

    print("Loading model, scaler, and data...")
    model, scaler, df = load_artifacts()

    num_classes = len(np.unique(df["decision"]))

    background_df = df.groupby("decision", group_keys=False).apply(
        lambda group: group.sample(min(50, len(group)), random_state=42)
    )

    explain_df = df.groupby("decision", group_keys=False).apply(
        lambda group: group.sample(min(75, len(group)), random_state=0)
    )

    X_background = scaler.transform(
        background_df.drop("decision", axis=1)
    ).astype(np.float32)

    X_explain = scaler.transform(
        explain_df.drop("decision", axis=1)
    ).astype(np.float32)

    y_explain = explain_df["decision"].values

    print("Running SHAP explainer...")

    try:
        print("Trying DeepExplainer...")
        explainer = shap.DeepExplainer(model, X_background)
        shap_values_raw = explainer.shap_values(X_explain)
        expected_values = explainer.expected_value
        print("DeepExplainer worked.")
    except Exception as error:
        print("DeepExplainer failed. Using GradientExplainer instead.")
        print(f"Reason: {error}")

        explainer = shap.GradientExplainer(model, X_background)
        shap_values_raw = explainer.shap_values(X_explain)

        expected_values = model.predict(X_background, verbose=0).mean(axis=0)

    shap_values = normalize_shap_values(shap_values_raw, num_classes)

    print("Generating SHAP plots...")

    # -----------------------------
    # 1. Global bar plot
    # -----------------------------
    mean_abs_per_class = np.array([
        np.abs(class_values)
        for class_values in shap_values
    ])

    global_importance = mean_abs_per_class.mean(axis=(0, 1))
    sorted_idx = np.argsort(global_importance)

    plt.figure(figsize=(8, 5))
    plt.barh(
        [FEATURE_NAMES[i] for i in sorted_idx],
        global_importance[sorted_idx],
        color="#5C6BC0",
    )
    plt.xlabel("Mean absolute SHAP value")
    plt.title("Global Feature Importance")
    plt.tight_layout()
    plt.savefig("outputs/plots/shap_bar_global.png", dpi=150)
    plt.close()

    print("Saved: outputs/plots/shap_bar_global.png")

    # -----------------------------
    # 2. Beeswarm plot for BRAKE
    # -----------------------------
    shap_brake = np.array(shap_values[DECISION_BRAKE])

    shap.summary_plot(
        shap_brake,
        X_explain,
        feature_names=FEATURE_NAMES,
        show=False,
        plot_type="dot",
        max_display=8,
    )

    plt.title("SHAP Beeswarm - BRAKE Class")
    plt.tight_layout()
    plt.savefig("outputs/plots/shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved: outputs/plots/shap_beeswarm.png")

    # -----------------------------
    # 3. Force plot for one BRAKE sample
    # -----------------------------
    brake_indices = np.where(y_explain == DECISION_BRAKE)[0]

    if len(brake_indices) > 0:
        brake_idx = brake_indices[0]

        base_value = float(
            np.array(expected_values[DECISION_BRAKE]).reshape(-1)[0]
        )

        sample_shap = np.array(
            shap_values[DECISION_BRAKE][brake_idx]
        ).reshape(-1)

        sample_features = np.array(
            X_explain[brake_idx]
        ).reshape(-1)

        shap.force_plot(
            base_value,
            sample_shap,
            sample_features,
            feature_names=FEATURE_NAMES,
            matplotlib=True,
            show=False,
        )

        plt.title("SHAP Force Plot - Single BRAKE Decision")
        plt.savefig(
            "outputs/plots/shap_force_brake.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        print("Saved: outputs/plots/shap_force_brake.png")
    else:
        print("No BRAKE sample found. Force plot skipped.")

    # -----------------------------
    # Console feature ranking
    # -----------------------------
    print("\nTop features driving BRAKE decisions:")

    brake_importance = np.abs(shap_values[DECISION_BRAKE]).mean(axis=0)
    max_value = max(brake_importance)

    ranked_features = sorted(
        zip(FEATURE_NAMES, brake_importance),
        key=lambda item: -item[1],
    )

    for feature_name, value in ranked_features:
        bar_length = int((value / max_value) * 20) if max_value > 0 else 0
        bar = "#" * bar_length
        print(f"{feature_name:<20} {bar:<20} {value:.4f}")

    print("\nSHAP analysis complete.")


if __name__ == "__main__":
    main()