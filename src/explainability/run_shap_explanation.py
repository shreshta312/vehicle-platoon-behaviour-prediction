import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import shap
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
tf.get_logger().setLevel("ERROR")

DECISION_BRAKE = 0

print("Loading safety-aware model, scaler, and dataset...")

try:
    model = tf.keras.models.load_model("models/FINAL_OPTIMIZED_MODEL.h5")
    scaler = joblib.load("models/scaler.joblib")
    df = pd.read_csv("data/processed/platoon_data.csv")
except FileNotFoundError as e:
    print(f"Error: Missing file: {e.filename}")
    exit()

print("Loaded model, scaler, and data successfully.")

df = df.dropna()

# Pick safety-critical BRAKE samples
victim_df = df[(df["decision"] == DECISION_BRAKE) & (df["time_gap"] < 1.4)]

if victim_df.empty:
    print("Error: Could not find a suitable BRAKE sample in the data.")
    exit()

# Use only one victim sample for force plot
X_victim_unscaled_df = victim_df.drop("decision", axis=1).iloc[0:1]
X_victim_scaled = scaler.transform(X_victim_unscaled_df).astype(np.float32)

# Background samples for SHAP
background_df = df.drop("decision", axis=1).sample(100, random_state=42)
background_scaled = scaler.transform(background_df).astype(np.float32)

print("Created victim and background data for SHAP.")

print("\nRunning SHAP analysis...")
print("This may take some time.")

explainer = shap.KernelExplainer(model.predict, background_scaled)
shap_values = explainer.shap_values(X_victim_scaled)

print("SHAP analysis complete.")

# SHAP output shape can differ depending on SHAP version.
# For multi-class model, we need SHAP values for class 0 = BRAKE.
if isinstance(shap_values, list):
    shap_values_sample = shap_values[DECISION_BRAKE][0]
    expected_value = explainer.expected_value[DECISION_BRAKE]
else:
    shap_values_sample = shap_values[0, :, DECISION_BRAKE]
    expected_value = explainer.expected_value[DECISION_BRAKE]

features_sample = X_victim_unscaled_df.iloc[0]

print("\nGenerating SHAP force plot...")

shap.force_plot(
    expected_value,
    shap_values_sample,
    features_sample,
    matplotlib=True,
    show=False
)

plt.title("SHAP Explanation for BRAKE Decision")
plt.savefig("outputs/plots/shap_force_plot.png", bbox_inches="tight")
plt.close()

print("\n--- SHAP EXPLANATION COMPLETE ---")
print("Saved plot to: outputs/plots/shap_force_plot.png")
print("This plot explains which features pushed the model toward the BRAKE decision.")