import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
tf.get_logger().setLevel("ERROR")

DECISION_BRAKE = 0
DECISION_LANE_LEFT = 2

decision_map = {
    0: "BRAKE (Safe)",
    1: "FOLLOW",
    2: "CHANGE_LANE_LEFT (Unsafe Target)",
    3: "CHANGE_LANE_RIGHT"
}

print("Loading safety-aware model, scaler, and dataset...")

try:
    safe_model = tf.keras.models.load_model("models/FINAL_OPTIMIZED_MODEL.h5")
    scaler = joblib.load("models/scaler.joblib")
    df = pd.read_csv("data/processed/platoon_data.csv")
except FileNotFoundError as e:
    print(f"Error: Missing file: {e.filename}")
    exit()

print("Loaded model, scaler, and data successfully.")

df = df.dropna()

# Pick a safety-critical BRAKE sample
victim_df = df[(df["decision"] == DECISION_BRAKE) & (df["time_gap"] < 1.4)]

if victim_df.empty:
    print("Error: Could not find a suitable BRAKE sample in the data.")
    exit()

X_victim_unscaled = victim_df.drop("decision", axis=1).iloc[0:1]
X_victim_scaled = scaler.transform(X_victim_unscaled).astype(np.float32)

# Original prediction
pred_victim = safe_model.predict(X_victim_scaled, verbose=0)
pred_victim_class = int(np.argmax(pred_victim))

print("\n--- 1. PRE-ATTACK VERIFICATION ---")
print(f"True Label: BRAKE")
print(
    f"Model Prediction: {decision_map[pred_victim_class]} "
    f"(Confidence: {np.max(pred_victim):.2%})"
)

print("\nLaunching FGSM adversarial attack...")

epsilon = 0.01
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

X_victim_tensor = tf.Variable(X_victim_scaled, dtype=tf.float32)

with tf.GradientTape() as tape:
    tape.watch(X_victim_tensor)
    prediction = safe_model(X_victim_tensor, training=False)

    # We try to push the model toward an unsafe lane-left prediction
    malicious_target = tf.convert_to_tensor([DECISION_LANE_LEFT], dtype=tf.int64)
    loss = loss_object(malicious_target, prediction)

gradient = tape.gradient(loss, X_victim_tensor)
perturbation_direction = tf.sign(gradient)

# Targeted FGSM: subtract gradient direction to move toward malicious target
X_attack_scaled = X_victim_tensor - epsilon * perturbation_direction
X_attack_scaled = tf.clip_by_value(X_attack_scaled, -3.0, 3.0)

# Prediction after attack
pred_attack = safe_model.predict(X_attack_scaled, verbose=0)
pred_attack_class = int(np.argmax(pred_attack))

print("\n--- 2. POST-ATTACK RESULT ---")
print(f"Original Prediction: {decision_map[pred_victim_class]}")
print(
    f"Attacked Prediction: {decision_map[pred_attack_class]} "
    f"(Confidence: {np.max(pred_attack):.2%})"
)

print("\n--- 3. FINAL ANALYSIS ---")

if pred_attack_class == pred_victim_class:
    print("SUCCESS: The attack failed.")
    print("The safety-aware model ignored the adversarial perturbation.")
    print("Result: Model appears robust for this tested safety-critical sample.")
else:
    print("WARNING: The attack changed the model prediction.")
    print("The model may still be vulnerable under adversarial perturbation.")
    print("Result: More robustness testing is needed.")