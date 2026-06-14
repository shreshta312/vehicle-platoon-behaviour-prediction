import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

print("Loading model, scaler, and dataset...")

df = pd.read_csv("data/processed/platoon_data.csv")
model = tf.keras.models.load_model("models/FINAL_OPTIMIZED_MODEL.h5")
scaler = joblib.load("models/scaler.joblib")

df = df.dropna()

X = df.drop("decision", axis=1)
y = df["decision"]

decision_names = ["BRAKE", "FOLLOW", "CHANGE_LANE_LEFT", "CHANGE_LANE_RIGHT"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_test_scaled = scaler.transform(X_test)

print("Running predictions...")

y_pred_probs = model.predict(X_test_scaled, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

print("Generating classification report...")

report = classification_report(
    y_test,
    y_pred,
    target_names=decision_names
)

print(report)

with open("outputs/reports/classification_report.txt", "w") as f:
    f.write(report)

print("Saved classification report to outputs/reports/classification_report.txt")

print("Generating confusion matrix...")

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=decision_names
)

disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix - Safety-Aware Model")
plt.tight_layout()
plt.savefig("outputs/plots/confusion_matrix.png")
plt.close()

print("Saved confusion matrix to outputs/plots/confusion_matrix.png")

print("Generating class distribution plot...")

class_counts = y.value_counts().sort_index()

plt.figure(figsize=(8, 5))
plt.bar(decision_names, class_counts.values)
plt.title("Class Distribution in Platoon Dataset")
plt.xlabel("Decision Class")
plt.ylabel("Number of Samples")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("outputs/plots/class_distribution.png")
plt.close()

print("Saved class distribution plot to outputs/plots/class_distribution.png")

print("\n--- EVALUATION COMPLETE ---")