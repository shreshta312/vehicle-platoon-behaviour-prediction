import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import joblib
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
tf.get_logger().setLevel("ERROR")

print("Loading data, scaler, and the baseline vulnerable model...")

try:
    df = pd.read_csv("data/processed/platoon_data.csv")
    scaler = joblib.load("models/scaler.joblib")
    vulnerable_model = tf.keras.models.load_model("models/good_vulnerable_model.h5")
except FileNotFoundError as e:
    print(f"Error: Missing file: {e.filename}")
    exit()

df = df.dropna()

X_df = df.drop("decision", axis=1)
y_series = df["decision"]

num_classes = len(np.unique(y_series))

print(f"Dataset loaded successfully.")
print(f"Total samples: {len(df)}")
print(f"Number of classes: {num_classes}")
print(f"Features: {X_df.columns.tolist()}")

print("\nCalculating class weights...")

weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_series),
    y=y_series
)

class_weights_dict = dict(zip(np.unique(y_series), weights))
print(f"Class weights: {class_weights_dict}")

X_train_df, X_test_df, y_train, y_test = train_test_split(
    X_df,
    y_series,
    test_size=0.2,
    random_state=42,
    stratify=y_series
)

X_train = scaler.transform(X_train_df).astype(np.float32)
X_test = scaler.transform(X_test_df).astype(np.float32)

print("\nUsing FGSM adversarial training.")
epsilon = 0.01
num_attack_samples = min(2000, len(X_train))
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()


def generate_fgsm_attack(X_batch, y_batch):
    X_tensor = tf.Variable(X_batch, dtype=tf.float32)
    y_true_tensor = tf.convert_to_tensor(y_batch, dtype=tf.int64)

    with tf.GradientTape() as tape:
        tape.watch(X_tensor)
        prediction = vulnerable_model(X_tensor, training=False)
        loss = loss_object(y_true_tensor, prediction)

    gradient = tape.gradient(loss, X_tensor)
    perturbation_direction = tf.sign(gradient)

    X_attack = X_tensor + epsilon * perturbation_direction
    X_attack = tf.clip_by_value(X_attack, -3.0, 3.0)

    return X_attack.numpy()


print(f"Generating {num_attack_samples} FGSM adversarial samples...")

X_train_to_attack = X_train[:num_attack_samples]
y_train_to_attack = y_train.iloc[:num_attack_samples].to_numpy()

X_train_adversarial = generate_fgsm_attack(
    X_train_to_attack,
    y_train_to_attack
)

X_train_robust = np.concatenate([X_train, X_train_adversarial], axis=0)
y_train_robust = np.concatenate([y_train.to_numpy(), y_train_to_attack], axis=0)

indices = np.arange(len(X_train_robust))
np.random.shuffle(indices)

X_train_robust = X_train_robust[indices]
y_train_robust = y_train_robust[indices]

print("\nBuilding final safety-aware model...")

safe_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(num_classes, activation="softmax")
])

safe_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

print("\nTraining safety-aware model with EarlyStopping...")

safe_model.fit(
    X_train_robust,
    y_train_robust,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    class_weight=class_weights_dict,
    callbacks=[early_stopping_callback],
    verbose=1
)

print("\nEvaluating safety-aware model on clean test data...")

test_loss, test_accuracy = safe_model.evaluate(X_test, y_test, verbose=0)

print(f"\nFinal safety-aware model test accuracy: {test_accuracy * 100:.2f}%")

safe_model.save("models/FINAL_OPTIMIZED_MODEL.h5")

print("\n--- FINAL MODEL TRAINING COMPLETE ---")
print("Saved 'models/FINAL_OPTIMIZED_MODEL.h5'.")