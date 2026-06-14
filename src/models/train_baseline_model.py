import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import joblib

print("Loading NEW platoon_data.csv...")
df = pd.read_csv('data/processed/platoon_data.csv')
df = df.dropna()

X = df.drop('decision', axis=1)
y = df['decision']
feature_names = X.columns.tolist()
num_classes = len(np.unique(y))

print(f"Features (X): {feature_names}")

print("\nCalculating class weights...")
weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y),
    y=y
)
class_weights_dict = dict(enumerate(weights))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nBuilding 'good' vulnerable model...")
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nTraining 'good' vulnerable model...")
model.fit(
    X_train_scaled,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test_scaled, y_test),
    class_weight=class_weights_dict,
    verbose=1
)

print("\nEvaluating 'good' model on test data...")
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")

model.save('models/good_vulnerable_model.h5')
joblib.dump(scaler, 'models/scaler.joblib')

print("\n--- 'GOOD' VULNERABLE MODEL CREATED ---")
print("Saved 'models/good_vulnerable_model.h5' and 'models/scaler.joblib'.")