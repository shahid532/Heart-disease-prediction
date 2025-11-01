# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# ------------------ LOAD DATA ------------------
data = pd.read_csv("dataset/heart_disease_uci.csv")

# Drop unnecessary columns
data = data.drop(columns=["id", "dataset"])

# Map categorical to numeric
data["sex"] = data["sex"].map({"Male": 1, "Female": 0})
data["cp"] = data["cp"].map({
    "typical angina": 0,
    "atypical angina": 1,
    "non-anginal": 2,
    "asymptomatic": 3
})
data["fbs"] = data["fbs"].map({True: 1, False: 0})
data["restecg"] = data["restecg"].map({
    "normal": 0,
    "lv hypertrophy": 1,
    "ST-T abnormality": 2
})
data["exang"] = data["exang"].map({True: 1, False: 0})
data["slope"] = data["slope"].map({
    "upsloping": 0,
    "flat": 1,
    "downsloping": 2
})
data["thal"] = data["thal"].map({
    "normal": 0,
    "fixed defect": 1,
    "reversable defect": 2
})

# ------------------ FEATURES & TARGET ------------------
X = data.drop("num", axis=1)
y = data["num"]

# ------------------ TRAIN / TEST SPLIT ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------ SCALING ------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------ MODEL TRAINING ------------------
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train_scaled, y_train)

# ------------------ EVALUATION ------------------
y_pred = model.predict(X_test_scaled)
print("✅ Model trained successfully!")
print(f"🎯 Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ------------------ SAVE MODEL ------------------
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/heart_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
print("💾 Model and scaler saved inside /model folder!")
