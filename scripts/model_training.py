# model_training.py
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# === Step 1: Load processed data ===
train_path = "../data/processed/train_clean.csv"
test_path = "../data/processed/test_clean.csv"

print("🔹 Loading processed datasets...")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print(f"✅ Train shape: {train_df.shape}")
print(f"✅ Test shape: {test_df.shape}")

# === Step 2: Split features and labels ===
X_train = train_df.drop(['label', 'attack_type'], axis=1)
y_train = train_df['attack_type']

X_test = test_df.drop(['label', 'attack_type'], axis=1)
y_test = test_df['attack_type']

# === Step 3: Train models ===
print("\n🌳 Training Decision Tree Classifier...")
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

print("🌲 Training Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
rf_model.fit(X_train, y_train)

# === Step 4: Evaluate models ===
models = {'Decision Tree': dt_model, 'Random Forest': rf_model}

results = {}
for name, model in models.items():
    print(f"\n🔍 Evaluating {name}...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"✅ Accuracy: {acc*100:.2f}%")
    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=['attack', 'normal'])
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Attack', 'Normal'], yticklabels=['Attack', 'Normal'])
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"../results/{name.replace(' ', '_')}_cm.png")
    plt.close()

# === Step 5: Compare model performance ===
print("\n📈 Model Comparison:")
for name, acc in results.items():
    print(f"{name}: {acc*100:.2f}%")

# === Step 6: Save models ===
os.makedirs("../models", exist_ok=True)
joblib.dump(dt_model, "../models/decision_tree_model.pkl")
joblib.dump(rf_model, "../models/random_forest_model.pkl")

print("\n💾 Models saved in ../models/")
print("📊 Confusion matrices saved in ../results/")
print("✅ Phase 3 completed successfully!")

