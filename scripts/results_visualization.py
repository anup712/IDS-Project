# results_visualization.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import os

# === Step 1: Load processed data ===
train_df = pd.read_csv("../data/processed/train_clean.csv")
test_df = pd.read_csv("../data/processed/test_clean.csv")

X_test = test_df.drop(['label', 'attack_type'], axis=1)
y_test = test_df['attack_type']

# === Step 2: Load trained models ===
dt_model = joblib.load("../models/decision_tree_model.pkl")
rf_model = joblib.load("../models/random_forest_model.pkl")

# === Step 3: Evaluate both models ===
models = {
    "Decision Tree": dt_model,
    "Random Forest": rf_model
}

results = []

for name, model in models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append({"Model": name, "Accuracy": acc * 100})

results_df = pd.DataFrame(results)
print("📈 Model Accuracy Comparison:\n")
print(results_df)

# === Step 4: Plot accuracy comparison ===
plt.figure(figsize=(6,4))
sns.barplot(data=results_df, x="Model", y="Accuracy", palette="Blues_d")
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig("../results/model_accuracy_comparison.png")
plt.close()

# === Step 5: Show attack distribution ===
plt.figure(figsize=(5,4))
sns.countplot(x='attack_type', data=test_df, palette='coolwarm')
plt.title("Attack Type Distribution in Test Data")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("../results/attack_distribution.png")
plt.close()

print("\n✅ Charts saved successfully in ../results/")
print("   - model_accuracy_comparison.png")
print("   - attack_distribution.png")
print("\n✅ Phase 4 completed successfully!")

