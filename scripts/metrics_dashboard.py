import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

MODEL_PATH = "models/random_forest_fixed_41.pkl"
SCALER_PATH = "models/scaler_fixed_41.pkl"
DATA_PATH = "data/processed/train_clean.csv"

FEATURES = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent",
    "hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
    "count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
    "dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate"
]

print("Loading data + model...")
df = pd.read_csv(DATA_PATH)
X = df[FEATURES].astype(float)
y = df["label"]

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

Xs = scaler.transform(X)

print("Evaluating...")
y_pred = model.predict(Xs)
acc = accuracy_score(y, y_pred)
print(f"\nAccuracy on train_clean.csv: {acc:.4f}\n")

print("Classification report:")
print(classification_report(y, y_pred))

print("Plotting confusion matrix...")
cm = confusion_matrix(y, y_pred)
fig, ax = plt.subplots(figsize=(8,6))
im = ax.imshow(cm, interpolation='nearest')
ax.set_title("Confusion Matrix")
plt.colorbar(im)
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
plt.tight_layout()

print("Plotting feature importance...")
importances = model.feature_importances_
idx = np.argsort(importances)[::-1][:15]
fig2, ax2 = plt.subplots(figsize=(8,6))
ax2.barh(range(len(idx)), importances[idx][::-1])
ax2.set_yticks(range(len(idx)))
ax2.set_yticklabels([FEATURES[i] for i in idx][::-1])
ax2.set_title("Top 15 Feature Importances")
plt.tight_layout()

plt.show()

