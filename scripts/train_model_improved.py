import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ================================
# 1. LOAD DATA
# ================================
df_train = pd.read_csv("../data/processed/train_clean.csv")
df_test = pd.read_csv("../data/processed/test_clean.csv")

df = pd.concat([df_train, df_test], axis=0)

# ================================
# 2. SELECT BEST 22 FEATURES
# ================================
best_features = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes",
    "wrong_fragment", "urgent",
    "hot", "num_failed_logins",
    "logged_in", "num_compromised",
    "count", "srv_count",
    "serror_rate", "rerror_rate",
    "same_srv_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_serror_rate", "dst_host_rerror_rate"
]

X = df[best_features]
y = df["label"]

# ================================
# 3. ENCODE CATEGORICAL FEATURES
# ================================
cat_cols = ["protocol_type", "service", "flag"]
le = LabelEncoder()

for col in cat_cols:
    X[col] = le.fit_transform(X[col])

# Save encoder
joblib.dump(le, "../models/label_encoder.pkl")

# ================================
# 4. SCALE DATA
# ================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "../models/scaler_improved.pkl")

# ================================
# 5. TRAIN / TEST SPLIT
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ================================
# 6. TUNED RANDOM FOREST MODEL
# ================================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=25,
    min_samples_split=3,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42
)

model.fit(X_train, y_train)

# ================================
# 7. EVALUATION
# ================================
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ================================
# 8. SAVE MODEL
# ================================
joblib.dump(model, "../models/random_forest_improved.pkl")

print("\n✅ Improved Model Saved Successfully!")

