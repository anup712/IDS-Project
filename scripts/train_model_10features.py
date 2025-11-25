import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Load dataset
train = pd.read_csv("../data/processed/train_clean.csv")
test = pd.read_csv("../data/processed/test_clean.csv")
df = pd.concat([train, test], ignore_index=True)
# Select only the 10 numeric features used in your GUI

selected_features = [
    'duration',
    'src_bytes',
    'dst_bytes',
    'wrong_fragment',
    'urgent',
    'count',
    'srv_count',
    'serror_rate',
    'rerror_rate',
    'same_srv_rate'
]

# Ensure label column exists
if 'label' not in df.columns:
    df.rename(columns={'class': 'label'}, inplace=True)

# Encode labels if necessary
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

# Prepare features and target
X = df[selected_features]
y = df['label']

# Normalize numeric features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train RandomForest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate
y_pred = rf.predict(X_test)
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model and scaler
os.makedirs("../models", exist_ok=True)
joblib.dump(rf, "../models/random_forest_model_10.pkl")
joblib.dump(scaler, "../models/scaler_10.pkl")

print("\n✅ Model and Scaler saved successfully!")
