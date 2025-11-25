# save_scaler.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

os.makedirs("../models", exist_ok=True)

# Load processed train set
train_df = pd.read_csv("../data/processed/train_clean.csv")

# Features are all columns except 'label' and 'attack_type'
feature_cols = train_df.drop(['label', 'attack_type'], axis=1).columns

scaler = MinMaxScaler()
scaler.fit(train_df[feature_cols])

# Save scaler
joblib.dump(scaler, "../models/scaler.pkl")
print("✅ Scaler saved to ../models/scaler.pkl")

