# data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# === Step 1: Load Dataset ===
train_path = "../data/raw/KDDTrain+.txt"
test_path = "../data/raw/KDDTest+.txt"

# The dataset has 41 features + 1 label
col_names = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes',
    'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
    'num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login',
    'count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate',
    'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate','label'
]

print("🔹 Loading NSL-KDD dataset...")
train_df = pd.read_csv(train_path, names=col_names)
test_df = pd.read_csv(test_path, names=col_names)

print(f"✅ Train shape: {train_df.shape}")
print(f"✅ Test shape: {test_df.shape}")

# === Step 2: Simplify label classes (Normal vs Attack Types) ===
train_df['attack_type'] = train_df['label'].apply(lambda x: 'normal' if x == 'normal' else 'attack')
test_df['attack_type'] = test_df['label'].apply(lambda x: 'normal' if x == 'normal' else 'attack')

# === Step 3: Encode categorical columns safely ===
cat_cols = ['protocol_type', 'service', 'flag']

for col in cat_cols:
    enc = LabelEncoder()
    enc.fit(list(train_df[col].values))

    mapping = dict(zip(enc.classes_, enc.transform(enc.classes_)))
    train_df[col] = train_df[col].map(mapping)
    test_df[col] = test_df[col].map(mapping)
    test_df[col] = test_df[col].fillna(-1)

# === Step 4: Ensure all feature columns are numeric ===
num_cols = train_df.drop(['label', 'attack_type'], axis=1).columns
train_df[num_cols] = train_df[num_cols].apply(pd.to_numeric, errors='coerce')
test_df[num_cols] = test_df[num_cols].apply(pd.to_numeric, errors='coerce')

# Fill any missing values (if conversion created NaN)
train_df[num_cols] = train_df[num_cols].fillna(0)
test_df[num_cols] = test_df[num_cols].fillna(0)

# === Step 5: Normalize numerical features ===
scaler = MinMaxScaler()
train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
test_df[num_cols] = scaler.transform(test_df[num_cols])

# === Step 6: Save preprocessed data ===
train_df.to_csv("../data/processed/train_clean.csv", index=False)
test_df.to_csv("../data/processed/test_clean.csv", index=False)

print("💾 Preprocessed data saved in: ../data/processed/")
print("✅ Phase 2 completed successfully!")


