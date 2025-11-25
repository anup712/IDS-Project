import streamlit as st
import pandas as pd
import joblib
import numpy as np

MODEL_PATH = "models/random_forest_fixed_41.pkl"
SCALER_PATH = "models/scaler_fixed_41.pkl"

FEATURES = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent",
    "hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
    "count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
    "dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate"
]

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_model()

st.title("Web-based IDS Demo (NSL-KDD, 41 Features)")

vals = []
cols = st.columns(3)

for i, f in enumerate(FEATURES):
    with cols[i % 3]:
        v = st.number_input(f, value=0.0, format="%.6f")
        vals.append(v)

if st.button("Predict"):
    row = pd.DataFrame([vals], columns=FEATURES)
    Xs = scaler.transform(row.astype(float))
    pred = model.predict(Xs)[0]
    prob = model.predict_proba(Xs)[0].max()
    st.write(f"**Prediction:** `{pred}`  —  Confidence: `{prob:.2f}`")

