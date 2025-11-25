# test_samples.py
import numpy as np
import joblib
import os

# Load model + scaler
model = joblib.load("../models/random_forest_model.pkl")
scaler = joblib.load("../models/scaler.pkl")

# Two sample feature vectors (must match your realtime feature order; 34 values)
normal_sample = np.array([
    0,491,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,
    0.0,0.0,0.0,0.0,1.0,0.0,0.0,255,5,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
]).reshape(1,-1)

attack_sample = np.array([
    0,99999,0,0,0,0,0,0,0,0,0,0,0,0,0,1000,1000,
    1.0,1.0,1.0,1.0,0.0,0.0,0.0,255,255,0.0,1.0,0.0,0.0,1.0,1.0,1.0,1.0
]).reshape(1,-1)

for name, s in [("Normal", normal_sample), ("Attack", attack_sample)]:
    s_scaled = scaler.transform(s)
    pred = model.predict(s_scaled)[0]
    # model might output strings ('attack'/'normal') or labels — handle both
    if isinstance(pred, str):
        label = pred
    else:
        # if model used numeric labels, try to infer: 0->normal else attack
        label = "normal" if pred == 0 else "attack"
    print(f"{name} sample -> model prediction: {label}")
