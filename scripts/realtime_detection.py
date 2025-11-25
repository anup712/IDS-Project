# realtime_detection.py
import joblib
import numpy as np

print("\n🔹 Loading model and scaler...")
model = joblib.load("../models/random_forest_model.pkl")
scaler = joblib.load("../models/scaler.pkl")
print("✅ Model and scaler loaded successfully!\n")

# Expected number of features
EXPECTED_FEATURES = scaler.n_features_in_

def pad_features(features, target_len=EXPECTED_FEATURES):
    """Ensure user input has the correct number of features."""
    if len(features) < target_len:
        features = np.pad(features, (0, target_len - len(features)))
    elif len(features) > target_len:
        features = features[:target_len]
    return features

print("🧠 Enter sample network feature values to simulate detection:")
print("(For quick testing, just press Enter to use default values.)\n")

# Example default feature values (simplified)
default_features = [
    0, 491, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2,
    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 255, 5, 1.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0
]

# Ask user input (optional)
user_input = []
try:
    for i, val in enumerate(default_features[:10]):  # only ask first 10 for speed
        x = input(f"→ feature_{i+1} (default {val}): ")
        if x.strip() == "":
            user_input.append(val)
        else:
            user_input.append(float(x))
except KeyboardInterrupt:
    print("\n❌ Interrupted by user.")

# Fill remaining with defaults
if len(user_input) < len(default_features):
    user_input += default_features[len(user_input):]

# Pad up to correct number of features (41)
user_input = pad_features(np.array(user_input))

# Scale + Predict
scaled = scaler.transform(user_input.reshape(1, -1))
prediction = model.predict(scaled)[0]

# Display result
print("\n🔍 IDS Prediction Result:")
if prediction == 0:
    print("✅ Normal Traffic — No intrusion detected.")
else:
    print("🚨 ALERT! Possible Attack detected.")


