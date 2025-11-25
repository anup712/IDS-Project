from flask import Flask, render_template, request, jsonify, send_file
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt

BASE = os.path.dirname(__file__)
MODEL = joblib.load(os.path.join(BASE, "models/random_forest_fixed_41.pkl"))
SCALER = joblib.load(os.path.join(BASE, "models/scaler_fixed_41.pkl"))

FEATURES = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment",
    "urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted",
    "num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
    "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate"
]

LOG_PATH = os.path.join(BASE, "web_logs.csv")

app = Flask(__name__)

# ----------------------------
# Mapping categorical values
# ----------------------------
def map_categoricals(df):
    proto_map = {"tcp":0, "udp":1, "icmp":2}
    flag_map = {"SF":0, "S0":1, "REJ":2}
    service_map = {"http":0, "ftp":1, "smtp":2, "other":3}

    df["protocol_type"] = df["protocol_type"].map(proto_map).fillna(0)
    df["flag"] = df["flag"].map(flag_map).fillna(0)
    df["service"] = df["service"].map(service_map).fillna(0)
    return df.astype(float)


# ----------------------------
# Home Page
# ----------------------------
@app.route("/")
def index():
    return render_template("index.html", features=FEATURES)


# ----------------------------
# Predict Route (POST)
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = {}

    for f in FEATURES:
        v = request.form.get(f, "0")
        data[f] = v

    df = pd.DataFrame([data], columns=FEATURES)
    df = map_categoricals(df)
    scaled = SCALER.transform(df)

    pred = MODEL.predict(scaled)[0]
    prob = MODEL.predict_proba(scaled)[0].max()

    # log
    df["prediction"] = pred
    df["confidence"] = round(prob * 100, 2)
    df["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df.to_csv(LOG_PATH, mode="a", header=not os.path.exists(LOG_PATH), index=False)

    return jsonify({
        "prediction": int(pred),
        "confidence": float(round(prob * 100, 2))
    })


# ----------------------------
# Plot Logs
# ----------------------------
@app.route("/plot")
def plot():
    if not os.path.exists(LOG_PATH):
        plt.figure()
        plt.text(0.5,0.5,"No logs yet", ha='center')
    else:
        logs = pd.read_csv(LOG_PATH).tail(100)
        plt.figure(figsize=(6,3))
        plt.plot(logs["prediction"])
        plt.title("Recent Predictions")
        plt.ylabel("Label")
        plt.xlabel("Entry")

    img_path = os.path.join(BASE, "static/plot.png")
    plt.savefig(img_path, dpi=100, bbox_inches='tight')
    plt.close()
    return send_file(img_path, mimetype='image/png')


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,         # turn off debug mode
        use_reloader=False   # fix TTY crash
    )

