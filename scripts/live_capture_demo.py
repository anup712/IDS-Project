from scapy.all import sniff, IP, TCP, UDP, ICMP
import pandas as pd
import joblib
import numpy as np
from datetime import datetime

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

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def packet_to_row(pkt):
    # VERY rough mapping just for demo
    proto_val = 0.0
    if TCP in pkt: proto_val = 0.5
    elif UDP in pkt: proto_val = 1.0
    elif ICMP in pkt: proto_val = 1.5

    length = len(pkt)
    src_bytes = length * 0.6
    dst_bytes = length * 0.4

    row = {f: 0.0 for f in FEATURES}
    row["duration"] = 0.0
    row["protocol_type"] = proto_val
    row["service"] = 0.5
    row["flag"] = 0.5
    row["src_bytes"] = src_bytes
    row["dst_bytes"] = dst_bytes
    row["count"] = 1.0
    row["srv_count"] = 1.0
    return pd.DataFrame([row], columns=FEATURES)

def handle_packet(pkt):
    try:
        row = packet_to_row(pkt)
        Xs = scaler.transform(row.astype(float))
        pred = model.predict(Xs)[0]
        prob = model.predict_proba(Xs)[0].max()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Packet -> label {pred}, conf={prob:.2f}")
    except Exception as e:
        print("Error:", e)

print("Sniffing 50 packets (needs sudo)...")
sniff(count=50, prn=handle_packet)

