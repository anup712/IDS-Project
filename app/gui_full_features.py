import tkinter as tk
from tkinter import ttk
import joblib
import numpy as np
import pandas as pd

# ----------------------
# Load Model + Scaler
# ----------------------
model = joblib.load("../models/random_forest_improved.pkl")
scaler = joblib.load("../models/scaler_improved.pkl")

# ----------------------
# EXACT Feature List (42 features)
# ----------------------
feature_names = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate"
]

# ----------------------
# GUI Window
# ----------------------
root = tk.Tk()
root.title("IDS Prediction Dashboard (42 Features)")
root.geometry("1050x850")
root.configure(bg="#1e1e1e")

# Apply dark theme to ttk widgets
style = ttk.Style()
style.theme_use("clam")
style.configure("TLabel", background="#1e1e1e", foreground="white")
style.configure("TEntry", fieldbackground="#2c2c2c", foreground="white")
style.configure("TFrame", background="#1e1e1e")

entries = {}

canvas = tk.Canvas(root, bg="#1e1e1e", highlightthickness=0)
scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)

scrollable_frame = ttk.Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas_frame = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

# ----------------------
# Create All Input Fields
# ----------------------
for idx, feature in enumerate(feature_names):
    label = ttk.Label(scrollable_frame, text=feature)
    label.grid(row=idx, column=0, padx=10, pady=5, sticky="w")

    entry = ttk.Entry(scrollable_frame, width=25)
    entry.grid(row=idx, column=1, padx=10, pady=5)
    entries[feature] = entry

# ----------------------
# Prediction Function
# ----------------------
def predict():
    try:
        input_values = []

        for feature in feature_names:
            value = entries[feature].get().strip()
            if value == "":
                value = 0  # default
            input_values.append(float(value))

        # Convert to dataframe with correct columns
        df = pd.DataFrame([input_values], columns=feature_names)

        # Scale + Predict
        scaled = scaler.transform(df)
        pred = model.predict(scaled)[0]

        if pred == 1:
            result_label.config(text="🚨 ATTACK DETECTED 🚨", fg="red")
        else:
            result_label.config(text="✔ NORMAL TRAFFIC", fg="green")

    except Exception as e:
        result_label.config(text=f"Error: {e}", fg="yellow")

# ----------------------
# Predict Button
# ----------------------
predict_btn = tk.Button(
    root,
    text="PREDICT",
    command=predict,
    bg="#007acc",
    fg="white",
    font=("Arial", 16, "bold"),
    padx=20,
    pady=10
)
predict_btn.pack(pady=10)

# ----------------------
# Result Label
# ----------------------
result_label = tk.Label(
    root,
    text="Enter all feature values and click Predict",
    bg="#1e1e1e",
    fg="white",
    font=("Arial", 18, "bold")
)
result_label.pack(pady=10)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

root.mainloop()

