# ids_tk_app.py
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE, "models", "random_forest_improved.pkl")
SCALER_PATH = os.path.join(BASE, "models", "scaler_improved.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE, "models", "label_encoder.pkl")
TRAIN_CSV = os.path.join(BASE, "data", "processed", "train_clean.csv")
LOG_PATH = os.path.join(BASE, "results", "realtime_log.csv")

# Load model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = None
if os.path.exists(LABEL_ENCODER_PATH):
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

# Define full feature list (from your train header)
feature_names = [
"duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent",
"hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root",
"num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
"count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
"diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
"dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
"dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate"
]

# Load categorical options from CSV (protocol, service, flag) if available
protocol_options = ["tcp","udp","icmp"]
service_options = ["http","smtp","ftp","domain_u","other"]
flag_options = ["SF","S0","REJ","RSTR","RSTO"]

if os.path.exists(TRAIN_CSV):
    df = pd.read_csv(TRAIN_CSV)
    if "protocol_type" in df.columns:
        protocol_options = sorted(df["protocol_type"].dropna().unique().tolist())
    if "service" in df.columns:
        service_options = sorted(df["service"].dropna().unique().tolist())
    if "flag" in df.columns:
        flag_options = sorted(df["flag"].dropna().unique().tolist())

# Preset examples (map feature_name -> value; leave missing keys as 0)
PRESETS = {
    "Normal": {
        "duration": 0.3, "src_bytes": 200, "dst_bytes": 250, "count": 5, "srv_count": 5,
        "serror_rate": 0.0, "rerror_rate": 0.01, "same_srv_rate": 0.9, "protocol_type":"tcp", "service":"http", "flag":"SF"
    },
    "DoS_SYN": {
        "duration": 0.1, "src_bytes":20, "dst_bytes":0, "count":300, "srv_count":250,
        "serror_rate":0.95, "rerror_rate":0.02, "same_srv_rate":1.0, "protocol_type":"tcp", "service":"other", "flag":"S0"
    },
    "Probe": {
        "duration":0.5,"src_bytes":150,"dst_bytes":30,"count":50,"srv_count":1,
        "serror_rate":0.01,"rerror_rate":0.8,"same_srv_rate":0.02,"protocol_type":"tcp","service":"http","flag":"SF"
    },
    "R2L": {
        "duration":5,"src_bytes":100,"dst_bytes":120,"count":20,"srv_count":10,
        "serror_rate":0.0,"rerror_rate":0.10,"same_srv_rate":0.4,"protocol_type":"tcp","service":"ftp","flag":"REJ"
    },
    "U2R": {
        "duration":2,"src_bytes":50,"dst_bytes":5,"count":5,"srv_count":5,
        "serror_rate":0.05,"rerror_rate":0.40,"same_srv_rate":0.8,"protocol_type":"tcp","service":"other","flag":"SF"
    }
}

# GUI
root = tk.Tk()
root.title("IDS Desktop Dashboard")
root.geometry("1100x800")
root.configure(bg="#0f1720")

mainframe = ttk.Frame(root, padding=10)
mainframe.pack(fill="both", expand=True)

# Left frame: inputs
left = ttk.Frame(mainframe)
left.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

# Right frame: charts + log
right = ttk.Frame(mainframe)
right.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)

# input widgets dict
widgets = {}

# Create inputs with labels. For protocol/service/flag use Combobox
for i, fname in enumerate(feature_names):
    r = i
    label = ttk.Label(left, text=fname, width=20)
    label.grid(row=r, column=0, sticky="w", pady=2)
    # dropdowns for categorical
    if fname == "protocol_type":
        cb = ttk.Combobox(left, values=protocol_options, width=18)
        cb.set(protocol_options[0])
        cb.grid(row=r, column=1, pady=2)
        widgets[fname] = cb
    elif fname == "service":
        cb = ttk.Combobox(left, values=service_options, width=18)
        cb.set(service_options[0])
        cb.grid(row=r, column=1, pady=2)
        widgets[fname] = cb
    elif fname == "flag":
        cb = ttk.Combobox(left, values=flag_options, width=18)
        cb.set(flag_options[0])
        cb.grid(row=r, column=1, pady=2)
        widgets[fname] = cb
    else:
        e = ttk.Entry(left, width=20)
        e.grid(row=r, column=1, pady=2)
        e.insert(0, "0")
        widgets[fname] = e

# presets dropdown
preset_var = tk.StringVar(value="Select preset")
preset_combo = ttk.Combobox(left, values=list(PRESETS.keys()), textvariable=preset_var, state="readonly")
preset_combo.grid(row=len(feature_names)+1, column=0, pady=8)
def load_preset(evt=None):
    name = preset_var.get()
    if name in PRESETS:
        preset = PRESETS[name]
        for k,v in preset.items():
            if k in widgets:
                w = widgets[k]
                if isinstance(w, ttk.Combobox):
                    w.set(v)
                else:
                    w.delete(0, tk.END)
                    w.insert(0, str(v))
preset_combo.bind("<<ComboboxSelected>>", load_preset)

# predict button + result
def predict_action():
    try:
        # collect inputs
        data = []
        temp_row = {}
        for f in feature_names:
            w = widgets[f]
            if isinstance(w, ttk.Combobox):
                val = w.get()
                temp_row[f] = val
                # if encoder exists, we will map later
                data.append(val)
            else:
                txt = w.get().strip()
                if txt == "":
                    txt = "0"
                temp_row[f] = float(txt)
                data.append(float(txt))

        # Build DataFrame to use same column names
        df_row = pd.DataFrame([temp_row], columns=feature_names)

        # Encode categorical columns using label encoder used earlier (fallback to mapping from train csv)
        if label_encoder is not None:
            # our saved label_encoder earlier was a single LabelEncoder used incorrectly in script;
            # safe approach: rebuild local mapping from training csv
            pass

        # To be robust: load mapping from training file for protocol/service/flag
        # create mapping based on train_clean.csv (consistent with earlier training)
        if os.path.exists(TRAIN_CSV):
            train_df = pd.read_csv(TRAIN_CSV)
            for col in ["protocol_type","service","flag"]:
                if col in train_df.columns and col in df_row.columns:
                    mapping = {k:i for i,k in enumerate(sorted(train_df[col].dropna().unique().tolist()))}
                    df_row[col] = df_row[col].map(mapping).fillna(-1).astype(int)

        # Ensure numeric types
        df_row = df_row.astype(float)

        # Scale
        scaled = scaler.transform(df_row[feature_names])

        prob = model.predict_proba(scaled)[0]
        pred = model.predict(scaled)[0]
        conf = round(np.max(prob)*100, 2)

        label_text = ("🚨 ATTACK" if pred != 21 and pred != "normal" else "✅ NORMAL")
        # Note: in your trained model y values were string labels earlier; we used label names in training script.
        # We'll show both predicted class and confidence:
        result_text = f"Predicted class: {pred}   Confidence: {conf}%"
        result_var.set(result_text)

        # log to CSV
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        rowlog = df_row.copy()
        rowlog["prediction"] = pred
        rowlog["confidence"] = conf
        rowlog["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not os.path.exists(LOG_PATH):
            rowlog.to_csv(LOG_PATH, index=False)
        else:
            rowlog.to_csv(LOG_PATH, mode="a", header=False, index=False)

        # update simple chart (pie of last 20 predictions)
        update_chart()

    except Exception as e:
        messagebox.showerror("Error", str(e))

predict_btn = ttk.Button(left, text="Predict", command=predict_action)
predict_btn.grid(row=len(feature_names)+2, column=0, pady=8)

result_var = tk.StringVar(value="No prediction yet")
result_label = ttk.Label(left, textvariable=result_var, foreground="white")
result_label.grid(row=len(feature_names)+2, column=1, pady=8)

# Right: chart area and log view
fig, ax = plt.subplots(figsize=(5,4))
canvas_fig = FigureCanvasTkAgg(fig, master=right)
canvas_fig.get_tk_widget().pack()

def update_chart():
    # read last N logs and draw pie
    if not os.path.exists(LOG_PATH):
        ax.clear()
        ax.text(0.5, 0.5, "No logs yet", ha="center")
        canvas_fig.draw()
        return
    logs = pd.read_csv(LOG_PATH)
    last = logs.tail(50)
    counts = last["prediction"].value_counts()
    ax.clear()
    counts.plot.pie(ax=ax, autopct="%1.1f%%", ylabel="")
    ax.set_title("Last 50 predictions (distribution)")
    canvas_fig.draw()

# initial chart
update_chart()

# log open button
def open_log():
    if os.path.exists(LOG_PATH):
        filedialog.askopenfilename(initialdir=os.path.dirname(LOG_PATH))
    else:
        messagebox.showinfo("No logs", "No log file found yet.")

log_btn = ttk.Button(right, text="Open Logs", command=open_log)
log_btn.pack(pady=6)

root.mainloop()
