# ids_tk_full_dark.py
import os
import joblib
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ---------------------------
# Paths (project-relative)
# ---------------------------
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE, "models", "random_forest_improved.pkl")
SCALER_PATH = os.path.join(BASE, "models", "scaler_improved.pkl")
TRAIN_CSV = os.path.join(BASE, "data", "processed", "train_clean.csv")
LOG_PATH = os.path.join(BASE, "results", "realtime_log.csv")

# ---------------------------
# Load model and scaler
# ---------------------------
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise SystemExit(f"Model or scaler missing. Expected:\n{MODEL_PATH}\n{SCALER_PATH}")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ---------------------------
# Feature list (from your data header)
# ---------------------------
FEATURES = [
"duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent",
"hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root",
"num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
"count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
"diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
"dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
"dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate"
]

# ---------------------------
# Get categorical options from train CSV (if available)
# ---------------------------
protocol_options = ["tcp","udp","icmp"]
service_options = ["http","smtp","ftp","other"]
flag_options = ["SF","S0","REJ","RSTR","RSTO"]

if os.path.exists(TRAIN_CSV):
    try:
        tmp = pd.read_csv(TRAIN_CSV)
        if "protocol_type" in tmp.columns:
            protocol_options = sorted(tmp["protocol_type"].dropna().unique().tolist())
        if "service" in tmp.columns:
            service_options = sorted(tmp["service"].dropna().unique().tolist())
        if "flag" in tmp.columns:
            flag_options = sorted(tmp["flag"].dropna().unique().tolist())
    except Exception:
        pass

# ---------------------------
# Preset examples
# ---------------------------
PRESETS = {
    "Normal": {
        "duration":0.3,"protocol_type":"tcp","service":"http","flag":"SF","src_bytes":200,"dst_bytes":250,
        "count":5,"srv_count":5,"serror_rate":0.0,"rerror_rate":0.01,"same_srv_rate":0.9
    },
    "DoS_SYN": {
        "duration":0.1,"protocol_type":"tcp","service":"other","flag":"S0","src_bytes":20,"dst_bytes":0,
        "count":300,"srv_count":250,"serror_rate":0.95,"rerror_rate":0.02,"same_srv_rate":1.0
    },
    "Probe": {
        "duration":0.5,"protocol_type":"tcp","service":"http","flag":"SF","src_bytes":150,"dst_bytes":30,
        "count":50,"srv_count":1,"serror_rate":0.01,"rerror_rate":0.8,"same_srv_rate":0.02
    },
    "R2L": {
        "duration":5,"protocol_type":"tcp","service":"ftp","flag":"REJ","src_bytes":100,"dst_bytes":120,
        "count":20,"srv_count":10,"serror_rate":0.0,"rerror_rate":0.10,"same_srv_rate":0.4
    },
    "U2R": {
        "duration":2,"protocol_type":"tcp","service":"other","flag":"SF","src_bytes":50,"dst_bytes":5,
        "count":5,"srv_count":5,"serror_rate":0.05,"rerror_rate":0.40,"same_srv_rate":0.8
    }
}

# ---------------------------
# GUI Setup (dark theme)
# ---------------------------
root = tk.Tk()
root.title("IDS Desktop - Dark Dashboard")
root.geometry("1200x820")
root.configure(bg="#0b1220")

style = ttk.Style(root)
style.theme_use('clam')
style.configure('TLabel', background='#0b1220', foreground='#e6eef6', font=('Segoe UI', 10))
style.configure('TButton', background='#1f6feb', foreground='white')
style.configure('TEntry', fieldbackground='#111827', foreground='#e6eef6')
style.configure('TCombobox', fieldbackground='#111827', foreground='#e6eef6')

main = ttk.Frame(root, padding=10)
main.pack(fill='both', expand=True)

left = ttk.Frame(main)
left.grid(row=0, column=0, sticky='nsw', padx=6, pady=6)

right = ttk.Frame(main)
right.grid(row=0, column=1, sticky='nse', padx=6, pady=6)

# Scrollable left panel (many fields)
canvas = tk.Canvas(left, width=420, bg='#0b1220', highlightthickness=0)
scrollbar = ttk.Scrollbar(left, orient="vertical", command=canvas.yview)
scrollable = ttk.Frame(canvas)

scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
canvas.create_window((0,0), window=scrollable, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

widgets = {}

for i, fname in enumerate(FEATURES):
    lbl = ttk.Label(scrollable, text=fname+':', width=22, anchor='w')
    lbl.grid(row=i, column=0, padx=4, pady=3, sticky='w')
    if fname == "protocol_type":
        cb = ttk.Combobox(scrollable, values=protocol_options, width=28)
        cb.set(protocol_options[0])
        cb.grid(row=i, column=1, padx=4, pady=3)
        widgets[fname] = cb
    elif fname == "service":
        cb = ttk.Combobox(scrollable, values=service_options, width=28)
        cb.set(service_options[0])
        cb.grid(row=i, column=1, padx=4, pady=3)
        widgets[fname] = cb
    elif fname == "flag":
        cb = ttk.Combobox(scrollable, values=flag_options, width=28)
        cb.set(flag_options[0])
        cb.grid(row=i, column=1, padx=4, pady=3)
        widgets[fname] = cb
    else:
        ent = ttk.Entry(scrollable, width=30)
        ent.insert(0, "0")
        ent.grid(row=i, column=1, padx=4, pady=3)
        widgets[fname] = ent

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# Preset selection and action
preset_lbl = ttk.Label(scrollable, text="Load Preset:", width=22, anchor='w')
preset_lbl.grid(row=len(FEATURES)+1, column=0, pady=(10,2), sticky='w')
preset_var = tk.StringVar()
preset_combo = ttk.Combobox(scrollable, textvariable=preset_var, values=list(PRESETS.keys()), state='readonly', width=28)
preset_combo.grid(row=len(FEATURES)+1, column=1, pady=(10,2))

def load_preset(event=None):
    name = preset_var.get()
    if name in PRESETS:
        p = PRESETS[name]
        for k,v in p.items():
            if k in widgets:
                w = widgets[k]
                if isinstance(w, ttk.Combobox):
                    w.set(v)
                else:
                    w.delete(0, tk.END)
                    w.insert(0, str(v))
preset_combo.bind("<<ComboboxSelected>>", load_preset)

# Predict button
def predict_action():
    try:
        # collect row dict
        row = {}
        for f in FEATURES:
            w = widgets[f]
            if isinstance(w, ttk.Combobox):
                row[f] = w.get()
            else:
                txt = w.get().strip()
                if txt == "" or txt.lower() == "nan":
                    txt = "0"
                row[f] = float(txt)

        df_row = pd.DataFrame([row], columns=FEATURES)

        # Map categorical features consistent with training CSV
        if os.path.exists(TRAIN_CSV):
            train_df = pd.read_csv(TRAIN_CSV)
            for col in ["protocol_type","service","flag"]:
                if col in train_df.columns:
                    mapping = {k:i for i,k in enumerate(sorted(train_df[col].dropna().unique().tolist()))}
                    df_row[col] = df_row[col].map(mapping).fillna(-1).astype(float)

        # Ensure numeric and scale
        df_row = df_row.astype(float)
        scaled = scaler.transform(df_row[FEATURES])

        proba = model.predict_proba(scaled)[0]
        pred = model.predict(scaled)[0]
        conf = round(np.max(proba)*100, 2)

        # Nice popup
        if isinstance(pred, (int, np.integer)):
            is_normal = (str(pred) == "21") or (pred == 21)  # depends on label encoding; keep fallback
        else:
            is_normal = (str(pred).lower() == "normal")

        if is_normal:
            txt = f"✅ NORMAL\nConfidence: {conf}%"
            messagebox.showinfo("Prediction Result", txt)
        else:
            txt = f"🚨 ATTACK\nPredicted class: {pred}\nConfidence: {conf}%"
            messagebox.showwarning("Prediction Result", txt)

        # Log
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        log_row = df_row.copy()
        log_row["prediction"] = pred
        log_row["confidence"] = conf
        log_row["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not os.path.exists(LOG_PATH):
            log_row.to_csv(LOG_PATH, index=False)
        else:
            log_row.to_csv(LOG_PATH, mode="a", header=False, index=False)

        update_chart()
    except Exception as e:
        messagebox.showerror("Error", str(e))

predict_btn = ttk.Button(right, text="Predict", command=predict_action)
predict_btn.pack(pady=12)

# Chart area (matplotlib) - show distribution of last 50 predictions
fig, ax = plt.subplots(figsize=(6,5), facecolor='#0b1220')
fig.patch.set_facecolor('#0b1220')
ax.set_facecolor('#0b1220')
canvas_fig = FigureCanvasTkAgg(fig, master=right)
canvas_fig.get_tk_widget().pack()

def update_chart():
    ax.clear()
    if not os.path.exists(LOG_PATH):
        ax.text(0.5,0.5,"No logs yet", ha='center', color='white')
        canvas_fig.draw()
        return
    try:
        logs = pd.read_csv(LOG_PATH)
        last = logs.tail(100)
        counts = last["prediction"].value_counts()
        labels = [str(x) for x in counts.index.tolist()]
        values = counts.values.tolist()
        ax.pie(values, labels=labels, autopct="%1.1f%%", textprops={"color":"white"})
        ax.set_title("Last 100 predictions (by class)", color='white')
        canvas_fig.draw()
    except Exception as e:
        ax.text(0.5,0.5,"Error plotting logs", ha='center', color='white')
        canvas_fig.draw()

update_chart()

# Open logs button
def open_logs():
    if os.path.exists(LOG_PATH):
        os.system(f'xdg-open "{LOG_PATH}"')
    else:
        messagebox.showinfo("Logs", "No logs found yet.")
open_btn = ttk.Button(right, text="Open logs (CSV)", command=open_logs)
open_btn.pack(pady=6)

# Refresh chart button
refresh_btn = ttk.Button(right, text="Refresh Chart", command=update_chart)
refresh_btn.pack(pady=6)

root.mainloop()
