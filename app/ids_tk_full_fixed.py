# ids_tk_full_fixed.py
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
# Paths
# ---------------------------
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE, "models", "random_forest_improved.pkl")
SCALER_PATH = os.path.join(BASE, "models", "scaler_improved.pkl")
TRAIN_CSV = os.path.join(BASE, "data", "processed", "train_clean.csv")
LOG_PATH = os.path.join(BASE, "results", "realtime_log.csv")

# ---------------------------
# Check model & scaler presence
# ---------------------------
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise SystemExit(f"Missing model or scaler. Expected:\n{MODEL_PATH}\n{SCALER_PATH}")

# Load model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ---------------------------
# Exact feature order (41) — (matches train_clean.csv header)
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
# Load train CSV to get categorical options + label->attack_type mapping
# ---------------------------
protocol_options = ["tcp","udp","icmp"]
service_options = []
flag_options = []

label_to_attack = {}  # mapping numeric label -> most common attack_type string

if os.path.exists(TRAIN_CSV):
    train_df = pd.read_csv(TRAIN_CSV)
    # categorical options (sorted so mapping is deterministic)
    if "protocol_type" in train_df.columns:
        protocol_options = sorted(train_df["protocol_type"].dropna().unique().tolist())
    if "service" in train_df.columns:
        service_options = sorted(train_df["service"].dropna().unique().tolist())
    if "flag" in train_df.columns:
        flag_options = sorted(train_df["flag"].dropna().unique().tolist())

    # Build label -> attack_type mapping (use most common attack_type for each label value)
    if "label" in train_df.columns and "attack_type" in train_df.columns:
        mapping = train_df.groupby("label")["attack_type"].agg(lambda s: s.mode().iat[0] if not s.mode().empty else str(s.iloc[0]))
        label_to_attack = mapping.to_dict()

# If service_options empty, add fallback small list
if not service_options:
    service_options = ["http","smtp","ftp","other"]

# ---------------------------
# Build categorical mappings (value -> int) consistent with training CSV
# We'll compute on-demand inside predict to ensure mapping reflects training CSV
# ---------------------------

# ---------------------------
# Presets (autofill) - values for a subset of features; rest default to 0
# ---------------------------
PRESETS = {
    "Normal": {
        "duration":0.3,"protocol_type":"tcp","service":"http","flag":"SF",
        "src_bytes":200,"dst_bytes":250,"count":5,"srv_count":5,"serror_rate":0.0,"rerror_rate":0.01,"same_srv_rate":0.9
    },
    "DoS_SYN": {
        "duration":0.1,"protocol_type":"tcp","service":"other","flag":"S0",
        "src_bytes":20,"dst_bytes":0,"count":300,"srv_count":250,"serror_rate":0.95,"rerror_rate":0.02,"same_srv_rate":1.0
    },
    "Probe": {
        "duration":0.5,"protocol_type":"tcp","service":"http","flag":"SF",
        "src_bytes":150,"dst_bytes":30,"count":50,"srv_count":1,"serror_rate":0.01,"rerror_rate":0.8,"same_srv_rate":0.02
    },
    "R2L": {
        "duration":5,"protocol_type":"tcp","service":"ftp","flag":"REJ",
        "src_bytes":100,"dst_bytes":120,"count":20,"srv_count":10,"serror_rate":0.0,"rerror_rate":0.10,"same_srv_rate":0.4
    },
    "U2R": {
        "duration":2,"protocol_type":"tcp","service":"other","flag":"SF",
        "src_bytes":50,"dst_bytes":5,"count":5,"srv_count":5,"serror_rate":0.05,"rerror_rate":0.40,"same_srv_rate":0.8
    }
}

# ---------------------------
# Build GUI (dark theme)
# ---------------------------
root = tk.Tk()
root.title("IDS Desktop — Fixed (41 features)")
root.geometry("1200x820")
root.configure(bg="#0b1220")

style = ttk.Style(root)
style.theme_use('clam')
style.configure('TLabel', background='#0b1220', foreground='#e6eef6', font=('Segoe UI', 10))
style.configure('TButton', background='#1f6feb', foreground='white')
style.configure('TEntry', fieldbackground='#111827', foreground='#e6eef6')
style.configure('TCombobox', fieldbackground='#111827', foreground='#e6eef6')

main = ttk.Frame(root, padding=8)
main.pack(fill='both', expand=True)

left = ttk.Frame(main)
left.grid(row=0, column=0, sticky='nsw', padx=6, pady=6)

right = ttk.Frame(main)
right.grid(row=0, column=1, sticky='nse', padx=6, pady=6)

# Scrollable left panel for inputs
canvas = tk.Canvas(left, width=460, bg='#0b1220', highlightthickness=0)
scrollbar = ttk.Scrollbar(left, orient="vertical", command=canvas.yview)
scrollable = ttk.Frame(canvas)
scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
canvas.create_window((0,0), window=scrollable, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

widgets = {}

for i, fname in enumerate(FEATURES):
    lbl = ttk.Label(scrollable, text=fname+':', width=26, anchor='w')
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
        # protect if no flags present
        fopts = flag_options if flag_options else ["SF","S0","REJ"]
        cb = ttk.Combobox(scrollable, values=fopts, width=28)
        cb.set(fopts[0])
        cb.grid(row=i, column=1, padx=4, pady=3)
        widgets[fname] = cb
    else:
        ent = ttk.Entry(scrollable, width=30)
        ent.insert(0, "0")
        ent.grid(row=i, column=1, padx=4, pady=3)
        widgets[fname] = ent

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# Preset combo
preset_lbl = ttk.Label(scrollable, text="Load Preset:", width=26, anchor='w')
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

# Right side: predict button, chart, logs
predict_btn = ttk.Button(right, text="Predict", width=18)
predict_btn.pack(pady=10)

# Chart area
fig, ax = plt.subplots(figsize=(6,5), facecolor='#0b1220')
fig.patch.set_facecolor('#0b1220')
ax.set_facecolor('#0b1220')
canvas_fig = FigureCanvasTkAgg(fig, master=right)
canvas_fig.get_tk_widget().pack(pady=6)

# status / result label
status_var = tk.StringVar(value="No predictions yet")
status_lbl = ttk.Label(right, textvariable=status_var, foreground='white')
status_lbl.pack(pady=6)

# utility: read_last_logs
def read_logs():
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame()
    try:
        return pd.read_csv(LOG_PATH)
    except Exception:
        return pd.DataFrame()

def update_chart():
    ax.clear()
    logs = read_logs()
    if logs.empty:
        ax.text(0.5,0.5,"No logs yet", ha='center', color='white')
        canvas_fig.draw()
        return
    last = logs.tail(100)
    # use prediction column if exists, else class index
    if "prediction" in last.columns:
        counts = last["prediction"].value_counts()
    else:
        counts = last.iloc[:,-2].value_counts()  # fallback
    labels = [str(x) for x in counts.index.tolist()]
    values = counts.values.tolist()
    # draw pie with white labels
    ax.pie(values, labels=labels, autopct="%1.1f%%", textprops={"color":"white"})
    ax.set_title("Last 100 predictions (by class)", color='white')
    canvas_fig.draw()

# Prediction function
def predict_action():
    try:
        # Collect row dict in correct order
        row = {}
        for f in FEATURES:
            w = widgets[f]
            if isinstance(w, ttk.Combobox):
                val = w.get()
                row[f] = val
            else:
                txt = w.get().strip()
                if txt == "" or txt.lower() == "nan":
                    txt = "0"
                # safe convert
                try:
                    row[f] = float(txt)
                except:
                    messagebox.showerror("Invalid input", f"Field {f} requires a numeric value.")
                    return

        df_row = pd.DataFrame([row], columns=FEATURES)

        # Map categorical values using training CSV unique mapping
        if os.path.exists(TRAIN_CSV):
            tdf = pd.read_csv(TRAIN_CSV)
            for col in ["protocol_type","service","flag"]:
                if col in tdf.columns and col in df_row.columns:
                    uniq = sorted(tdf[col].dropna().unique().tolist())
                    mapping = {k:i for i,k in enumerate(uniq)}
                    df_row[col] = df_row[col].map(mapping).fillna(-1).astype(float)
        else:
            # fallback: try to cast protocol/service/flag to numeric (if they are numeric)
            for col in ["protocol_type","service","flag"]:
                if col in df_row.columns:
                    try:
                        df_row[col] = df_row[col].astype(float)
                    except:
                        df_row[col] = 0.0

        # Ensure float dtype
        df_row = df_row.astype(float)

        # scale
        scaled = scaler.transform(df_row[FEATURES])

        # predict & confidence
        prob = model.predict_proba(scaled)[0]
        pred = model.predict(scaled)[0]
        conf = round(np.max(prob)*100,2)

        # human readable label via mapping if available
        human = None
        try:
            # if label_to_attack mapping exists as dict (label numeric -> attack string)
            if label_to_attack:
                human = label_to_attack.get(pred, None)
        except Exception:
            human = None

        # display popup
        if human:
            if str(human).lower() == "normal":
                messagebox.showinfo("Prediction", f"✅ NORMAL\nPredicted label: {pred} ({human})\nConfidence: {conf}%")
            else:
                messagebox.showwarning("Prediction", f"🚨 ATTACK\nPredicted label: {pred} ({human})\nConfidence: {conf}%")
        else:
            if str(pred).lower() == "normal" or (isinstance(pred, (int, np.integer)) and str(pred) == "21"):
                messagebox.showinfo("Prediction", f"✅ NORMAL\nPredicted label: {pred}\nConfidence: {conf}%")
            else:
                messagebox.showwarning("Prediction", f"🚨 ATTACK\nPredicted label: {pred}\nConfidence: {conf}%")

        # Log to CSV
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        log_row = df_row.copy()
        log_row["prediction"] = pred
        log_row["confidence"] = conf
        log_row["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # write header if new
        if not os.path.exists(LOG_PATH):
            log_row.to_csv(LOG_PATH, index=False)
        else:
            log_row.to_csv(LOG_PATH, mode="a", header=False, index=False)

        status_var.set(f"Last: {pred} ({human if human else 'class'}) — {conf}%")
        update_chart()

    except Exception as e:
        messagebox.showerror("Error", str(e))

# bind predict button
predict_btn.config(command=predict_action)

# initial chart
update_chart()

# helper: open logs (system)
def open_logs():
    if os.path.exists(LOG_PATH):
        try:
            os.system(f'xdg-open "{LOG_PATH}"')
        except:
            messagebox.showinfo("Logs", f"Log file at: {LOG_PATH}")
    else:
        messagebox.showinfo("Logs", "No logs found yet.")

open_btn = ttk.Button(right, text="Open logs (CSV)", command=open_logs)
open_btn.pack(pady=6)

refresh_btn = ttk.Button(right, text="Refresh Chart", command=update_chart)
refresh_btn.pack(pady=6)

root.mainloop()
