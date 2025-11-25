#!/usr/bin/env python3
"""
ids_tk_full_complete.py  (Final GUI)
- 41 features (correct order)
- Presets based on train_clean.csv group means
- Simulation mode
- Logging to results/realtime_log.csv
- Pie + line chart
- Big colored banner (Normal vs Attack)
- Shows numeric label + attack name (e.g., 21 (normal), 18 (neptune))
"""

import os
import sys
import time
import joblib
import threading
import random
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ------------------- Paths -------------------
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE, "models", "random_forest_fixed_41.pkl")
SCALER_PATH = os.path.join(BASE, "models", "scaler_fixed_41.pkl")
TRAIN_CSV = os.path.join(BASE, "data", "processed", "train_clean.csv")
LOG_DIR = os.path.join(BASE, "results")
LOG_PATH = os.path.join(LOG_DIR, "realtime_log.csv")

# ----------------- Load model ----------------
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    msg = f"Missing model or scaler.\nExpected:\n{MODEL_PATH}\n{SCALER_PATH}"
    print(msg)
    raise SystemExit(msg)

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# 41 features in exact training order
FEATURES = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent",
    "hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
    "count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
    "dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate"
]

# ----------------- Presets + label→attack mapping -----------------
presets_means = {}
global_means = None
label_to_attack = {}   # e.g. {21: 'normal', 18: 'neptune', ...}

if os.path.exists(TRAIN_CSV):
    try:
        df_train = pd.read_csv(TRAIN_CSV)

        # label -> attack_type mapping for display
        if "label" in df_train.columns and "attack_type" in df_train.columns:
            mapping_series = df_train.groupby("label")["attack_type"].agg(
                lambda s: s.mode().iat[0] if not s.mode().empty else str(s.iloc[0])
            )
            label_to_attack = mapping_series.to_dict()

        # global feature means (fallback)
        global_means = df_train[FEATURES].mean().to_dict()

        # Attack-type based presets if we have attack_type
        if "attack_type" in df_train.columns:
            df_train["attack_type_lc"] = df_train["attack_type"].astype(str).str.lower()

            # Normal
            mask_normal = df_train["attack_type_lc"] == "normal"
            if mask_normal.any():
                presets_means["Normal"] = df_train[mask_normal][FEATURES].mean().to_dict()

            # DoS
            dos_mask = df_train["attack_type_lc"].str.contains(
                "dos|smurf|neptune|back|teardrop|land|pod", na=False
            )
            if dos_mask.any():
                presets_means["DoS"] = df_train[dos_mask][FEATURES].mean().to_dict()

            # Probe
            probe_mask = df_train["attack_type_lc"].str.contains(
                "probe|ipsweep|satan|nmap|portsweep|mscan|saint", na=False
            )
            if probe_mask.any():
                presets_means["Probe"] = df_train[probe_mask][FEATURES].mean().to_dict()

            # R2L
            r2l_mask = df_train["attack_type_lc"].str.contains(
                "r2l|ftp_write|guess_passwd|imap|phf|spy|warezclient|warezmaster", na=False
            )
            if r2l_mask.any():
                presets_means["R2L"] = df_train[r2l_mask][FEATURES].mean().to_dict()

            # U2R
            u2r_mask = df_train["attack_type_lc"].str.contains(
                "u2r|buffer_overflow|loadmodule|perl|rootkit|sqlattack", na=False
            )
            if u2r_mask.any():
                presets_means["U2R"] = df_train[u2r_mask][FEATURES].mean().to_dict()

        if global_means is None:
            global_means = {f: 0.0 for f in FEATURES}

    except Exception as e:
        print("Warning reading train CSV:", e)
        global_means = {f: 0.0 for f in FEATURES}
else:
    global_means = {f: 0.0 for f in FEATURES}

# Fallback label_to_attack (in case CSV missing)
if not label_to_attack:
    label_to_attack = {21: "normal"}

# Ensure presets exist
for p in ["Normal", "DoS", "Probe", "R2L", "U2R"]:
    if p not in presets_means:
        presets_means[p] = global_means.copy()

# ----------------- GUI Layout -----------------
root = tk.Tk()
root.title("IDS Desktop — Full Dashboard (final)")
root.geometry("1250x860")
root.configure(bg="#0b1220")

style = ttk.Style(root)
style.theme_use("clam")
style.configure('TLabel', background='#0b1220', foreground='#e6eef6', font=('Segoe UI', 10))
style.configure('TButton', background='#1f6feb', foreground='white')
style.configure('TEntry', fieldbackground='#111827', foreground='#e6eef6')
style.configure('TCombobox', fieldbackground='#111827', foreground='#e6eef6')

sidebar = ttk.Frame(root, width=360)
sidebar.pack(side="left", fill="y", padx=8, pady=8)

center = ttk.Frame(root)
center.pack(side="left", fill="both", expand=True, padx=8, pady=8)

right = ttk.Frame(root, width=360)
right.pack(side="right", fill="y", padx=8, pady=8)

# --- Sidebar scrollable form ---
canvas = tk.Canvas(sidebar, bg='#0b1220', highlightthickness=0)
vscroll = ttk.Scrollbar(sidebar, orient="vertical", command=canvas.yview)
form_frame = ttk.Frame(canvas)
form_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
canvas.create_window((0, 0), window=form_frame, anchor="nw")
canvas.configure(yscrollcommand=vscroll.set, width=360, height=780)
canvas.pack(side="left", fill="y", expand=False)
vscroll.pack(side="right", fill="y")

widgets = {}
r = 0
for fname in FEATURES:
    lbl = ttk.Label(form_frame, text=f"{fname}:", width=24, anchor="w")
    lbl.grid(row=r, column=0, padx=6, pady=2, sticky="w")
    ent = ttk.Entry(form_frame, width=22)
    ent.grid(row=r, column=1, padx=6, pady=2)
    ent.insert(0, str(round(global_means.get(fname, 0.0), 6)))
    widgets[fname] = ent
    r += 1

# Preset selector
preset_lbl = ttk.Label(form_frame, text="Load Preset:", width=24, anchor="w")
preset_lbl.grid(row=r, column=0, padx=6, pady=(10, 4), sticky="w")
preset_var = tk.StringVar()
preset_combo = ttk.Combobox(
    form_frame, textvariable=preset_var,
    values=list(presets_means.keys()),
    state="readonly", width=20
)
preset_combo.grid(row=r, column=1, padx=6, pady=(10, 4))
r += 1

def load_preset(event=None):
    key = preset_var.get()
    if key and key in presets_means:
        meanmap = presets_means[key]
        for f in FEATURES:
            val = meanmap.get(f, global_means.get(f, 0.0))
            widgets[f].delete(0, tk.END)
            widgets[f].insert(0, str(round(float(val), 6)))

preset_combo.bind("<<ComboboxSelected>>", load_preset)

# Simulate / Predict / Export buttons
sim_frame = ttk.Frame(form_frame)
sim_frame.grid(row=r, column=0, columnspan=2, pady=(8, 8))
start_sim_btn = ttk.Button(sim_frame, text="Start Simulate", width=14)
start_sim_btn.grid(row=0, column=0, padx=4)
stop_sim_btn = ttk.Button(sim_frame, text="Stop Simulate", width=14)
stop_sim_btn.grid(row=0, column=1, padx=4)
r += 1

predict_btn = ttk.Button(form_frame, text="Predict Now", width=24)
predict_btn.grid(row=r, column=0, columnspan=2, pady=(8, 6))
r += 1

export_csv_btn = ttk.Button(form_frame, text="Export Logs CSV", width=24)
export_csv_btn.grid(row=r, column=0, columnspan=2, pady=(4, 6))
r += 1

open_logs_btn = ttk.Button(form_frame, text="Open Logs Folder", width=24)
open_logs_btn.grid(row=r, column=0, columnspan=2, pady=(4, 12))
r += 1

# --- Center: charts + banner ---
fig1, ax1 = plt.subplots(figsize=(5.2, 4), facecolor='#0b1220')
fig1.patch.set_facecolor('#0b1220')
ax1.set_facecolor('#0b1220')
pie_canvas = FigureCanvasTkAgg(fig1, master=center)
pie_canvas.get_tk_widget().pack(pady=8)

fig2, ax2 = plt.subplots(figsize=(5.2, 2.6), facecolor='#0b1220')
fig2.patch.set_facecolor('#0b1220')
ax2.set_facecolor('#0b1220')
line_canvas = FigureCanvasTkAgg(fig2, master=center)
line_canvas.get_tk_widget().pack(pady=8)

status_var = tk.StringVar(value="Ready")
status_lbl = ttk.Label(center, textvariable=status_var, font=("Segoe UI", 11))
status_lbl.pack(pady=(2, 4))

# big colored banner
result_banner = tk.Label(
    center,
    text="Awaiting prediction...",
    bg="#111827",
    fg="#e5e7eb",
    font=("Segoe UI", 16, "bold"),
    pady=10
)
result_banner.pack(fill="x", padx=10, pady=(0, 10))

# --- Right: recent predictions list ---
logs_frame = ttk.Frame(right)
logs_frame.pack(fill="both", expand=True)
logs_title = ttk.Label(logs_frame, text="Recent Predictions (last 20)", font=("Segoe UI", 10, "bold"))
logs_title.pack(pady=(6, 4))
logs_listbox = tk.Listbox(logs_frame, height=18, width=48, bg="#0b1220", fg="#e6eef6")
logs_listbox.pack(padx=6, pady=6)

# ----------------- Logging helpers -----------------
def ensure_log_dir():
    os.makedirs(LOG_DIR, exist_ok=True)

def append_log(df_row, pred, conf):
    ensure_log_dir()
    log_row = df_row.copy()
    log_row["prediction"] = pred
    log_row["confidence"] = conf
    log_row["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not os.path.exists(LOG_PATH):
        log_row.to_csv(LOG_PATH, index=False)
    else:
        log_row.to_csv(LOG_PATH, mode="a", header=False, index=False)

def read_logs():
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame()
    try:
        return pd.read_csv(LOG_PATH)
    except Exception:
        return pd.DataFrame()

# ----------------- Prediction core -----------------
def collect_input_row():
    row = {}
    for f in FEATURES:
        txt = widgets[f].get().strip()
        if txt == "" or txt.lower() == "nan":
            txt = "0"
        try:
            row[f] = float(txt)
        except:
            t = txt.replace(",", ".").strip()
            try:
                row[f] = float(t)
            except:
                raise ValueError(f"Field {f} must be numeric (got '{txt}')")
    return pd.DataFrame([row], columns=FEATURES)

def do_predict(row_df):
    X = row_df[FEATURES].astype(float)
    Xs = scaler.transform(X)
    probs = model.predict_proba(Xs)[0]
    pred = model.predict(Xs)[0]
    conf = round(float(np.max(probs)) * 100, 2)
    return pred, conf, probs

def update_ui_after_prediction(pred, conf):
    # Get human name if available
    human = label_to_attack.get(pred)
    text_core = f"{pred}" + (f" ({human})" if human else "")

    # Decide normal vs attack
    is_normal = False
    if human and str(human).lower() == "normal":
        is_normal = True
    elif not human and str(pred) == "21":
        is_normal = True

    status_var.set(f"Last: {text_core} — {conf}%")

    if is_normal:
        banner_text = f"✔ NORMAL TRAFFIC — {text_core} [{conf}%]"
        result_banner.config(text=banner_text, bg="#065f46", fg="#d1fae5")
    else:
        banner_text = f"⚠ ATTACK DETECTED — {text_core} [{conf}%]"
        result_banner.config(text=banner_text, bg="#7f1d1d", fg="#fee2e2")

    ts = datetime.now().strftime("%H:%M:%S")
    logs_listbox.insert(0, f"{ts} → {text_core} [{conf}%]")
    if logs_listbox.size() > 50:
        logs_listbox.delete(50, tk.END)

def predict_action(event=None):
    try:
        row_df = collect_input_row()
        pred, conf, probs = do_predict(row_df)
        append_log(row_df, pred, conf)
        update_ui_after_prediction(pred, conf)
        refresh_charts()

        human = label_to_attack.get(pred)
        is_normal = False
        if human and str(human).lower() == "normal":
            is_normal = True
        elif not human and str(pred) == "21":
            is_normal = True

        if is_normal:
            messagebox.showinfo(
                "Prediction",
                f"✅ NORMAL TRAFFIC\n\nLabel: {pred} ({human or 'normal'})\nConfidence: {conf}%"
            )
        else:
            messagebox.showwarning(
                "Prediction",
                f"🚨 ATTACK DETECTED\n\nLabel: {pred}" +
                (f" ({human})" if human else "") +
                f"\nConfidence: {conf}%"
            )
    except Exception as e:
        messagebox.showerror("Error", str(e))

predict_btn.config(command=predict_action)

# ----------------- Charts -----------------
def refresh_charts():
    logs = read_logs()

    ax1.clear()
    ax1.set_facecolor('#0b1220')
    if logs.empty:
        ax1.text(0.5, 0.5, "No logs yet", color='white', ha='center')
    else:
        last = logs.tail(100)
        counts = last["prediction"].value_counts()
        labels = [str(x) for x in counts.index.tolist()]
        vals = counts.values.tolist()
        if vals:
            ax1.pie(vals, labels=labels, autopct="%1.1f%%", textprops={"color": "white"})
            ax1.set_title("Last 100 predictions (by class)", color='white')
    pie_canvas.draw()

    ax2.clear()
    ax2.set_facecolor('#0b1220')
    if logs.empty:
        ax2.text(0.5, 0.5, "No logs yet", color='white', ha='center')
    else:
        last = logs.tail(80)
        try:
            y = last["prediction"].astype(float).tolist()
            x = list(range(len(y)))
            ax2.plot(x, y, marker='o', color='#66b2ff')
            ax2.set_ylabel("Pred class", color='white')
        except Exception:
            ax2.text(0.5, 0.5, "Unable to draw line", color='white')
    line_canvas.draw()

refresh_charts()

# ----------------- Simulation -----------------
simulate_flag = threading.Event()

def simulate_loop(interval=1.0):
    while simulate_flag.is_set():
        choice = random.choice(list(presets_means.keys()))
        meanmap = presets_means[choice]
        for f in FEATURES:
            base = float(meanmap.get(f, global_means.get(f, 0.0)))
            noise = base * random.uniform(-0.15, 0.15) if abs(base) > 1e-6 else random.uniform(0, 1)
            val = round(max(0.0, base + noise), 6)
            widgets[f].delete(0, tk.END)
            widgets[f].insert(0, str(val))
        try:
            row_df = collect_input_row()
            pred, conf, _ = do_predict(row_df)
            append_log(row_df, pred, conf)
            root.after(0, lambda p=pred, c=conf: update_ui_after_prediction(p, c))
            root.after(0, refresh_charts)
        except Exception as e:
            print("Sim error:", e)
        time.sleep(interval)

def start_sim():
    if simulate_flag.is_set():
        return
    simulate_flag.set()
    t = threading.Thread(target=simulate_loop, args=(1.0,), daemon=True)
    t.start()
    status_var.set("Simulation running...")

def stop_sim():
    simulate_flag.clear()
    status_var.set("Ready")

start_sim_btn.config(command=start_sim)
stop_sim_btn.config(command=stop_sim)

# ----------------- Export / open logs -----------------
def export_csv():
    df = read_logs()
    if df.empty:
        messagebox.showinfo("Export", "No logs found.")
        return
    fname = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV", "*.csv")],
        initialfile="ids_export.csv"
    )
    if not fname:
        return
    df.to_csv(fname, index=False)
    messagebox.showinfo("Export", f"Saved: {fname}")

def open_logs_folder():
    ensure_log_dir()
    try:
        os.system(f'xdg-open \"{LOG_DIR}\"')
    except Exception:
        messagebox.showinfo("Logs", f"Logs at: {LOG_DIR}")

export_csv_btn.config(command=export_csv)
open_logs_btn.config(command=open_logs_folder)

# ----------------- Shortcuts -----------------
root.bind("<Control-s>", lambda e: export_csv())
root.bind("<Control-q>", lambda e: root.quit())
root.bind("<Return>", predict_action)

# ----------------- Run -----------------
root.mainloop()

