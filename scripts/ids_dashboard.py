import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np

# Load model and scaler
#model = joblib.load("../models/random_forest_model.pkl")
#scaler = joblib.load("../models/scaler.pkl")

#model = joblib.load("../models/random_forest_model_10.pkl")
#scaler = joblib.load("../models/scaler_10.pkl")

model = joblib.load("../models/random_forest_improved.pkl")
scaler = joblib.load("../models/scaler_improved.pkl")



# Feature names and hints
feature_info = [
    ("Duration (sec)", "e.g. 0.1"),
    ("Src Bytes", "e.g. 120"),
    ("Dst Bytes", "e.g. 100"),
    ("Wrong Fragment", "e.g. 0"),
    ("Urgent", "e.g. 0"),
    ("Count (connections)", "e.g. 3"),
    ("Srv Count", "e.g. 3"),
    ("Serror Rate", "e.g. 0.02"),
    ("Rerror Rate", "e.g. 0.01"),
    ("Same Srv Rate", "e.g. 0.9"),
]

# GUI window
root = tk.Tk()
root.title("IDS Dashboard")
root.geometry("500x500")
root.config(bg="#f0f0f0")

tk.Label(root, text="Intrusion Detection System", font=("Helvetica", 16, "bold"), bg="#f0f0f0").pack(pady=15)

entries = []

def clear_hint(event, entry, hint):
    if entry.get() == hint:
        entry.delete(0, tk.END)
        entry.config(fg="black")

for name, hint in feature_info:
    frame = tk.Frame(root, bg="#f0f0f0")
    frame.pack(pady=4)
    tk.Label(frame, text=name + ":", width=20, anchor="w", bg="#f0f0f0").pack(side=tk.LEFT)
    e = tk.Entry(frame, width=10, fg="gray")
    e.pack(side=tk.LEFT)
    e.insert(0, hint)
    e.bind("<FocusIn>", lambda event, entry=e, h=hint: clear_hint(event, entry, h))
    entries.append(e)


result_label = tk.Label(root, text="", font=("Helvetica", 14, "bold"), bg="#f0f0f0")
result_label.pack(pady=20)

def predict():
    values = []
    for e in entries:
        val = e.get().strip()
        # Check if empty or not numeric
        try:
            val = float(val)
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter numeric values only.")
            return
        values.append(val)

    # Proceed only if all are valid numbers
    features = np.array([values])
    prediction = model.predict(features)
    result = "⚠️ Attack Detected!" if prediction[0] == 1 else "✅ Normal Traffic"
    messagebox.showinfo("Prediction Result", result)


tk.Button(root, text="Predict", command=predict, bg="blue", fg="white", width=15).pack(pady=10)

root.mainloop()

