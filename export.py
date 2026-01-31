# ============================================
# 🚀 MULTI-CHANNEL CSV EXPORT FOR POWER BI
# ============================================

import os
import ast
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# --------------------------------------------
# PATHS
# --------------------------------------------
DATASET_PATH = r"C:\Users\arsha\Downloads\archive (2)"
LABELED_FILE = os.path.join(DATASET_PATH, "labeled_anomalies.csv")
DATA_FOLDER = os.path.join(DATASET_PATH, "data", "data", "test")

# --------------------------------------------
# PARAMETERS
# --------------------------------------------
ROLLING_WINDOW = 50
RECURRING_THRESHOLD = 10

# --------------------------------------------
# LOAD LABELED DATA
# --------------------------------------------
labeled_df = pd.read_csv(LABELED_FILE)
labeled_df["anomaly_sequences"] = labeled_df["anomaly_sequences"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

channels = sorted(labeled_df["chan_id"].unique())

# --------------------------------------------
# MODEL PIPELINE
# --------------------------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("iforest", IsolationForest(
        n_estimators=200,
        contamination=0.02,
        max_samples=0.8,
        random_state=42
    ))
])

# --------------------------------------------
# STORAGE
# --------------------------------------------
all_channel_data = []
all_segments = []

# ============================================
# 🔁 LOOP THROUGH ALL CHANNELS
# ============================================
for channel in channels:
    print(f"Processing channel: {channel}")

    # ---------- Load sensor data ----------
    values = np.load(os.path.join(DATA_FOLDER, f"{channel}.npy"))
    df = pd.DataFrame({"value": values[:, 0]})

    # ---------- Feature engineering ----------
    df["rolling_mean"] = df["value"].rolling(ROLLING_WINDOW).mean()
    df["rolling_std"] = df["value"].rolling(ROLLING_WINDOW).std()
    df["zscore"] = (df["value"] - df["rolling_mean"]) / df["rolling_std"]
    df["lag_1"] = df["value"].shift(1)
    df.dropna(inplace=True)

    X = df[["value", "rolling_mean", "rolling_std", "zscore", "lag_1"]]

    # ---------- Train model ----------
    pipeline.fit(X)
    preds = pipeline.predict(X)
    df["anomaly_ml"] = (preds == -1).astype(int)

    # ---------- Recurring logic ----------
    df["recurring_count"] = df["anomaly_ml"].rolling(ROLLING_WINDOW * 4).sum()
    df["recurring_flag"] = df["recurring_count"] > RECURRING_THRESHOLD

    # ---------- Add channel info ----------
    df.reset_index(inplace=True)
    df.rename(columns={"index": "Time_Index"}, inplace=True)
    df["Channel"] = channel
    all_channel_data.append(df)

    # ---------- Build recurring segments ----------
    segments, start = [], None
    for i in range(len(df)):
        if df.loc[i, "recurring_flag"] and start is None:
            start = i
        elif not df.loc[i, "recurring_flag"] and start is not None:
            segments.append((start, i - 1))
            start = None
    if start is not None:
        segments.append((start, len(df) - 1))

    # ---------- NASA labeled segments ----------
    labeled_sequences = labeled_df[labeled_df["chan_id"] == channel].iloc[0]["anomaly_sequences"]

    # Add segments to list
    for s, e in segments:
        overlaps = False
        for ns, ne in labeled_sequences:
            if not (e < ns or s > ne):
                overlaps = True
                break
        all_segments.append({
            "Channel": channel,
            "Start_Index": s,
            "End_Index": e,
            "Duration": e - s,
            "Source": "Recurring (ML)" if overlaps else "ML Only"
        })

    for s, e in labeled_sequences:
        all_segments.append({
            "Channel": channel,
            "Start_Index": s,
            "End_Index": e,
            "Duration": e - s,
            "Source": "NASA Labeled"
        })

# ============================================
# 📤 EXPORT CSV FILES
# ============================================
pd.concat(all_channel_data).to_csv("channel_data_all.csv", index=False)
pd.DataFrame(all_segments).to_csv("all_segments.csv", index=False)

print("✅ channel_data_all.csv exported")
print("✅ all_segments.csv exported")
