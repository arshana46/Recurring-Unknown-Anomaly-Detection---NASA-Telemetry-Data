import os
import ast
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# =========================
# Config
# =========================
DATASET_PATH = r"C:\Users\arsha\Downloads\archive (2)"
LABELED_FILE = os.path.join(DATASET_PATH, "labeled_anomalies.csv")
DATA_FOLDER = os.path.join(DATASET_PATH, "data", "data", "test")

ROLLING_WINDOW = 50
RECURRING_THRESHOLD = 10

# =========================
# Load labeled data
# =========================
labeled_df = pd.read_csv(LABELED_FILE)
labeled_df["anomaly_sequences"] = labeled_df["anomaly_sequences"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)
channels = sorted(labeled_df["chan_id"].unique())

# =========================
# Prepare storage
# =========================
all_channel_data = []
ml_segments_only = []
nasa_labels_for_validation = []

# =========================
# Process each channel
# =========================
for channel in channels:
    print(f"🔄 Processing {channel}...")

    # Load ML data
    values = np.load(os.path.join(DATA_FOLDER, f"{channel}.npy"))
    channel_data = pd.DataFrame({"value": values[:, 0]})

    # Feature engineering
    channel_data["rolling_mean"] = channel_data["value"].rolling(ROLLING_WINDOW).mean()
    channel_data["rolling_std"] = channel_data["value"].rolling(ROLLING_WINDOW).std()
    channel_data["zscore"] = (channel_data["value"] - channel_data["rolling_mean"]) / channel_data["rolling_std"]
    channel_data["lag_1"] = channel_data["value"].shift(1)
    channel_data.dropna(inplace=True)

    X = channel_data[["value", "rolling_mean", "rolling_std", "zscore", "lag_1"]]

    # GridSearchCV with IsolationForest
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("iforest", IsolationForest(random_state=42))
    ])
    param_grid = {
        "iforest__n_estimators": [100, 200],
        "iforest__max_samples": [0.6, 0.8],
        "iforest__contamination": [0.01, 0.02, 0.05],
        "iforest__max_features": [1.0]
    }

    # Dummy target for unsupervised scoring
    y_dummy = np.zeros(len(X))
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(X, y_dummy)

    # Predictions
    preds = grid.best_estimator_.predict(X)
    channel_data["anomaly_ml"] = pd.Series(preds, index=channel_data.index).map({1: 0, -1: 1})

    total_anomalies = channel_data["anomaly_ml"].sum()
    print(f"  ✅ {channel}: {total_anomalies} anomalies")

    # Recurring ML anomalies
    channel_data["recurring_count"] = channel_data["anomaly_ml"].rolling(ROLLING_WINDOW * 4).sum()
    channel_data["recurring_flag"] = channel_data["recurring_count"] > RECURRING_THRESHOLD

    # Add channel info
    channel_data["Channel"] = channel
    channel_data["Time_Index"] = channel_data.index

    # ML segments
    segments, start = [], None
    for i in range(len(channel_data)):
        if channel_data["recurring_flag"].iloc[i] and start is None:
            start = i
        elif not channel_data["recurring_flag"].iloc[i] and start is not None:
            segments.append((start, i - 1))
            start = None
    if start is not None:
        segments.append((start, len(channel_data) - 1))
    for s, e in segments:
        ml_segments_only.append({
            "Channel": channel,
            "Start_Index": s,
            "End_Index": e,
            "Duration": e - s + 1,
            "Anomalies_in_Segment": channel_data["anomaly_ml"].iloc[s:e+1].sum(),
            "Source": "ML_Recurring"
        })

    # Append ML data
    all_channel_data.append(channel_data)

    # =========================
    # Create NASA validation table (separate)
    # =========================
    label_row = labeled_df[labeled_df["chan_id"] == channel]
    if not label_row.empty:
        anomalies_idx = label_row.iloc[0]["anomaly_sequences"]
        for i in anomalies_idx:
            nasa_labels_for_validation.append({
                "Channel": channel,
                "Time_Index": i,
                "anomaly_nasa": 1
            })

# =========================
# Export
# =========================
pd.concat(all_channel_data).to_csv("channel_data_final.csv", index=False)
pd.DataFrame(ml_segments_only).to_csv("ml_segments_final.csv", index=False)
pd.DataFrame(nasa_labels_for_validation).to_csv("nasa_labels_validation.csv", index=False)

print("\n✅ SUCCESS! Files created:")
print("📁 channel_data_final.csv (ML data, counts match Streamlit)")
print("📁 ml_segments_final.csv (ML recurring segments)")
print("📁 nasa_labels_validation.csv (separate NASA anomalies for validation)")
