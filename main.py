# ============================================
# 🚀 NASA Recurring Unknown Anomaly Detection
# UI ENHANCED | LOGIC PRESERVED | COLORS + BG
# ============================================

import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import base64

# --------------------------------------------
# Page config
# --------------------------------------------
st.set_page_config(page_title="NASA Anomaly Detection", layout="wide")

# --------------------------------------------
# Background Image
# --------------------------------------------
def set_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/jpeg;base64,{encoded});
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg("bg.jpeg")

# --------------------------------------------
# UI Styling
# --------------------------------------------
st.markdown("""
<style>
.card {
    background: rgba(255,255,255,0.9);
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.12);
    transition: transform 0.25s ease;
    margin-bottom: 20px;
}
.card:hover {
    transform: translateY(-6px);
}
.metric {
    font-size: 34px;
    font-weight: 700;
}
.label {
    color: #6b7b8c;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------
# Dataset paths
# --------------------------------------------
DATASET_PATH = r"C:\Users\arsha\Downloads\archive (2)"
LABELED_FILE = os.path.join(DATASET_PATH, "labeled_anomalies.csv")
DATA_FOLDER = os.path.join(DATASET_PATH, "data", "data", "test")

# --------------------------------------------
# Load labeled anomaly metadata
# --------------------------------------------
@st.cache_data
def load_labeled_data():
    df = pd.read_csv(LABELED_FILE)
    df["anomaly_sequences"] = df["anomaly_sequences"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    return df

labeled_df = load_labeled_data()

# --------------------------------------------
# Sidebar
# --------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    channels = sorted(labeled_df["chan_id"].unique())
    selected_channel = st.selectbox("📡 Channel", channels)
    ROLLING_WINDOW = st.slider("🪟 Rolling Window", 10, 200, 50)
    RECURRING_THRESHOLD = st.slider("♻️ Recurring Threshold", 1, 50, 10)
    show_nasa_labels = st.checkbox("🧪 Show NASA Labeled", True)

# --------------------------------------------
# Header
# --------------------------------------------
st.markdown("""
<div class="card" style="text-align:center;">
<h2 style='color:#17becf;'>🚀 Recurring Unknown Anomaly Detection – NASA Sensor Data</h2>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------
# Load sensor data
# --------------------------------------------
@st.cache_data
def load_sensor(channel):
    return np.load(os.path.join(DATA_FOLDER, f"{channel}.npy"))

values = load_sensor(selected_channel)
channel_data = pd.DataFrame({"value": values[:, 0]})

# --------------------------------------------
# Feature engineering
# --------------------------------------------
channel_data["rolling_mean"] = channel_data["value"].rolling(ROLLING_WINDOW).mean()
channel_data["rolling_std"] = channel_data["value"].rolling(ROLLING_WINDOW).std()
channel_data["zscore"] = (channel_data["value"] - channel_data["rolling_mean"]) / channel_data["rolling_std"]
channel_data["lag_1"] = channel_data["value"].shift(1)
channel_data.dropna(inplace=True)

X = channel_data[["value", "rolling_mean", "rolling_std", "zscore", "lag_1"]]

# --------------------------------------------
# Train model
# --------------------------------------------
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

with st.spinner("🔄 Training model..."):
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring="roc_auc", n_jobs=-1)
    grid.fit(X)

channel_data["anomaly_ml"] = grid.best_estimator_.predict(X)
channel_data["anomaly_ml"] = channel_data["anomaly_ml"].map({1: 0, -1: 1})

# --------------------------------------------
# Recurring anomaly logic
# --------------------------------------------
channel_data["recurring_count"] = channel_data["anomaly_ml"].rolling(ROLLING_WINDOW * 4).sum()
channel_data["recurring_flag"] = channel_data["recurring_count"] > RECURRING_THRESHOLD

segments, start = [], None
for i in range(len(channel_data)):
    if channel_data["recurring_flag"].iloc[i] and start is None:
        start = i
    elif not channel_data["recurring_flag"].iloc[i] and start is not None:
        segments.append((start, i - 1))
        start = None
if start is not None:
    segments.append((start, len(channel_data) - 1))

# --------------------------------------------
# NASA labeled anomalies
# --------------------------------------------
channel_info = labeled_df[labeled_df["chan_id"] == selected_channel].iloc[0]
labeled_sequences = channel_info["anomaly_sequences"]

# --------------------------------------------
# Metrics
# --------------------------------------------
m1, m2 = st.columns(2)

with m1:
    st.markdown(f"""
    <div class="card">
        <div class="label" style="color:#a6cee3;">🔴 Total ML Anomalies</div>
        <div class="metric" style="color:#ff7f0e;">{channel_data["anomaly_ml"].sum()}</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    rate = len(segments) / len(channel_data) * 100
    st.markdown(f"""
    <div class="card">
        <div class="label" style="color:#b2df8a;">♻️ Recurrence Rate</div>
        <div class="metric" style="color:#1f77b4;">{rate:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# 📊 Visualization
# ============================================
col1, col2 = st.columns(2)

# 📈 Raw Signal
with col1:
    st.markdown("<div class='card'><h4 style='color:#17becf;'>📈 Raw Sensor Signal</h4>", unsafe_allow_html=True)
    fig1, ax1 = plt.subplots(figsize=(6,4))
    ax1.plot(channel_data["value"], color="#1f77b4")
    st.pyplot(fig1)
    st.markdown("</div>", unsafe_allow_html=True)

# 🚨 ML + Recurring
with col2:
    st.markdown("<div class='card'><h4 style='color:#17becf;'>🚨 ML Detected Anomalies</h4>", unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(6,4))
    ax2.plot(channel_data["value"], alpha=0.7)
    mask = channel_data["anomaly_ml"].astype(bool)
    ax2.scatter(channel_data.index[mask], channel_data["value"][mask], color="red", s=10, label="ML Anomaly")

    for i, (s, e) in enumerate(segments):
        ax2.axvspan(s, e, color="orange", alpha=0.25,
                    label="Recurring Segment" if i == 0 else None)

    ax2.legend(loc="upper right")
    st.pyplot(fig2)
    st.markdown("</div>", unsafe_allow_html=True)

# 🧪 ML vs NASA
st.markdown("<div class='card'><h4 style='color:#17becf;'>🧪 ML vs NASA Labeled Anomalies</h4>", unsafe_allow_html=True)
fig3, ax3 = plt.subplots(figsize=(14,5))
ax3.plot(channel_data["value"], alpha=0.7, label="Sensor Value")
ax3.scatter(channel_data.index[mask], channel_data["value"][mask], color="red", s=10, label="ML Anomalies")

for i, (s, e) in enumerate(segments):
    ax3.axvspan(s, e, color="orange", alpha=0.25,
                label="Recurring Anomaly" if i == 0 else None)

if show_nasa_labels:
    for i, (s, e) in enumerate(labeled_sequences):
        ax3.axvspan(s, e, color="cyan", alpha=0.25,
                    label="NASA Labeled" if i == 0 else None)

ax3.legend(loc="upper right")
st.pyplot(fig3)
st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------
# 📋 Recurring segment table
# --------------------------------------------
st.markdown("<div class='card'><h4 style='color:#17becf;'>📋 Recurring Anomaly Segments</h4>", unsafe_allow_html=True)
segment_df = pd.DataFrame(segments, columns=["Start Index", "End Index"])
segment_df["Duration"] = segment_df["End Index"] - segment_df["Start Index"]

def style_table(df):
    return df.style.set_table_styles(
        [{'selector':'th','props':[('background-color','#2c3e50'),('color','#ffffff'),('font-weight','bold')]}]
    )
st.dataframe(style_table(segment_df), use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------
# 🆕 ML-only anomalies table (not labeled by NASA)
# --------------------------------------------
st.markdown("<div class='card'><h4 style='color:#17becf;'>🆕 ML-Detected Anomalies Not Labeled by NASA</h4>", unsafe_allow_html=True)
ml_only_segments = []
for s, e in segments:
    overlaps = False
    for ns, ne in labeled_sequences:
        if not (e < ns or s > ne):
            overlaps = True
            break
    if not overlaps:
        ml_only_segments.append((s, e))
ml_only_df = pd.DataFrame(ml_only_segments, columns=["Start Index", "End Index"])
ml_only_df["Duration"] = ml_only_df["End Index"] - ml_only_df["Start Index"]
if not ml_only_df.empty:
    st.dataframe(style_table(ml_only_df), use_container_width=True)
else:
    st.markdown(
        "<span style='color:#32CD32; font-size:18px; font-weight:bold;'>✅ No ML-only anomalies found for this channel.</span>",
        unsafe_allow_html=True
    )
st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------
# 💡 Insights Section
# --------------------------------------------
st.markdown("<div class='card'><h4 style='color:#17becf;'>💡 Insights</h4>", unsafe_allow_html=True)
if ml_only_segments:
    for i, (s, e) in enumerate(ml_only_segments, 1):
        st.markdown(
            f"<span style='color:#32CD32; font-size:18px; font-weight:bold;'>**Anomaly {i}:** ML detected anomalies in indices {s}–{e}, which were not labeled by NASA.</span>",
            unsafe_allow_html=True
        )
else:
    st.markdown(
        "<span style='color:#32CD32; font-size:18px; font-weight:bold;'>All ML-detected anomalies match NASA-labeled anomalies for this channel.</span>",
        unsafe_allow_html=True
    )
st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------
# ⬇️ Download report
# --------------------------------------------
report_df = channel_data.copy()
csv = report_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download Anomaly Report (CSV)",
    data=csv,
    file_name="nasa_anomaly_report.csv",
    mime="text/csv"
)