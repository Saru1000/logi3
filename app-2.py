# --- Safe import for Plotly: auto-install if missing ------------------------
import importlib.util
import subprocess
import sys
import streamlit as st

if importlib.util.find_spec("plotly") is None:
    st.warning("Plotly not found. Installing now… (this may take ~30 s)")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly>=5.16.0"])
    except subprocess.CalledProcessError:
        st.error(
            "Automatic installation failed.\n"
            "Add 'plotly>=5.16.0' to requirements.txt and redeploy."
        )
        st.stop()

import plotly.express as px
# ---------------------------------------------------------------------------


# ────────────────────────────────────────────────────────────────────────────
# Rest of your original dashboard code starts here
# (Everything that was already in app.py follows unchanged)
# ────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import datetime as dt
import altair as alt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import streamlit as st

st.set_page_config(
    page_title="FastTrack Logistics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # quick type optimisation
    cat_cols = df.select_dtypes(include="object").columns
    df[cat_cols] = df[cat_cols].astype("category")
    return df


DATA_PATH = "fasttrack_survey_synthetic.csv"  # adjust if needed
df = load_data(DATA_PATH)

# Sidebar filters (example)
st.sidebar.header("Filters")
origins = st.sidebar.multiselect(
    "Origin city", options=df["origin_city"].cat.categories, default=[]
)
if origins:
    df = df[df["origin_city"].isin(origins)]

# ---------------------------------------------------------------------------
# 2. Tabs
# ---------------------------------------------------------------------------
t1, t2, t3, t4 = st.tabs(
    [
        "Demand & Market Viability",
        "Operational Feasibility",
        "Financial Viability",
        "Competitive Benchmarking",
    ]
)

# --- Tab 1 — Demand & Market Viability -------------------------------------
with t1:
    st.subheader("High-urgency demand clusters")
    cluster_cols = ["pct_same_day", "pct_4hr", "shipments_per_day"]
    kmeans = KMeans(n_clusters=4, random_state=42).fit(df[cluster_cols])
    df["demand_cluster"] = kmeans.labels_

    fig = px.scatter_3d(
        df,
        x="pct_same_day",
        y="pct_4hr",
        z="shipments_per_day",
        color="demand_cluster",
        title="3-D demand segmentation",
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Tab 2 — Operational Feasibility ---------------------------------------
with t2:
    st.subheader("Distance vs delivery time regression")
    X = df[["avg_distance_km"]]
    y = df["cur_delivery_time_hr"]
    model = LinearRegression().fit(X, y)
    pred = model.predict(X)

    chart_df = pd.DataFrame(
        {"Distance (km)": X.squeeze(), "Actual hrs": y, "Predicted hrs": pred}
    )
    fig = px.scatter(
        chart_df,
        x="Distance (km)",
        y="Actual hrs",
        opacity=0.5,
        title="Delivery Time Model",
        trendline="ols",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.write(f"R² = **{model.score(X, y):.3f}**")

# --- Tab 3 — Financial Viability -------------------------------------------
with t3:
    st.subheader("Cost-per-km distribution")
    fig = px.histogram(
        df,
        x="cur_cost_aed",
        nbins=50,
        title="Current cost per parcel (AED)",
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Tab 4 — Competitive Benchmarking --------------------------------------
with t4:
    st.subheader("Churn vs delay rate")
    fig = px.scatter(
        df,
        x="delay_rate_pct",
        y="delay_churn_pct",
        trendline="ols",
        labels={"delay_rate_pct": "Delay rate (%)", "delay_churn_pct": "Churn (%)"},
    )
    st.plotly_chart(fig, use_container_width=True)

st.success("Dashboard loaded successfully!")
