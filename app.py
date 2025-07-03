# ────────────────────────────────────────────────────────────────────────────
# FastTrack Logistics • Streamlit Dashboard (v2)
# Robust to column-name variations in the CSV
# ────────────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# ---- Optional Plotly import with graceful fallback ------------------------
PLOTLY_OK = True
try:
    import plotly.express as px
except ModuleNotFoundError:
    PLOTLY_OK = False
    st.warning(
        "⚠️ Plotly isn’t installed. Charts will fall back to Altair.\n"
        "Ensure `plotly>=5.16.0` is in requirements.txt."
    )
import altair as alt
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FastTrack Logistics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# 1. Load & harmonise data
# ---------------------------------------------------------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # normalise column names: strip → lower → spaces→underscores
    def normalise(col):
        col = col.strip()
        col = re.sub(r"\\s+", "_", col)
        return col.lower()

    df.columns = [normalise(c) for c in df.columns]

    # down-cast object columns to category
    obj_cols = df.select_dtypes(include="object").columns
    df[obj_cols] = df[obj_cols].astype("category")
    return df


DATA_PATH = "fasttrack_survey_10k.csv"  # change if your file differs
try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"CSV file '{DATA_PATH}' not found. "
             "Place it in the same folder or update DATA_PATH.")
    st.stop()

# verify mandatory columns exist
REQUIRED = [
    "origin_city", "pct_same_day", "pct_4hr", "shipments_per_day",
    "avg_distance_km", "cur_delivery_time_hr",
    "cur_cost_aed", "delay_rate_pct", "delay_churn_pct"
]
missing = [c for c in REQUIRED if c not in df.columns]
if missing:
    st.error("Missing columns in CSV: " + ", ".join(missing))
    st.stop()

# ---------------------------------------------------------------------------
# 2. Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")

origin_opts = df["origin_city"].cat.categories.tolist()
selected_origins = st.sidebar.multiselect(
    "Origin city", options=origin_opts, default=[]
)
if selected_origins:
    df = df[df["origin_city"].isin(selected_origins)]

# ---------------------------------------------------------------------------
# 3. Dashboard Tabs
# ---------------------------------------------------------------------------
t1, t2, t3, t4 = st.tabs(
    [
        "Demand & Market Viability",
        "Operational Feasibility",
        "Financial Viability",
        "Competitive Benchmarking",
    ]
)

# ---------------------------------------------------------------------------
# Tab 1 – Demand & Market Viability
# ---------------------------------------------------------------------------
with t1:
    st.subheader("High-urgency demand clusters (K-Means)")

    cluster_cols = ["pct_same_day", "pct_4hr", "shipments_per_day"]
    kmeans = KMeans(n_clusters=4, random_state=42).fit(df[cluster_cols])
    df["demand_cluster"] = kmeans.labels_

    if PLOTLY_OK:
        fig = px.scatter_3d(
            df,
            x="pct_same_day",
            y="pct_4hr",
            z="shipments_per_day",
            color="demand_cluster",
            opacity=0.6,
            title="3-D Demand Segmentation",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        chart = (
            alt.Chart(df)
            .mark_circle(opacity=0.6)
            .encode(
                x="pct_same_day",
                y="pct_4hr",
                color="demand_cluster:N",
                size="shipments_per_day",
            )
            .properties(width="container")
        )
        st.altair_chart(chart, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 2 – Operational Feasibility
# ---------------------------------------------------------------------------
with t2:
    st.subheader("Distance vs Delivery-time Regression")

    X = df[["avg_distance_km"]]
    y = df["cur_delivery_time_hr"]
    model = LinearRegression().fit(X, y)
    pred = model.predict(X)
    r2 = model.score(X, y)

    chart_df = pd.DataFrame(
        {"avg_distance_km": X.squeeze(), "actual": y, "predicted": pred}
    )

    if PLOTLY_OK:
        fig = px.scatter(
            chart_df,
            x="avg_distance_km",
            y="actual",
            opacity=0.5,
            title=f"Delivery Time Model  (R² = {r2:.3f})",
            trendline="ols",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        base = alt.Chart(chart_df).mark_circle(opacity=0.5).encode(
            x="avg_distance_km", y="actual"
        )
        line = (
            alt.Chart(chart_df)
            .transform_regression("avg_distance_km", "actual")
            .mark_line()
            .encode(x="avg_distance_km", y="actual")
        )
        st.altair_chart(
            (base + line).properties(title=f"R² = {r2:.3f}"),
            use_container_width=True,
        )

# ---------------------------------------------------------------------------
# Tab 3 – Financial Viability
# ---------------------------------------------------------------------------
with t3:
    st.subheader("Cost-per-parcel Distribution")

    if PLOTLY_OK:
        fig = px.histogram(
            df,
            x="cur_cost_aed",
            nbins=50,
            title="Current cost per parcel (AED)",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(x=alt.X("cur_cost_aed:Q", bin=alt.Bin(maxbins=50)), y="count()")
            .properties(width="container")
        )
        st.altair_chart(chart, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 4 – Competitive Benchmarking
# ---------------------------------------------------------------------------
with t4:
    st.subheader("Delay rate vs Customer Churn")

    if PLOTLY_OK:
        fig = px.scatter(
            df,
            x="delay_rate_pct",
            y="delay_churn_pct",
            trendline="ols",
            labels={"delay_rate_pct": "Delay rate (%)", "delay_churn_pct": "Churn (%)"},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        chart = (
            alt.Chart(df)
            .mark_circle(opacity=0.6)
            .encode(x="delay_rate_pct", y="delay_churn_pct")
            .properties(width="container")
        )
        st.altair_chart(chart, use_container_width=True)

st.success("Dashboard loaded successfully!")
