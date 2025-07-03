# FastTrack Logistics • Streamlit Dashboard (v3)
# Robust handling of optional columns, including fail_rate_band / fail_rate_pct

import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# ── Plotly import (Altair fallback) ────────────────────────────────────────
try:
    import plotly.express as px
    PLOTLY_OK = True
except ModuleNotFoundError:
    PLOTLY_OK = False
    st.warning("⚠️ Plotly not installed – using Altair fallback.")

import altair as alt

st.set_page_config(
    page_title="FastTrack Logistics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 1 Load & normalise data ───────────────────────────────────────────────
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # standardise headers → lower_snake_case
    df.columns = [re.sub(r"\s+", "_", c.strip().lower()) for c in df.columns]

    # convert object cols to category
    obj_cols = df.select_dtypes(include="object").columns
    df[obj_cols] = df[obj_cols].astype("category")
    return df


DATA_PATH = "fasttrack_survey_10k.csv"
df = load_data(DATA_PATH)

# ── 2 Create numeric fail_rate_pct no matter what ─────────────────────────
if "fail_rate_pct" in df.columns:
    df["fail_rate_pct"] = df["fail_rate_pct"].astype(float)

elif "fail_rate_band" in df.columns:
    BAND_MAP = {
        "<1_%": 0.5,
        "1-3_%": 2.0,
        "3-5_%": 4.0,
        "5-10_%": 7.5,
        ">10_%": 12.5,
    }
    df["fail_rate_pct"] = (
        df["fail_rate_band"].astype(str).str.replace(" ", "").map(BAND_MAP)
    )
else:
    df["fail_rate_pct"] = np.nan  # placeholder so later charts don’t crash

# ── 3 Sidebar filters ─────────────────────────────────────────────────────
st.sidebar.header("Filters")

if "city" in df.columns:
    choices = sorted(df["city"].dropna().unique())
    sel = st.sidebar.multiselect("City", choices, default=choices)
    df = df[df["city"].isin(sel)]

if "sector" in df.columns:
    choices = sorted(df["sector"].dropna().unique())
    sel = st.sidebar.multiselect("Sector", choices, default=choices)
    df = df[df["sector"].isin(sel)]

if df.empty:
    st.error("No data after applying filters.")
    st.stop()

# ── 4 Tab layout ──────────────────────────────────────────────────────────
tab_mv, tab_ops, tab_fin, tab_comp = st.tabs(
    [
        "Demand & Market Viability",
        "Operational Feasibility",
        "Financial Viability",
        "Competitive Benchmarking",
    ]
)

# ── Tab 1 – Demand & Market Viability ─────────────────────────────────────
with tab_mv:
    st.subheader("High-urgency demand clusters")

    cluster_cols = ["pct_ultra_urgent", "pct_same_day", "orders_per_week"]
    if all(c in df.columns for c in cluster_cols):
        kmeans = KMeans(n_clusters=4, random_state=42).fit(df[cluster_cols])
        df["demand_cluster"] = kmeans.labels_

        if PLOTLY_OK:
            fig = px.scatter_3d(
                df,
                x=cluster_cols[0],
                y=cluster_cols[1],
                z=cluster_cols[2],
                color="demand_cluster",
                opacity=0.65,
                title="3-D Urgency Segmentation",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            chart = (
                alt.Chart(df)
                .mark_circle(opacity=0.6)
                .encode(
                    x=cluster_cols[0],
                    y=cluster_cols[1],
                    size=cluster_cols[2],
                    color="demand_cluster:N",
                )
            )
            st.altair_chart(chart, use_container_width=True)
    else:
        st.info("CSV lacks one of: pct_ultra_urgent, pct_same_day, orders_per_week.")

# ── Tab 2 – Operational Feasibility ───────────────────────────────────────
with tab_ops:
    st.subheader("Distance vs Actual delivery time")

    if {"avg_distance_km", "curr_time_hr"}.issubset(df.columns):
        X = df[["avg_distance_km"]]
        y = df["curr_time_hr"]
        if len(X) > 20:
            lr = LinearRegression().fit(X, y)
            r2 = lr.score(X, y)
            df_plot = df.assign(pred=lr.predict(X))

            if PLOTLY_OK:
                fig = px.scatter(
                    df_plot,
                    x="avg_distance_km",
                    y="curr_time_hr",
                    trendline="ols",
                    color="traffic_level"
                    if "traffic_level" in df.columns
                    else None,
                    opacity=0.5,
                    title=f"Distance vs Time (R² {r2:.2f})",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                base = (
                    alt.Chart(df_plot)
                    .mark_circle(opacity=0.5)
                    .encode(x="avg_distance_km", y="curr_time_hr")
                )
                reg = (
                    alt.Chart(df_plot)
                    .transform_regression("avg_distance_km", "curr_time_hr")
                    .mark_line()
                    .encode(x="avg_distance_km", y="curr_time_hr")
                )
                st.altair_chart(base + reg, use_container_width=True)
        else:
            st.info("Insufficient rows for regression in current filter.")
    else:
        st.info("CSV lacks avg_distance_km or curr_time_hr.")

# ── Tab 3 – Financial Viability ───────────────────────────────────────────
with tab_fin:
    st.subheader("Cost-per-parcel distribution")

    if "curr_cost_aed" in df.columns:
        if PLOTLY_OK:
            fig = px.histogram(
                df, x="curr_cost_aed", nbins=50, title="Current cost per parcel (AED)"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            chart = (
                alt.Chart(df)
                .mark_bar()
                .encode(x=alt.X("curr_cost_aed:Q", bin=alt.Bin(maxbins=50)), y="count()")
            )
            st.altair_chart(chart, use_container_width=True)
    else:
        st.info("CSV lacks curr_cost_aed.")

# ── Tab 4 – Competitive Benchmarking ──────────────────────────────────────
with tab_comp:
    st.subheader("Failure rate vs Delay-induced churn")

    if {"fail_rate_pct", "delay_churn"}.issubset(df.columns):
        if PLOTLY_OK:
            fig = px.scatter(
                df,
                x="fail_rate_pct",
                y="delay_churn",
                color="current_provider"
                if "current_provider" in df.columns
                else None,
                trendline="ols",
                labels={
                    "fail_rate_pct": "Failure rate (%)",
                    "delay_churn": "Churn score",
                },
                title="Fail % vs Churn",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            chart = (
                alt.Chart(df)
                .mark_circle(opacity=0.6)
                .encode(
                    x="fail_rate_pct",
                    y="delay_churn",
                    color="current_provider"
                    if "current_provider" in df.columns
                    else alt.value("steelblue"),
                )
            )
            st.altair_chart(chart, use_container_width=True)
    else:
        st.info("CSV lacks fail_rate_pct or delay_churn.")

st.success("Dashboard loaded successfully!")
