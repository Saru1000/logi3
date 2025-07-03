# FastTrack Logistics • Streamlit Dashboard
# (auto-adapts to the 50-column fasttrack_survey_10k.csv)

import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# ── Safe Plotly import (fallback to Altair if missing) ─────────────────────
try:
    import plotly.express as px
    PLOTLY_OK = True
except ModuleNotFoundError:
    import altair as alt
    PLOTLY_OK = False
    st.warning(
        "⚠️ Plotly isn't installed. Visuals will fall back to Altair.\n"
        "Add 'plotly>=5.16.0' to requirements.txt to enable 3-D plots."
    )

# Altair fallback (used even if Plotly exists for some charts)
import altair as alt

st.set_page_config(
    page_title="FastTrack Logistics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────
# 1 Load and harmonise data
# ──────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # normalise column names: strip → lower → spaces→underscores
    def norm(col): return re.sub(r"\s+", "_", col.strip().lower())
    df.columns = [norm(c) for c in df.columns]

    # category-cast object cols for efficiency
    obj_cols = df.select_dtypes(include="object").columns
    df[obj_cols] = df[obj_cols].astype("category")
    return df


DATA_PATH = "fasttrack_survey_10k.csv"  # same folder as app.py
df = load_data(DATA_PATH)

# map fail-rate bands (strings) to numeric mid-points
BAND_MAP = {"<1_%": 0.5, "1-3_%": 2, "3-5_%": 4, "5-10_%": 7.5, ">10_%": 12.5}
df["fail_rate_pct"] = (
    df["fail_rate_band"].astype(str).str.replace(" ", "").map(BAND_MAP).astype(float)
)

# ──────────────────────────────────────────────────────────────────────────
# 2 Sidebar filters
# ──────────────────────────────────────────────────────────────────────────
st.sidebar.header("Filters")

city_opts = sorted(df["city"].dropna().unique())
chosen_cities = st.sidebar.multiselect("City", options=city_opts, default=city_opts)
df_filt = df[df["city"].isin(chosen_cities)]

sector_opts = sorted(df["sector"].dropna().unique())
chosen_sectors = st.sidebar.multiselect(
    "Sector", options=sector_opts, default=sector_opts
)
df_filt = df_filt[df_filt["sector"].isin(chosen_sectors)]

if df_filt.empty:
    st.warning("No data for selected filters.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────
# 3 Tabs
# ──────────────────────────────────────────────────────────────────────────
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
    kmeans = KMeans(n_clusters=4, random_state=42).fit(df_filt[cluster_cols])
    df_filt["demand_cluster"] = kmeans.labels_

    if PLOTLY_OK:
        fig = px.scatter_3d(
            df_filt,
            x="pct_ultra_urgent",
            y="pct_same_day",
            z="orders_per_week",
            color="demand_cluster",
            opacity=0.65,
            title="3-D Urgency Segmentation",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        chart = (
            alt.Chart(df_filt)
            .mark_circle(opacity=0.6)
            .encode(
                x="pct_ultra_urgent",
                y="pct_same_day",
                size="orders_per_week",
                color="demand_cluster:N",
            )
        )
        st.altair_chart(chart, use_container_width=True)

# ── Tab 2 – Operational Feasibility ───────────────────────────────────────
with tab_ops:
    st.subheader("Distance vs Actual delivery time")

    X = df_filt[["avg_distance_km"]]
    y = df_filt["curr_time_hr"]
    if len(X) > 20:
        lm = LinearRegression().fit(X, y)
        r2 = lm.score(X, y)
        df_plot = df_filt.assign(pred=lm.predict(X))

        if PLOTLY_OK:
            fig = px.scatter(
                df_plot,
                x="avg_distance_km",
                y="curr_time_hr",
                trendline="ols",
                color="traffic_level",
                opacity=0.5,
                title=f"Distance vs Time (R² {r2:.2f})",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            base = (
                alt.Chart(df_plot)
                .mark_circle(opacity=0.5)
                .encode(x="avg_distance_km", y="curr_time_hr", color="traffic_level")
            )
            line = (
                alt.Chart(df_plot)
                .transform_regression("avg_distance_km", "curr_time_hr")
                .mark_line()
                .encode(x="avg_distance_km", y="curr_time_hr")
            )
            st.altair_chart((base + line), use_container_width=True)
    else:
        st.info("Not enough data for regression in current filter.")

# ── Tab 3 – Financial Viability ───────────────────────────────────────────
with tab_fin:
    st.subheader("Cost-per-parcel distribution")

    if PLOTLY_OK:
        fig = px.histogram(
            df_filt, x="curr_cost_aed", nbins=50, title="Current cost per parcel (AED)"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        chart = (
            alt.Chart(df_filt)
            .mark_bar()
            .encode(x=alt.X("curr_cost_aed:Q", bin=alt.Bin(maxbins=50)), y="count()")
        )
        st.altair_chart(chart, use_container_width=True)

# ── Tab 4 – Competitive Benchmarking ──────────────────────────────────────
with tab_comp:
    st.subheader("Failure rate vs Delay-induced churn")

    if PLOTLY_OK:
        fig = px.scatter(
            df_filt,
            x="fail_rate_pct",
            y="delay_churn",
            color="current_provider",
            trendline="ols",
            labels={"fail_rate_pct": "Failure rate (%)", "delay_churn": "Churn score"},
            title="Fail % vs Churn",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        chart = (
            alt.Chart(df_filt)
            .mark_circle(opacity=0.6)
            .encode(
                x="fail_rate_pct",
                y="delay_churn",
                color="current_provider",
            )
        )
        st.altair_chart(chart, use_container_width=True)

st.success("Dashboard loaded successfully!")
