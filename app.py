# FastTrack Logistics • Streamlit Dashboard (v4)
# Rich visuals + robust fallbacks

import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# ── Plotly (Altair fallback) ───────────────────────────────────────────────
try:
    import plotly.express as px
    PLOTLY = True
except ModuleNotFoundError:
    PLOTLY = False
    st.warning(
        "⚠️ Plotly not installed – using Altair fallback. "
        "Add 'plotly>=5.16.0' to requirements.txt for richer visuals."
    )
import altair as alt

st.set_page_config(
    page_title="FastTrack Logistics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 1 Load & normalise data ───────────────────────────────────────────────
@st.cache_data
def load(path="fasttrack_survey_10k.csv"):
    df = pd.read_csv(path)
    df.columns = [re.sub(r"\s+", "_", c.strip().lower()) for c in df.columns]
    cat = df.select_dtypes(include="object").columns
    df[cat] = df[cat].astype("category")
    return df


df = load()

# Numeric fail-rate column
if "fail_rate_pct" not in df.columns:
    if "fail_rate_band" in df.columns:
        band_map = {
            "<1_%": 0.5,
            "1-3_%": 2.0,
            "3-5_%": 4.0,
            "5-10_%": 7.5,
            ">10_%": 12.5,
        }
        df["fail_rate_pct"] = (
            df["fail_rate_band"].astype(str).str.replace(" ", "").map(band_map)
        )
    else:
        df["fail_rate_pct"] = np.nan

# ── 2 Sidebar filters ─────────────────────────────────────────────────────
st.sidebar.title("Filters")

if "city" in df.columns:
    cities = sorted(df["city"].dropna().unique())
    sel_city = st.sidebar.multiselect("City", cities, default=cities)
    df = df[df["city"].isin(sel_city)]

if "sector" in df.columns:
    sectors = sorted(df["sector"].dropna().unique())
    sel_sector = st.sidebar.multiselect("Sector", sectors, default=sectors)
    df = df[df["sector"].isin(sel_sector)]

if df.empty:
    st.error("No data after applying filters.")
    st.stop()

# ── 3 Tab layout ──────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Demand & Market Viability",
        "Operational Feasibility",
        "Financial Viability",
        "Competitive Benchmarking",
    ]
)

# ──────────────────────────────────────────────────────────────────────────
# TAB 1 – Demand & Market Viability
# ──────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("High-urgency demand clusters")

    clust_cols = ["pct_ultra_urgent", "pct_same_day", "orders_per_week"]
    if all(c in df.columns for c in clust_cols):
        data = df[clust_cols].dropna()
        if len(data) >= 10:
            km = KMeans(n_clusters=4, random_state=1).fit(data)
            df["cluster"] = km.labels_

            if PLOTLY:
                fig = px.scatter_3d(
                    df,
                    x=clust_cols[0],
                    y=clust_cols[1],
                    z=clust_cols[2],
                    color="cluster",
                    opacity=0.65,
                    title="3-D urgency segmentation",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                chart = (
                    alt.Chart(df)
                    .mark_circle(opacity=0.6)
                    .encode(
                        x=clust_cols[0],
                        y=clust_cols[1],
                        size=clust_cols[2],
                        color="cluster:N",
                    )
                )
                st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Not enough rows for clustering.")
    else:
        st.info("CSV lacks: " + ", ".join(clust_cols))

    st.subheader("Same-day demand by city")
    if {"city", "pct_same_day"}.issubset(df.columns):
        by_city = df.groupby("city")["pct_same_day"].mean().reset_index()
        if PLOTLY:
            bar = px.bar(
                by_city, x="city", y="pct_same_day", title="Avg same-day demand (%)"
            )
            st.plotly_chart(bar, use_container_width=True)
        else:
            chart = alt.Chart(by_city).mark_bar().encode(x="city", y="pct_same_day")
            st.altair_chart(chart, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────
# TAB 2 – Operational Feasibility
# ──────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("Operational performance")

    if {"avg_distance_km", "curr_time_hr"}.issubset(df.columns):
        X = df[["avg_distance_km"]].values
        y = df["curr_time_hr"].values
        reg = LinearRegression().fit(X, y)
        r2 = reg.score(X, y)
        df["pred_time"] = reg.predict(X)

        if PLOTLY:
            fig = px.scatter(
                df,
                x="avg_distance_km",
                y="curr_time_hr",
                trendline="ols",
                color="traffic_level"
                if "traffic_level" in df.columns
                else None,
                opacity=0.5,
                title=f"Distance vs Time (R²={r2:.2f})",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            base = (
                alt.Chart(df)
                .mark_circle(opacity=0.5)
                .encode(x="avg_distance_km", y="curr_time_hr")
            )
            regline = (
                alt.Chart(df)
                .transform_regression("avg_distance_km", "curr_time_hr")
                .mark_line()
            )
            st.altair_chart(base + regline, use_container_width=True)

    st.subheader("Stops per route distribution")
    if "stops_per_route" in df.columns:
        if PLOTLY:
            hist = px.histogram(
                df,
                x="stops_per_route",
                nbins=30,
                title="Stops per route distribution",
            )
            st.plotly_chart(hist, use_container_width=True)
        else:
            chart = (
                alt.Chart(df)
                .mark_bar()
                .encode(
                    x=alt.X("stops_per_route:Q", bin=alt.Bin(maxbins=30)), y="count()"
                )
            )
            st.altair_chart(chart, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────
# TAB 3 – Financial Viability
# ──────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("Financial metrics")

    if "curr_cost_aed" in df.columns:
        if PLOTLY:
            fig = px.violin(
                df,
                y="curr_cost_aed",
                box=True,
                points="all",
                title="Current cost per parcel (AED)",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            chart = alt.Chart(df).mark_violin().encode(y="curr_cost_aed")
            st.altair_chart(chart, use_container_width=True)

    if {"avg_distance_km", "curr_cost_aed"}.issubset(df.columns):
        if PLOTLY:
            fig = px.scatter(
                df,
                x="avg_distance_km",
                y="curr_cost_aed",
                trendline="ols",
                opacity=0.5,
                title="Cost vs Distance",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            scatter = (
                alt.Chart(df)
                .mark_circle(opacity=0.5)
                .encode(x="avg_distance_km", y="curr_cost_aed")
            )
            st.altair_chart(scatter, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────
# TAB 4 – Competitive Benchmarking
# ──────────────────────────────────────────────────────────────────────────
with tab4:
    st.header("Competitive insights")

    if {"fail_rate_pct", "delay_churn"}.issubset(df.columns):
        if PLOTLY:
            fig = px.scatter(
                df,
                x="fail_rate_pct",
                y="delay_churn",
                color="current_provider"
                if "current_provider" in df.columns
                else None,
                trendline="ols",
                opacity=0.6,
                title="Fail rate vs Delay-induced churn",
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

    if {"current_provider", "overall_csat"}.issubset(df.columns):
        csat = df.groupby("current_provider")["overall_csat"].mean().reset_index()
        if PLOTLY:
            fig = px.bar(csat, x="current_provider", y="overall_csat", title="Avg CSAT")
            st.plotly_chart(fig, use_container_width=True)
        else:
            chart = (
                alt.Chart(csat).mark_bar().encode(x="current_provider", y="overall_csat")
            )
            st.altair_chart(chart, use_container_width=True)

st.success("Dashboard loaded successfully!")
