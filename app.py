# FastTrack Mini-Suite – works with 9-column CSV
import streamlit as st, pandas as pd, numpy as np, re
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# ── Plotly try-import ──────────────────────────────────────────────────────
try:
    import plotly.express as px
    PLOTLY = True
except ModuleNotFoundError:
    import altair as alt
    PLOTLY = False
    st.warning("Plotly not installed – using Altair fallback.")

st.set_page_config(page_title="FastTrack Mini-Dashboard", layout="wide")

# ── Load CSV & normalise headers ───────────────────────────────────────────
@st.cache_data
def load(path="fasttrack_survey_10k.csv"):
    df = pd.read_csv(path)
    df.columns = [re.sub(r"\s+", "_", c.lower().strip()) for c in df.columns]
    return df

df = load()
st.sidebar.write("🔍 Columns:", list(df.columns))

# ── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["Demand & Market Viability",
     "Operational Feasibility",
     "Financial Viability",
     "Benchmarking"]
)

# ======================================================================== #
# Tab 1 – Market Viability (clustering only)                               #
# ======================================================================== #
with tab1:
    st.header("Demand clusters (simple K-Means)")
    # use pct_ultra_urgent if present, else fall back to pct_4hr
    urgency_col = "pct_ultra_urgent" if "pct_ultra_urgent" in df.columns else "pct_4hr"
    need = [urgency_col, "pct_same_day", "shipments_per_day"]
    if all(c in df.columns for c in need):
        km = KMeans(n_clusters=3, random_state=42).fit(df[need])
        df["cluster"] = km.labels_
        if PLOTLY:
            fig = px.scatter_3d(df, x=need[0], y=need[1], z=need[2],
                                color="cluster", opacity=0.65)
            st.plotly_chart(fig, use_container_width=True)
        else:
            chart = (alt.Chart(df).mark_circle(opacity=0.6)
                     .encode(x=need[0], y=need[1], size=need[2], color="cluster:N"))
            st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Need cols " + ", ".join(need))

# ======================================================================== #
# Tab 2 – Operational Feasibility (distance → time regression)             #
# ======================================================================== #
with tab2:
    st.header("Distance vs delivery time")
    need = {"avg_distance_km", "cur_delivery_time_hr"}
    if need.issubset(df.columns):
        X = df[["avg_distance_km"]]
        y = df["cur_delivery_time_hr"]
        lm = LinearRegression().fit(X, y)
        pred = lm.predict(X)
        st.metric("R²", f"{r2_score(y, pred):.2f}")
        st.metric("MAE (hr)", f"{mean_absolute_error(y, pred):.2f}")
        if PLOTLY:
            fig = px.scatter(df, x="avg_distance_km", y="cur_delivery_time_hr",
                             trendline="ols", opacity=0.5)
            st.plotly_chart(fig, use_container_width=True)
        else:
            scatter = alt.Chart(df).mark_circle(opacity=0.5).encode(
                x="avg_distance_km", y="cur_delivery_time_hr")
            st.altair_chart(scatter, use_container_width=True)
    else:
        st.info("avg_distance_km or cur_delivery_time_hr missing.")

# ======================================================================== #
# Tab 3 – Financial Viability (cost hist + distance→cost)                 #
# ======================================================================== #
with tab3:
    st.header("Cost metrics")
    if "cur_cost_aed" in df.columns:
        if PLOTLY:
            st.plotly_chart(px.histogram(df, x="cur_cost_aed", nbins=40,
                                         title="Cost per parcel (AED)"),
                            use_container_width=True)
        else:
            st.altair_chart(
                alt.Chart(df).mark_bar().encode(
                    x=alt.X("cur_cost_aed:Q", bin=alt.Bin(maxbins=40)), y="count()"),
                use_container_width=True)

    if {"avg_distance_km", "cur_cost_aed"}.issubset(df.columns):
        lm = LinearRegression().fit(df[["avg_distance_km"]], df["cur_cost_aed"])
        st.write(f"Coef cost/km ≈ **{lm.coef_[0]:.2f} AED**  (intercept {lm.intercept_:.2f})")

# ======================================================================== #
# Tab 4 – Benchmarking (fail % vs churn)                                   #
# ======================================================================== #
with tab4:
    st.header("Delay vs churn")
    if {"delay_rate_pct", "delay_churn_pct"}.issubset(df.columns):
        if PLOTLY:
            fig = px.scatter(df, x="delay_rate_pct", y="delay_churn_pct",
                             trendline="ols", opacity=0.6)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.altair_chart(
                alt.Chart(df).mark_circle(opacity=0.6)
                .encode(x="delay_rate_pct", y="delay_churn_pct"), use_container_width=True)
    else:
        st.info("delay_rate_pct or delay_churn_pct missing.")

st.success("Loaded minimal dashboard ✔")
