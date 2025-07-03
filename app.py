# FastTrack Logistics • Mini-Dashboard (trendline-safe)
# Works with 9-column CSV and no statsmodels dependency
# ───────────────────────────────────────────────────────
import streamlit as st, pandas as pd, numpy as np, re
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# ---------- Plotly (optional) ----------
try:
    import plotly.express as px
    PLOTLY = True
    # Check if statsmodels is available for px trendlines
    try:
        import statsmodels.api as sm  # noqa: F401
        TREND_OK = True
    except ModuleNotFoundError:
        TREND_OK = False
except ModuleNotFoundError:
    import altair as alt
    PLOTLY, TREND_OK = False, False
    st.warning("Plotly not installed – charts fall back to Altair.")

st.set_page_config(page_title="FastTrack Mini-Dashboard", layout="wide")

# ---------- Load CSV ----------
@st.cache_data
def load(path="fasttrack_survey_10k.csv"):
    df = pd.read_csv(path)
    df.columns = [re.sub(r"\s+", "_", c.lower().strip()) for c in df.columns]
    return df

df = load()
st.sidebar.write("🔍 **Columns:**", list(df.columns))

# ---------- Tabs ----------
tab1, tab2, tab3, tab4 = st.tabs(
    ["Market Viability", "Operational Feasibility", "Financial Viability", "Benchmark"]
)

# ==================================================================== #
# Tab 1 – Market Viability                                             #
# ==================================================================== #
with tab1:
    st.header("Demand clusters (K-Means)")
    need = ["pct_4hr", "pct_same_day", "shipments_per_day"]
    if all(c in df.columns for c in need) and len(df) >= 10:
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
        st.info(f"Need cols {need}")
        st.dataframe(df.head())

# ==================================================================== #
# Tab 2 – Operational Feasibility                                      #
# ==================================================================== #
with tab2:
    st.header("Distance vs delivery time")

    if {"avg_distance_km", "cur_delivery_time_hr"}.issubset(df.columns):
        X = df[["avg_distance_km"]]
        y = df["cur_delivery_time_hr"]
        lm = LinearRegression().fit(X, y)
        pred = lm.predict(X)
        st.metric("R²", f"{r2_score(y, pred):.2f}")
        st.metric("MAE (h)", f"{mean_absolute_error(y, pred):.2f}")

        if PLOTLY:
            fig = px.scatter(
                df, x="avg_distance_km", y="cur_delivery_time_hr",
                opacity=0.5,
                trendline="ols" if TREND_OK else None,
                title="Distance vs time" + ("" if TREND_OK else "  (no trend-line)")
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            chart = alt.Chart(df).mark_circle(opacity=0.5).encode(
                x="avg_distance_km", y="cur_delivery_time_hr")
            st.altair_chart(chart, use_container_width=True)
    else:
        st.info("avg_distance_km or cur_delivery_time_hr missing.")

# ==================================================================== #
# Tab 3 – Financial Viability                                          #
# ==================================================================== #
with tab3:
    st.header("Cost vs distance")

    if {"avg_distance_km", "cur_cost_aed"}.issubset(df.columns):
        if PLOTLY:
            fig = px.scatter(
                df, x="avg_distance_km", y="cur_cost_aed",
                opacity=0.5,
                trendline="ols" if TREND_OK else None,
                title="Cost vs distance" + ("" if TREND_OK else "  (no trend-line)")
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            chart = alt.Chart(df).mark_circle(opacity=0.5).encode(
                x="avg_distance_km", y="cur_cost_aed")
            st.altair_chart(chart, use_container_width=True)
    else:
        st.info("avg_distance_km or cur_cost_aed missing.")

# ==================================================================== #
# Tab 4 – Benchmarking                                                 #
# ==================================================================== #
with tab4:
    st.header("Delay vs churn")

    if {"delay_rate_pct", "delay_churn_pct"}.issubset(df.columns):
        if PLOTLY:
            fig = px.scatter(
                df, x="delay_rate_pct", y="delay_churn_pct",
                trendline="ols" if TREND_OK else None,
                opacity=0.6,
                title="Delay % vs churn" + ("" if TREND_OK else "  (no trend-line)")
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            chart = alt.Chart(df).mark_circle(opacity=0.6).encode(
                x="delay_rate_pct", y="delay_churn_pct")
            st.altair_chart(chart, use_container_width=True)
    else:
        st.info("delay_rate_pct or delay_churn_pct missing.")

st.success("Dashboard loaded ✔ (trend-line steps disabled if statsmodels missing)")
