# FastTrack Logistics • Feasibility Analytics Dashboard
# ——————————————————————————————————————————————————————
# Tabs:
#   • Demand & Market Viability
#   • Operational Feasibility & AI Advantage
#   • Financial Viability
#   • Strategic Positioning & Competitive Benchmarking
#
# Libs needed (add to requirements.txt):
#   streamlit, pandas, numpy, scikit-learn, plotly, altair, mlxtend
# ——————————————————————————————————————————————————————

import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, classification_report
from sklearn.ensemble import RandomForestClassifier
from mlxtend.frequent_patterns import apriori, association_rules

# ── optional Plotly import (fallback to Altair) ────────────────────────────
try:
    import plotly.express as px

    PLOTLY = True
except ModuleNotFoundError:
    import altair as alt

    PLOTLY = False
    st.warning("Plotly not installed – charts will use Altair.")

st.set_page_config(page_title="FastTrack Logistics Feasibility Suite",
                   layout="wide", initial_sidebar_state="expanded")

# ── 1 Load & clean data ───────────────────────────────────────────────────
@st.cache_data
def load_csv(path="fasttrack_survey_10k.csv"):
    df_ = pd.read_csv(path)
    df_.columns = [re.sub(r"\s+", "_", c.lower().strip()) for c in df_.columns]
    obj = df_.select_dtypes(include="object").columns
    df_[obj] = df_[obj].astype("category")
    # derive numeric fail_rate_pct if needed
    if "fail_rate_pct" not in df_.columns and "fail_rate_band" in df_.columns:
        band_map = {"<1_%": 0.5, "1-3_%": 2, "3-5_%": 4, "5-10_%": 7.5, ">10_%": 12.5}
        df_["fail_rate_pct"] = (
            df_["fail_rate_band"].astype(str).str.replace(" ", "").map(band_map)
        )
    return df_


df = load_csv()

# ── 2 Sidebar filters ─────────────────────────────────────────────────────
st.sidebar.header("Filters")
if "city" in df.columns:
    cities = sorted(df["city"].dropna().unique())
    sel_city = st.sidebar.multiselect("City", cities, default=cities)
    df = df[df["city"].isin(sel_city)]

if "sector" in df.columns:
    sectors = sorted(df["sector"].dropna().unique())
    sel_sector = st.sidebar.multiselect("Sector", sectors, default=sectors)
    df = df[df["sector"].isin(sel_sector)]

if df.empty:
    st.error("No data after filters.")
    st.stop()

# ── 3 Tab layout ──────────────────────────────────────────────────────────
tab_mv, tab_ops, tab_fin, tab_comp = st.tabs(
    [
        "Demand & Market Viability",
        "Operational Feasibility & AI Advantage",
        "Financial Viability",
        "Strategic Positioning & Competitive Benchmarking",
    ]
)

# ==========================================================================#
# 3-A  DEMAND & MARKET VIABILITY                                            #
# ==========================================================================#
with tab_mv:
    st.header("Market-viability analytics")

    # --- Clustering: high-density, high-urgency zones ---------------------
    cols_needed = ["pct_ultra_urgent", "pct_same_day", "orders_per_week"]
    if all(c in df.columns for c in cols_needed):
        X = df[cols_needed].fillna(0)
        km = KMeans(n_clusters=4, random_state=42).fit(X)
        df["urgency_cluster"] = km.labels_

        if PLOTLY:
            fig = px.scatter_3d(
                df,
                x="pct_ultra_urgent",
                y="pct_same_day",
                z="orders_per_week",
                color="urgency_cluster",
                opacity=0.6,
                title="Urgency–density clusters",
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            import altair as alt

            chart = (
                alt.Chart(df)
                .mark_circle(opacity=0.6)
                .encode(
                    x="pct_ultra_urgent",
                    y="pct_same_day",
                    size="orders_per_week",
                    color="urgency_cluster:N",
                )
            )
            st.altair_chart(chart, use_container_width=True)

    # --- Classification: who needs urgent logistics? ----------------------
    st.subheader("Urgency-segment classifier")
    if {"sector", "pct_ultra_urgent"}.issubset(df.columns):
        df["urgent_flag"] = (df["pct_ultra_urgent"] > 10).astype(int)
        X = pd.get_dummies(df[["sector"]], drop_first=True)
        y = df["urgent_flag"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        clf = RandomForestClassifier(max_depth=5, n_estimators=200, random_state=42)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        st.code(classification_report(y_test, pred, zero_division=0))

    # --- Peak-time analysis ------------------------------------------------
    if "peak_hours" in df.columns:
        st.subheader("Peak request windows")
        peak_expanded = (
            df["peak_hours"]
            .str.split(",", expand=True)
            .stack()
            .str.strip()
            .reset_index(drop=True)
        )
        peak_df = peak_expanded.value_counts().reset_index()
        peak_df.columns = ["window", "count"]

        if PLOTLY:
            fig = px.bar(peak_df, x="window", y="count", title="Request peaks")
            st.plotly_chart(fig, use_container_width=True)
        else:
            chart = alt.Chart(peak_df).mark_bar().encode(x="window", y="count")
            st.altair_chart(chart, use_container_width=True)

# ==========================================================================#
# 3-B  OPERATIONAL FEASIBILITY & AI ADVANTAGE                                #
# ==========================================================================#
with tab_ops:
    st.header("Operational Feasibility & AI Advantage")

    # --- Regression: distance → time --------------------------------------
    if {"avg_distance_km", "curr_time_hr"}.issubset(df.columns):
        X = df[["avg_distance_km", "stops_per_route"]].fillna(0)
        y = df["curr_time_hr"]
        reg = LinearRegression().fit(X, y)
        df["pred_time_hr"] = reg.predict(X)
        r2 = r2_score(y, df["pred_time_hr"])
        mae = mean_absolute_error(y, df["pred_time_hr"])

        st.metric("R² (dist+stops→time)", f"{r2:.2f}")
        st.metric("MAE (hr)", f"{mae:.2f}")

        if PLOTLY:
            fig = px.scatter(
                df,
                x="avg_distance_km",
                y="curr_time_hr",
                color="traffic_level"
                if "traffic_level" in df.columns
                else None,
                trendline="ols",
                title="Distance vs delivery time",
            )
            st.plotly_chart(fig, use_container_width=True)

    # --- A/B simulation: AI vs traditional --------------------------------
    st.subheader("A/B: AI-optimized route vs traditional")
    if "curr_time_hr" in df.columns:
        # naive simulation: AI is 20 % faster on avg
        df["ai_time_hr"] = df["curr_time_hr"] * 0.8
        improv = 100 * (df["curr_time_hr"].mean() - df["ai_time_hr"].mean()) / df[
            "curr_time_hr"
        ].mean()
        st.write(f"**Average time saved**: {improv:.1f} % (simulated)")

        if PLOTLY:
            ab = df.melt(
                value_vars=["curr_time_hr", "ai_time_hr"],
                var_name="method",
                value_name="time_hr",
            )
            fig = px.box(ab, x="method", y="time_hr", title="Delivery time A/B")
            st.plotly_chart(fig, use_container_width=True)

    # --- Route archetype clustering ---------------------------------------
    st.subheader("Route archetypes (distance & stops)")
    if {"avg_distance_km", "stops_per_route"}.issubset(df.columns):
        scl = StandardScaler()
        feat = scl.fit_transform(df[["avg_distance_km", "stops_per_route"]].fillna(0))
        db = DBSCAN(eps=0.8, min_samples=10).fit(feat)
        df["route_type"] = db.labels_

        if PLOTLY:
            fig = px.scatter(
                df,
                x="avg_distance_km",
                y="stops_per_route",
                color="route_type",
                title="Route clusters",
            )
            st.plotly_chart(fig, use_container_width=True)

# ==========================================================================#
# 3-C  FINANCIAL VIABILITY                                                  #
# ==========================================================================#
with tab_fin:
    st.header("Financial Viability")

    if {"avg_distance_km", "curr_cost_aed", "fuel_price"}.issubset(df.columns):
        X = df[["avg_distance_km", "fuel_price"]]
        y = df["curr_cost_aed"]
        ridge = Ridge(alpha=1.0).fit(X, y)
        r2 = ridge.score(X, y)
        st.metric("R² (cost model)", f"{r2:.2f}")

        # sensitivity: vary fuel ±20 %
        base = df["fuel_price"].median()
        dist = df["avg_distance_km"].median()
        for delta in [-0.2, 0, 0.2]:
            pred_cost = ridge.predict([[dist, base * (1 + delta)]])[0]
            st.write(f"Fuel {delta:+.0%} ⇒ cost = **AED {pred_cost:.2f}**")

    if "curr_cost_aed" in df.columns:
        df["ai_cost_aed"] = df["curr_cost_aed"] * 0.9  # 10 % saving hypothesis
        saving = df["curr_cost_aed"].mean() - df["ai_cost_aed"].mean()
        st.write(f"**Avg cost saved per parcel (sim.)**: AED {saving:.2f}")

        if PLOTLY:
            cost = df.melt(
                value_vars=["curr_cost_aed", "ai_cost_aed"],
                var_name="method",
                value_name="cost",
            )
            fig = px.box(cost, x="method", y="cost", title="Cost A/B")
            st.plotly_chart(fig, use_container_width=True)

# ==========================================================================#
# 3-D  STRATEGIC POSITIONING & COMPETITIVE BENCHMARKING                      #
# ==========================================================================#
with tab_comp:
    st.header("Strategic Positioning & Competitive Benchmarking")

    # --- Classification: will AI be preferred? ---------------------------
    if {"current_provider", "overall_csat", "ai_worry"}.issubset(df.columns):
        X = pd.get_dummies(
            df[["current_provider", "ai_worry"]],
            drop_first=True,
        )
        y = (df["overall_csat"] < 6).astype(int)  # unhappy segment
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        rf = RandomForestClassifier(max_depth=6, n_estimators=200, random_state=42)
        rf.fit(X_train, y_train)
        st.code(classification_report(y_test, rf.predict(X_test), zero_division=0))

    # --- Association rule mining: pain → churn ----------------------------
    st.subheader("Pain-point association rules")
    if "top_pain" in df.columns:
        trans = (
            df["top_pain"]
            .astype(str)
            .str.get_dummies(sep=",")
            .astype(bool)
        )
        freq = apriori(trans, min_support=0.05, use_colnames=True)
        rules = association_rules(freq, metric="lift", min_threshold=1.2)
        high = rules.sort_values("lift", ascending=False).head(10)
        st.dataframe(high[["antecedents", "consequents", "support", "confidence", "lift"]])

    # --- ROI calculator ---------------------------------------------------
    st.subheader("ROI calculator (per 1 000 shipments)")
    col1, col2 = st.columns(2)
    with col1:
        cost_now = st.number_input("Current cost/parcel (AED)", 5.0, 100.0, 20.0)
        time_now = st.number_input("Current avg time (hr)", 1.0, 72.0, 12.0)
    with col2:
        cost_ai = st.number_input("AI cost/parcel (AED)", 5.0, 100.0, 18.0)
        time_ai = st.number_input("AI avg time (hr)", 1.0, 72.0, 9.6)

    saving_aed = (cost_now - cost_ai) * 1000
    time_saved = (time_now - time_ai) / time_now * 100
    st.write(f"**AED saved / 1 000 parcels:** `{saving_aed:,.0f}`")
    st.write(f"**Time saved:** `{time_saved:.1f}%`")

st.success("Dashboard loaded successfully!")
