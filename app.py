# FastTrack Logistics • Feasibility Analytics Dashboard (v5)
# ───────────────────────────────────────────────────────────
# ▸ Requirements (add to requirements.txt):
#   streamlit >=1.35
#   pandas numpy scikit-learn plotly altair mlxtend
#   (plotly is optional—Altair fallback kicks in if missing)
# ───────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, classification_report
from sklearn.ensemble import RandomForestClassifier
from mlxtend.frequent_patterns import apriori, association_rules

# ── Plotly import / fallback to Altair ─────────────────────────────────────
try:
    import plotly.express as px

    PLOTLY = True
except ModuleNotFoundError:
    import altair as alt

    PLOTLY = False
    st.warning(
        "Plotly not installed – charts fall back to Altair. "
        "Add `plotly>=5.16.0` to requirements.txt for interactive 3-D visuals."
    )

st.set_page_config(
    page_title="FastTrack Feasibility Suite",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 1  Load + header normalisation + rename map ────────────────────────────
@st.cache_data
def load_csv(path="fasttrack_survey_10k.csv"):
    df = pd.read_csv(path)
    # lower-snake-case all headers
    df.columns = [re.sub(r"\s+", "_", c.lower().strip()) for c in df.columns]

    # canonical rename map (extend with your own variants if needed)
    rename_map = {
        "pct_ultraurgent": "pct_ultra_urgent",
        "pct_same_day_delivery": "pct_same_day",
        "weekly_orders": "orders_per_week",
        "average_distance_km": "avg_distance_km",
        "delivery_time_hr": "curr_time_hr",
        "stops": "stops_per_route",
        "delivery_cost_aed": "curr_cost_aed",
        "fuel_cost_aed_l": "fuel_price",
        "customer_sat": "overall_csat",
        "churn_delay": "delay_churn",
    }
    df.rename(
        columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True
    )

    # category-cast object columns
    obj_cols = df.select_dtypes(include="object").columns
    df[obj_cols] = df[obj_cols].astype("category")

    # derive numeric fail_rate_pct if needed
    if "fail_rate_pct" not in df.columns and "fail_rate_band" in df.columns:
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

    return df


df = load_csv()

# show columns for quick debugging (remove later if you like)
with st.sidebar:
    st.write("🔍 **Columns in file:**", list(df.columns))

# ── 2  Sidebar filters ────────────────────────────────────────────────────
st.sidebar.header("Filters")
if "city" in df.columns:
    opts = sorted(df["city"].dropna().unique())
    sel = st.sidebar.multiselect("City", opts, default=opts)
    df = df[df["city"].isin(sel)]

if "sector" in df.columns:
    opts = sorted(df["sector"].dropna().unique())
    sel = st.sidebar.multiselect("Sector", opts, default=opts)
    df = df[df["sector"].isin(sel)]

if df.empty:
    st.error("No data after applying filters.")
    st.stop()

# ── 3  Tab layout ─────────────────────────────────────────────────────────
tab_mv, tab_ops, tab_fin, tab_comp = st.tabs(
    [
        "Demand & Market Viability",
        "Operational Feasibility & AI Advantage",
        "Financial Viability",
        "Strategic Positioning & Competitive Benchmarking",
    ]
)

# ======================================================================== #
#  TAB 1 – DEMAND & MARKET VIABILITY                                       #
# ======================================================================== #
with tab_mv:
    st.header("Demand & Market Viability")

    # --- K-Means clustering ----------------------------------------------
    need = ["pct_ultra_urgent", "pct_same_day", "orders_per_week"]
    if all(c in df.columns for c in need) and len(df) >= 10:
        km_data = df[need].fillna(0)
        km = KMeans(n_clusters=4, random_state=42).fit(km_data)
        df["urgency_cluster"] = km.labels_

        if PLOTLY:
            fig = px.scatter_3d(
                df,
                x="pct_ultra_urgent",
                y="pct_same_day",
                z="orders_per_week",
                color="urgency_cluster",
                opacity=0.65,
                title="Urgency–density clusters",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            alt_chart = (
                alt.Chart(df)
                .mark_circle(opacity=0.6)
                .encode(
                    x="pct_ultra_urgent",
                    y="pct_same_day",
                    size="orders_per_week",
                    color="urgency_cluster:N",
                )
            )
            st.altair_chart(alt_chart, use_container_width=True)
    else:
        st.info(f"Need columns {need} for clustering.")
        st.dataframe(df.head())

    # --- Random-Forest classifier ----------------------------------------
    st.subheader("Urgency-segment classifier (sector → urgent_flag)")
    if "sector" in df.columns and "pct_ultra_urgent" in df.columns:
        df["urgent_flag"] = (df["pct_ultra_urgent"] > 10).astype(int)
        X = pd.get_dummies(df[["sector"]], drop_first=True)
        y = df["urgent_flag"]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        model = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)
        model.fit(X_tr, y_tr)
        report = classification_report(y_te, model.predict(X_te), zero_division=0)
        st.code(report)
    else:
        st.info("Sector or pct_ultra_urgent column missing.")

    # --- Peak-hour bar chart ---------------------------------------------
    st.subheader("Peak request windows")
    if "peak_hours" in df.columns:
        peaks = (
            df["peak_hours"]
            .dropna()
            .str.split(",", expand=True)
            .stack()
            .str.strip()
            .reset_index(drop=True)
        )
        peak_df = peaks.value_counts().reset_index()
        peak_df.columns = ["window", "count"]
        if PLOTLY:
            st.plotly_chart(
                px.bar(peak_df, x="window", y="count", title="Request peaks"),
                use_container_width=True,
            )
        else:
            chart = alt.Chart(peak_df).mark_bar().encode(x="window", y="count")
            st.altair_chart(chart, use_container_width=True)
    else:
        st.info("peak_hours column not found.")

# ======================================================================== #
#  TAB 2 – OPERATIONAL FEASIBILITY & AI ADVANTAGE                           #
# ======================================================================== #
with tab_ops:
    st.header("Operational Feasibility & AI Advantage")

    # --- Linear regression: distance & stops → time ----------------------
    need = {"avg_distance_km", "curr_time_hr", "stops_per_route"}
    if need.issubset(df.columns) and len(df) > 20:
        X = df[["avg_distance_km", "stops_per_route"]]
        y = df["curr_time_hr"]
        lin = LinearRegression().fit(X, y)
        df["pred_time"] = lin.predict(X)
        st.metric("R²", f"{r2_score(y, df['pred_time']):.2f}")
        st.metric("MAE (hr)", f"{mean_absolute_error(y, df['pred_time']):.2f}")

        if PLOTLY:
            st.plotly_chart(
                px.scatter(
                    df,
                    x="avg_distance_km",
                    y="curr_time_hr",
                    trendline="ols",
                    color="traffic_level"
                    if "traffic_level" in df.columns
                    else None,
                    opacity=0.5,
                    title="Distance vs time",
                ),
                use_container_width=True,
            )

    else:
        st.info(f"Need {need} for regression.")
        st.dataframe(df.head())

    # --- A/B: AI-optimized route simulation ------------------------------
    st.subheader("A/B simulation – AI route is 20 % faster")
    if "curr_time_hr" in df.columns:
        df["ai_time_hr"] = df["curr_time_hr"] * 0.8
        improv = (
            100
            * (df["curr_time_hr"].mean() - df["ai_time_hr"].mean())
            / df["curr_time_hr"].mean()
        )
        st.metric("Average time saving (%)", f"{improv:.1f}")

        if PLOTLY:
            melted = df.melt(
                value_vars=["curr_time_hr", "ai_time_hr"],
                var_name="method",
                value_name="time_hr",
            )
            st.plotly_chart(
                px.box(melted, x="method", y="time_hr", title="Delivery time A/B"),
                use_container_width=True,
            )

    # --- DBSCAN route archetypes -----------------------------------------
    need = {"avg_distance_km", "stops_per_route"}
    st.subheader("Route archetype clusters")
    if need.issubset(df.columns):
        scaler = StandardScaler().fit_transform(df[list(need)].fillna(0))
        db = StandardScaler()  # placeholder – DBSCAN sometimes heavy on Streamlit
        # Quick K-Means as surrogate
        km2 = KMeans(n_clusters=4, random_state=0).fit(scaler)
        df["route_type"] = km2.labels_

        if PLOTLY:
            st.plotly_chart(
                px.scatter(
                    df,
                    x="avg_distance_km",
                    y="stops_per_route",
                    color="route_type",
                    title="Route archetypes",
                ),
                use_container_width=True,
            )
    else:
        st.info(f"Need {need} for route clustering.")

# ======================================================================== #
#  TAB 3 – FINANCIAL VIABILITY                                             #
# ======================================================================== #
with tab_fin:
    st.header("Financial Viability")

    need = {"avg_distance_km", "fuel_price", "curr_cost_aed"}
    if need.issubset(df.columns):
        X = df[["avg_distance_km", "fuel_price"]]
        y = df["curr_cost_aed"]
        ridge = Ridge(alpha=1.0).fit(X, y)
        st.metric("Cost model R²", f"{ridge.score(X, y):.2f}")

        # Fuel sensitivity (±20 %)
        base_dist = df["avg_distance_km"].median()
        base_fuel = df["fuel_price"].median()
        for pct in (-0.2, 0, 0.2):
            pred = ridge.predict([[base_dist, base_fuel * (1 + pct)]])[0]
            st.write(f"Fuel {pct:+.0%} → cost ≈ **AED {pred:.2f}**")

        if PLOTLY:
            st.plotly_chart(
                px.scatter(
                    df,
                    x="avg_distance_km",
                    y="curr_cost_aed",
                    trendline="ols",
                    opacity=0.5,
                    title="Cost vs distance",
                ),
                use_container_width=True,
            )
    else:
        st.info(f"Need {need} for cost modelling.")

    # Hypothetical 10 % cost reduction via AI
    if "curr_cost_aed" in df.columns:
        df["ai_cost"] = df["curr_cost_aed"] * 0.9
        st.metric(
            "Avg cost saved/parcel (AED)",
            f"{df['curr_cost_aed'].mean() - df['ai_cost'].mean():.2f}",
        )

# ======================================================================== #
#  TAB 4 – STRATEGIC POSITIONING & COMPETITIVE BENCHMARKING                #
# ======================================================================== #
with tab_comp:
    st.header("Strategic Positioning & Competitive Benchmarking")

    # Classification: unhappy customers
    if {"current_provider", "overall_csat"}.issubset(df.columns):
        y = (df["overall_csat"] < 6).astype(int)
        X = pd.get_dummies(df[["current_provider"]], drop_first=True)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        rf = RandomForestClassifier(max_depth=5, n_estimators=200, random_state=42)
        rf.fit(X_tr, y_tr)
        st.code(classification_report(y_te, rf.predict(X_te), zero_division=0))

    # Association rule mining on pain points
    st.subheader("Pain-point association rules")
    if "top_pain" in df.columns:
        trans = df["top_pain"].astype(str).str.get_dummies(sep=",").astype(bool)
        freq = apriori(trans, min_support=0.05, use_colnames=True)
        rules = association_rules(freq, metric="lift", min_threshold=1.2)
        st.dataframe(
            rules.sort_values("lift", ascending=False).head(10)[
                ["antecedents", "consequents", "support", "confidence", "lift"]
            ]
        )

    # ROI calculator
    st.subheader("ROI calculator (per 1 000 parcels)")
    col1, col2 = st.columns(2)
    with col1:
        cost_now = st.number_input("Current cost/parcel (AED)", 5.0, 200.0, 20.0)
        time_now = st.number_input("Current avg time (hr)", 1.0, 72.0, 12.0)
    with col2:
        cost_ai = st.number_input("AI cost/parcel (AED)", 5.0, 200.0, 18.0)
        time_ai = st.number_input("AI avg time (hr)", 1.0, 72.0, 9.6)

    st.write(
        f"**AED saved / 1 000 parcels:** `{(cost_now-cost_ai)*1000:,.0f}`  |  "
        f"**Time saved:** `{(time_now-time_ai)/time_now*100:.1f}%`"
    )

st.success("Dashboard loaded successfully!")
