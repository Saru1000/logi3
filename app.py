# FastTrack Logistics • End-to-End Feasibility Suite  (2025-v1)
# ────────────────────────────────────────────────────────────────────────────
# Requires  ↠  streamlit pandas numpy scikit-learn plotly altair mlxtend
#            (optional) statsmodels  → enables px trend-lines
# Dataset   ↠  fasttrack_survey_10k.csv  in the same folder.
# ────────────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import re, warnings, itertools
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score, mean_absolute_error, classification_report,
    confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from mlxtend.frequent_patterns import apriori, association_rules

# ── Plotly import (Altair fallback) ─────────────────────────────────────────
try:
    import plotly.express as px
    import plotly.graph_objects as go

    PLOTLY = True
    try:               # trend-line support
        import statsmodels.api as _sm      # noqa: F401
        TREND_OK = True
    except ModuleNotFoundError:
        TREND_OK = False
except ModuleNotFoundError:
    import altair as alt                   # noqa: F401
    PLOTLY = TREND_OK = False
    st.warning(
        "Plotly not installed – charts will use Altair fallback. "
        "Add `plotly>=5.16.0` + `statsmodels` for full interactivity."
    )

warnings.filterwarnings("ignore", category=FutureWarning)
st.set_page_config(page_title="FastTrack Feasibility Suite",
                   layout="wide", initial_sidebar_state="expanded")

# ── 1 Load + header normalisation + rename map ─────────────────────────────
@st.cache_data
def load_csv(path: str = "fasttrack_survey_10k.csv") -> pd.DataFrame:
    df = pd.read_csv(path)

    # lower-snake-case headers
    df.columns = [re.sub(r"\s+", "_", c.strip().lower()) for c in df.columns]

    # canonical rename map (extend if needed)
    rename = {
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
    df.rename(columns={k: v for k, v in rename.items() if k in df.columns},
              inplace=True)

    # category-cast object cols
    obj = df.select_dtypes(include="object").columns
    df[obj] = df[obj].astype("category")

    # derive fail_rate_pct if only band is present
    if "fail_rate_pct" not in df.columns and "fail_rate_band" in df.columns:
        band_map = {"<1_%": .5, "1-3_%": 2, "3-5_%": 4,
                    "5-10_%": 7.5, ">10_%": 12.5}
        df["fail_rate_pct"] = (df["fail_rate_band"].astype(str)
                               .str.replace(" ", "").map(band_map))

    return df


df = load_csv()

# Sidebar column list (can comment out after debugging)
with st.sidebar:
    st.write("🔍 **Columns detected:**", list(df.columns))

# ── 2 Sidebar filters ──────────────────────────────────────────────────────
st.sidebar.header("Filters")
if "city" in df.columns:
    city_opts = sorted(df["city"].dropna().unique())
    df = df[df["city"].isin(st.sidebar.multiselect("City", city_opts,
                                                   default=city_opts))]
if "sector" in df.columns:
    sec_opts = sorted(df["sector"].dropna().unique())
    df = df[df["sector"].isin(st.sidebar.multiselect("Sector", sec_opts,
                                                     default=sec_opts))]

if df.empty:
    st.error("No data after filters — widen your selection.")
    st.stop()

# ── 3 Tab layout ───────────────────────────────────────────────────────────
tab_mv, tab_ops, tab_fin, tab_comp = st.tabs(
    ["Demand & Market Viability",
     "Operational Feasibility & AI Advantage",
     "Financial Viability",
     "Strategic Positioning & Competitive Benchmarking"]
)

# helper: quick Altair fallback scatter
def alt_scatter(data, x, y, c=None, size=None, title=None):
    import altair as _alt
    enc = {"x": x, "y": y}
    if c: enc["color"] = c
    if size: enc["size"] = size
    chart = _alt.Chart(data).mark_circle(opacity=.6).encode(**enc)
    st.altair_chart(chart.properties(title=title), use_container_width=True)

# ======================================================================== #
#  TAB 1 – DEMAND & MARKET VIABILITY                                        #
# ======================================================================== #
with tab_mv:
    st.header("Demand & Market Viability")

    # 1-A  Urgency-density clustering  (K-Means)
    need = ["pct_ultra_urgent", "pct_same_day", "orders_per_week"]
    if all(c in df.columns for c in need) and len(df) >= 15:
        km = KMeans(n_clusters=4, random_state=0).fit(df[need].fillna(0))
        df["urgency_cluster"] = km.labels_

        st.subheader("Urgency-density clusters")
        if PLOTLY:
            st.plotly_chart(
                px.scatter_3d(df, x=need[0], y=need[1], z=need[2],
                              color="urgency_cluster", opacity=.65),
                use_container_width=True)
        else:
            alt_scatter(df, need[0], need[1], c="urgency_cluster:N",
                        size=need[2], title="Clusters")

    else:
        st.info(f"Need columns {need} (≥15 rows) for urgency clustering.")

    # 1-B  Sector-urgency box-plot
    if {"sector", "pct_ultra_urgent"}.issubset(df.columns):
        st.subheader("Verticals vs ultra-urgent demand")
        if PLOTLY:
            fig = px.box(df, x="sector", y="pct_ultra_urgent",
                         points="all", title="Sector urgency box-plot")
            st.plotly_chart(fig, use_container_width=True)
        else:
            import altair as alt
            st.altair_chart(
                alt.Chart(df).mark_boxplot().encode(x="sector", y="pct_ultra_urgent"),
                use_container_width=True)
    else:
        st.info("sector or pct_ultra_urgent missing.")

    # 1-C  Geo heat-map (optional lat/lon)
    if {"lat", "lon", "pct_same_day"}.issubset(df.columns) and PLOTLY:
        st.subheader("Same-day request density")
        fig = px.density_mapbox(df, lat="lat", lon="lon", z="pct_same_day",
                                radius=20, zoom=7, mapbox_style="carto-positron")
        st.plotly_chart(fig, use_container_width=True)

# ======================================================================== #
#  TAB 2 – OPERATIONAL FEASIBILITY & AI ADVANTAGE                           #
# ======================================================================== #
with tab_ops:
    st.header("Operational Feasibility & AI Advantage")

    # 2-A  Regression distance+stops+traffic → time
    need = {"avg_distance_km", "curr_time_hr"}
    X_cols = ["avg_distance_km"]
    if "stops_per_route" in df.columns:
        X_cols.append("stops_per_route")
    if "traffic_level" in df.columns:
        # encode traffic as ordinal: Low=0,Med=1,High=2
        traffic_map = {"low": 0, "medium": 1, "med": 1, "high": 2}
        df["traffic_ord"] = df["traffic_level"].astype(str).str.lower().map(traffic_map)
        X_cols.append("traffic_ord")
        need.add("traffic_level")

    if need.issubset(df.columns):
        X = df[X_cols].fillna(0)
        y = df["curr_time_hr"]
        lin = LinearRegression().fit(X, y)
        df["pred_time"] = lin.predict(X)
        st.metric("R²", f"{r2_score(y, df['pred_time']):.2f}")
        st.metric("MAE", f"{mean_absolute_error(y, df['pred_time']):.2f} hr")

        # scatter
        if PLOTLY:
            fig = px.scatter(df, x="avg_distance_km", y="curr_time_hr",
                             color="traffic_level" if "traffic_level" in df.columns else None,
                             trendline="ols" if TREND_OK else None,
                             opacity=.5, title="Distance vs time")
            st.plotly_chart(fig, use_container_width=True)
        else:
            alt_scatter(df, "avg_distance_km", "curr_time_hr",
                        c="traffic_level:N" if "traffic_level" in df.columns else None,
                        title="Distance vs time")
    else:
        st.info(f"Missing columns for regression: {need - set(df.columns)}")

    # 2-B  A/B cumulative distribution  (AI 20 % faster sim)
    if "curr_time_hr" in df.columns:
        df["ai_time_hr"] = df["curr_time_hr"] * 0.8
        st.subheader("Delivery-time CDF  (AI vs traditional)")
        if PLOTLY:
            cdf = df.melt(value_vars=["curr_time_hr", "ai_time_hr"],
                          var_name="method", value_name="time")
            fig = px.ecdf(cdf, x="time", color="method")
            st.plotly_chart(fig, use_container_width=True)

    # 2-C  Route archetypes via DBSCAN + PCA plot
    if {"avg_distance_km", "stops_per_route"}.issubset(df.columns) and len(df) > 30:
        st.subheader("Route archetypes (PCA-2D view)")
        feat = df[["avg_distance_km", "stops_per_route"]].fillna(0)
        scaled = StandardScaler().fit_transform(feat)
        # DBSCAN often noisy; we use K-Means for visual clarity
        km = KMeans(n_clusters=4, random_state=1).fit(scaled)
        df["route_cluster"] = km.labels_
        pca = PCA(n_components=2).fit_transform(scaled)
        df["pc1"], df["pc2"] = pca[:, 0], pca[:, 1]
        if PLOTLY:
            fig = px.scatter(df, x="pc1", y="pc2",
                             color="route_cluster", opacity=0.7,
                             title="Route clusters (PCA)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            alt_scatter(df, "pc1", "pc2", c="route_cluster:N",
                        title="Route clusters (PCA)")

# ======================================================================== #
#  TAB 3 – FINANCIAL VIABILITY                                             #
# ======================================================================== #
with tab_fin:
    st.header("Financial Viability")

    # 3-A  Ridge regression cost model
    need = {"avg_distance_km", "fuel_price", "curr_cost_aed"}
    if need.issubset(df.columns):
        X = df[["avg_distance_km", "fuel_price"]]
        y = df["curr_cost_aed"]
        ridge = Ridge(alpha=1.0).fit(X, y)
        st.metric("Cost model R²", f"{ridge.score(X, y):.2f}")

        # Tornado sensitivity ±20 %
        st.subheader("Sensitivity – cost/parcel vs fuel price")
        base_dist, base_fuel = df["avg_distance_km"].median(), df["fuel_price"].median()
        sens = []
        for pct in (-0.2, 0, .2):
            cost = ridge.predict([[base_dist, base_fuel * (1 + pct)]])[0]
            sens.append((pct, cost))
        st.table(pd.DataFrame(sens, columns=["fuel_price_change", "pred_cost"]))

    # 3-B  Waterfall cost breakdown if components exist
    comp_cols = ["fuel_cost", "wage_cost", "maint_cost", "overhead_cost"]
    if all(c in df.columns for c in comp_cols):
        st.subheader("Cost breakdown – waterfall")
        avg = df[comp_cols].mean()
        wf = pd.DataFrame({"component": comp_cols + ["Total"],
                           "value": list(avg) + [avg.sum()]})
        if PLOTLY:
            st.plotly_chart(
                px.waterfall(wf, x="component", y="value",
                             title="Avg cost/parcel breakdown"),
                use_container_width=True)

# ======================================================================== #
#  TAB 4 – STRATEGIC POSITIONING & BENCHMARKING                            #
# ======================================================================== #
with tab_comp:
    st.header("Strategic Positioning & Competitive Benchmarking")

    # 4-A  Confusion matrix for unhappy-customer classifier
    need = {"current_provider", "overall_csat"}
    if need.issubset(df.columns):
        y = (df["overall_csat"] < 6).astype(int)
        X = pd.get_dummies(df[["current_provider"]], drop_first=True)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, random_state=1, stratify=y)
        rf = RandomForestClassifier(max_depth=4, n_estimators=300,
                                    random_state=1).fit(X_tr, y_tr)
        cm = confusion_matrix(y_te, rf.predict(X_te))
        st.subheader("Churn-risk model – confusion matrix")
        if PLOTLY:
            st.plotly_chart(px.imshow(cm, text_auto=True,
                                      color_continuous_scale="Blues",
                                      labels=dict(x="Predicted", y="Actual")),
                            use_container_width=True)
        st.code(classification_report(y_te, rf.predict(X_te), zero_division=0))
    else:
        st.info("Need current_provider & overall_csat for churn model")

    # 4-B  Pain-point association rules
    if "top_pain" in df.columns:
        st.subheader("Pain-point association rules (lift >1.2)")
        trans = df["top_pain"].astype(str).str.get_dummies(sep=",").astype(bool)
        freq = apriori(trans, min_support=0.05, use_colnames=True)
        rules = association_rules(freq, metric="lift", min_threshold=1.2)
        st.dataframe(rules.sort_values("lift", ascending=False)
                     [["antecedents", "consequents", "confidence", "lift"]]
                     .head(10))

    # 4-C  ROI calculator & map
    st.subheader("Quick ROI calculator")
    col1, col2 = st.columns(2)
    with col1:
        cost_now = st.number_input("Current cost/parcel (AED)", 5.0, 200.0, 20.0)
        time_now = st.number_input("Current avg time (hr)", 1.0, 72.0, 12.0)
    with col2:
        cost_ai = st.number_input("AI cost/parcel (AED)", 5.0, 200.0, 18.0)
        time_ai = st.number_input("AI avg time (hr)", 1.0, 72.0, 9.6)
    st.write(f"**AED saved / 1 000 parcels:** `{(cost_now-cost_ai)*1000:,.0f}`")
    st.write(f"**Time saved:** `{(time_now-time_ai)/time_now*100:.1f}%`")

    # Heat-map of ROI by city
    if {"lat", "lon", "curr_cost_aed"}.issubset(df.columns) and PLOTLY:
        df["roi_aed"] = (df["curr_cost_aed"] - df.get("ai_cost", df["curr_cost_aed"]*0.9))
        st.subheader("Geo ROI – AED saved/parcel")
        fig = px.scatter_mapbox(df, lat="lat", lon="lon",
                                color="roi_aed", size="roi_aed",
                                color_continuous_scale="RdYlGn_r", zoom=7,
                                mapbox_style="carto-positron")
        st.plotly_chart(fig, use_container_width=True)

st.success("Dashboard loaded successfully ✔")
