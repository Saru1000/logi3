import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import seaborn as sns

st.set_page_config(page_title="FastTrack Logistics – Feasibility Dashboard", layout="wide")

# ---------- Helper ----------
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

def descriptive_insights(df):
    st.subheader("Key Descriptive Insights")

    st.markdown("**1. Sector distribution**")
    st.plotly_chart(px.histogram(df, x="sector", color="sector"), use_container_width=True)

    st.markdown("**2. City distribution**")
    st.plotly_chart(px.histogram(df, x="city", color="city"), use_container_width=True)

    st.markdown("**3. Orders per week (log‑scale)**")
    st.plotly_chart(px.histogram(df, x="orders_per_week", log_y=True), use_container_width=True)

    st.markdown("**4. Average urgency composition**")
    urg = df[["pct_ultra_urgent","pct_same_day","pct_next_day"]].mean().reset_index()
    st.plotly_chart(px.pie(urg, values=0, names="index"), use_container_width=True)

    st.markdown("**5. Distance vs current delivery time**")
    st.plotly_chart(px.scatter(df, x="avg_distance_km", y="curr_time_hr", trendline="ols",
                               labels={"avg_distance_km":"Distance (km)","curr_time_hr":"Time (h)"}), use_container_width=True)

    st.markdown("**6. Distance vs cost**")
    st.plotly_chart(px.scatter(df, x="avg_distance_km", y="curr_cost_aed", trendline="ols"), use_container_width=True)

    st.markdown("**7. Numeric correlation heat‑map**")
    corr = df.select_dtypes("number").corr()
    st.plotly_chart(px.imshow(corr, aspect="auto", color_continuous_scale="Viridis"), use_container_width=True)

    st.markdown("**8. Fuel price by strategy**")
    st.plotly_chart(px.box(df, x="fuel_strategy", y="fuel_price"), use_container_width=True)

    st.markdown("**9. CSAT by provider**")
    st.plotly_chart(px.box(df, x="current_provider", y="overall_csat"), use_container_width=True)

    st.markdown("**10. NPS distribution**")
    st.plotly_chart(px.histogram(df, x="nps", nbins=11), use_container_width=True)

def preprocess(df, target):
    X = df.drop(columns=[target])
    y = df[target]
    cat = X.select_dtypes("object").columns.tolist()
    num = X.select_dtypes("number").columns.tolist()
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat),
                             ("num", StandardScaler(), num)])
    return X, y, pre

def build_classifiers(X_train, y_train, prep):
    models = {
        "KNN": KNeighborsClassifier(5),
        "Decision Tree": DecisionTreeClassifier(max_depth=6),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8),
        "GBRT": GradientBoostingClassifier(n_estimators=200)
    }
    pipes = {}
    for n, m in models.items():
        p = Pipeline([("prep", prep), ("model", m)])
        p.fit(X_train, y_train)
        pipes[n] = p
    return pipes

def metrics_df(models, X_test, y_test):
    rows = []
    for n, p in models.items():
        y_pred = p.predict(X_test)
        rows.append({
            "Model": n,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "F1": f1_score(y_test, y_pred, average='weighted', zero_division=0)
        })
    return pd.DataFrame(rows).set_index("Model").round(3)

def plot_cm(cm, labels):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    st.pyplot(fig)

# ---------- Sidebar ----------
st.sidebar.header("Global Settings")
csv_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
df = load_data(csv_file) if csv_file else load_data("data/fasttrack_survey_10k.csv")

# Tabs
tabs = st.tabs(["Data Visualisation","Classification","Clustering","Association Rules","Regression"])

# ---------- Visualisation ----------
with tabs[0]:
    st.header("Exploratory Dashboard")
    with st.expander("Filters"):
        c_city = st.multiselect("City", df["city"].unique())
        c_sector = st.multiselect("Sector", df["sector"].unique())
    df_viz = df.copy()
    if c_city:
        df_viz = df_viz[df_viz["city"].isin(c_city)]
    if c_sector:
        df_viz = df_viz[df_viz["sector"].isin(c_sector)]
    descriptive_insights(df_viz)

# ---------- Classification ----------
with tabs[1]:
    st.header("Classification")
    class_target = st.selectbox("Target variable", ["switch_past","beta_opt_in"])
    test_size = st.slider("Test size", 0.1, 0.4, 0.25, 0.05)
    X, y, prep = preprocess(df, class_target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    cl_models = build_classifiers(X_train, y_train, prep)
    st.subheader("Performance")
    st.dataframe(metrics_df(cl_models, X_test, y_test))

    if y.nunique()==2:
        fig, ax = plt.subplots()
        for n, p in cl_models.items():
            y_prob = p.predict_proba(X_test)[:,1]
            fpr,tpr,_ = roc_curve(y_test.astype(int), y_prob)
            ax.plot(fpr,tpr,label=f"{n} (AUC={auc(fpr,tpr):.2f})")
        ax.plot([0,1],[0,1],'k--'); ax.legend(); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        st.pyplot(fig)

    model_cm = st.selectbox("Confusion matrix for", list(cl_models.keys()))
    cm = confusion_matrix(y_test, cl_models[model_cm].predict(X_test))
    plot_cm(cm, cl_models[model_cm].classes_)

    st.subheader("Batch prediction")
    up_file = st.file_uploader("Upload new data (no target)", key="pred")
    if up_file:
        new_df = pd.read_csv(up_file)
        preds = cl_models[model_cm].predict(new_df)
        new_df[class_target] = preds
        st.dataframe(new_df.head())
        st.download_button("Download predictions", new_df.to_csv(index=False).encode(), "predictions.csv")

# ---------- Clustering ----------
with tabs[2]:
    st.header("K‑Means Clustering")
    num_cols = st.multiselect("Numeric columns", df.select_dtypes("number").columns.tolist(),
                              default=["avg_distance_km","orders_per_week","curr_cost_aed"])
    k = st.slider("Clusters (k)", 2, 10, 4)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[num_cols])
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    df["cluster"] = km.fit_predict(X_scaled)
    if st.checkbox("Show Elbow Chart"):
        inertias=[]
        for i in range(2,11):
            inertias.append(KMeans(n_clusters=i, random_state=42, n_init="auto").fit(X_scaled).inertia_)
        fig=plt.figure(); plt.plot(range(2,11), inertias,"o-"); plt.xlabel("k"); plt.ylabel("Inertia")
        st.pyplot(fig)
    st.subheader("Cluster profiles")
    st.dataframe(df.groupby("cluster")[num_cols].mean().round(2))
    st.download_button("Download clustered data", df.to_csv(index=False).encode(), "clustered.csv")

# ---------- Association Rules ----------
with tabs[3]:
    st.header("Association Rule Mining")
    trans_cols = st.multiselect("Columns to treat as transactions",
                                ["peak_hours","season_peaks","top_pain","fuel_strategy"])
    min_sup = st.slider("Min support", 0.01, 0.3, 0.05, 0.01)
    min_conf = st.slider("Min confidence", 0.1, 1.0, 0.4, 0.05)
    if st.button("Run Apriori") and len(trans_cols)>=2:
        baskets=[]
        for c in trans_cols:
            for items in df[c].fillna("").str.split(","):
                baskets.append([f"{c}={i.strip()}" for i in items if i.strip()])
        te=TransactionEncoder()
        arr=te.fit_transform(baskets)
        trans_df=pd.DataFrame(arr, columns=te.columns_)
        freq=apriori(trans_df, min_support=min_sup, use_colnames=True)
        rules=association_rules(freq, metric="confidence", min_threshold=min_conf)
        st.dataframe(rules.sort_values("confidence", ascending=False).head(10))

# ---------- Regression ----------
with tabs[4]:
    st.header("Regression Modelling")
    reg_target = st.selectbox("Target", ["curr_cost_aed","curr_time_hr","orders_per_week"])
    feat_cols = st.multiselect("Features", [c for c in df.columns if c!=reg_target],
                               default=["avg_distance_km","orders_per_week","pct_ultra_urgent","fuel_price"])
    tst = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    X=df[feat_cols]; y=df[reg_target]
    cat=X.select_dtypes("object").columns.tolist(); num=X.select_dtypes("number").columns.tolist()
    prep_r=ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore"),cat),
                              ("num",StandardScaler(),num)])
    X_tr,X_te,y_tr,y_te=train_test_split(X,y,test_size=tst,random_state=42)
    r_models={"Linear":LinearRegression(),"Ridge":Ridge(alpha=1.0),"Lasso":Lasso(alpha=0.01),
              "DecisionTree":DecisionTreeRegressor(max_depth=6)}
    res={}
    for n,m in r_models.items():
        p=Pipeline([("prep",prep_r),("m",m)]).fit(X_tr,y_tr)
        pred=p.predict(X_te)
        res[n]={"RMSE":np.sqrt(((y_te-pred)**2).mean()),"R2":p.score(X_te,y_te)}
    st.dataframe(pd.DataFrame(res).T.round(3))
    best=max(res,key=lambda k: res[k]["R2"])
    st.markdown(f"**Actual vs Predicted – {best}**")
    best_pipe=Pipeline([("prep",prep_r),("m",r_models[best])]).fit(X_tr,y_tr)
    pred=best_pipe.predict(X_te)
    fig=px.scatter(x=y_te,y=pred, labels={"x":"Actual","y":"Predicted"})
    fig.add_shape(type="line",x0=y_te.min(),y0=y_te.min(),x1=y_te.max(),y1=y_te.max())
    st.plotly_chart(fig, use_container_width=True)
