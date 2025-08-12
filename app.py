import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title="Social Media Ads BI + Prediction", layout="wide")
st.title("📊 Social Media Advertising — BI + Prediction")

REQUIRED = ["Date", "Channel_Used", "Target_Audience", "Impressions", "Clicks", "Conversion_Rate", "ROI"]
OPTIONAL = ["Company", "Location", "Campaign_Goal", "Engagement_Score", "Acquisition_Cost"]

st.sidebar.header("Upload CSV")
up = st.sidebar.file_uploader("Choose a CSV (must include required columns)", type=["csv"])

# --- load data
if up:
    df = pd.read_csv(up, encoding="utf-8", encoding_errors="replace")
else:
    path = "data/Social_Media_Advertising.csv"
    if os.path.exists(path):
        df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    else:
        st.info("Upload a CSV or place Social_Media_Advertising.csv in /data")
        st.stop()

if not set(REQUIRED).issubset(df.columns):
    st.error("Missing columns: " + ", ".join([c for c in REQUIRED if c not in df.columns]))
    st.dataframe(df.head(), use_container_width=True)
    st.stop()

# --- basic prep
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
for c in ["Impressions", "Clicks", "Conversion_Rate", "ROI", "Engagement_Score"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
# CTR
df["CTR"] = np.where(df["Impressions"] > 0, df["Clicks"] / df["Impressions"], np.nan)

tab1, tab2 = st.tabs(["📈 Dashboard", "🤖 Prediction"])

with tab1:
    st.subheader("Filters")
    dmin, dmax = df["Date"].min(), df["Date"].max()
    date_range = st.date_input("Date range", (dmin, dmax))
    f = df.copy()
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        f = f[(f["Date"] >= start) & (f["Date"] <= end)]

    def opts(col):
        return ["All"] + sorted(f[col].dropna().unique().tolist()) if col in f.columns else ["All"]

    colA, colB, colC, colD = st.columns(4)
    sel_channel  = colA.selectbox("Channel",  opts("Channel_Used"))
    sel_audience = colB.selectbox("Audience", opts("Target_Audience"))
    sel_company  = colC.selectbox("Company",  opts("Company") if "Company" in f.columns else ["All"])
    sel_loc      = colD.selectbox("Location", opts("Location") if "Location" in f.columns else ["All"])

    if sel_channel != "All":  f = f[f["Channel_Used"] == sel_channel]
    if sel_audience != "All": f = f[f["Target_Audience"] == sel_audience]
    if sel_company != "All" and "Company" in f.columns:  f = f[f["Company"] == sel_company]
    if sel_loc != "All" and "Location" in f.columns:     f = f[f["Location"] == sel_loc]

    # KPIs
    total_impr  = int(f["Impressions"].sum())
    total_clicks = int(f["Clicks"].sum())
    avg_ctr = (f["CTR"].mean() * 100) if f["CTR"].notna().any() else 0
    avg_roi = f["ROI"].mean() if "ROI" in f.columns else np.nan
    avg_cr  = (f["Conversion_Rate"].mean() * 100) if "Conversion_Rate" in f.columns else np.nan

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Impressions", f"{total_impr:,}")
    k2.metric("Clicks", f"{total_clicks:,}")
    k3.metric("Avg CTR", f"{avg_ctr:.2f}%")
    k4.metric("Avg ROI", f"{avg_roi:.2f}" if pd.notna(avg_roi) else "—")

    st.subheader("Trends")
    daily = f.groupby("Date").agg(Impressions=("Impressions","sum"),
                                  Clicks=("Clicks","sum"),
                                  ROI=("ROI","mean"),
                                  CR=("Conversion_Rate","mean")).reset_index()
    if len(daily):
        daily["CTR"] = np.where(daily["Impressions"]>0, daily["Clicks"]/daily["Impressions"], np.nan)
        st.plotly_chart(px.line(daily, x="Date", y="CTR", markers=True, title="CTR over time"), use_container_width=True)
        st.plotly_chart(px.line(daily, x="Date", y="ROI", markers=True, title="ROI over time"), use_container_width=True)
    else:
        st.info("No rows to plot after filters.")

    st.subheader("Channel performance")
    if "Channel_Used" in f.columns and len(f):
        by_ch = f.groupby("Channel_Used").agg(Impressions=("Impressions","sum"),
                                              Clicks=("Clicks","sum"),
                                              ROI=("ROI","mean")).reset_index()
        by_ch["CTR"] = np.where(by_ch["Impressions"]>0, by_ch["Clicks"]/by_ch["Impressions"], np.nan)
        cA, cB = st.columns(2)
        cA.plotly_chart(px.bar(by_ch, x="Channel_Used", y="CTR", title="CTR by channel"), use_container_width=True)
        cB.plotly_chart(px.bar(by_ch, x="Channel_Used", y="ROI", title="ROI by channel"), use_container_width=True)

    st.subheader("Audience × Channel (Conversion Rate)")
    if {"Target_Audience","Channel_Used","Conversion_Rate"}.issubset(f.columns):
        heat = f.pivot_table(index="Target_Audience", columns="Channel_Used",
                             values="Conversion_Rate", aggfunc="mean").round(3)
        st.plotly_chart(px.imshow(heat, text_auto=True, aspect="auto", title="Avg Conversion Rate heatmap"),
                        use_container_width=True)

    st.subheader("Top & Bottom performers (by ROI)")
    segment_cols = [c for c in ["Company","Target_Audience","Channel_Used","Campaign_Goal","Location"] if c in f.columns]
    if "ROI" in f.columns and segment_cols:
        seg = f.groupby(segment_cols).agg(Impressions=("Impressions","sum"),
                                          Clicks=("Clicks","sum"),
                                          ROI=("ROI","mean")).reset_index()
        seg = seg[seg["Impressions"] > 0]
        seg["CTR"] = seg["Clicks"] / seg["Impressions"]
        top = seg.sort_values("ROI", ascending=False).head(10)
        low = seg.sort_values("ROI", ascending=True).head(10)
        tA, tB = st.columns(2)
        tA.markdown("**Top 10**"); tA.dataframe(top, use_container_width=True)
        tB.markdown("**Bottom 10**"); tB.dataframe(low, use_container_width=True)

with tab2:
    st.subheader("Train a simple model and predict")
    target = st.selectbox("Target to predict", ["ROI", "Conversion_Rate", "CTR"], index=0)

    feature_cats = [c for c in ["Channel_Used","Target_Audience","Campaign_Goal","Company","Location"] if c in df.columns]
    feature_nums = [c for c in ["Impressions","Clicks","Engagement_Score"] if c in df.columns]

    usable = df.dropna(subset=[target]).copy()
    if target == "CTR":
        # avoid trivial leakage: if predicting CTR, drop CTR components from features if desired
        # keep it simple: allow Impressions/Clicks but note it's optimistic
        pass

    X = usable[feature_cats + feature_nums].copy()
    y = usable[target].astype(float)

    for c in feature_nums:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.dropna()

    y = y.loc[X.index]

    if len(X) < 200:
        st.info("Not enough rows to train. Adjust target or provide more data.")
        st.stop()

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), feature_cats),
        ("num", "passthrough", feature_nums)
    ])

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    pipe = Pipeline([("prep", pre), ("rf", model)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    m1, m2 = st.columns(2)
    m1.metric("R² (test)", f"{r2:.3f}")
    m2.metric("MAE (test)", f"{mae:.3f}")

    st.markdown("---")
    st.markdown("**Enter campaign parameters to get a prediction**")

    def pick(col, fallback=""):
        if col not in df.columns: return fallback
        vals = sorted(df[col].dropna().unique().tolist())
        return st.selectbox(col, vals) if vals else fallback

    in_channel  = pick("Channel_Used")
    in_aud      = pick("Target_Audience")
    in_goal     = pick("Campaign_Goal")
    in_company  = pick("Company")
    in_loc      = pick("Location")

    def num_input(name, default_col=None):
        default = float(df[default_col].median()) if default_col in df.columns else 0.0
        return st.number_input(name, value=float(default), step=1.0, format="%.0f") if name in ["Impressions","Clicks"] else \
               st.number_input(name, value=float(df[default_col].median()) if default_col in df.columns else 0.0, step=0.1)

    in_impr = num_input("Impressions", "Impressions") if "Impressions" in feature_nums else None
    in_clicks = num_input("Clicks", "Clicks") if "Clicks" in feature_nums else None
    in_eng = st.number_input("Engagement_Score", value=float(df["Engagement_Score"].median()) if "Engagement_Score" in df.columns else 0.0, step=0.1) if "Engagement_Score" in feature_nums else None

    input_row = pd.DataFrame([{
        "Channel_Used": in_channel,
        "Target_Audience": in_aud,
        "Campaign_Goal": in_goal,
        "Company": in_company,
        "Location": in_loc,
        "Impressions": in_impr if in_impr is not None else np.nan,
        "Clicks": in_clicks if in_clicks is not None else np.nan,
        "Engagement_Score": in_eng if in_eng is not None else np.nan
    }])

    if st.button("Predict"):
        pred = pipe.predict(input_row[feature_cats + feature_nums])[0]
        label = {"ROI":"Predicted ROI", "Conversion_Rate":"Predicted Conversion Rate", "CTR":"Predicted CTR"}[target]
        if target in ["Conversion_Rate", "CTR"]:
            st.success(f"{label}: {pred:.4f} ({pred*100:.2f}%)")
        else:
            st.success(f"{label}: {pred:.4f}")

    st.caption("Note: Simple Random Forest on structured features. For CTR target, using Clicks/Impressions as inputs can be optimistic; you can remove them to be stricter.")
