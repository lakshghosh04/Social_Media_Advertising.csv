import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Social Media Ads BI", layout="wide")
st.title("📊 Social Media Advertising — BI Dashboard")
st.caption("Build v1.1 (with safe ROI prediction)")

REQUIRED = ["Date", "Channel_Used", "Target_Audience", "Impressions", "Clicks", "Conversion_Rate", "ROI"]
OPTIONAL = ["Company", "Location", "Campaign_Goal"]

# -------------------------------
# Load CSV (upload or /data file)
# -------------------------------
st.sidebar.header("Upload CSV")
up = st.sidebar.file_uploader("Choose a CSV (must have required columns)", type=["csv"])

if up:
    df = pd.read_csv(up, encoding="utf-8", encoding_errors="replace")
else:
    path = "data/Social_Media_Advertising.csv"
    if os.path.exists(path):
        df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    else:
        st.info("Upload a CSV or place Social_Media_Advertising.csv in /data")
        st.stop()

# -------------------------------
# Quick debug info (very helpful)
# -------------------------------
with st.expander("🔍 Debug: dataset snapshot"):
    st.write("Rows:", len(df))
    st.write("Columns:", list(df.columns))
    st.write("Dtypes:", df.dtypes.astype(str).to_dict())
    st.dataframe(df.head(), use_container_width=True)

# -------------------------------
# Basic validation & cleaning
# -------------------------------
missing = [c for c in REQUIRED if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {', '.join(missing)}")
    st.stop()

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
for c in ["Impressions", "Clicks", "Conversion_Rate", "ROI"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["CTR"] = np.where(df["Impressions"] > 0, df["Clicks"] / df["Impressions"], np.nan)

# -------------------------------
# Filters
# -------------------------------
st.sidebar.header("Filters")
dmin, dmax = df["Date"].min(), df["Date"].max()
date_range = st.sidebar.date_input("Date range", (dmin, dmax))

f = df.copy()
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    f = f[(f["Date"] >= start) & (f["Date"] <= end)]

def opts(col):
    return ["All"] + sorted(f[col].dropna().unique().tolist()) if col in f.columns else ["All"]

sel_channel  = st.sidebar.selectbox("Channel",  opts("Channel_Used"))
sel_audience = st.sidebar.selectbox("Audience", opts("Target_Audience"))
sel_company  = st.sidebar.selectbox("Company",  opts("Company") if "Company" in f.columns else ["All"])
sel_loc      = st.sidebar.selectbox("Location", opts("Location") if "Location" in f.columns else ["All"])

if sel_channel != "All":  f = f[f["Channel_Used"] == sel_channel]
if sel_audience != "All": f = f[f["Target_Audience"] == sel_audience]
if sel_company != "All" and "Company" in f.columns:  f = f[f["Company"] == sel_company]
if sel_loc != "All" and "Location" in f.columns:     f = f[f["Location"] == sel_loc]

# -------------------------------
# KPIs
# -------------------------------
total_impr  = int(f["Impressions"].sum())
total_clicks = int(f["Clicks"].sum())
avg_ctr = (f["CTR"].mean() * 100) if f["CTR"].notna().any() else 0
avg_roi = f["ROI"].mean() if "ROI" in f.columns else np.nan
avg_cr  = (f["Conversion_Rate"].mean() * 100) if "Conversion_Rate" in f.columns else np.nan

c1, c2, c3, c4 = st.columns(4)
c1.metric("Impressions", f"{total_impr:,}")
c2.metric("Clicks", f"{total_clicks:,}")
c3.metric("Avg CTR", f"{avg_ctr:.2f}%")
c4.metric("Avg ROI", f"{avg_roi:.2f}" if pd.notna(avg_roi) else "—")

# -------------------------------
# Trends
# -------------------------------
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

# -------------------------------
# Channel performance
# -------------------------------
st.subheader("Channel performance")
if "Channel_Used" in f.columns and len(f):
    by_ch = f.groupby("Channel_Used").agg(Impressions=("Impressions","sum"),
                                          Clicks=("Clicks","sum"),
                                          ROI=("ROI","mean")).reset_index()
    by_ch["CTR"] = np.where(by_ch["Impressions"]>0, by_ch["Clicks"]/by_ch["Impressions"], np.nan)
    colA, colB = st.columns(2)
    colA.plotly_chart(px.bar(by_ch, x="Channel_Used", y="CTR", title="CTR by channel"), use_container_width=True)
    colB.plotly_chart(px.bar(by_ch, x="Channel_Used", y="ROI", title="ROI by channel"), use_container_width=True)

# -------------------------------
# Audience × Channel heatmap
# -------------------------------
st.subheader("Audience × Channel (Conversion Rate)")
if {"Target_Audience","Channel_Used","Conversion_Rate"}.issubset(f.columns):
    heat = f.pivot_table(index="Target_Audience", columns="Channel_Used", values="Conversion_Rate", aggfunc="mean")
    heat = heat.round(3)
    st.plotly_chart(px.imshow(heat, text_auto=True, aspect="auto", title="Avg Conversion Rate heatmap"),
                    use_container_width=True)

# -------------------------------
# Top & Bottom performers
# -------------------------------
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
    colT, colL = st.columns(2)
    colT.markdown("**Top 10**"); colT.dataframe(top, use_container_width=True)
    colL.markdown("**Bottom 10**"); colL.dataframe(low, use_container_width=True)

# -------------------------------
# Recommendations
# -------------------------------
st.subheader("Recommendations")
recs = []
if len(f):
    if "ROI" in f.columns:
        by_ch_roi = f.groupby("Channel_Used")["ROI"].mean().sort_values(ascending=False)
        if len(by_ch_roi):
            recs.append(f"Shift budget toward **{by_ch_roi.index[0]}** (highest avg ROI).")
    if {"Impressions","Clicks","Channel_Used"}.issubset(f.columns):
        by_ch_ctr = f.groupby("Channel_Used").apply(lambda x: (x["Clicks"].sum()/x["Impressions"].sum()) if x["Impressions"].sum()>0 else np.nan)
        by_ch_ctr = by_ch_ctr.sort_values(ascending=False)
        if len(by_ch_ctr):
            recs.append(f"Use **{by_ch_ctr.index[0]}** for awareness/traffic (best CTR).")
    if {"Target_Audience","Channel_Used","Conversion_Rate"}.issubset(f.columns):
        hot = f.groupby(["Target_Audience","Channel_Used"])["Conversion_Rate"].mean().sort_values(ascending=False).head(3)
        if len(hot):
            pairs = ", ".join([f"{a}×{c}" for (a,c) in hot.index])
            recs.append(f"Double-down on high-CR combos: {pairs}.")
if recs:
    for r in recs: st.markdown(f"- {r}")
else:
    st.write("No strong signals yet. Adjust filters or collect more data.")

# ===================================================
# ROI Prediction (safe, guarded; won't crash the app)
# ===================================================
st.divider()
st.header("🤖 ROI Prediction")

try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import r2_score, mean_absolute_error

    feature_pool = ["Channel_Used", "Target_Audience", "Campaign_Goal", "Impressions", "Clicks"]
    available = [c for c in feature_pool if c in df.columns]

    if "ROI" not in df.columns:
        st.info("ROI column not found — cannot train ROI predictor.")
    elif len(available) < 3:
        st.info("Need at least 3 of these features to train: Channel_Used, Target_Audience, Campaign_Goal, Impressions, Clicks.")
    else:
        df_model = df[available + ["ROI"]].dropna()
        if len(df_model) < 200:
            st.info("Not enough rows to train a stable model (need ≥ 200 after dropping NA).")
        else:
            X = df_model[available]
            y = df_model["ROI"]

            cat_cols = [c for c in ["Channel_Used", "Target_Audience", "Campaign_Goal"] if c in available]
            num_cols = [c for c in ["Impressions", "Clicks"] if c in available]

            pre = ColumnTransformer([
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                ("num", "passthrough", num_cols)
            ])

            pipe = Pipeline([
                ("prep", pre),
                ("model", RandomForestRegressor(n_estimators=200, random_state=42))
            ])

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            m1, m2 = st.columns(2)
            m1.metric("R² (test)", f"{r2:.2f}")
            m2.metric("MAE (test)", f"{mae:.2f}")

            st.subheader("Try a prediction")
            col1, col2, col3 = st.columns(3)
            with col1:
                ch_val = st.selectbox("Channel_Used", sorted(df["Channel_Used"].dropna().unique().tolist()))
                au_val = st.selectbox("Target_Audience", sorted(df["Target_Audience"].dropna().unique().tolist()))
            with col2:
                if "Campaign_Goal" in df.columns:
                    go_val = st.selectbox("Campaign_Goal", sorted(df["Campaign_Goal"].dropna().unique().tolist()))
                else:
                    go_val = ""
                imp_val = st.number_input("Impressions", min_value=0, value=int(df["Impressions"].median()))
            with col3:
                clk_val = st.number_input("Clicks", min_value=0, value=int(df["Clicks"].median()))

            if st.button("Predict ROI"):
                row = {"Channel_Used": ch_val, "Target_Audience": au_val}
                if "Campaign_Goal" in available: row["Campaign_Goal"] = go_val
                if "Impressions" in available:   row["Impressions"] = imp_val
                if "Clicks" in available:        row["Clicks"] = clk_val
                pred = float(pipe.predict(pd.DataFrame([row]))[0])
                st.success(f"Predicted ROI: {pred:.2f}")

except Exception as e:
    st.error(f"Prediction module error: {e}")
    st.caption("Tip: Ensure 'scikit-learn' is in requirements.txt and your CSV has enough rows & required columns.")
