import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Social Media Ads BI", layout="wide")
st.title("📊 Social Media Advertising — BI Dashboard")

REQUIRED = ["Date", "Channel_Used", "Target_Audience", "Impressions", "Clicks", "Conversion_Rate", "ROI"]
OPTIONAL = ["Company", "Location", "Campaign_Goal"]

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

if not set(REQUIRED).issubset(df.columns):
    st.error(f"Missing columns: {', '.join([c for c in REQUIRED if c not in df.columns])}")
    st.dataframe(df.head(), use_container_width=True)
    st.stop()

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
for c in ["Impressions", "Clicks", "Conversion_Rate", "ROI"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["CTR"] = np.where(df["Impressions"] > 0, df["Clicks"] / df["Impressions"], np.nan)

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
    colA, colB = st.columns(2)
    colA.plotly_chart(px.bar(by_ch, x="Channel_Used", y="CTR", title="CTR by channel"), use_container_width=True)
    colB.plotly_chart(px.bar(by_ch, x="Channel_Used", y="ROI", title="ROI by channel"), use_container_width=True)

st.subheader("Audience × Channel (Conversion Rate)")
if {"Target_Audience","Channel_Used","Conversion_Rate"}.issubset(f.columns):
    heat = f.pivot_table(index="Target_Audience", columns="Channel_Used", values="Conversion_Rate", aggfunc="mean")
    heat = heat.round(3)
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
    colT, colL = st.columns(2)
    colT.markdown("**Top 10**"); colT.dataframe(top, use_container_width=True)
    colL.markdown("**Bottom 10**"); colL.dataframe(low, use_container_width=True)

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
