import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Social Media Ads BI", layout="wide")
st.title("📊 Social Media Advertising — BI Dashboard")
st.caption("Build v1")

# ---------- 1) Load data ----------
os.makedirs("data", exist_ok=True)
DEFAULT_PATHS = [
    "data/Social_Media_Advertising.csv",
    "Social_Media_Advertising.csv"
]

def load_data():
    path = next((p for p in DEFAULT_PATHS if os.path.exists(p)), None)
    if path is None:
        up = st.file_uploader("Upload Social_Media_Advertising.csv", type=["csv"])
        if up:
            path = "Social_Media_Advertising.csv"
            with open(path, "wb") as f: f.write(up.read())
        else:
            st.stop()
    df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    return df

df = load_data()

# ---------- 2) Basic cleaning / feature engineering ----------
# parse date
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
else:
    st.error("Dataset must contain a 'Date' column.")
    st.stop()

# ensure numeric fields
num_cols = ["Conversion_Rate", "ROI", "Clicks", "Impressions", "Engagement_Score"]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Acquisition_Cost comes like "$500.00" → float
if "Acquisition_Cost" in df.columns:
    df["Acquisition_Cost_Num"] = (
        df["Acquisition_Cost"]
        .astype(str)
        .str.replace(r"[^0-9.\-]", "", regex=True)
        .replace("", np.nan)
        .astype(float)
    )
else:
    df["Acquisition_Cost_Num"] = np.nan

# KPIs
df["CTR"] = np.where(df["Impressions"]>0, df["Clicks"]/df["Impressions"], np.nan)
# approximate spend if not given: Acquisition_Cost ≈ CPA; if Conversion_Rate>0, est conversions = CR*Clicks (or use CR * Impressions?)
# we'll estimate spend conservatively as: CPA * (Conversion_Rate * Impressions)
est_conversions = np.where((df["Conversion_Rate"].notna()) & (df["Impressions"].notna()),
                           df["Conversion_Rate"] * df["Impressions"], np.nan)
df["Est_Spend"] = df["Acquisition_Cost_Num"] * est_conversions

# ---------- 3) Filters ----------
st.sidebar.header("Filters")
min_d, max_d = pd.to_datetime(df["Date"]).min(), pd.to_datetime(df["Date"]).max()
date_range = st.sidebar.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)

channels = ["All"] + sorted(df["Channel_Used"].dropna().unique().tolist()) if "Channel_Used" in df.columns else ["All"]
audiences = ["All"] + sorted(df["Target_Audience"].dropna().unique().tolist()) if "Target_Audience" in df.columns else ["All"]
companies = ["All"] + sorted(df["Company"].dropna().unique().tolist()) if "Company" in df.columns else ["All"]
locations = ["All"] + sorted(df["Location"].dropna().unique().tolist()) if "Location" in df.columns else ["All"]

sel_channel = st.sidebar.selectbox("Channel", channels, index=0)
sel_aud = st.sidebar.selectbox("Audience", audiences, index=0)
sel_company = st.sidebar.selectbox("Company", companies, index=0)
sel_loc = st.sidebar.selectbox("Location", locations, index=0)

f = df.copy()
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    f = f[(f["Date"] >= start) & (f["Date"] <= end)]

if sel_channel != "All" and "Channel_Used" in f.columns:
    f = f[f["Channel_Used"] == sel_channel]
if sel_aud != "All" and "Target_Audience" in f.columns:
    f = f[f["Target_Audience"] == sel_aud]
if sel_company != "All" and "Company" in f.columns:
    f = f[f["Company"] == sel_company]
if sel_loc != "All" and "Location" in f.columns:
    f = f[f["Location"] == sel_loc]

# ---------- 4) KPI cards ----------
total_impr = int(f["Impressions"].sum()) if "Impressions" in f.columns else 0
total_clicks = int(f["Clicks"].sum()) if "Clicks" in f.columns else 0
ctr = (f["CTR"].mean()*100) if f["CTR"].notna().any() else 0
avg_roi = f["ROI"].mean() if "ROI" in f.columns else np.nan
avg_cr = (f["Conversion_Rate"].mean()*100) if "Conversion_Rate" in f.columns else np.nan
est_spend = f["Est_Spend"].sum() if f["Est_Spend"].notna().any() else np.nan

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Impressions", f"{total_impr:,}")
c2.metric("Clicks", f"{total_clicks:,}")
c3.metric("Avg CTR", f"{ctr:.2f}%")
c4.metric("Avg ROI", f"{avg_roi:.2f}" if pd.notna(avg_roi) else "—")
c5.metric("Est. Spend", f"${est_spend:,.0f}" if pd.notna(est_spend) else "—")
st.caption("Note: Est. Spend derived from Acquisition_Cost × estimated conversions.")

st.divider()

# ---------- 5) Trends ----------
st.subheader("📈 Trends over time")
# CTR trend
daily = f.groupby("Date").agg(
    Impressions=("Impressions","sum"),
    Clicks=("Clicks","sum"),
    ROI=("ROI","mean"),
    CR=("Conversion_Rate","mean")
).reset_index()
if len(daily):
    daily["CTR"] = np.where(daily["Impressions"]>0, daily["Clicks"]/daily["Impressions"], np.nan)

    fig_ctr = px.line(daily, x="Date", y="CTR", markers=True, title="CTR over time")
    st.plotly_chart(fig_ctr, use_container_width=True)

    fig_roi = px.line(daily, x="Date", y="ROI", markers=True, title="ROI over time")
    st.plotly_chart(fig_roi, use_container_width=True)
else:
    st.info("No rows after filters to plot trends.")

st.divider()

# ---------- 6) Channel comparison ----------
st.subheader("📊 Channel performance")
if "Channel_Used" in f.columns and len(f):
    by_ch = f.groupby("Channel_Used").agg(
        Impressions=("Impressions","sum"),
        Clicks=("Clicks","sum"),
        ROI=("ROI","mean"),
        CR=("Conversion_Rate","mean")
    ).reset_index()
    by_ch["CTR"] = np.where(by_ch["Impressions"]>0, by_ch["Clicks"]/by_ch["Impressions"], np.nan)

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**CTR by channel**")
        fig1 = px.bar(by_ch, x="Channel_Used", y="CTR", title="CTR by channel")
        st.plotly_chart(fig1, use_container_width=True)
    with colB:
        st.markdown("**ROI by channel**")
        fig2 = px.bar(by_ch, x="Channel_Used", y="ROI", title="ROI by channel")
        st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Channel column not available or empty after filters.")

st.divider()

# ---------- 7) Audience × Channel heatmap ----------
st.subheader("🧩 Audience × Channel (Conversion Rate)")
if {"Target_Audience","Channel_Used","Conversion_Rate"}.issubset(f.columns):
    heat = f.pivot_table(
        index="Target_Audience",
        columns="Channel_Used",
        values="Conversion_Rate",
        aggfunc="mean"
    )
    heat = heat.round(3)
    heat = heat.sort_index()
    fig_h = px.imshow(heat, text_auto=True, aspect="auto",
                      title="Avg Conversion Rate heatmap")
    st.plotly_chart(fig_h, use_container_width=True)
else:
    st.info("Need Target_Audience, Channel_Used, and Conversion_Rate for the heatmap.")

st.divider()

# ---------- 8) Top & bottom segments ----------
st.subheader("🏆 Top & Bottom performers (by ROI)")
segment_cols = [c for c in ["Company","Target_Audience","Channel_Used","Campaign_Goal","Location"] if c in f.columns]
if "ROI" in f.columns and len(segment_cols):
    seg = f.groupby(segment_cols).agg(
        Impressions=("Impressions","sum"),
        Clicks=("Clicks","sum"),
        ROI=("ROI","mean"),
        CR=("Conversion_Rate","mean")
    ).reset_index()
    seg = seg[seg["Impressions"] > 0]
    seg["CTR"] = seg["Clicks"] / seg["Impressions"]

    top = seg.sort_values("ROI", ascending=False).head(10)
    low = seg.sort_values("ROI", ascending=True).head(10)

    colT, colL = st.columns(2)
    with colT:
        st.markdown("**Top 10 by ROI**")
        st.dataframe(top, use_container_width=True)
    with colL:
        st.markdown("**Bottom 10 by ROI**")
        st.dataframe(low, use_container_width=True)
else:
    st.info("ROI or segment columns missing.")

st.divider()

# ---------- 9) Recommendations (simple rules) ----------
st.subheader("🧭 Recommendations")
recs = []

# channel winners
if "Channel_Used" in f.columns and len(f):
    if "ROI" in f.columns:
        by_ch2 = f.groupby("Channel_Used")["ROI"].mean().sort_values(ascending=False)
        if len(by_ch2) >= 2:
            best = by_ch2.index[0]
            recs.append(f"Shift budget toward **{best}** (highest avg ROI). Test +10–20% reallocation for next cycle.")
    if "Clicks" in f.columns and "Impressions" in f.columns:
        by_ch_ctr = f.groupby("Channel_Used").apply(lambda x: (x["Clicks"].sum()/x["Impressions"].sum()) if x["Impressions"].sum()>0 else np.nan)
        by_ch_ctr = by_ch_ctr.sort_values(ascending=False)
        if len(by_ch_ctr) >= 1:
            ctr_best = by_ch_ctr.index[0]
            recs.append(f"Use **{ctr_best}** for reach/engagement campaigns (best CTR).")

# audience × channel hotspots
if {"Target_Audience","Channel_Used","Conversion_Rate"}.issubset(f.columns):
    hot = f.groupby(["Target_Audience","Channel_Used"])["Conversion_Rate"].mean().sort_values(ascending=False).head(3)
    if len(hot):
        pairs = ", ".join([f"{a}×{c}" for (a,c) in hot.index])
        recs.append(f"Double-down on high-CR combos: {pairs}.")

# goal-specific nudge
if "Campaign_Goal" in f.columns and "ROI" in f.columns:
    goal_roi = f.groupby("Campaign_Goal")["ROI"].mean().sort_values(ascending=False)
    if len(goal_roi):
        top_goal = goal_roi.index[0]
        recs.append(f"For **{top_goal}** objectives, replicate top creatives/targeting from best channels.")

# cost sanity
if f["Est_Spend"].notna().any() and "ROI" in f.columns:
    cost_roi = f[["Est_Spend","ROI"]].dropna()
    if len(cost_roi) > 100:
        corr = cost_roi.corr(numeric_only=True).loc["Est_Spend","ROI"]
        if pd.notna(corr) and corr < 0:
            recs.append("High spend correlates with lower ROI — consider narrowing targeting or improving creative quality.")

if recs:
    for r in recs:
        st.markdown(f"- {r}")
else:
    st.write("No strong signals found. Keep monitoring and A/B test new creatives and narrower segments.")
