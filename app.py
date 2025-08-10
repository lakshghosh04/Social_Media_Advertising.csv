# app.py — Upload-first Social Media Ads BI (CSV/ZIP + column mapper)
import os, io, zipfile
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Social Media Ads BI", layout="wide")
st.title("📊 Social Media Advertising — BI Dashboard")
st.caption("Build v13 — upload your own CSV/ZIP")

# -------------------------
# 1) Upload / load helpers
# -------------------------
REQUIRED = [
    "Date", "Channel_Used", "Target_Audience",
    "Impressions", "Clicks", "Conversion_Rate", "ROI"
]
OPTIONAL = ["Company", "Location", "Campaign_Goal", "Engagement_Score", "Acquisition_Cost"]

def read_uploaded(file) -> pd.DataFrame | None:
    """Read CSV directly or first CSV inside a ZIP."""
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file, encoding="utf-8", encoding_errors="replace")
    if name.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(file.read())) as z:
            csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not csvs: 
                st.error("ZIP has no CSV inside.")
                return None
            with z.open(csvs[0]) as f:
                return pd.read_csv(f, encoding="utf-8", encoding_errors="replace")
    st.error("Please upload a .csv or .zip containing a CSV.")
    return None

def coerce_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def parse_date(df, col):
    df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

# -------------------------
# 2) Sidebar: Upload + mapping
# -------------------------
st.sidebar.header("📥 Upload your data")
up = st.sidebar.file_uploader("Upload CSV or ZIP", type=["csv", "zip"])

if "df" not in st.session_state:
    st.session_state.df = None

if up:
    raw_df = read_uploaded(up)
    if raw_df is not None and len(raw_df):
        st.session_state.df = raw_df

if st.session_state.df is None:
    st.info("Upload a CSV or a ZIP with a CSV to begin. Expected fields include: " + ", ".join(REQUIRED + OPTIONAL))
    st.stop()

df = st.session_state.df.copy()
st.write("### Preview")
st.dataframe(df.head(), use_container_width=True)

# -------------------------
# 3) Column mapping (simple)
# -------------------------
st.write("### Column mapping")
mapping = {}
for need in REQUIRED + OPTIONAL:
    candidates = ["(none)"] + df.columns.tolist()
    default = candidates.index(need) if need in df.columns else 0
    mapping[need] = st.selectbox(f"Select column for **{need}**", candidates, index=default)

# Ensure required mapped
missing = [k for k,v in mapping.items() if k in REQUIRED and v == "(none)"]
if missing:
    st.error(f"Please map required fields: {', '.join(missing)}")
    st.stop()

# Build normalized frame
norm = pd.DataFrame()
for k,v in mapping.items():
    if v != "(none)":
        norm[k] = df[v]

# Clean/parse types
if "Date" in norm.columns:
    norm = parse_date(norm, "Date")
num_cols = ["Conversion_Rate", "ROI", "Clicks", "Impressions", "Engagement_Score"]
norm = coerce_numeric(norm, num_cols)

# Acquisition_Cost may be like "$500"
if "Acquisition_Cost" in norm.columns:
    norm["Acquisition_Cost_Num"] = (
        norm["Acquisition_Cost"]
        .astype(str)
        .str.replace(r"[^0-9.\-]", "", regex=True)
        .replace("", np.nan)
        .astype(float)
    )
else:
    norm["Acquisition_Cost_Num"] = np.nan

# Derived metrics
norm["CTR"] = np.where(norm["Impressions"] > 0, norm["Clicks"] / norm["Impressions"], np.nan)
est_conversions = np.where(
    (norm["Conversion_Rate"].notna()) & (norm["Impressions"].notna()),
    norm["Conversion_Rate"] * norm["Impressions"],
    np.nan
)
norm["Est_Spend"] = norm["Acquisition_Cost_Num"] * est_conversions

# -------------------------
# 4) Filters
# -------------------------
st.sidebar.header("🔎 Filters")
if norm["Date"].notna().any():
    dmin, dmax = norm["Date"].min(), norm["Date"].max()
    date_range = st.sidebar.date_input("Date range", (dmin.date() if pd.notna(dmin) else None,
                                                      dmax.date() if pd.notna(dmax) else None))
else:
    date_range = None

def opt_values(col):
    return ["All"] + sorted(norm[col].dropna().unique().tolist()) if col in norm.columns else ["All"]

sel_channel  = st.sidebar.selectbox("Channel",  opt_values("Channel_Used"))
sel_audience = st.sidebar.selectbox("Audience", opt_values("Target_Audience"))
sel_company  = st.sidebar.selectbox("Company",  opt_values("Company") if "Company" in norm.columns else ["All"])
sel_loc      = st.sidebar.selectbox("Location", opt_values("Location") if "Location" in norm.columns else ["All"])

f = norm.copy()
if date_range and isinstance(date_range, (list, tuple)) and len(date_range) == 2 and all(date_range):
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    f = f[(f["Date"] >= start) & (f["Date"] <= end)]
if sel_channel != "All":
    f = f[f["Channel_Used"] == sel_channel]
if sel_audience != "All":
    f = f[f["Target_Audience"] == sel_audience]
if sel_company != "All" and "Company" in f.columns:
    f = f[f["Company"] == sel_company]
if sel_loc != "All" and "Location" in f.columns:
    f = f[f["Location"] == sel_loc]

# -------------------------
# 5) KPI cards
# -------------------------
st.write("### KPIs")
total_impr = int(f["Impressions"].sum())
total_clicks = int(f["Clicks"].sum())
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
st.caption("Est. Spend ≈ Acquisition_Cost × (Conversion_Rate × Impressions)")

st.divider()

# -------------------------
# 6) Trends
# -------------------------
st.subheader("📈 Trends over time")
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

# -------------------------
# 7) Channel comparison
# -------------------------
st.subheader("📊 Channel performance")
if len(f):
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
        st.plotly_chart(px.bar(by_ch, x="Channel_Used", y="CTR", title="CTR by channel"), use_container_width=True)
    with colB:
        st.markdown("**ROI by channel**")
        st.plotly_chart(px.bar(by_ch, x="Channel_Used", y="ROI", title="ROI by channel"), use_container_width=True)
else:
    st.info("No channel data after filters.")

st.divider()

# -------------------------
# 8) Audience × Channel heatmap
# -------------------------
st.subheader("🧩 Audience × Channel (Conversion Rate)")
if {"Target_Audience","Channel_Used","Conversion_Rate"}.issubset(f.columns):
    heat = f.pivot_table(index="Target_Audience", columns="Channel_Used", values="Conversion_Rate", aggfunc="mean")
    heat = heat.round(3)
    st.plotly_chart(px.imshow(heat, text_auto=True, aspect="auto", title="Avg Conversion Rate heatmap"),
                    use_container_width=True)
else:
    st.info("Need Target_Audience, Channel_Used, and Conversion_Rate for the heatmap.")

st.divider()

# -------------------------
# 9) Top & bottom performers
# -------------------------
st.subheader("🏆 Top & Bottom performers (by ROI)")
segment_cols = [c for c in ["Company","Target_Audience","Channel_Used","Campaign_Goal","Location"] if c in f.columns]
if "ROI" in f.columns and segment_cols:
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
    colT.markdown("**Top 10 by ROI**"); colT.dataframe(top, use_container_width=True)
    colL.markdown("**Bottom 10 by ROI**"); colL.dataframe(low, use_container_width=True)
else:
    st.info("ROI or segment columns missing.")

st.divider()

# -------------------------
# 10) Recommendations (simple rules)
# -------------------------
st.subheader("🧭 Recommendations")
recs = []

if len(f):
    # Channel winners
    by_ch2 = f.groupby("Channel_Used")["ROI"].mean().sort_values(ascending=False) if "ROI" in f.columns else pd.Series(dtype=float)
    if len(by_ch2) >= 1:
        best = by_ch2.index[0]
        recs.append(f"Shift budget toward **{best}** (highest average ROI). Test +10–20% reallocation next cycle.")

    # CTR leader for awareness/traffic
    if {"Impressions","Clicks","Channel_Used"}.issubset(f.columns):
        by_ch_ctr = f.groupby("Channel_Used").apply(lambda x: (x["Clicks"].sum()/x["Impressions"].sum()) if x["Impressions"].sum()>0 else np.nan)
        by_ch_ctr = by_ch_ctr.sort_values(ascending=False)
        if len(by_ch_ctr) >= 1:
            ctr_best = by_ch_ctr.index[0]
            recs.append(f"Use **{ctr_best}** for awareness/traffic (best CTR).")

    # Audience × Channel hotspots
    if {"Target_Audience","Channel_Used","Conversion_Rate"}.issubset(f.columns):
        hot = f.groupby(["Target_Audience","Channel_Used"])["Conversion_Rate"].mean().sort_values(ascending=False).head(3)
        if len(hot):
            pairs = ", ".join([f"{a}×{c}" for (a,c) in hot.index])
            recs.append(f"Double-down on high-CR combos: {pairs}.")

    # Goal-specific
    if {"Campaign_Goal","ROI"}.issubset(f.columns):
        goal_roi = f.groupby("Campaign_Goal")["ROI"].mean().sort_values(ascending=False)
        if len(goal_roi):
            top_goal = goal_roi.index[0]
            recs.append(f"For **{top_goal}** goals, replicate top creatives/targeting from best channels.")

    # Spend vs ROI sanity (if Acquisition_Cost given)
    if f["Est_Spend"].notna().any() and "ROI" in f.columns:
        cost_roi = f[["Est_Spend","ROI"]].dropna()
        if len(cost_roi) > 50:
            corr = cost_roi.corr(numeric_only=True).loc["Est_Spend","ROI"]
            if pd.notna(corr) and corr < 0:
                recs.append("High spend correlates with lower ROI — narrow targeting or improve creatives.")

if recs:
    for r in recs: st.markdown(f"- {r}")
else:
    st.write("No strong signals yet. Keep monitoring and A/B test creatives + narrower segments.")
