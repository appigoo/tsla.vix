import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import time
from datetime import datetime, timedelta

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TSLA × VIX Correlation Tracker",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

:root {
    --bg: #0a0a0f;
    --surface: #111118;
    --surface2: #1a1a24;
    --accent1: #e8ff47;   /* electric yellow-green */
    --accent2: #ff3d6b;   /* hot red */
    --accent3: #3df5b0;   /* teal */
    --text: #e8e8f0;
    --muted: #6b6b82;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif;
}

.stApp { background: var(--bg) !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid #222230;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] .stSlider > div > div { background: var(--accent1) !important; }

/* Metric cards */
.metric-card {
    background: var(--surface2);
    border: 1px solid #2a2a38;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 4px 0;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.metric-card.yellow::before { background: var(--accent1); }
.metric-card.red::before    { background: var(--accent2); }
.metric-card.teal::before   { background: var(--accent3); }

.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 6px;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 32px;
    font-weight: 800;
    line-height: 1;
}
.metric-value.yellow { color: var(--accent1); }
.metric-value.red    { color: var(--accent2); }
.metric-value.teal   { color: var(--accent3); }
.metric-sub {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: var(--muted);
    margin-top: 4px;
}

/* Section headers */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--muted);
    border-bottom: 1px solid #222230;
    padding-bottom: 8px;
    margin: 24px 0 16px 0;
}

/* Main title */
.main-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 42px;
    line-height: 1.1;
    background: linear-gradient(135deg, var(--accent1) 0%, var(--accent3) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 4px;
}
.main-subtitle {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: var(--muted);
    letter-spacing: 2px;
}

/* Correlation verdict */
.verdict-box {
    background: linear-gradient(135deg, #1a1a24, #0f1520);
    border: 1px solid var(--accent1);
    border-radius: 12px;
    padding: 20px 24px;
    margin: 16px 0;
}
.verdict-title {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 2px;
    color: var(--accent1);
    margin-bottom: 8px;
}
.verdict-text {
    font-size: 15px;
    line-height: 1.6;
    color: var(--text);
}

/* Live badge */
.live-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(232, 255, 71, 0.08);
    border: 1px solid rgba(232, 255, 71, 0.3);
    border-radius: 20px;
    padding: 4px 12px;
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    color: var(--accent1);
    letter-spacing: 1px;
}
.live-dot {
    width: 6px; height: 6px;
    background: var(--accent1);
    border-radius: 50%;
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.2; }
}

/* Plotly chart containers */
.stPlotlyChart { border-radius: 12px; overflow: hidden; }

/* Buttons */
.stButton > button {
    background: var(--accent1) !important;
    color: #000 !important;
    border: none !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 12px !important;
    letter-spacing: 1px !important;
    border-radius: 8px !important;
    padding: 8px 20px !important;
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Helper Functions ────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def fetch_data(period: str, interval: str):
    tsla = yf.download("TSLA", period=period, interval=interval, progress=False)
    vix  = yf.download("^VIX",  period=period, interval=interval, progress=False)
    return tsla, vix

def compute_correlation(tsla_close, vix_close):
    df = pd.DataFrame({"TSLA": tsla_close, "VIX": vix_close}).dropna()
    if len(df) < 5:
        return None, None, None, df
    r, p = stats.pearsonr(df["TSLA"], df["VIX"])
    return r, p, len(df), df

def corr_verdict(r):
    ar = abs(r)
    if ar >= 0.7:
        strength = "STRONG"
    elif ar >= 0.4:
        strength = "MODERATE"
    else:
        strength = "WEAK"
    direction = "NEGATIVE" if r < 0 else "POSITIVE"
    return strength, direction

def rolling_corr(df, window=20):
    return df["TSLA"].rolling(window).corr(df["VIX"])

# ─── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-header">⚙ SETTINGS</div>', unsafe_allow_html=True)

    period_map = {
        "5 Days":   ("5d",  "5m"),
        "1 Month":  ("1mo", "1h"),
        "3 Months": ("3mo", "1d"),
        "6 Months": ("6mo", "1d"),
        "1 Year":   ("1y",  "1d"),
        "3 Years":  ("3y",  "1wk"),
    }
    period_label = st.selectbox("Time Period", list(period_map.keys()), index=2)
    period, interval = period_map[period_label]

    roll_window = st.slider("Rolling Corr Window (bars)", 5, 60, 20)
    auto_refresh = st.checkbox("Auto-refresh (60s)", value=False)
    normalize = st.checkbox("Normalize prices (z-score)", value=True)

    st.markdown('<div class="section-header">ℹ ABOUT</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Space Mono',monospace;font-size:11px;color:#6b6b82;line-height:1.8">
    TSLA tends to move <span style="color:#ff3d6b">inversely</span> with VIX.<br><br>
    When fear spikes (↑VIX), high-beta growth stocks like TSLA sell off (↓TSLA).<br><br>
    This dashboard quantifies that relationship in real time.
    </div>
    """, unsafe_allow_html=True)

# ─── Main Layout ─────────────────────────────────────────────────────────────────
col_title, col_badge = st.columns([6, 1])
with col_title:
    st.markdown('<div class="main-title">TSLA × VIX</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">REAL-TIME CORRELATION TRACKER</div>', unsafe_allow_html=True)
with col_badge:
    st.markdown("""
    <div style="padding-top:20px">
        <div class="live-badge"><div class="live-dot"></div>LIVE</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ─── Fetch Data ──────────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(0.1)
    st.cache_data.clear()

with st.spinner("Fetching market data..."):
    tsla_df, vix_df = fetch_data(period, interval)

if tsla_df.empty or vix_df.empty:
    st.error("⚠ Could not fetch data. Check your internet connection.")
    st.stop()

# Flatten MultiIndex if needed
tsla_close = tsla_df["Close"].squeeze()
vix_close  = vix_df["Close"].squeeze()

r, p, n, df = compute_correlation(tsla_close, vix_close)

# ─── Metrics Row ─────────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)

with m1:
    tsla_last = float(tsla_close.iloc[-1]) if len(tsla_close) else 0
    tsla_chg  = float((tsla_close.iloc[-1] - tsla_close.iloc[-2]) / tsla_close.iloc[-2] * 100) if len(tsla_close) > 1 else 0
    sign = "▲" if tsla_chg >= 0 else "▼"
    clr  = "teal" if tsla_chg >= 0 else "red"
    st.markdown(f"""
    <div class="metric-card yellow">
        <div class="metric-label">TSLA Last Price</div>
        <div class="metric-value yellow">${tsla_last:,.2f}</div>
        <div class="metric-sub" style="color:{'#3df5b0' if tsla_chg>=0 else '#ff3d6b'}">{sign} {abs(tsla_chg):.2f}%</div>
    </div>""", unsafe_allow_html=True)

with m2:
    vix_last = float(vix_close.iloc[-1]) if len(vix_close) else 0
    vix_chg  = float((vix_close.iloc[-1] - vix_close.iloc[-2]) / vix_close.iloc[-2] * 100) if len(vix_close) > 1 else 0
    sign2 = "▲" if vix_chg >= 0 else "▼"
    st.markdown(f"""
    <div class="metric-card red">
        <div class="metric-label">VIX Fear Index</div>
        <div class="metric-value red">{vix_last:.2f}</div>
        <div class="metric-sub" style="color:{'#ff3d6b' if vix_chg>=0 else '#3df5b0'}">{sign2} {abs(vix_chg):.2f}%</div>
    </div>""", unsafe_allow_html=True)

with m3:
    if r is not None:
        strength, direction = corr_verdict(r)
        r_clr = "red" if r < 0 else "teal"
        st.markdown(f"""
        <div class="metric-card teal">
            <div class="metric-label">Pearson Correlation (r)</div>
            <div class="metric-value {r_clr}">{r:.3f}</div>
            <div class="metric-sub">{strength} {direction} · n={n}</div>
        </div>""", unsafe_allow_html=True)

with m4:
    if p is not None:
        p_str = f"{p:.2e}" if p < 0.001 else f"{p:.4f}"
        sig = "✓ SIGNIFICANT" if p < 0.05 else "✗ NOT SIGNIFICANT"
        sig_clr = "#3df5b0" if p < 0.05 else "#ff3d6b"
        st.markdown(f"""
        <div class="metric-card yellow">
            <div class="metric-label">P-Value (Statistical)</div>
            <div class="metric-value yellow" style="font-size:22px">{p_str}</div>
            <div class="metric-sub" style="color:{sig_clr}">{sig} (α=0.05)</div>
        </div>""", unsafe_allow_html=True)

# ─── Verdict Box ─────────────────────────────────────────────────────────────────
if r is not None:
    strength, direction = corr_verdict(r)
    ar = abs(r)
    if r < -0.4:
        msg = (f"With r = <strong>{r:.3f}</strong> and p = {p:.2e}, there is a <strong>statistically significant negative correlation</strong> "
               f"between TSLA and VIX over the selected period. When fear spikes, TSLA sells off — confirming its high-beta, "
               f"risk-on character. The relationship explains <strong>{ar**2*100:.1f}%</strong> of the variance (R²).")
    elif r < 0:
        msg = (f"r = <strong>{r:.3f}</strong> suggests a weak-to-moderate negative relationship. "
               f"TSLA tends to move against fear, but the signal is noisy over this timeframe.")
    else:
        msg = (f"r = <strong>{r:.3f}</strong> shows an unusual positive relationship in this window. "
               f"This may reflect a short-term anomaly or specific macro regime.")

    st.markdown(f"""
    <div class="verdict-box">
        <div class="verdict-title">⚡ ANALYSIS VERDICT</div>
        <div class="verdict-text">{msg}</div>
    </div>""", unsafe_allow_html=True)

# ─── Chart 1: Price / Normalized Series ──────────────────────────────────────────
st.markdown('<div class="section-header">01 — PRICE SERIES</div>', unsafe_allow_html=True)

fig1 = make_subplots(specs=[[{"secondary_y": True}]])

if normalize:
    tsla_plot = (df["TSLA"] - df["TSLA"].mean()) / df["TSLA"].std()
    vix_plot  = (df["VIX"]  - df["VIX"].mean())  / df["VIX"].std()
    y_title_l, y_title_r = "Z-Score", "Z-Score"
else:
    tsla_plot = df["TSLA"]
    vix_plot  = df["VIX"]
    y_title_l, y_title_r = "TSLA Price (USD)", "VIX Level"

fig1.add_trace(go.Scatter(
    x=df.index, y=tsla_plot,
    name="TSLA", line=dict(color="#e8ff47", width=2),
    fill="tozeroy", fillcolor="rgba(232,255,71,0.05)"
), secondary_y=False)

fig1.add_trace(go.Scatter(
    x=df.index, y=vix_plot,
    name="VIX", line=dict(color="#ff3d6b", width=2, dash="dot"),
), secondary_y=True if not normalize else False)

fig1.update_layout(
    paper_bgcolor="#111118", plot_bgcolor="#0d0d15",
    font=dict(family="Space Mono", color="#6b6b82", size=10),
    legend=dict(bgcolor="#111118", bordercolor="#222230", borderwidth=1,
                font=dict(color="#e8e8f0")),
    margin=dict(l=10, r=10, t=20, b=10),
    height=320,
    xaxis=dict(gridcolor="#1a1a24", zeroline=False, showline=False),
    yaxis=dict(gridcolor="#1a1a24", zeroline=False, title=y_title_l),
    yaxis2=dict(gridcolor="rgba(0,0,0,0)", zeroline=False, title=y_title_r,
                showgrid=False) if not normalize else {},
    hovermode="x unified",
)

st.plotly_chart(fig1, use_container_width=True)

# ─── Chart 2: Rolling Correlation + Scatter ──────────────────────────────────────
col_roll, col_scat = st.columns([3, 2])

with col_roll:
    st.markdown('<div class="section-header">02 — ROLLING CORRELATION</div>', unsafe_allow_html=True)
    roll = rolling_corr(df, window=roll_window).dropna()

    fig2 = go.Figure()
    fig2.add_hrect(y0=-1, y1=-0.4, fillcolor="rgba(255,61,107,0.07)", line_width=0)
    fig2.add_hrect(y0=0.4,  y1=1, fillcolor="rgba(61,245,176,0.05)", line_width=0)
    fig2.add_hline(y=0, line_color="#2a2a38", line_width=1)
    fig2.add_hline(y=-0.4, line_color="#ff3d6b", line_width=1, line_dash="dash")
    fig2.add_hline(y=0.4,  line_color="#3df5b0", line_width=1, line_dash="dash")

    # Color the line by value
    fig2.add_trace(go.Scatter(
        x=roll.index, y=roll.values,
        mode="lines",
        line=dict(width=2.5, color="#3df5b0"),
        name=f"Rolling r ({roll_window}-bar)",
        fill="tozeroy",
        fillcolor="rgba(61,245,176,0.05)",
    ))

    fig2.add_annotation(
        x=roll.index[-1], y=float(roll.iloc[-1]),
        text=f"  r={float(roll.iloc[-1]):.3f}",
        showarrow=False, font=dict(color="#e8ff47", size=12, family="Space Mono"),
        xanchor="left"
    )

    fig2.update_layout(
        paper_bgcolor="#111118", plot_bgcolor="#0d0d15",
        font=dict(family="Space Mono", color="#6b6b82", size=10),
        margin=dict(l=10, r=10, t=10, b=10),
        height=300,
        yaxis=dict(range=[-1.05, 1.05], gridcolor="#1a1a24", zeroline=False,
                   title="Pearson r"),
        xaxis=dict(gridcolor="#1a1a24", zeroline=False),
        showlegend=False,
        hovermode="x unified",
    )
    st.plotly_chart(fig2, use_container_width=True)

with col_scat:
    st.markdown('<div class="section-header">03 — SCATTER PLOT</div>', unsafe_allow_html=True)

    # Regression line
    slope, intercept, *_ = stats.linregress(df["VIX"], df["TSLA"])
    x_line = np.linspace(df["VIX"].min(), df["VIX"].max(), 100)
    y_line = slope * x_line + intercept

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=df["VIX"], y=df["TSLA"],
        mode="markers",
        marker=dict(
            color=df["TSLA"].values,
            colorscale=[[0, "#ff3d6b"], [0.5, "#6b6b82"], [1, "#e8ff47"]],
            size=5, opacity=0.7, line=dict(width=0)
        ),
        name="Observations",
    ))
    fig3.add_trace(go.Scatter(
        x=x_line, y=y_line,
        mode="lines",
        line=dict(color="#e8ff47", width=2, dash="dot"),
        name=f"OLS fit (slope={slope:.2f})",
    ))

    fig3.update_layout(
        paper_bgcolor="#111118", plot_bgcolor="#0d0d15",
        font=dict(family="Space Mono", color="#6b6b82", size=10),
        margin=dict(l=10, r=10, t=10, b=10),
        height=300,
        xaxis=dict(gridcolor="#1a1a24", zeroline=False, title="VIX"),
        yaxis=dict(gridcolor="#1a1a24", zeroline=False, title="TSLA"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#e8e8f0", size=9)),
        hovermode="closest",
    )
    st.plotly_chart(fig3, use_container_width=True)

# ─── Stats Table ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">04 — STATISTICS SUMMARY</div>', unsafe_allow_html=True)

stats_data = {
    "Metric": ["Pearson r", "R² (explained variance)", "P-Value", "OLS Slope", "OLS Intercept",
               "TSLA Mean", "VIX Mean", "TSLA Std Dev", "VIX Std Dev", "Sample Size"],
    "Value": [
        f"{r:.4f}",
        f"{r**2:.4f} ({r**2*100:.1f}%)",
        f"{p:.2e}",
        f"{slope:.4f}",
        f"{intercept:.4f}",
        f"{df['TSLA'].mean():.2f}",
        f"{df['VIX'].mean():.2f}",
        f"{df['TSLA'].std():.2f}",
        f"{df['VIX'].std():.2f}",
        str(n),
    ],
    "Interpretation": [
        "Strong negative → inverse relationship" if r < -0.4 else ("Weak" if abs(r) < 0.4 else "Moderate"),
        f"{r**2*100:.1f}% of TSLA variance explained by VIX",
        "Highly significant" if p < 0.001 else ("Significant" if p < 0.05 else "Not significant"),
        f"TSLA moves {slope:.2f} per unit VIX change",
        "Baseline TSLA level when VIX=0",
        "Average TSLA over period",
        "Average fear level over period",
        "TSLA volatility",
        "VIX volatility",
        "Number of data points",
    ]
}
stats_df = pd.DataFrame(stats_data)
st.dataframe(
    stats_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Metric": st.column_config.TextColumn("Metric", width=180),
        "Value": st.column_config.TextColumn("Value", width=180),
        "Interpretation": st.column_config.TextColumn("Interpretation"),
    }
)

# ─── Footer ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="font-family:'Space Mono',monospace;font-size:10px;color:#3a3a50;
            text-align:center;margin-top:32px;padding:16px;border-top:1px solid #1a1a24">
    Data via Yahoo Finance · Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ·
    Period: {period_label} ({interval} bars) · Not financial advice.
</div>
""", unsafe_allow_html=True)

# ─── Auto-refresh ────────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(60)
    st.rerun()
