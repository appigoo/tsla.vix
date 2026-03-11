"""
TSLA × VIX 实时相关性追踪器
- 1分钟级别实时数据（含盘前/盘后）
- 自动每30秒刷新
- 全中文界面
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import time
import datetime as dt
from datetime import datetime
import pytz

# ══════════════════════════════════════════════════════════
# 页面配置
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="TSLA × VIX 实时追踪",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════
# 全局样式
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Noto+Sans+SC:wght@400;700;900&display=swap');

:root {
    --bg:     #07080f;
    --surf:   #0e0f1a;
    --surf2:  #141525;
    --border: #1e1f35;
    --acc1:   #e8ff47;
    --acc2:   #ff3d6b;
    --acc3:   #3df5b0;
    --acc4:   #7b7bff;
    --text:   #dde1f5;
    --muted:  #5a5c78;
}
html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Noto Sans SC', 'Space Mono', sans-serif;
}
.stApp { background: var(--bg) !important; }

[data-testid="stSidebar"] {
    background: var(--surf) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* 指标卡 */
.kcard {
    background: var(--surf2);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px 22px 16px;
    position: relative;
    overflow: hidden;
    margin-bottom: 6px;
}
.kcard::after {
    content:''; position:absolute;
    top:0; left:0; right:0; height:3px;
    border-radius:14px 14px 0 0;
}
.kcard.y::after { background: var(--acc1); }
.kcard.r::after { background: var(--acc2); }
.kcard.t::after { background: var(--acc3); }
.kcard.p::after { background: var(--acc4); }
.klabel {
    font-size:10px; letter-spacing:2px;
    color:var(--muted); margin-bottom:5px;
    font-family:'Space Mono',monospace;
}
.kval {
    font-size:30px; font-weight:900; line-height:1;
}
.kval.y { color:var(--acc1); }
.kval.r { color:var(--acc2); }
.kval.t { color:var(--acc3); }
.kval.p { color:var(--acc4); }
.ksub {
    font-size:11px; color:var(--muted);
    margin-top:5px; font-family:'Space Mono',monospace;
}

/* 分节标题 */
.sec {
    font-family:'Space Mono',monospace;
    font-size:10px; letter-spacing:3px;
    color:var(--muted);
    border-bottom:1px solid var(--border);
    padding-bottom:6px; margin:20px 0 12px;
}

/* 主标题 */
.htitle {
    font-size:46px; font-weight:900; line-height:1;
    background: linear-gradient(120deg, var(--acc1), var(--acc3));
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text;
}
.hsub {
    font-family:'Space Mono',monospace;
    font-size:10px; letter-spacing:2px; color:var(--muted);
}

/* 结论框 */
.verdict {
    background: linear-gradient(135deg,#141525,#0a0b15);
    border:1px solid var(--acc1);
    border-radius:14px; padding:18px 22px; margin:12px 0;
}
.vt { font-family:'Space Mono',monospace; font-size:9px;
      letter-spacing:2px; color:var(--acc1); margin-bottom:8px; }
.vb { font-size:14px; line-height:1.7; color:var(--text); }

/* 交易时段徽章 */
.badge {
    display:inline-flex; align-items:center; gap:6px;
    border-radius:20px; padding:4px 14px;
    font-family:'Space Mono',monospace;
    font-size:10px; letter-spacing:1px;
}
.badge.pre  { background:rgba(123,123,255,.12); border:1px solid rgba(123,123,255,.4); color:var(--acc4); }
.badge.open { background:rgba(61,245,176,.10);  border:1px solid rgba(61,245,176,.4);  color:var(--acc3); }
.badge.post { background:rgba(255,61,107,.10);  border:1px solid rgba(255,61,107,.35); color:var(--acc2); }
.badge.clos { background:rgba(90,92,120,.15);   border:1px solid rgba(90,92,120,.4);   color:var(--muted); }
.dot { width:6px; height:6px; border-radius:50%; animation:blink 1.4s infinite; display:inline-block; }
.dot.g { background:var(--acc3); }
.dot.b { background:var(--acc4); }
@keyframes blink { 0%,100%{opacity:1}50%{opacity:.15} }

#MainMenu, footer, header { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════

ET = pytz.timezone("America/New_York")


def get_market_session():
    """返回 (key, 中文标签, ET时间字符串)"""
    now = datetime.now(ET)
    t   = now.time()
    wd  = now.weekday()
    ts  = now.strftime("%H:%M ET")

    if wd >= 5:
        return "closed", "休市（周末）", ts
    if t < dt.time(4, 0) or t >= dt.time(20, 0):
        return "closed", "休市（场外）", ts
    if dt.time(4, 0) <= t < dt.time(9, 30):
        return "pre",    "盘前交易", ts
    if dt.time(9, 30) <= t < dt.time(16, 0):
        return "open",   "正常交易", ts
    return "post", "盘后交易", ts


@st.cache_data(ttl=30)
def fetch_1min(days_back: int = 5):
    """1 分钟 K 线，含盘前盘后（prepost=True）"""
    tsla = yf.download("TSLA", period=f"{days_back}d", interval="1m",
                        prepost=True, progress=False, auto_adjust=True)
    vix  = yf.download("^VIX",  period=f"{days_back}d", interval="1m",
                        prepost=True, progress=False, auto_adjust=True)
    return tsla, vix


@st.cache_data(ttl=60)
def fetch_history(period: str, interval: str):
    tsla = yf.download("TSLA", period=period, interval=interval,
                        progress=False, auto_adjust=True)
    vix  = yf.download("^VIX",  period=period, interval=interval,
                        progress=False, auto_adjust=True)
    return tsla, vix


def fetch_rt_price(ticker: str):
    """最新报价及昨日收盘"""
    try:
        fi = yf.Ticker(ticker).fast_info
        return getattr(fi, "last_price", None), getattr(fi, "previous_close", None)
    except Exception:
        return None, None


def build_df(tsla_raw, vix_raw):
    tc = tsla_raw["Close"].squeeze().rename("TSLA")
    vc = vix_raw["Close"].squeeze().rename("VIX")
    return pd.concat([tc, vc], axis=1).dropna()


def pearson_corr(df):
    if len(df) < 5:
        return None, None
    r, p = stats.pearsonr(df["TSLA"].values, df["VIX"].values)
    return r, p


def strength_zh(r):
    a = abs(r)
    s = "强" if a >= 0.7 else ("中等" if a >= 0.4 else "弱")
    d = "负" if r < 0 else "正"
    return s, d


def pct(cur, prev):
    if cur and prev and prev != 0:
        return (cur - prev) / prev * 100
    return 0.0


def tag_session(index, tz=ET):
    """给时间索引打交易时段标签"""
    if hasattr(index, "tz_convert"):
        idx_et = index.tz_convert(tz)
    else:
        idx_et = index
    labels = []
    for t in idx_et:
        tt = t.time()
        if dt.time(4, 0) <= tt < dt.time(9, 30):
            labels.append("pre")
        elif dt.time(9, 30) <= tt < dt.time(16, 0):
            labels.append("open")
        elif dt.time(16, 0) <= tt < dt.time(20, 0):
            labels.append("post")
        else:
            labels.append("closed")
    return labels


# ══════════════════════════════════════════════════════════
# 侧边栏
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sec">⚙ 控制面板</div>', unsafe_allow_html=True)

    auto_refresh = st.checkbox("🔄 自动刷新（每30秒）", value=True)

    hist_map = {
        "今日（1分钟）":   ("1d",  "1m"),
        "近5日（1分钟）":  ("5d",  "1m"),
        "近1月（1小时）":  ("1mo", "1h"),
        "近3月（日线）":   ("3mo", "1d"),
        "近1年（日线）":   ("1y",  "1d"),
    }
    period_label = st.selectbox("历史对比时段", list(hist_map.keys()), index=1)
    hist_period, hist_interval = hist_map[period_label]

    roll_window = st.slider("滚动相关窗口（根 K 线）", 5, 60, 20)
    normalize   = st.checkbox("标准化叠加（Z-Score）", value=True)

    st.markdown('<div class="sec">📌 背景知识</div>', unsafe_allow_html=True)
    st.markdown("""
<div style="font-size:12px;color:#5a5c78;line-height:1.9">
<b style="color:#dde1f5">TSLA 与 VIX 为何负相关？</b><br><br>
· <b style="color:#ff3d6b">VIX 上升</b>（恐慌加剧）→ 资金避险撤出成长股 → <b style="color:#ff3d6b">TSLA 下跌</b><br><br>
· <b style="color:#3df5b0">VIX 下降</b>（情绪平稳）→ 风险偏好回升 → <b style="color:#3df5b0">TSLA 上涨</b><br><br>
TSLA 贝塔值约 <b style="color:#e8ff47">2.0</b>，对市场情绪高度敏感，是追踪此负相关的理想标的。
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# 顶栏
# ══════════════════════════════════════════════════════════
session_key, session_zh_label, et_time = get_market_session()
badge_cls = {"pre": "pre", "open": "open", "post": "post", "closed": "clos"}[session_key]

col_h, col_b = st.columns([5, 2])
with col_h:
    st.markdown('<div class="htitle">TSLA × VIX</div>', unsafe_allow_html=True)
    st.markdown('<div class="hsub">实时相关性追踪 · 1分钟级别 · 含盘前/盘后交易时段</div>',
                unsafe_allow_html=True)
with col_b:
    dot = '<span class="dot g"></span>' if session_key == "open" else (
          '<span class="dot b"></span>' if session_key == "pre" else "●")
    st.markdown(f"""
    <div style="padding-top:24px;text-align:right">
      <div class="badge {badge_cls}">{dot} {session_zh_label} · {et_time}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════
# 数据加载
# ══════════════════════════════════════════════════════════
now_str = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S ET")

with st.spinner("⏳ 正在拉取 1 分钟实时行情（含盘前数据）…"):
    tsla_1m_raw, vix_1m_raw = fetch_1min(days_back=5)

if tsla_1m_raw.empty or vix_1m_raw.empty:
    st.error("⚠ 数据获取失败，请检查网络或稍后重试。")
    st.stop()

df1m = build_df(tsla_1m_raw, vix_1m_raw)
sessions_1m = tag_session(df1m.index)

r1m, p1m = pearson_corr(df1m)

# ── 实时报价
tsla_rt, tsla_prev = fetch_rt_price("TSLA")
vix_rt,  vix_prev  = fetch_rt_price("^VIX")

# ── 使用1分钟最后一根兜底
if not tsla_rt and len(df1m):
    tsla_rt = float(df1m["TSLA"].iloc[-1])
if not vix_rt and len(df1m):
    vix_rt  = float(df1m["VIX"].iloc[-1])
if not tsla_prev and len(df1m) > 390:
    tsla_prev = float(df1m["TSLA"].iloc[-391])
if not vix_prev and len(df1m) > 390:
    vix_prev  = float(df1m["VIX"].iloc[-391])

tsla_chg = pct(tsla_rt, tsla_prev)
vix_chg  = pct(vix_rt,  vix_prev)

# ══════════════════════════════════════════════════════════
# 指标卡
# ══════════════════════════════════════════════════════════
c1, c2, c3, c4 = st.columns(4)

with c1:
    v = f"${tsla_rt:,.2f}" if tsla_rt else "—"
    s = "▲" if tsla_chg >= 0 else "▼"
    c = "#3df5b0" if tsla_chg >= 0 else "#ff3d6b"
    st.markdown(f"""
    <div class="kcard y">
      <div class="klabel">TSLA 实时价格</div>
      <div class="kval y">{v}</div>
      <div class="ksub" style="color:{c}">{s} {abs(tsla_chg):.2f}%&nbsp;&nbsp;较昨日收盘</div>
    </div>""", unsafe_allow_html=True)

with c2:
    v2 = f"{vix_rt:.2f}" if vix_rt else "—"
    s2 = "▲" if vix_chg >= 0 else "▼"
    c2c = "#ff3d6b" if vix_chg >= 0 else "#3df5b0"
    st.markdown(f"""
    <div class="kcard r">
      <div class="klabel">VIX 恐慌指数</div>
      <div class="kval r">{v2}</div>
      <div class="ksub" style="color:{c2c}">{s2} {abs(vix_chg):.2f}%&nbsp;&nbsp;较昨日收盘</div>
    </div>""", unsafe_allow_html=True)

with c3:
    if r1m is not None:
        st_zh, d_zh = strength_zh(r1m)
        rc = "r" if r1m < 0 else "t"
        st.markdown(f"""
        <div class="kcard t">
          <div class="klabel">皮尔逊相关系数 r（1分钟线）</div>
          <div class="kval {rc}">{r1m:.3f}</div>
          <div class="ksub">{st_zh}{d_zh}相关 · 样本 {len(df1m)} 根</div>
        </div>""", unsafe_allow_html=True)

with c4:
    if p1m is not None:
        p_str = f"{p1m:.2e}" if p1m < 0.001 else f"{p1m:.4f}"
        sig   = "✓ 极度显著" if p1m < 0.001 else ("✓ 显著" if p1m < 0.05 else "✗ 不显著")
        sc    = "#3df5b0" if p1m < 0.05 else "#ff3d6b"
        r2p   = r1m**2 * 100
        st.markdown(f"""
        <div class="kcard p">
          <div class="klabel">P 值 / R² 解释方差</div>
          <div class="kval p" style="font-size:22px">{p_str}</div>
          <div class="ksub" style="color:{sc}">{sig} · R²={r2p:.1f}%</div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# 结论框
# ══════════════════════════════════════════════════════════
if r1m is not None:
    ar = abs(r1m)
    if r1m <= -0.5:
        verdict_msg = (
            f"基于 <b>{len(df1m)}</b> 根 1 分钟 K 线，皮尔逊 r = <b>{r1m:.3f}</b>，"
            f"p 值 = {p1m:.2e}（远低于 0.05 显著性门槛），证明 TSLA 与 VIX 之间存在"
            f"<b>统计上高度显著的强负相关</b>。<br>"
            f"VIX 可解释 TSLA 价格 <b>{ar**2*100:.1f}%</b> 的方差（R²），"
            f"充分验证了市场恐慌加剧 → 特斯拉抛压显著的系统性规律。"
            f"当前交易时段：<b>{session_zh_label}</b>。"
        )
    elif r1m < -0.3:
        verdict_msg = (
            f"r = <b>{r1m:.3f}</b>，当前呈<b>中等负相关</b>（R²={ar**2*100:.1f}%）。"
            f"信号有效但存在噪音，可能受{'盘前流动性偏低' if session_key=='pre' else '盘中事件驱动'}影响，"
            f"负相关规律依然成立。"
        )
    elif r1m < 0:
        verdict_msg = (
            f"r = <b>{r1m:.3f}</b>，负相关信号偏弱。"
            f"当前处于 <b>{session_zh_label}</b>，可能有个股特定催化剂（如宏观数据、财报）"
            f"短暂压过 VIX 的系统性影响，建议延长观测窗口。"
        )
    else:
        verdict_msg = (
            f"r = <b>{r1m:.3f}</b>，当前出现<b>正相关异常</b>。"
            f"这属于短期特殊情况，通常由重大事件（财报、政策发布）主导个股走势所致，"
            f"历史长期负相关规律不受影响。"
        )

    st.markdown(f"""
    <div class="verdict">
      <div class="vt">⚡ 实时分析结论 · {now_str}</div>
      <div class="vb">{verdict_msg}</div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# 图1：1分钟走势图（含盘前/盘后着色带）
# ══════════════════════════════════════════════════════════
st.markdown('<div class="sec">01 — 1分钟实时走势（含盘前 04:00 / 盘后 16:00 ET）</div>',
            unsafe_allow_html=True)

# 标准化
if normalize:
    t_plot = (df1m["TSLA"] - df1m["TSLA"].mean()) / df1m["TSLA"].std()
    v_plot = (df1m["VIX"]  - df1m["VIX"].mean())  / df1m["VIX"].std()
    y_lab  = "Z-Score（标准化）"
    use_sec2 = False
else:
    t_plot, v_plot = df1m["TSLA"], df1m["VIX"]
    y_lab  = "价格 / 指数"
    use_sec2 = True

fig1 = make_subplots(specs=[[{"secondary_y": use_sec2}]])

# 盘前/盘后背景色块（按日分组，避免过多 vrect）
sess_series = pd.Series(sessions_1m, index=df1m.index)
in_pre  = sess_series == "pre"
in_post = sess_series == "post"

def add_bands(fig, mask, color, label):
    start = None
    idx = df1m.index
    vals = mask.values
    for i, v in enumerate(vals):
        if v and start is None:
            start = idx[i]
        elif not v and start is not None:
            fig.add_vrect(x0=start, x1=idx[i-1],
                          fillcolor=color, line_width=0,
                          annotation_text=label,
                          annotation_font=dict(size=8,
                              color="#7b7bff" if "盘前" in label else "#ff3d6b"),
                          annotation_position="top left")
            start = None
    if start is not None:
        fig.add_vrect(x0=start, x1=idx[-1],
                      fillcolor=color, line_width=0)

add_bands(fig1, in_pre,  "rgba(123,123,255,0.07)", "盘前")
add_bands(fig1, in_post, "rgba(255,61,107,0.06)",  "盘后")

fig1.add_trace(go.Scatter(
    x=df1m.index, y=t_plot,
    name="TSLA", mode="lines",
    line=dict(color="#e8ff47", width=1.8),
    fill="tozeroy", fillcolor="rgba(232,255,71,0.04)",
    hovertemplate="TSLA: %{y:.3f}<extra></extra>",
), secondary_y=False)

fig1.add_trace(go.Scatter(
    x=df1m.index, y=v_plot,
    name="VIX", mode="lines",
    line=dict(color="#ff3d6b", width=1.8, dash="dot"),
    hovertemplate="VIX: %{y:.3f}<extra></extra>",
), secondary_y=use_sec2)

# 标注最新值
if len(df1m):
    fig1.add_annotation(
        x=df1m.index[-1], y=float(t_plot.iloc[-1]),
        text=f"  ¥{float(df1m['TSLA'].iloc[-1]):.2f}",
        showarrow=False,
        font=dict(color="#e8ff47", size=11, family="Space Mono"),
        xanchor="left",
    )

fig1.update_layout(
    paper_bgcolor="#0e0f1a", plot_bgcolor="#09090f",
    font=dict(family="'Noto Sans SC',monospace", color="#5a5c78", size=10),
    legend=dict(bgcolor="#0e0f1a", bordercolor="#1e1f35", borderwidth=1,
                font=dict(color="#dde1f5")),
    margin=dict(l=8, r=8, t=16, b=8), height=360,
    xaxis=dict(gridcolor="#141525", zeroline=False, rangeslider=dict(visible=False)),
    yaxis=dict(gridcolor="#141525", zeroline=False, title=y_lab),
    yaxis2=dict(gridcolor="rgba(0,0,0,0)", zeroline=False,
                title="VIX", showgrid=False) if use_sec2 else {},
    hovermode="x unified",
)
st.plotly_chart(fig1, use_container_width=True)

# ══════════════════════════════════════════════════════════
# 图2 + 图3：滚动相关 & 散点图
# ══════════════════════════════════════════════════════════
col_roll, col_scat = st.columns([3, 2])

with col_roll:
    st.markdown(f'<div class="sec">02 — 滚动皮尔逊相关（窗口 {roll_window} 根 K 线）</div>',
                unsafe_allow_html=True)

    roll = df1m["TSLA"].rolling(roll_window).corr(df1m["VIX"]).dropna()

    fig2 = go.Figure()
    fig2.add_hrect(y0=-1,   y1=-0.4, fillcolor="rgba(255,61,107,0.07)", line_width=0)
    fig2.add_hrect(y0=-0.4, y1=0.4,  fillcolor="rgba(90,92,120,0.03)",  line_width=0)
    fig2.add_hrect(y0=0.4,  y1=1,    fillcolor="rgba(61,245,176,0.05)", line_width=0)
    fig2.add_hline(y=-0.4, line_color="#ff3d6b", line_dash="dash", line_width=1,
                   annotation_text="强负相关阈值 −0.4",
                   annotation_font=dict(color="#ff3d6b", size=9))
    fig2.add_hline(y=0,    line_color="#1e1f35",  line_width=1)
    fig2.add_hline(y=0.4,  line_color="#3df5b0",  line_dash="dash", line_width=1,
                   annotation_text="强正相关阈值 +0.4",
                   annotation_font=dict(color="#3df5b0", size=9))

    fig2.add_trace(go.Scatter(
        x=roll.index, y=roll.values, mode="lines",
        line=dict(width=2, color="#7b7bff"),
        fill="tozeroy", fillcolor="rgba(123,123,255,0.06)",
        name=f"滚动 r（{roll_window}根）",
        hovertemplate="时间: %{x}<br>r = %{y:.3f}<extra></extra>",
    ))

    if len(roll):
        cur_r = float(roll.iloc[-1])
        fig2.add_annotation(
            x=roll.index[-1], y=cur_r,
            text=f"  当前 r = {cur_r:.3f}",
            showarrow=False,
            font=dict(color="#e8ff47", size=11, family="Space Mono"),
            xanchor="left",
        )

    fig2.update_layout(
        paper_bgcolor="#0e0f1a", plot_bgcolor="#09090f",
        font=dict(color="#5a5c78", size=10),
        margin=dict(l=8, r=8, t=8, b=8), height=310,
        yaxis=dict(range=[-1.05, 1.05], gridcolor="#141525",
                   zeroline=False, title="皮尔逊 r"),
        xaxis=dict(gridcolor="#141525", zeroline=False),
        showlegend=False, hovermode="x unified",
    )
    st.plotly_chart(fig2, use_container_width=True)

with col_scat:
    st.markdown('<div class="sec">03 — 散点图 + OLS 线性回归</div>', unsafe_allow_html=True)

    slope_v, intercept_v, *_ = stats.linregress(df1m["VIX"].values, df1m["TSLA"].values)
    xr = np.linspace(df1m["VIX"].min(), df1m["VIX"].max(), 200)
    yr = slope_v * xr + intercept_v

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=df1m["VIX"], y=df1m["TSLA"],
        mode="markers",
        marker=dict(
            color=df1m["TSLA"].values,
            colorscale=[[0, "#ff3d6b"], [0.5, "#5a5c78"], [1, "#e8ff47"]],
            size=3, opacity=0.5, line=dict(width=0),
        ),
        name="观测点",
        hovertemplate="VIX=%{x:.2f}  TSLA=%{y:.2f}<extra></extra>",
    ))
    fig3.add_trace(go.Scatter(
        x=xr, y=yr, mode="lines",
        line=dict(color="#e8ff47", width=2, dash="dot"),
        name=f"OLS（斜率={slope_v:.2f}）",
    ))

    fig3.update_layout(
        paper_bgcolor="#0e0f1a", plot_bgcolor="#09090f",
        font=dict(color="#5a5c78", size=10),
        margin=dict(l=8, r=8, t=8, b=8), height=310,
        xaxis=dict(gridcolor="#141525", zeroline=False, title="VIX 恐慌指数"),
        yaxis=dict(gridcolor="#141525", zeroline=False, title="TSLA 股价（美元）"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#dde1f5", size=9)),
        hovermode="closest",
    )
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════
# 图4：历史视角
# ══════════════════════════════════════════════════════════
st.markdown(f'<div class="sec">04 — 历史相关性视角（{period_label}）</div>',
            unsafe_allow_html=True)

with st.spinner("加载历史数据…"):
    th_raw, vh_raw = fetch_history(hist_period, hist_interval)

if not th_raw.empty and not vh_raw.empty:
    dfh = build_df(th_raw, vh_raw)
    rh, ph = pearson_corr(dfh)

    fig4 = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35], vertical_spacing=0.03,
    )

    tp2 = (dfh["TSLA"] - dfh["TSLA"].mean()) / dfh["TSLA"].std() if normalize else dfh["TSLA"]
    vp2 = (dfh["VIX"]  - dfh["VIX"].mean())  / dfh["VIX"].std()  if normalize else dfh["VIX"]

    fig4.add_trace(go.Scatter(x=dfh.index, y=tp2, name="TSLA（历史）",
                              line=dict(color="#e8ff47", width=1.4),
                              fill="tozeroy", fillcolor="rgba(232,255,71,0.04)"),
                   row=1, col=1)
    fig4.add_trace(go.Scatter(x=dfh.index, y=vp2, name="VIX（历史）",
                              line=dict(color="#ff3d6b", width=1.4, dash="dot")),
                   row=1, col=1)

    roll_h = dfh["TSLA"].rolling(roll_window).corr(dfh["VIX"]).dropna()
    fig4.add_trace(go.Scatter(x=roll_h.index, y=roll_h.values,
                              name=f"滚动r（{roll_window}根）",
                              line=dict(color="#7b7bff", width=1.8),
                              fill="tozeroy", fillcolor="rgba(123,123,255,0.06)"),
                   row=2, col=1)
    fig4.add_hline(y=-0.4, line_color="#ff3d6b", line_dash="dash",
                   line_width=1, row=2, col=1)
    fig4.add_hline(y=0, line_color="#1e1f35", line_width=1, row=2, col=1)

    title_txt = (f"历史皮尔逊 r = {rh:.3f}  |  R² = {rh**2*100:.1f}%  |  p = {ph:.2e}"
                 if rh else "")
    fig4.update_layout(
        paper_bgcolor="#0e0f1a", plot_bgcolor="#09090f",
        font=dict(color="#5a5c78", size=10),
        title=dict(text=title_txt,
                   font=dict(color="#e8ff47", size=11, family="Space Mono"), x=0.01),
        margin=dict(l=8, r=8, t=34, b=8), height=440,
        legend=dict(bgcolor="#0e0f1a", bordercolor="#1e1f35", borderwidth=1,
                    font=dict(color="#dde1f5")),
        hovermode="x unified",
        xaxis=dict(gridcolor="#141525", zeroline=False),
        xaxis2=dict(gridcolor="#141525", zeroline=False),
        yaxis=dict(gridcolor="#141525", zeroline=False,
                   title="Z-Score" if normalize else "价格/指数"),
        yaxis2=dict(gridcolor="#141525", zeroline=False,
                    range=[-1.05, 1.05], title="r 值"),
    )
    st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════
# 统计摘要表
# ══════════════════════════════════════════════════════════
st.markdown('<div class="sec">05 — 完整统计摘要</div>', unsafe_allow_html=True)

if r1m is not None:
    rows = [
        ("皮尔逊相关系数 r（1分钟线）",  f"{r1m:.4f}",
         "强负相关 → 高度反向联动" if r1m < -0.5 else
         ("中等负相关" if r1m < -0.3 else ("弱负相关" if r1m < 0 else "正相关（异常）"))),
        ("R² 解释方差（1分钟线）",       f"{r1m**2:.4f}（{r1m**2*100:.1f}%）",
         f"VIX 解释了 TSLA {r1m**2*100:.1f}% 的价格波动"),
        ("P 值（1分钟线）",              f"{p1m:.2e}",
         "极度显著（p<0.001）" if p1m < 0.001 else
         ("显著（p<0.05）" if p1m < 0.05 else "不显著")),
        ("OLS 回归斜率（TSLA/VIX）",    f"{slope_v:.4f}",
         f"VIX 每升 1 点 → TSLA 理论变动 {slope_v:.2f} 美元"),
        ("OLS 截距",                    f"{intercept_v:.4f}",
         "VIX=0 时 TSLA 的理论基准值"),
        ("样本量（1分钟 K 线数）",       str(len(df1m)),
         f"约覆盖 {len(df1m)//390 + 1} 个交易日"),
        ("TSLA 均值",                   f"${df1m['TSLA'].mean():.2f}", ""),
        ("TSLA 标准差",                 f"${df1m['TSLA'].std():.2f}", ""),
        ("VIX 均值",                    f"{df1m['VIX'].mean():.2f}",  ""),
        ("VIX 标准差",                  f"{df1m['VIX'].std():.2f}",   ""),
        ("当前交易时段",                session_zh_label,              et_time),
        ("实时 TSLA 价格",             f"${tsla_rt:.2f}" if tsla_rt else "—",
         f"较昨日 {'▲' if tsla_chg>=0 else '▼'} {abs(tsla_chg):.2f}%"),
        ("实时 VIX 指数",              f"{vix_rt:.2f}" if vix_rt else "—",
         f"较昨日 {'▲' if vix_chg>=0 else '▼'} {abs(vix_chg):.2f}%"),
        ("数据更新时间",               now_str,
         "每30秒自动刷新（已开启）" if auto_refresh else "自动刷新已关闭"),
    ]
    df_stats = pd.DataFrame(rows, columns=["指标", "数值", "解读"])
    st.dataframe(df_stats, use_container_width=True, hide_index=True,
                 column_config={
                     "指标": st.column_config.TextColumn("指标", width=240),
                     "数值": st.column_config.TextColumn("数值", width=220),
                     "解读": st.column_config.TextColumn("解读"),
                 })

# ══════════════════════════════════════════════════════════
# 页脚
# ══════════════════════════════════════════════════════════
refresh_info = "下次刷新：30 秒后" if auto_refresh else "自动刷新已关闭"
st.markdown(f"""
<div style="font-family:'Space Mono',monospace;font-size:10px;color:#22233a;
            text-align:center;margin-top:32px;padding:14px;border-top:1px solid #141525">
    数据来源：Yahoo Finance · 更新：{now_str} · {refresh_info} · 本页面仅供学术研究，不构成任何投资建议。
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# 自动刷新
# ══════════════════════════════════════════════════════════
if auto_refresh:
    time.sleep(30)
    st.cache_data.clear()
    st.rerun()
