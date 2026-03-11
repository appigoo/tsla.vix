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
import requests
import json

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

/* 告警面板 */
.alert-box {
    background: linear-gradient(135deg,#1a0f18,#0f0a18);
    border: 1px solid #ff3d6b;
    border-radius: 14px; padding: 14px 18px; margin: 6px 0;
    border-left: 4px solid #ff3d6b;
}
.alert-box.div-up {
    border-color: #ff8c00;
    border-left-color: #ff8c00;
    background: linear-gradient(135deg,#1a1200,#0f0c00);
}
.alert-box.div-down {
    border-color: #7b7bff;
    border-left-color: #7b7bff;
    background: linear-gradient(135deg,#0d0d1a,#080812);
}
.alert-time {
    font-family:'Space Mono',monospace; font-size:9px;
    color:var(--muted); margin-bottom:4px; letter-spacing:1px;
}
.alert-msg { font-size:13px; line-height:1.6; color:var(--text); }
.alert-badge {
    display:inline-block; border-radius:4px; padding:1px 8px;
    font-family:'Space Mono',monospace; font-size:9px; font-weight:700;
    margin-bottom:6px;
}
.alert-badge.warn  { background:#ff3d6b22; color:#ff3d6b; border:1px solid #ff3d6b55; }
.alert-badge.divup { background:#ff8c0022; color:#ff8c00; border:1px solid #ff8c0055; }
.alert-badge.divdn { background:#7b7bff22; color:#7b7bff; border:1px solid #7b7bff55; }
.tg-status {
    display:inline-flex; align-items:center; gap:6px;
    border-radius:8px; padding:5px 12px; margin-top:4px;
    font-family:'Space Mono',monospace; font-size:10px;
}
.tg-status.ok  { background:#3df5b010; border:1px solid #3df5b055; color:#3df5b0; }
.tg-status.err { background:#ff3d6b10; border:1px solid #ff3d6b55; color:#ff3d6b; }
.tg-status.off { background:#5a5c7820; border:1px solid #5a5c7855; color:#5a5c78; }

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
# Telegram 告警函数
# ══════════════════════════════════════════════════════════

def tg_send(bot_token: str, chat_id: str, text: str) -> tuple[bool, str]:
    """发送 Telegram 消息，返回 (成功, 信息)"""
    if not bot_token or not chat_id:
        return False, "未配置 Token / Chat ID"
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try:
        resp = requests.post(url, json={
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
        }, timeout=8)
        data = resp.json()
        if data.get("ok"):
            return True, "发送成功"
        return False, data.get("description", "未知错误")
    except Exception as e:
        return False, str(e)


def tg_test(bot_token: str, chat_id: str) -> tuple[bool, str]:
    """测试 Telegram 连接"""
    return tg_send(bot_token, chat_id,
                   "✅ <b>TSLA × VIX 监控系统</b>\n\n连接测试成功！告警通知已就绪。")


def detect_divergence(
    df: pd.DataFrame,
    window: int = 5,
    vix_thresh: float = 1.0,
    tsla_thresh: float = 0.5,
) -> dict | None:
    """
    检测 VIX 与 TSLA 的背离信号。
    使用最近 window 根 K 线的涨跌幅。

    背离类型：
    - 「恐慌背离」：VIX 显著上升，但 TSLA 未跟随下跌（异常强势）
    - 「情绪背离」：VIX 显著下降，但 TSLA 未跟随上涨（异常弱势）

    返回 None（无背离）或 dict（背离详情）。
    """
    if len(df) < window + 1:
        return None

    recent = df.iloc[-window:]
    base   = df.iloc[-(window + 1)]

    vix_chg_pct  = (float(recent["VIX"].iloc[-1])  - float(base["VIX"]))  / float(base["VIX"])  * 100
    tsla_chg_pct = (float(recent["TSLA"].iloc[-1]) - float(base["TSLA"])) / float(base["TSLA"]) * 100

    # 恐慌背离：VIX↑ 但 TSLA 未跌（TSLA 反而持平或涨）
    if vix_chg_pct >= vix_thresh and tsla_chg_pct >= -tsla_thresh:
        return {
            "type":     "panic_divergence",
            "label":    "🟠 恐慌背离",
            "badge":    "divup",
            "css":      "div-up",
            "emoji":    "🟠",
            "vix_chg":  vix_chg_pct,
            "tsla_chg": tsla_chg_pct,
            "msg": (
                f"⚠️ <b>恐慌背离警报</b>\n\n"
                f"📈 VIX 在过去 {window} 分钟内上涨 <b>{vix_chg_pct:+.2f}%</b>（恐慌加剧），\n"
                f"但 TSLA 未出现预期跌幅，反而变动 <b>{tsla_chg_pct:+.2f}%</b>。\n\n"
                f"📌 <b>意义</b>：TSLA 出现异常强势，可能有正面催化剂（利好消息、机构托盘）支撑，\n"
                f"或市场尚未充分定价恐慌情绪。\n\n"
                f"💡 关注 TSLA 是否后续补跌，或 VIX 是否快速回落。\n\n"
                f"🕐 时间：{datetime.now(ET).strftime('%Y-%m-%d %H:%M:%S ET')}"
            ),
            "desc_html": (
                f"VIX <b>{vix_chg_pct:+.2f}%</b>↑，"
                f"但 TSLA 仅 <b>{tsla_chg_pct:+.2f}%</b>（应跌未跌）"
            ),
        }

    # 情绪背离：VIX↓ 但 TSLA 未涨（TSLA 反而持平或跌）
    if vix_chg_pct <= -vix_thresh and tsla_chg_pct <= tsla_thresh:
        return {
            "type":     "calm_divergence",
            "label":    "🔵 情绪背离",
            "badge":    "divdn",
            "css":      "div-down",
            "emoji":    "🔵",
            "vix_chg":  vix_chg_pct,
            "tsla_chg": tsla_chg_pct,
            "msg": (
                f"⚠️ <b>情绪背离警报</b>\n\n"
                f"📉 VIX 在过去 {window} 分钟内下跌 <b>{vix_chg_pct:+.2f}%</b>（恐慌缓解），\n"
                f"但 TSLA 未出现预期涨幅，反而变动 <b>{tsla_chg_pct:+.2f}%</b>。\n\n"
                f"📌 <b>意义</b>：TSLA 出现异常弱势，可能有负面催化剂（利空消息、卖盘压力）\n"
                f"压制其跟随市场反弹。\n\n"
                f"💡 关注 TSLA 是否持续承压，或市场情绪是否真的改善。\n\n"
                f"🕐 时间：{datetime.now(ET).strftime('%Y-%m-%d %H:%M:%S ET')}"
            ),
            "desc_html": (
                f"VIX <b>{vix_chg_pct:+.2f}%</b>↓，"
                f"但 TSLA 仅 <b>{tsla_chg_pct:+.2f}%</b>（应涨未涨）"
            ),
        }

    return None


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

    # ── Telegram 告警配置 ────────────────────────────────
    st.markdown('<div class="sec">📲 Telegram 告警设置</div>', unsafe_allow_html=True)

    tg_enabled = st.checkbox("启用 Telegram 告警", value=False)

    # ── 从 st.secrets 读取（优先），否则显示手动输入框 ──
    _secret_token   = st.secrets.get("telegram", {}).get("bot_token", "")
    _secret_chat_id = st.secrets.get("telegram", {}).get("chat_id", "")

    if _secret_token and _secret_chat_id:
        # Secrets 已配置：显示已加载状态，不暴露明文
        tg_token   = _secret_token
        tg_chat_id = _secret_chat_id
        st.markdown("""
<div style="background:#3df5b010;border:1px solid #3df5b055;border-radius:8px;
            padding:8px 12px;font-family:'Space Mono',monospace;font-size:10px;
            color:#3df5b0;line-height:1.8">
  ✅ <b>已从 Streamlit Secrets 加载</b><br>
  <span style="color:#5a5c78">bot_token ••••••••<br>
  chat_id ••••••••</span>
</div>""", unsafe_allow_html=True)
    else:
        # Secrets 未配置：降级到手动输入
        st.markdown("""
<div style="background:#ff3d6b0a;border:1px solid #ff3d6b33;border-radius:8px;
            padding:8px 12px;font-family:'Space Mono',monospace;font-size:10px;
            color:#ff3d6b;line-height:1.6;margin-bottom:8px">
  ⚠ 未检测到 Secrets，请手动输入<br>
  <span style="color:#5a5c78">建议配置 .streamlit/secrets.toml</span>
</div>""", unsafe_allow_html=True)
        tg_token   = st.text_input("Bot Token",
                                   placeholder="110201543:AAHdqTcvCH1vGWJxfSeofSz326CKEqt4VN",
                                   type="password",
                                   help="从 @BotFather 获取")
        tg_chat_id = st.text_input("Chat ID",
                                   placeholder="-1001234567890 或 你的 user_id",
                                   help="可用 @userinfobot 查询")

    # 背离检测参数
    st.markdown("**背离检测灵敏度**")
    div_window     = st.slider("观察窗口（分钟数）", 3, 30, 5,
                               help="用最近 N 根 1 分钟 K 线计算涨跌幅")
    vix_div_thresh = st.slider("VIX 变动阈值（%）", 0.5, 5.0, 1.0, step=0.1,
                               help="VIX 变动超过此值才触发检测")
    tsla_div_thresh= st.slider("TSLA 容忍阈值（%）", 0.2, 3.0, 0.5, step=0.1,
                               help="TSLA 变动在此范围内视为「未跟随」")

    # 冷却时间（防止重复告警）
    alert_cooldown = st.slider("告警冷却时间（分钟）", 1, 60, 15,
                               help="同类型告警的最短间隔，防止刷屏")

    col_test, col_clear = st.columns(2)
    with col_test:
        if st.button("📡 测试连接", use_container_width=True):
            if tg_token and tg_chat_id:
                ok, msg = tg_test(tg_token, tg_chat_id)
                if ok:
                    st.success(f"✅ {msg}")
                else:
                    st.error(f"❌ {msg}")
            else:
                st.warning("未检测到有效的 Token / Chat ID")
    with col_clear:
        if st.button("🗑 清除记录", use_container_width=True):
            st.session_state.alert_history = []
            st.session_state.last_alert_time = {}
            st.success("已清除")

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

# ── Session state 初始化（告警历史 & 冷却）
if "alert_history" not in st.session_state:
    st.session_state.alert_history = []
if "last_alert_time" not in st.session_state:
    st.session_state.last_alert_time = {}  # {type_str: datetime}

# ══════════════════════════════════════════════════════════
# 背离检测 & Telegram 告警
# ══════════════════════════════════════════════════════════
divergence = detect_divergence(
    df1m,
    window=div_window,
    vix_thresh=vix_div_thresh,
    tsla_thresh=tsla_div_thresh,
)

tg_alert_fired = False
tg_alert_status = None

if divergence is not None:
    dtype = divergence["type"]
    now_dt = datetime.now(ET)

    # 冷却检查
    last_t = st.session_state.last_alert_time.get(dtype)
    cooldown_ok = (
        last_t is None or
        (now_dt - last_t).total_seconds() >= alert_cooldown * 60
    )

    # 写入告警历史（不受冷却限制，UI 始终展示）
    already_in_history = (
        st.session_state.alert_history and
        st.session_state.alert_history[0]["type"] == dtype and
        (now_dt - st.session_state.alert_history[0]["time"]).total_seconds() < 60
    )
    if not already_in_history:
        st.session_state.alert_history.insert(0, {
            "type":      dtype,
            "label":     divergence["label"],
            "badge":     divergence["badge"],
            "css":       divergence["css"],
            "desc_html": divergence["desc_html"],
            "time":      now_dt,
            "sent":      False,
        })
        st.session_state.alert_history = st.session_state.alert_history[:30]  # 最多保留30条

    # 发送 Telegram
    if tg_enabled and tg_token and tg_chat_id and cooldown_ok:
        ok, err = tg_send(tg_token, tg_chat_id, divergence["msg"])
        tg_alert_fired = ok
        tg_alert_status = (ok, err)
        if ok:
            st.session_state.last_alert_time[dtype] = now_dt
            if st.session_state.alert_history:
                st.session_state.alert_history[0]["sent"] = True

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
            f"充分验证了「市场恐慌加剧，特斯拉抛压显著」的系统性规律。"
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
# 告警状态栏 & 历史记录
# ══════════════════════════════════════════════════════════
st.markdown('<div class="sec">🚨 背离告警监控</div>', unsafe_allow_html=True)

alert_col1, alert_col2 = st.columns([3, 2])

with alert_col1:
    # 当前背离状态
    if divergence is not None:
        dtype = divergence["type"]
        last_t = st.session_state.last_alert_time.get(dtype)
        if last_t:
            mins_ago = int((datetime.now(ET) - last_t).total_seconds() / 60)
            sent_str = f"✅ Telegram 已发送（{mins_ago} 分钟前）" if tg_enabled else "📵 Telegram 未启用"
        else:
            sent_str = "🆕 首次触发"

        st.markdown(f"""
        <div class="alert-box {divergence['css']}">
          <div class="alert-badge {divergence['badge']}">{divergence['emoji']} {divergence['label']}</div>
          <div class="alert-msg">{divergence['desc_html']}</div>
          <div class="alert-time" style="margin-top:6px">{sent_str} · {datetime.now(ET).strftime('%H:%M:%S ET')}</div>
        </div>""", unsafe_allow_html=True)

        if tg_alert_status:
            ok, err = tg_alert_status
            if ok:
                st.markdown('<div class="tg-status ok">📲 Telegram 告警已推送</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="tg-status err">❌ 推送失败：{err}</div>', unsafe_allow_html=True)
    else:
        # 计算当前窗口涨跌幅展示
        if len(df1m) >= div_window + 1:
            base   = df1m.iloc[-(div_window + 1)]
            recent_last = df1m.iloc[-1]
            cur_vix_chg  = (float(recent_last["VIX"])  - float(base["VIX"]))  / float(base["VIX"])  * 100
            cur_tsla_chg = (float(recent_last["TSLA"]) - float(base["TSLA"])) / float(base["TSLA"]) * 100
            vix_arrow  = "▲" if cur_vix_chg  >= 0 else "▼"
            tsla_arrow = "▲" if cur_tsla_chg >= 0 else "▼"
            summary = (
                f"过去 {div_window} 分钟：VIX {vix_arrow} {abs(cur_vix_chg):.2f}%，"
                f"TSLA {tsla_arrow} {abs(cur_tsla_chg):.2f}%  ── 走势符合负相关预期，无背离"
            )
        else:
            summary = "数据积累中，请稍候…"

        st.markdown(f"""
        <div class="kcard t" style="border-left:4px solid #3df5b0">
          <div class="klabel">当前背离状态</div>
          <div class="kval t" style="font-size:20px">✅ 正常</div>
          <div class="ksub">{summary}</div>
        </div>""", unsafe_allow_html=True)

    # Telegram 连接状态
    if tg_enabled:
        if tg_token and tg_chat_id:
            st.markdown('<div class="tg-status ok" style="margin-top:6px">📡 Telegram 已配置 · 监控中</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="tg-status err" style="margin-top:6px">⚠ 已启用但缺少 Token 或 Chat ID</div>',
                        unsafe_allow_html=True)
    else:
        st.markdown('<div class="tg-status off" style="margin-top:6px">📵 Telegram 告警已关闭</div>',
                    unsafe_allow_html=True)

with alert_col2:
    st.markdown(f"""
    <div style="font-size:11px;color:#5a5c78;font-family:'Space Mono',monospace;
                background:#0e0f1a;border:1px solid #1e1f35;border-radius:10px;
                padding:14px 16px;line-height:2">
      <b style="color:#dde1f5">告警触发条件</b><br>
      🟠 <b style="color:#ff8c00">恐慌背离</b><br>
      VIX ≥ +{vix_div_thresh:.1f}%<br>
      且 TSLA 变动 ≥ −{tsla_div_thresh:.1f}%（未跌）<br><br>
      🔵 <b style="color:#7b7bff">情绪背离</b><br>
      VIX ≤ −{vix_div_thresh:.1f}%<br>
      且 TSLA 变动 ≤ +{tsla_div_thresh:.1f}%（未涨）<br><br>
      ⏱ 观察窗口：{div_window} 分钟<br>
      🔕 冷却时间：{alert_cooldown} 分钟
    </div>""", unsafe_allow_html=True)

# ── 告警历史
if st.session_state.alert_history:
    with st.expander(f"📋 告警历史记录（共 {len(st.session_state.alert_history)} 条）", expanded=False):
        for rec in st.session_state.alert_history:
            sent_icon = "📲" if rec.get("sent") else "🔕"
            st.markdown(f"""
            <div class="alert-box {rec['css']}" style="padding:10px 14px;margin:4px 0">
              <div style="display:flex;justify-content:space-between;align-items:center">
                <span class="alert-badge {rec['badge']}">{rec['label']}</span>
                <span class="alert-time">{sent_icon} {rec['time'].strftime('%m-%d %H:%M:%S ET')}</span>
              </div>
              <div class="alert-msg" style="font-size:12px;margin-top:4px">{rec['desc_html']}</div>
            </div>""", unsafe_allow_html=True)


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
