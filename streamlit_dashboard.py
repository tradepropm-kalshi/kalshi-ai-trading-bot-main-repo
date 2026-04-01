"""
Unified Trading Dashboard — Beast Mode + Lean Mode

Run with:
    streamlit run streamlit_dashboard.py

Communicates with the running bot orchestrator via bot_state.json:
  - Reads live status (positions, P&L, cycles, AI cost)
  - Writes toggle instructions (enable/disable/mode switches)
  - The orchestrator's _toggle_watcher() picks up changes within 5 seconds

Tabs:
  🏠 Overview      — combined portfolio, mode status cards, mode controls
  📊 Positions     — filterable by strategy; live prices from Kalshi
  📈 Performance   — P&L charts per strategy, win rates, AI cost breakdown
  🤖 AI Analysis   — recent LLM queries, cost tracking, confidence history
  ⚠️  Risk          — daily loss %, position concentration, circuit breaker
  🔧 System        — API health, data sources, log tail
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.clients.kalshi_client import KalshiClient
from src.config.settings import settings
from src.utils.database import DatabaseManager

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Kalshi AI Trading",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* Mode card containers */
.mode-card {
    padding: 1.2rem 1.4rem;
    border-radius: 10px;
    border: 2px solid transparent;
    margin-bottom: 0.5rem;
}
.mode-beast  { border-color: #ff4b4b; background: rgba(255,75,75,0.07); }
.mode-lean   { border-color: #00c4a0; background: rgba(0,196,160,0.07); }
.mode-off    { border-color: #888;    background: rgba(128,128,128,0.05); }

/* Status pill */
.pill-live   { background:#ff4b4b; color:#fff; padding:2px 10px; border-radius:99px; font-size:0.75rem; font-weight:700; }
.pill-paper  { background:#ffa500; color:#fff; padding:2px 10px; border-radius:99px; font-size:0.75rem; font-weight:700; }
.pill-off    { background:#555;    color:#fff; padding:2px 10px; border-radius:99px; font-size:0.75rem; }
.pill-run    { background:#00c4a0; color:#fff; padding:2px 10px; border-radius:99px; font-size:0.75rem; font-weight:700; }

/* Metric card */
.metric-box {
    background:#f7f8fa; border-radius:8px; padding:0.8rem 1rem;
    text-align:center; border:1px solid #e0e0e0;
}
.metric-box h3 { margin:0; font-size:1.6rem; }
.metric-box p  { margin:0; color:#666; font-size:0.8rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# State file path
# ---------------------------------------------------------------------------
_STATE_FILE = Path("bot_state.json")

# ---------------------------------------------------------------------------
# Data loaders (each creates its own event loop — Streamlit runs synchronously)
# ---------------------------------------------------------------------------

def _run_async(coro):
    """Run a coroutine synchronously (Streamlit is not async-native)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@st.cache_data(ttl=8)
def load_bot_state() -> Dict:
    """Read bot_state.json written by the orchestrator."""
    try:
        if _STATE_FILE.exists():
            return json.loads(_STATE_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        pass
    return {
        "beast_enabled": False, "lean_enabled": False,
        "beast_live": False,    "lean_live": False,
        "beast": {}, "lean": {}, "system": {},
    }


@st.cache_data(ttl=15)
def load_positions() -> List[Dict]:
    """Load open positions from the database."""
    async def _get():
        db = DatabaseManager()
        await db.initialize()
        rows = await db.get_open_positions()
        await db.close()
        return [
            {
                "market_id": getattr(r, "market_id", ""),
                "strategy":  getattr(r, "strategy", ""),
                "side":      getattr(r, "side", ""),
                "entry":     getattr(r, "entry_price", 0),
                "qty":       getattr(r, "quantity", 0),
                "confidence":getattr(r, "confidence", 0),
                "since":     str(getattr(r, "timestamp", ""))[:16],
                "rationale": (getattr(r, "rationale", "") or "")[:80],
            }
            for r in rows
        ]
    try:
        return _run_async(_get())
    except Exception:
        return []


@st.cache_data(ttl=30)
def load_performance() -> Dict:
    """Load strategy performance stats from the database."""
    async def _get():
        db = DatabaseManager()
        await db.initialize()
        raw = await db.get_performance_by_strategy()
        await db.close()
        return {
            str(k): {str(kk): (float(vv) if isinstance(vv, (int, float)) else str(vv))
                     for kk, vv in v.items()}
            for k, v in (raw or {}).items()
        }
    try:
        return _run_async(_get())
    except Exception:
        return {}


@st.cache_data(ttl=20)
def load_llm_queries(hours: int = 24) -> List[Dict]:
    """Load recent LLM queries from the database."""
    async def _get():
        db = DatabaseManager()
        await db.initialize()
        rows = await db.get_llm_queries(hours_back=hours, limit=200)
        await db.close()
        return [
            {
                "ts":         str(getattr(r, "timestamp", ""))[:16],
                "market":     getattr(r, "market_id", ""),
                "strategy":   getattr(r, "strategy", ""),
                "model":      getattr(r, "model", ""),
                "cost":       float(getattr(r, "cost", 0) or 0),
                "confidence": float(getattr(r, "confidence_extracted", 0) or 0),
                "decision":   getattr(r, "decision_extracted", ""),
                "query_type": getattr(r, "query_type", ""),
            }
            for r in rows
        ]
    try:
        return _run_async(_get())
    except Exception:
        return []


@st.cache_data(ttl=60)
def load_balance() -> Dict:
    """Fetch live Kalshi balance."""
    async def _get():
        kc = KalshiClient(
            api_key=settings.api.kalshi_api_key,
            base_url=settings.api.kalshi_base_url,
        )
        bal = await kc.get_balance()
        await kc.close()
        raw = float(bal.get("balance", 0) or bal.get("available_balance", 0) or 0)
        return raw / 100.0 if raw > 1.0 else raw
    try:
        return {"balance": _run_async(_get())}
    except Exception:
        return {"balance": 0.0}


# ---------------------------------------------------------------------------
# Toggle writer — writes to bot_state.json which the orchestrator polls
# ---------------------------------------------------------------------------

def write_toggle(strategy: str, enabled: bool, live: bool) -> None:
    """Write a toggle instruction for the orchestrator."""
    try:
        data = {}
        if _STATE_FILE.exists():
            data = json.loads(_STATE_FILE.read_text())
        if strategy == "beast":
            data["beast_enabled"] = enabled
            data["beast_live"]    = live
        elif strategy == "lean":
            data["lean_enabled"] = enabled
            data["lean_live"]    = live
        data["toggle_pending"] = True
        data["written_at"] = datetime.now().isoformat(timespec="seconds")
        _STATE_FILE.write_text(json.dumps(data, indent=2))
        # Bust the cache so the dashboard reflects the new state immediately
        load_bot_state.clear()
    except OSError as exc:
        st.error(f"Could not write toggle: {exc}")


# ---------------------------------------------------------------------------
# Sidebar — navigation + global status
# ---------------------------------------------------------------------------

def render_sidebar(state: Dict) -> str:
    st.sidebar.image(
        "https://cdn.prod.website-files.com/64e3b8e9cfc2b37c9aced8d5/"
        "64e3b9a5aabcfa69a5b90e0b_Kalshi%20Logo%20White.svg",
        width=120,
    )
    st.sidebar.title("Kalshi AI Trading")

    beast = state.get("beast", {})
    lean  = state.get("lean",  {})
    sys_  = state.get("system", {})

    # Quick status indicators
    def _pill(enabled, running, live):
        if not enabled:
            return '<span class="pill-off">OFF</span>'
        if running:
            label = "LIVE" if live else "PAPER"
            cls   = "pill-live" if live else "pill-run"
            return f'<span class="{cls}">{label} ▶</span>'
        return '<span class="pill-paper">PAUSED</span>'

    st.sidebar.markdown(
        f"**Beast Mode** {_pill(state.get('beast_enabled'), beast.get('running'), state.get('beast_live'))}  \n"
        f"**Lean Mode**  {_pill(state.get('lean_enabled'),  lean.get('running'),  state.get('lean_live'))}",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")

    # Live balance
    balance = load_balance().get("balance", 0)
    st.sidebar.metric("💰 Balance", f"${balance:,.2f}")
    st.sidebar.metric("📂 Open Positions", sys_.get("total_positions", 0))

    written = state.get("written_at", "")
    if written:
        st.sidebar.caption(f"Last bot update: {written[11:19]}")

    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Navigate",
        ["🏠 Overview", "📊 Positions", "📈 Performance",
         "🤖 AI Analysis", "⚠️ Risk", "🔧 System"],
        label_visibility="collapsed",
    )

    if st.sidebar.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    return page


# ---------------------------------------------------------------------------
# Overview tab — mode control cards
# ---------------------------------------------------------------------------

def render_overview(state: Dict) -> None:
    st.title("⚡ Kalshi AI Trading — Control Center")

    beast_on   = state.get("beast_enabled", False)
    lean_on    = state.get("lean_enabled",  False)
    beast_live = state.get("beast_live",    False)
    lean_live  = state.get("lean_live",     False)
    beast_st   = state.get("beast", {})
    lean_st    = state.get("lean",  {})
    sys_st     = state.get("system", {})

    # ── Top metrics row ──────────────────────────────────────────────
    balance = load_balance().get("balance", 0)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("💰 Balance",       f"${balance:,.2f}")
    m2.metric("📂 Positions",     sys_st.get("total_positions", 0))
    beast_pnl = beast_st.get("daily_pnl", 0)
    lean_pnl  = lean_st.get("daily_pnl",  0)
    m3.metric("🐉 Beast P&L",    f"${beast_pnl:+.2f}")
    m4.metric("🎯 Lean P&L",     f"${lean_pnl:+.2f}")
    m5.metric("💸 AI Cost Today",
              f"${(beast_st.get('ai_cost_today', 0) + lean_st.get('ai_cost_today', 0)):.3f}")

    st.markdown("---")

    # ── Mode control cards ───────────────────────────────────────────
    col1, col2 = st.columns(2, gap="large")

    # ── Beast Mode card ──
    with col1:
        card_cls = "mode-beast" if beast_on else "mode-off"
        st.markdown(f'<div class="{card_cls}">', unsafe_allow_html=True)
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown("### 🐉 Beast Mode")
            st.caption("5-model ensemble · full strategy suite")
        with c2:
            beast_toggle = st.toggle(
                "Enable", value=beast_on, key="beast_toggle",
                help="Toggle Beast Mode on/off without restarting the orchestrator"
            )

        if beast_toggle != beast_on:
            # Determine live mode from the radio below before writing
            pass  # Handled after radio renders

        b_live_choice = st.radio(
            "Trading mode",
            ["📄 Paper", "🔴 Live"],
            index=1 if beast_live else 0,
            key="beast_live_radio",
            horizontal=True,
            disabled=not beast_toggle,
        )
        want_beast_live = (b_live_choice == "🔴 Live")

        if beast_toggle != beast_on or want_beast_live != beast_live:
            write_toggle("beast", beast_toggle, want_beast_live)

        # Status metrics
        if beast_on:
            bm1, bm2, bm3 = st.columns(3)
            bm1.metric("Positions", beast_st.get("positions_open", 0))
            bm2.metric("Cycles",    beast_st.get("cycle_count", 0))
            bm3.metric("AI Cost",   f"${beast_st.get('ai_cost_today', 0):.3f}")
            if beast_st.get("last_error"):
                st.warning(f"⚠ {beast_st['last_error']}")
            last = beast_st.get("last_cycle_at", "")
            if last:
                st.caption(f"Last cycle: {last[11:19]}")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Lean Mode card ──
    with col2:
        card_cls = "mode-lean" if lean_on else "mode-off"
        st.markdown(f'<div class="{card_cls}">', unsafe_allow_html=True)
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown("### 🎯 Lean Mode")
            st.caption("Single Grok-3 · high-volume markets · real-world data")
        with c2:
            lean_toggle = st.toggle(
                "Enable", value=lean_on, key="lean_toggle",
                help="Toggle Lean Mode on/off without restarting the orchestrator"
            )

        l_live_choice = st.radio(
            "Trading mode",
            ["📄 Paper", "🔴 Live"],
            index=1 if lean_live else 0,
            key="lean_live_radio",
            horizontal=True,
            disabled=not lean_toggle,
        )
        want_lean_live = (l_live_choice == "🔴 Live")

        if lean_toggle != lean_on or want_lean_live != lean_live:
            write_toggle("lean", lean_toggle, want_lean_live)

        if lean_on:
            lm1, lm2, lm3 = st.columns(3)
            lm1.metric("Positions", lean_st.get("positions_open", 0))
            lm2.metric("Cycles",    lean_st.get("cycle_count", 0))
            lm3.metric("AI Cost",   f"${lean_st.get('ai_cost_today', 0):.3f}")
            if lean_st.get("last_error"):
                st.warning(f"⚠ {lean_st['last_error']}")
            last = lean_st.get("last_cycle_at", "")
            if last:
                st.caption(f"Last cycle: {last[11:19]}")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Compatibility note ───────────────────────────────────────────
    if beast_on and lean_on:
        st.info(
            "**Both modes running simultaneously.** "
            "Shared position lock prevents conflicting trades on the same market. "
            "Beast Mode: up to 7 positions · Lean Mode: up to 5 positions · Combined cap: 10.",
            icon="ℹ️",
        )

    # ── Uptime / system health strip ─────────────────────────────────
    uptime = sys_st.get("uptime_seconds", 0)
    if uptime > 0:
        h, m = divmod(int(uptime), 3600)
        m, s = divmod(m, 60)
        st.caption(f"Orchestrator uptime: {h:02d}:{m:02d}:{s:02d}")


# ---------------------------------------------------------------------------
# Positions tab
# ---------------------------------------------------------------------------

def render_positions(state: Dict) -> None:
    st.header("📊 Open Positions")

    positions = load_positions()
    if not positions:
        st.info("No open positions.")
        return

    df = pd.DataFrame(positions)

    # Filter by strategy
    strategies = ["All"] + sorted(df["strategy"].unique().tolist())
    filt = st.selectbox("Filter by strategy", strategies)
    if filt != "All":
        df = df[df["strategy"] == filt]

    # Colour-code strategy column
    def _style_strategy(val):
        if "lean" in str(val):
            return "color: #00c4a0; font-weight:bold"
        if "beast" in str(val) or "directional" in str(val):
            return "color: #ff4b4b; font-weight:bold"
        return ""

    styled = df.style.applymap(_style_strategy, subset=["strategy"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Summary
    col1, col2, col3 = st.columns(3)
    col1.metric("Total open", len(df))
    col2.metric("Beast", len(df[df["strategy"].str.contains("lean", na=False) == False]))
    col3.metric("Lean",  len(df[df["strategy"].str.contains("lean", na=False)]))


# ---------------------------------------------------------------------------
# Performance tab
# ---------------------------------------------------------------------------

def render_performance() -> None:
    st.header("📈 Strategy Performance")

    perf = load_performance()
    if not perf:
        st.info("No completed trades yet.")
        return

    rows = []
    for strat, stats in perf.items():
        rows.append({
            "Strategy":    strat.replace("_", " ").title(),
            "Trades":      int(stats.get("completed_trades", 0)),
            "Win Rate %":  round(float(stats.get("win_rate_pct", 0)), 1),
            "Total P&L":   round(float(stats.get("total_pnl", 0)), 2),
            "Avg P&L":     round(float(stats.get("avg_pnl", 0)), 2),
            "AI Cost":     round(float(stats.get("total_ai_cost", 0)), 3),
        })
    df = pd.DataFrame(rows)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            df, x="Strategy", y="Total P&L",
            color="Total P&L", color_continuous_scale="RdYlGn",
            title="P&L by Strategy",
        )
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.bar(
            df, x="Strategy", y="Win Rate %",
            color="Win Rate %", color_continuous_scale="Blues",
            title="Win Rate by Strategy",
        )
        fig2.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# AI Analysis tab
# ---------------------------------------------------------------------------

def render_ai_analysis() -> None:
    st.header("🤖 AI Analysis")

    hours = st.slider("Hours lookback", 1, 48, 24)
    queries = load_llm_queries(hours)

    if not queries:
        st.info("No AI queries in this window.")
        return

    df = pd.DataFrame(queries)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Queries",    len(df))
    col2.metric("Total Cost", f"${df['cost'].sum():.4f}")
    col3.metric("Avg Conf",   f"{df['confidence'].mean():.0%}")
    col4.metric("Buy rate",
                f"{len(df[df['decision']=='BUY'])/max(len(df),1):.0%}")

    # Cost by strategy
    cost_by_strat = df.groupby("strategy")["cost"].sum().reset_index()
    fig = px.pie(cost_by_strat, names="strategy", values="cost",
                 title="AI Cost by Strategy", hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

    # Confidence distribution
    fig2 = px.histogram(df, x="confidence", nbins=20,
                        title="Confidence Distribution", color="strategy")
    st.plotly_chart(fig2, use_container_width=True)

    # Recent queries table
    st.subheader("Recent queries")
    st.dataframe(
        df[["ts", "strategy", "market", "model", "cost", "confidence", "decision"]]
        .head(50),
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# Risk tab
# ---------------------------------------------------------------------------

def render_risk(state: Dict) -> None:
    st.header("⚠️ Risk Management")

    sys_st  = state.get("system", {})
    beast_st = state.get("beast", {})
    lean_st  = state.get("lean",  {})

    # Circuit breaker status
    cb = sys_st.get("circuit_breaker_active", False)
    if cb:
        st.error("🚨 CIRCUIT BREAKER ACTIVE — trading halted")
    else:
        st.success("✅ Circuit breaker: normal")

    # Daily loss %
    loss_pct = sys_st.get("daily_loss_pct", 0.0)
    limit_pct = settings.trading.max_daily_loss_pct
    col1, col2 = st.columns(2)
    col1.metric("Daily Loss %", f"{loss_pct:.1f}%", delta=f"Limit: {limit_pct:.0f}%")
    col2.progress(
        min(1.0, loss_pct / max(limit_pct, 1)),
        text=f"{loss_pct:.1f}% of {limit_pct:.0f}% daily limit",
    )

    # Position concentration
    positions = load_positions()
    if positions:
        df = pd.DataFrame(positions)
        by_strat = df.groupby("strategy").size().reset_index(name="count")
        fig = px.bar(by_strat, x="strategy", y="count",
                     title="Position Count by Strategy",
                     color="strategy", color_discrete_map={
                         "lean_directional": "#00c4a0",
                         "directional_trading": "#ff4b4b",
                     })
        st.plotly_chart(fig, use_container_width=True)

    # Daily AI budget
    daily_ai = beast_st.get("ai_cost_today", 0) + lean_st.get("ai_cost_today", 0)
    budget = settings.trading.daily_ai_budget
    st.subheader("AI Budget")
    col1, col2 = st.columns(2)
    col1.metric("Spent today", f"${daily_ai:.4f}", delta=f"Budget: ${budget:.2f}")
    col2.progress(min(1.0, daily_ai / max(budget, 0.01)),
                  text=f"${daily_ai:.4f} / ${budget:.2f}")


# ---------------------------------------------------------------------------
# System Health tab
# ---------------------------------------------------------------------------

def render_system(state: Dict) -> None:
    st.header("🔧 System Health")

    api = settings.api

    def _check(val, label):
        icon = "✅" if val else "❌"
        st.write(f"{icon} {label}: {'configured' if val else '**MISSING**'}")

    st.subheader("API Keys")
    col1, col2 = st.columns(2)
    with col1:
        _check(api.kalshi_api_key,    "KALSHI_API_KEY")
        _check(api.xai_api_key,       "XAI_API_KEY")
        _check(api.odds_api_key,      "ODDS_API_KEY")
        _check(api.fred_api_key,      "FRED_API_KEY")
        _check(api.newsapi_key,       "NEWSAPI_KEY")
    with col2:
        _check(api.metaculus_api_key, "METACULUS_API_KEY")
        _check(api.bls_api_key,       "BLS_API_KEY")
        st.write("✅ ESPN API (no auth required)")
        st.write("✅ Polymarket (no auth required)")
        st.write("✅ Manifold (no auth required)")
        st.write("✅ PredictIt (no auth required)")
        st.write("✅ MLB Stats API (no auth required)")
        st.write("✅ Jolpica F1 (no auth required)")
        st.write("✅ Open-Meteo (no auth required)")
        st.write("✅ NWS Weather (no auth required)")
        st.write("✅ CoinGecko (no auth required)")

    st.subheader("Configuration")
    col1, col2, col3 = st.columns(3)
    col1.metric("Max Positions",   settings.trading.max_positions)
    col2.metric("Daily AI Budget", f"${settings.trading.daily_ai_budget:.2f}")
    col3.metric("Kelly Cap",       f"{settings.trading.kelly_fraction:.0%}")

    # State file info
    st.subheader("State File")
    if _STATE_FILE.exists():
        mtime = datetime.fromtimestamp(_STATE_FILE.stat().st_mtime)
        st.write(f"📄 `bot_state.json` — last modified {mtime.strftime('%H:%M:%S')}")
    else:
        st.warning("bot_state.json not found — orchestrator may not be running")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    state = load_bot_state()
    page  = render_sidebar(state)

    if page == "🏠 Overview":
        render_overview(state)
    elif page == "📊 Positions":
        render_positions(state)
    elif page == "📈 Performance":
        render_performance()
    elif page == "🤖 AI Analysis":
        render_ai_analysis()
    elif page == "⚠️ Risk":
        render_risk(state)
    elif page == "🔧 System":
        render_system(state)

    # Auto-refresh every 10 seconds when a strategy is running
    beast_running = state.get("beast", {}).get("running", False)
    lean_running  = state.get("lean",  {}).get("running", False)
    if beast_running or lean_running:
        import time
        time.sleep(0.1)
        st.rerun()


if __name__ == "__main__":
    main()
