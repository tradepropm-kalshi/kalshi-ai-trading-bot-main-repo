import streamlit as st
import asyncio
import pandas as pd
import aiosqlite
from datetime import datetime
from src.config.settings import settings

st.set_page_config(page_title="Kalshi Phase Bot Dashboard", layout="wide")
st.title("🚀 Kalshi AI Trading Bot — Phase Profit Dashboard")
st.caption("Local web dashboard • Updates every 15 seconds • 100% on your computer")

DB_PATH = "trading_system.db"

async def get_data():
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        try:
            # Phase state
            cursor = await db.execute("SELECT * FROM phase_state WHERE id = 1")
            phase_row = await cursor.fetchone()
            phase = dict(phase_row) if phase_row else {"current_phase_profit": 0.0, "total_secured_profit": 0.0}
            
            # Positions (graceful if table missing)
            cursor = await db.execute("SELECT * FROM positions WHERE status = 'open'")
            positions = [dict(row) for row in await cursor.fetchall()]
            
            # Trades
            cursor = await db.execute("SELECT * FROM trade_logs ORDER BY exit_timestamp")
            trades = [dict(row) for row in await cursor.fetchall()]
            
            return phase, positions, trades, None
        except aiosqlite.OperationalError as e:
            if "no such table" in str(e).lower():
                return None, [], [], "Database tables not created yet"
            raise

try:
    phase, positions, trades, init_error = asyncio.run(get_data())
    
    if init_error:
        st.error("🚨 Database tables not created yet")
        st.info("✅ Please run the main bot in the other terminal with:\n**py cli.py run --phase**\n\nIt will create all tables on first run.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    
    current = phase.get('current_phase_profit', 0.0)
    secured = phase.get('total_secured_profit', 0.0)
    target = getattr(settings.trading, 'phase_profit_target', 2500.0)
    
    progress = min(current / target, 1.0) if target > 0 else 0.0
    
    col1.metric("Current Phase Profit", f"${current:,.2f}", f"/ ${target:,.2f}")
    col1.progress(progress)
    col1.caption(f"🔄 {progress:.0%} toward next $2,400 secure")
    
    col2.metric("Total Secured Profit", f"${secured:,.2f}", "all-time")
    
    if current >= target:
        col3.success("🎉 PHASE COMPLETE — $2,400 secured & reset!")
    else:
        col3.info(f"Remaining to next secure: ${target - current:,.2f}")

    # PnL Chart
    if trades:
        df = pd.DataFrame(trades)
        if not df.empty and 'exit_timestamp' in df.columns and 'pnl' in df.columns:
            df['cum_pnl'] = df['pnl'].cumsum()
            st.subheader("Cumulative Realized P&L")
            st.line_chart(df.set_index('exit_timestamp')['cum_pnl'])

    # Positions
    st.subheader("Active Positions")
    if positions:
        pos_data = []
        for p in positions:
            pos_data.append({
                'Market': p.get('market_id', 'N/A'),
                'Side': p.get('side', 'N/A'),
                'Entry': f"${p.get('entry_price', 0):.3f}",
                'Qty': p.get('quantity', 0),
                'Stop Loss': f"${p.get('stop_loss_price', 0):.3f}" if p.get('stop_loss_price') else "-",
                'Take Profit': f"${p.get('take_profit_price', 0):.3f}" if p.get('take_profit_price') else "-",
            })
        st.dataframe(pos_data, use_container_width=True)
    else:
        st.info("No open positions at the moment.")

except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.info("Make sure the main bot is running in the other terminal with 'py cli.py run --phase'")

st.caption("🔄 Auto-refreshing every 15 seconds")
st.rerun()