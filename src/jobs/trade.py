"""
Trade Job — Beast Mode entry point.

Beast Mode now leads with the order-flow copy-trade engine (FlowCopyTradeStrategy)
and falls back to the unified trading system for any remaining position capacity.

Flow copy trade is the *primary* alpha source:
  1. KalshiTradeFeed polls the public trade feed every 3 s.
  2. Block / momentum / imbalance / velocity signals are detected.
  3. AI confirms in < 200 tokens.
  4. Position is sized via Kelly and executed immediately.
  5. Outcome is fed back to the scanner so signal-type weights improve over time.

Unified trading system fills remaining slots (cap: BEAST_MAX_POSITIONS).

run_trading_job() accepts optional pre-built instances from the orchestrator.
"""

import asyncio
from typing import Dict, Optional

from src.utils.database import DatabaseManager
from src.clients.kalshi_client import KalshiClient
from src.clients.xai_client import XAIClient
from src.config.settings import settings
from src.utils.logging_setup import get_trading_logger
from src.strategies.unified_trading_system import run_unified_trading_system, TradingSystemConfig
from src.strategies.flow_copy_trade import FlowCopyTradeStrategy
from src.data.kalshi_trade_feed import KalshiTradeFeed

BEAST_MAX_POSITIONS = 7  # total hard cap for beast mode


async def run_trading_job(
    db_manager: Optional[DatabaseManager] = None,
    kalshi_client: Optional[KalshiClient] = None,
    xai_client: Optional[XAIClient] = None,
    trade_feed: Optional[KalshiTradeFeed] = None,
    flow_strategy: Optional[FlowCopyTradeStrategy] = None,
    live: bool = False,
) -> Dict:
    """
    Execute one Beast Mode trading cycle.

    Priority order:
      1. Flow copy trade (order-flow signals) — primary alpha engine
      2. Unified trading system — fills remaining position capacity

    Pass shared instances from the orchestrator to avoid connection churn.
    """
    logger = get_trading_logger("trade_job")
    logger.info("Starting Beast Mode trade cycle")

    _owns_kalshi_client = kalshi_client is None
    _flow_cycle_result: Dict = {}

    try:
        if db_manager is None:
            db_manager = DatabaseManager()
            await db_manager.initialize()

        if kalshi_client is None:
            kalshi_client = KalshiClient()

        if xai_client is None:
            xai_client = XAIClient(db_manager=db_manager)

        # ── 1. Flow copy trade (primary) ──────────────────────────────────────
        if trade_feed is not None and flow_strategy is not None:
            try:
                _flow_cycle_result = await flow_strategy.run_cycle()
                logger.info(
                    "Flow copy trade: +%d entries, -%d exits, %d open",
                    _flow_cycle_result.get("entries", 0),
                    _flow_cycle_result.get("exits", 0),
                    _flow_cycle_result.get("open_positions", 0),
                )
            except Exception as exc:
                logger.error("Flow copy trade cycle error: %s", exc)

        # ── 2. Unified trading system (fill remaining capacity) ───────────────
        flow_open = _flow_cycle_result.get("open_positions", 0)
        remaining_slots = max(0, BEAST_MAX_POSITIONS - flow_open)

        unified_result = {"total_positions": 0, "total_capital_used": 0.0}
        if remaining_slots > 0:
            try:
                config = TradingSystemConfig()
                results = await run_unified_trading_system(
                    db_manager=db_manager,
                    kalshi_client=kalshi_client,
                    xai_client=xai_client,
                    config=config,
                )
                unified_result = {
                    "total_positions": results.total_positions,
                    "total_capital_used": results.total_capital_used,
                    "capital_efficiency": results.capital_efficiency,
                }
            except Exception as exc:
                logger.error("Unified trading system error: %s", exc)

        total_positions = flow_open + unified_result.get("total_positions", 0)
        logger.info(
            "Beast Mode cycle done — %d total positions (%d flow + %d unified)",
            total_positions, flow_open, unified_result.get("total_positions", 0),
        )

        return {
            "total_positions": total_positions,
            "total_capital_used": unified_result.get("total_capital_used", 0.0),
            "capital_efficiency": unified_result.get("capital_efficiency", 0.0),
            "flow_entries": _flow_cycle_result.get("entries", 0),
            "flow_exits": _flow_cycle_result.get("exits", 0),
        }

    except Exception as exc:
        logger.error("Critical error in Beast Mode trade job: %s", exc)
        return {"total_positions": 0, "total_capital_used": 0.0}

    finally:
        if _owns_kalshi_client and kalshi_client is not None:
            try:
                await kalshi_client.close()
            except Exception:
                pass


async def run_trading_job_async():
    """Backward-compatible async wrapper (no feed / no flow strategy)."""
    return await run_trading_job()
