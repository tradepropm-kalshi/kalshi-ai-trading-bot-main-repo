"""
Claude AI Client — Anthropic API wrapper for dual-model consensus trading.

Role in the bot
---------------
Lean Mode  → claude-opus-4-6  with adaptive thinking
             Deep, reasoning-based market analysis run *after* Grok-3.
             Both models must agree before a position is opened.

Beast Mode → claude-haiku-4-5
             Instant (< $0.001) signal confirmation.
             Acts as a second veto layer on top of Grok-3's CONFIRM.

If ANTHROPIC_API_KEY is absent the client degrades gracefully:
  • get_trading_decision()  returns (None, 0.0)   → lean falls back to Grok-only
  • get_fast_confirmation() returns (None, 0.0)   → beast falls back to Grok-only
"""

import logging
from typing import Optional, Tuple

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

logger = logging.getLogger("claude_client")

# ---------------------------------------------------------------------------
# Model IDs
# ---------------------------------------------------------------------------
OPUS_MODEL  = "claude-opus-4-6"   # Lean Mode — deep analysis with thinking
HAIKU_MODEL = "claude-haiku-4-5"  # Beast Mode — fast signal confirmation

# Approximate cost per 1 M tokens (USD) — for cost tracking
_OPUS_IN   = 5.00    # $/1M input tokens
_OPUS_OUT  = 25.00   # $/1M output tokens
_HAIKU_IN  = 1.00
_HAIKU_OUT = 5.00


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class ClaudeClient:
    """
    Async Anthropic wrapper.

    Parameters
    ----------
    api_key : str
        Anthropic API key (from ANTHROPIC_API_KEY env var).
    db_manager : optional
        DatabaseManager for shared AI cost tracking.
    """

    def __init__(self, api_key: str, db_manager=None) -> None:
        self._db = db_manager
        self._total_cost: float = 0.0

        if not _ANTHROPIC_AVAILABLE:
            logger.warning(
                "anthropic package not installed — Claude calls disabled. "
                "Run: pip install anthropic"
            )
            self._client = None
            return

        if not api_key:
            logger.warning(
                "ANTHROPIC_API_KEY not set — Claude calls disabled. "
                "Add it to your .env to enable dual-model consensus."
            )
            self._client = None
            return

        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        logger.info("ClaudeClient initialised (Opus 4.6 + Haiku 4.5)")

    # ------------------------------------------------------------------
    # Lean Mode — deep analysis
    # ------------------------------------------------------------------

    async def get_trading_decision(
        self,
        prompt: str,
        market_id: str = "",
        strategy: str = "lean_directional",
    ) -> Tuple[Optional[str], float]:
        """
        Deep market analysis with Claude Opus 4.6 + adaptive thinking.

        Called after Grok-3 says BUY to get a second opinion.

        Returns ``(response_text, cost_usd)``.
        Returns ``(None, 0.0)`` on any error — caller falls back to Grok-only.
        """
        if self._client is None:
            return None, 0.0

        try:
            response = await self._client.messages.create(
                model=OPUS_MODEL,
                max_tokens=600,
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": prompt}],
            )

            # Collect text blocks (thinking blocks are separate and not included)
            text = "".join(
                block.text for block in response.content if block.type == "text"
            )

            cost = _token_cost(
                response.usage.input_tokens,
                response.usage.output_tokens,
                _OPUS_IN,
                _OPUS_OUT,
            )
            self._total_cost += cost
            await self._track_cost(cost)

            logger.debug(
                "Claude Opus decision  market=%s  tokens_in=%d  tokens_out=%d  cost=$%.5f",
                market_id,
                response.usage.input_tokens,
                response.usage.output_tokens,
                cost,
            )
            return text.strip() or None, cost

        except Exception as exc:  # noqa: BLE001
            _log_api_error("get_trading_decision", exc, market_id)
            return None, 0.0

    # ------------------------------------------------------------------
    # Beast Mode — fast signal confirmation
    # ------------------------------------------------------------------

    async def get_fast_confirmation(
        self,
        prompt: str,
        market_id: str = "",
        strategy: str = "flow_copy_trade",
    ) -> Tuple[Optional[str], float]:
        """
        Fast signal gate using Claude Haiku 4.5.

        Designed for Beast Mode: < $0.001 per call, typically < 1 s.

        Returns ``(response_text, cost_usd)``.
        Returns ``(None, 0.0)`` on any error — caller falls back to Grok-only.
        """
        if self._client is None:
            return None, 0.0

        try:
            response = await self._client.messages.create(
                model=HAIKU_MODEL,
                max_tokens=80,
                messages=[{"role": "user", "content": prompt}],
            )

            text = response.content[0].text if response.content else ""

            cost = _token_cost(
                response.usage.input_tokens,
                response.usage.output_tokens,
                _HAIKU_IN,
                _HAIKU_OUT,
            )
            self._total_cost += cost
            await self._track_cost(cost)

            logger.debug(
                "Claude Haiku confirm  market=%s  cost=$%.5f  response=%s",
                market_id,
                cost,
                text[:60],
            )
            return text.strip() or None, cost

        except Exception as exc:  # noqa: BLE001
            _log_api_error("get_fast_confirmation", exc, market_id)
            return None, 0.0

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """True if the Anthropic client was initialised successfully."""
        return self._client is not None

    @property
    def total_cost(self) -> float:
        """Cumulative USD cost charged to Anthropic this session."""
        return self._total_cost

    async def close(self) -> None:
        """No persistent connection — mirrors the XAIClient API."""
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _track_cost(self, cost: float) -> None:
        """Persist cost to the shared daily_cost_tracking table."""
        if self._db and cost > 0:
            try:
                await self._db.record_ai_cost(cost)
            except Exception:
                pass  # non-fatal


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

def _token_cost(
    input_tokens: int,
    output_tokens: int,
    input_rate: float,
    output_rate: float,
) -> float:
    """USD cost from token counts and per-1M-token rates."""
    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000


def _log_api_error(method: str, exc: Exception, market_id: str) -> None:
    if _ANTHROPIC_AVAILABLE:
        import anthropic as _a
        if isinstance(exc, _a.AuthenticationError):
            logger.error("Claude auth failed (%s) — check ANTHROPIC_API_KEY", method)
            return
        if isinstance(exc, _a.RateLimitError):
            logger.warning("Claude rate limit (%s, market=%s) — skipping", method, market_id)
            return
    logger.warning("Claude %s error for %s: %s", method, market_id, exc)
