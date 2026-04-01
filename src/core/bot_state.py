"""
Shared bot runtime state — the single source of truth for mode toggles.

Solves the core conflict: both BeastModeBot and LeanBot previously mutated
the global ``settings.trading.live_trading_enabled``, meaning the second bot
to start would silently override the first.

This module replaces that pattern with:
  1. An immutable settings layer (read-only access)
  2. A ``BotState`` dataclass with per-strategy flags
  3. ``asyncio.Event`` objects for live on/off signalling within the event loop
  4. A ``StateFile`` that persists toggle state to disk so the Streamlit
     dashboard (separate process) can read and write it

Architecture::

    BotOrchestrator
        │
        ├── BotState (in-memory, asyncio Events)
        │       ├── beast_enabled  → asyncio.Event
        │       ├── lean_enabled   → asyncio.Event
        │       ├── beast_live     → bool (True=live, False=paper)
        │       └── lean_live      → bool
        │
        └── StateFile  (bot_state.json on disk)
                ├── read by Streamlit dashboard
                └── written by both dashboard (toggles) and orchestrator (status)
"""

import asyncio
import json
import os
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

_STATE_FILE = Path("bot_state.json")


@dataclass
class StrategyStatus:
    """Live runtime status snapshot for one strategy."""
    enabled: bool = False
    live_mode: bool = False          # True = real orders, False = paper
    running: bool = False            # True = task is currently executing
    positions_open: int = 0
    daily_pnl: float = 0.0
    ai_cost_today: float = 0.0
    last_cycle_at: str = ""
    last_error: str = ""
    cycle_count: int = 0


@dataclass
class SystemStatus:
    """Overall system health snapshot."""
    portfolio_balance: float = 0.0
    available_cash: float = 0.0
    total_positions: int = 0
    daily_loss_pct: float = 0.0
    circuit_breaker_active: bool = False
    uptime_seconds: float = 0.0
    started_at: str = ""
    last_updated: str = ""


@dataclass
class BotState:
    """
    Runtime state for the bot orchestrator.

    This is the authoritative in-memory state.  It is serialised to
    ``bot_state.json`` after every mutation so the Streamlit dashboard
    (running in a separate process) can read it without polling the
    database.

    The ``beast_enable_event`` / ``lean_enable_event`` are asyncio Events
    used to wake the orchestrator's control loop when a toggle fires from
    the dashboard.
    """

    # ── Per-strategy mode flags (set at startup, can change at runtime) ───────
    beast_enabled: bool = False
    lean_enabled: bool = False
    beast_live: bool = False          # live vs paper
    lean_live: bool = False

    # ── Runtime status (updated by loops, read by dashboard) ─────────────────
    beast: StrategyStatus = field(default_factory=StrategyStatus)
    lean: StrategyStatus = field(default_factory=StrategyStatus)
    system: SystemStatus = field(default_factory=SystemStatus)

    # ── asyncio coordination (not serialised) ────────────────────────────────
    beast_enable_event: asyncio.Event = field(
        default_factory=asyncio.Event, compare=False, repr=False
    )
    lean_enable_event: asyncio.Event = field(
        default_factory=asyncio.Event, compare=False, repr=False
    )
    shutdown_event: asyncio.Event = field(
        default_factory=asyncio.Event, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        # Sync event state with enabled flags on construction
        if self.beast_enabled:
            self.beast_enable_event.set()
        if self.lean_enabled:
            self.lean_enable_event.set()

    # ------------------------------------------------------------------
    # Toggle helpers (call from orchestrator or dashboard file writer)
    # ------------------------------------------------------------------

    def enable_beast(self, live: bool = False) -> None:
        """Enable beast mode (optionally live)."""
        self.beast_enabled = True
        self.beast_live    = live
        self.beast.enabled  = True
        self.beast.live_mode = live
        self.beast_enable_event.set()
        self._persist()

    def disable_beast(self) -> None:
        """Disable beast mode gracefully."""
        self.beast_enabled = False
        self.beast.enabled  = False
        self.beast_enable_event.clear()
        self._persist()

    def enable_lean(self, live: bool = False) -> None:
        """Enable lean mode (optionally live)."""
        self.lean_enabled = True
        self.lean_live    = live
        self.lean.enabled  = True
        self.lean.live_mode = live
        self.lean_enable_event.set()
        self._persist()

    def disable_lean(self) -> None:
        """Disable lean mode gracefully."""
        self.lean_enabled = False
        self.lean.enabled  = False
        self.lean_enable_event.clear()
        self._persist()

    def set_live_mode(self, strategy: str, live: bool) -> None:
        """Switch a running strategy between live and paper without stopping it."""
        if strategy == "beast":
            self.beast_live      = live
            self.beast.live_mode = live
        elif strategy == "lean":
            self.lean_live      = live
            self.lean.live_mode = live
        self._persist()

    # ------------------------------------------------------------------
    # Status update helpers (called by bot loops every cycle)
    # ------------------------------------------------------------------

    def update_beast(self, **kwargs) -> None:
        """Update beast strategy status fields."""
        for k, v in kwargs.items():
            if hasattr(self.beast, k):
                setattr(self.beast, k, v)
        self.beast.last_cycle_at = datetime.now().isoformat(timespec="seconds")
        self._persist()

    def update_lean(self, **kwargs) -> None:
        """Update lean strategy status fields."""
        for k, v in kwargs.items():
            if hasattr(self.lean, k):
                setattr(self.lean, k, v)
        self.lean.last_cycle_at = datetime.now().isoformat(timespec="seconds")
        self._persist()

    def update_system(self, **kwargs) -> None:
        """Update system-level metrics."""
        for k, v in kwargs.items():
            if hasattr(self.system, k):
                setattr(self.system, k, v)
        self.system.last_updated = datetime.now().isoformat(timespec="seconds")
        self._persist()

    # ------------------------------------------------------------------
    # Disk persistence (JSON, Streamlit-readable)
    # ------------------------------------------------------------------

    def _persist(self) -> None:
        """Write current state to ``bot_state.json`` (non-blocking best-effort)."""
        try:
            data = {
                "beast_enabled": self.beast_enabled,
                "lean_enabled":  self.lean_enabled,
                "beast_live":    self.beast_live,
                "lean_live":     self.lean_live,
                "beast":  asdict(self.beast),
                "lean":   asdict(self.lean),
                "system": asdict(self.system),
                "written_at": datetime.now().isoformat(timespec="seconds"),
            }
            _STATE_FILE.write_text(json.dumps(data, indent=2))
        except OSError:
            pass  # Non-fatal — dashboard will retry on next read

    @classmethod
    def load_from_file(cls) -> Dict:
        """
        Read the last persisted state from disk.

        Used by the Streamlit dashboard process (which cannot share the
        asyncio event loop with the bot).

        Returns:
            Raw dict from JSON, or a default empty state dict.
        """
        try:
            if _STATE_FILE.exists():
                return json.loads(_STATE_FILE.read_text())
        except (OSError, json.JSONDecodeError):
            pass
        return {
            "beast_enabled": False,
            "lean_enabled":  False,
            "beast_live":    False,
            "lean_live":     False,
            "beast":  asdict(StrategyStatus()),
            "lean":   asdict(StrategyStatus()),
            "system": asdict(SystemStatus()),
            "written_at": "",
        }

    def write_toggle(self, strategy: str, enabled: bool, live: bool = False) -> None:
        """
        Write a toggle instruction to disk for the orchestrator to pick up.

        Called by the Streamlit dashboard when a user flips a toggle.  The
        orchestrator polls ``bot_state.json`` every few seconds and applies
        any pending toggle instructions.
        """
        data = self.load_from_file()
        if strategy == "beast":
            data["beast_enabled"] = enabled
            data["beast_live"]    = live
        elif strategy == "lean":
            data["lean_enabled"] = enabled
            data["lean_live"]    = live
        data["toggle_pending"] = True
        data["written_at"] = datetime.now().isoformat(timespec="seconds")
        try:
            _STATE_FILE.write_text(json.dumps(data, indent=2))
        except OSError:
            pass


# Module-level singleton — shared by orchestrator and strategy loops
_state: Optional[BotState] = None


def get_state() -> BotState:
    """Return the process-level :class:`BotState` singleton."""
    global _state
    if _state is None:
        _state = BotState()
    return _state


def init_state(
    beast_enabled: bool = False,
    lean_enabled: bool = False,
    beast_live: bool = False,
    lean_live: bool = False,
) -> BotState:
    """
    Initialise (or reset) the process-level state singleton.

    Called once by the orchestrator at startup.
    """
    global _state
    _state = BotState(
        beast_enabled=beast_enabled,
        lean_enabled=lean_enabled,
        beast_live=beast_live,
        lean_live=lean_live,
    )
    _state.beast.enabled   = beast_enabled
    _state.beast.live_mode = beast_live
    _state.lean.enabled    = lean_enabled
    _state.lean.live_mode  = lean_live
    _state.system.started_at = datetime.now().isoformat(timespec="seconds")
    _state._persist()
    return _state
