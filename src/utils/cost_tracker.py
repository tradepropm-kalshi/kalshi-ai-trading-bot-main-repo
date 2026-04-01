"""
Daily AI cost tracking with pickle-based persistence.

Provides CostTracker for recording and enforcing daily spending limits
on AI API calls across process restarts.
"""

import os
import pickle
from datetime import datetime
from typing import Optional


class CostTracker:
    """
    Tracks daily AI API costs and enforces a configurable spending limit.

    State is persisted to a pickle file so that totals survive process
    restarts within the same calendar day.
    """

    def __init__(self, storage_path: str = "logs/daily_ai_usage.pkl"):
        """
        Initialise the tracker, loading any existing state from disk.

        Args:
            storage_path: Path to the pickle file used for persistence.
        """
        self.storage_path = storage_path
        self._date: str = datetime.now().strftime("%Y-%m-%d")
        self._total_cost: float = 0.0
        self._request_count: int = 0
        self._is_exhausted: bool = False
        self._last_exhausted_time: Optional[datetime] = None
        self._load()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load persisted state from disk, resetting if it is from a previous day."""
        os.makedirs(os.path.dirname(self.storage_path) or ".", exist_ok=True)
        if not os.path.exists(self.storage_path):
            return

        try:
            with open(self.storage_path, "rb") as fh:
                state = pickle.load(fh)

            if isinstance(state, dict) and state.get("date") == datetime.now().strftime("%Y-%m-%d"):
                self._date = state["date"]
                self._total_cost = float(state.get("total_cost", 0.0))
                self._request_count = int(state.get("request_count", 0))
                self._is_exhausted = bool(state.get("is_exhausted", False))
                self._last_exhausted_time = state.get("last_exhausted_time")
        except (OSError, pickle.UnpicklingError, KeyError, TypeError):
            # Corrupt or unreadable file — start fresh
            pass

    def _save(self) -> None:
        """Persist current state to disk."""
        os.makedirs(os.path.dirname(self.storage_path) or ".", exist_ok=True)
        try:
            state = {
                "date": self._date,
                "total_cost": self._total_cost,
                "request_count": self._request_count,
                "is_exhausted": self._is_exhausted,
                "last_exhausted_time": self._last_exhausted_time,
            }
            with open(self.storage_path, "wb") as fh:
                pickle.dump(state, fh)
        except OSError:
            pass  # Non-fatal — metrics are best-effort

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset_if_new_day(self) -> bool:
        """
        Reset the tracker when the calendar date has changed.

        Returns:
            True if the tracker was reset (new day detected), False otherwise.
        """
        today = datetime.now().strftime("%Y-%m-%d")
        if self._date != today:
            self._date = today
            self._total_cost = 0.0
            self._request_count = 0
            self._is_exhausted = False
            self._last_exhausted_time = None
            self._save()
            return True
        return False

    def record_cost(self, amount: float) -> None:
        """
        Record a cost increment and persist state to disk.

        Args:
            amount: Cost in USD to add to the daily total.
        """
        self._total_cost += amount
        self._request_count += 1
        self._save()

    def get_daily_total(self) -> float:
        """
        Return the accumulated cost for the current calendar day.

        Returns:
            Total cost in USD recorded today.
        """
        return self._total_cost

    def get_request_count(self) -> int:
        """Return the number of requests recorded today."""
        return self._request_count

    def is_over_limit(self, limit: float) -> bool:
        """
        Check whether the daily total meets or exceeds *limit*.

        Also updates the internal exhaustion flag and persists the change when
        the limit is first crossed.

        Args:
            limit: Daily spending ceiling in USD.

        Returns:
            True if the daily total is >= limit.
        """
        if self._total_cost >= limit:
            if not self._is_exhausted:
                self._is_exhausted = True
                self._last_exhausted_time = datetime.now()
                self._save()
            return True

        # Limit may have been raised since exhaustion — un-exhaust
        if self._is_exhausted and self._total_cost < limit:
            self._is_exhausted = False
            self._save()

        return False

    @property
    def is_exhausted(self) -> bool:
        """True when the daily limit has been reached."""
        return self._is_exhausted

    @is_exhausted.setter
    def is_exhausted(self, value: bool) -> None:
        """Force the exhaustion flag and persist."""
        self._is_exhausted = value
        if value:
            self._last_exhausted_time = datetime.now()
        self._save()

    @property
    def date(self) -> str:
        """The calendar date string this tracker is active for (YYYY-MM-DD)."""
        return self._date

    @property
    def last_exhausted_time(self) -> Optional[datetime]:
        """Datetime when the limit was last reached, or None."""
        return self._last_exhausted_time
