"""
Calibration record management for the ensemble runner.

Stores ensemble probability predictions alongside final market outcomes so
that calibration curves (reliability diagrams) can be computed later.
Records are capped at 5 000 entries to prevent unbounded file growth.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from src.utils.logging_setup import get_trading_logger

logger = get_trading_logger("calibration")

_DEFAULT_PATH = Path("logs/ensemble_calibration.json")
_MAX_RECORDS = 5_000


class CalibrationManager:
    """
    Append-only store for ensemble calibration records.

    Each record captures the ensemble's probability estimate at decision
    time.  After a market resolves the ``resolved_yes`` field can be
    backfilled to compute accuracy and Brier scores.

    Args:
        file_path: Path to the JSON file used for persistence.
                   Defaults to ``logs/ensemble_calibration.json``.
        max_records: Maximum number of records to retain.  Older records
                     are dropped when the cap is exceeded.
    """

    def __init__(
        self,
        file_path: Path = _DEFAULT_PATH,
        max_records: int = _MAX_RECORDS,
    ) -> None:
        self.file_path = Path(file_path)
        self.max_records = max_records

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        market_data: dict,
        probability: float,
        confidence: float,
        disagreement: float,
        model_results: list,
    ) -> None:
        """
        Append one calibration record to the JSON store.

        Errors are logged but never raised so they cannot interrupt trading.

        Args:
            market_data:   Standard market data dict (title, ticker, yes_price).
            probability:   Ensemble weighted-average YES probability.
            confidence:    Aggregate ensemble confidence score.
            disagreement:  Std-dev of per-model probabilities.
            model_results: Raw per-model result dicts from the ensemble run.
        """
        entry = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "market_title": market_data.get("title", "")[:200],
            "market_ticker": market_data.get("ticker", ""),
            "yes_price": market_data.get("yes_price"),
            "ensemble_probability": probability,
            "ensemble_confidence": confidence,
            "disagreement": disagreement,
            "num_models": len([r for r in model_results if "error" not in r]),
            "model_probabilities": {
                r.get("_agent", "?"): r.get("probability")
                for r in model_results
                if "error" not in r and r.get("probability") is not None
            },
            "resolved_yes": None,  # Backfilled after market resolution
        }
        try:
            records = self._load()
            records.append(entry)
            # Enforce cap — keep the most recent records
            if len(records) > self.max_records:
                records = records[-self.max_records :]
            self._save(records)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to write calibration record", error=str(exc))

    def backfill_outcome(self, market_ticker: str, resolved_yes: bool) -> int:
        """
        Set ``resolved_yes`` for all records matching *market_ticker*.

        Args:
            market_ticker: Kalshi ticker string.
            resolved_yes:  ``True`` if the market resolved YES.

        Returns:
            Number of records updated.
        """
        try:
            records = self._load()
            updated = 0
            for r in records:
                if r.get("market_ticker") == market_ticker and r.get("resolved_yes") is None:
                    r["resolved_yes"] = resolved_yes
                    updated += 1
            if updated:
                self._save(records)
            return updated
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to backfill calibration outcome", error=str(exc))
            return 0

    def get_records(self) -> List[dict]:
        """Return all stored calibration records (may be empty)."""
        try:
            return self._load()
        except Exception:  # noqa: BLE001
            return []

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _load(self) -> list:
        """Load records from disk; return empty list on missing/corrupt file."""
        if not self.file_path.exists():
            return []
        try:
            data = json.loads(self.file_path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def _save(self, records: list) -> None:
        """Persist *records* to disk, creating parent directories as needed."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_path.write_text(
            json.dumps(records, indent=2), encoding="utf-8"
        )
