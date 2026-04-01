"""
LLM response parsing utilities for the trading bot.

Provides a single, well-tested place for JSON extraction and field
normalisation so that XAIClient and other callers do not need to
duplicate regex / repair logic.
"""

import json
import re
from typing import Any, Dict, Optional

from json_repair import repair_json


class ResponseParser:
    """
    Extracts structured data from raw LLM response text.

    All methods are static so the class can be used without instantiation.
    """

    # ------------------------------------------------------------------
    # JSON extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_json(text: str) -> Optional[Dict[str, Any]]:
        """
        Extract and parse the first JSON object found in *text*.

        Tries three strategies in order:
        1. JSON inside a ```json … ``` code fence.
        2. The first ``{…}`` block found via regex.
        3. The whole text passed through ``json_repair``.

        Args:
            text: Raw LLM response string.

        Returns:
            Parsed dict, or ``None`` if no valid JSON could be found.
        """
        if not text:
            return None

        # Strategy 1: fenced code block
        fence_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if fence_match:
            try:
                return json.loads(fence_match.group(1))
            except json.JSONDecodeError:
                try:
                    return json.loads(repair_json(fence_match.group(1)))
                except (json.JSONDecodeError, ValueError):
                    pass

        # Strategy 2: bare brace block
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            candidate = brace_match.group(0)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                try:
                    return json.loads(repair_json(candidate))
                except (json.JSONDecodeError, ValueError):
                    pass

        # Strategy 3: repair the whole response
        try:
            repaired = repair_json(text)
            if repaired:
                return json.loads(repaired)
        except (json.JSONDecodeError, ValueError):
            pass

        return None

    # ------------------------------------------------------------------
    # Field extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def extract_probability(data: Dict[str, Any]) -> Optional[float]:
        """
        Pull a probability value from a parsed LLM response dict.

        Looks for keys ``probability``, ``prob``, ``yes_probability``,
        ``confidence`` (as a fallback), clamping the result to [0.0, 1.0].

        Args:
            data: Parsed JSON dict from the LLM response.

        Returns:
            Probability in [0.0, 1.0], or ``None`` if not found.
        """
        for key in ("probability", "prob", "yes_probability", "p"):
            val = data.get(key)
            if val is not None:
                try:
                    return float(max(0.0, min(1.0, float(val))))
                except (TypeError, ValueError):
                    pass
        return None

    @staticmethod
    def extract_confidence(data: Dict[str, Any]) -> Optional[float]:
        """
        Pull a confidence score from a parsed LLM response dict.

        Checks ``confidence``, ``certainty``, and ``score``, clamping to
        [0.0, 1.0].

        Args:
            data: Parsed JSON dict from the LLM response.

        Returns:
            Confidence in [0.0, 1.0], or ``None`` if not found.
        """
        for key in ("confidence", "certainty", "score"):
            val = data.get(key)
            if val is not None:
                try:
                    return float(max(0.0, min(1.0, float(val))))
                except (TypeError, ValueError):
                    pass
        return None

    @staticmethod
    def extract_decision(data: Dict[str, Any]) -> Optional[str]:
        """
        Pull and normalise a YES/NO/SKIP trading decision from a parsed dict.

        Recognises ``action``, ``decision``, and ``recommendation`` keys.
        Maps common synonyms to canonical values:
        - BUY_YES / BUY_NO / BUY → ``"BUY"``
        - SKIP / HOLD / PASS / NO_TRADE → ``"SKIP"``

        Args:
            data: Parsed JSON dict from the LLM response.

        Returns:
            One of ``"BUY"``, ``"SKIP"``, or ``None`` if not found.
        """
        for key in ("action", "decision", "recommendation"):
            val = data.get(key)
            if val is not None:
                normalised = str(val).upper().strip()
                if normalised in ("BUY_YES", "BUY_NO", "BUY"):
                    return "BUY"
                if normalised in ("SKIP", "HOLD", "PASS", "NO_TRADE"):
                    return "SKIP"
                return normalised
        return None

    # ------------------------------------------------------------------
    # Trading decision parser
    # ------------------------------------------------------------------

    @staticmethod
    def parse_trading_decision(response_text: str) -> Optional[Dict[str, Any]]:
        """
        Parse a full trading decision from raw LLM response text.

        Extracts and normalises ``action``, ``side``, ``confidence``,
        ``limit_price``, and ``reasoning`` fields.

        Args:
            response_text: Raw LLM response string.

        Returns:
            Dict with keys ``action``, ``side``, ``confidence``,
            ``limit_price``, ``reasoning``, or ``None`` on parse failure.
        """
        data = ResponseParser.extract_json(response_text)
        if data is None:
            return None

        action = str(data.get("action", "SKIP")).upper()
        if action in ("BUY_YES", "BUY_NO", "BUY"):
            action = "BUY"
        elif action in ("SKIP", "HOLD", "PASS"):
            action = "SKIP"

        try:
            confidence = float(max(0.0, min(1.0, float(data.get("confidence", 0.5)))))
        except (TypeError, ValueError):
            confidence = 0.5

        try:
            limit_price = int(data.get("limit_price", 50))
            limit_price = max(1, min(99, limit_price))
        except (TypeError, ValueError):
            limit_price = 50

        return {
            "action": action,
            "side": str(data.get("side", "YES")).upper(),
            "confidence": confidence,
            "limit_price": limit_price,
            "reasoning": str(data.get("reasoning", "No reasoning provided")),
        }

    # ------------------------------------------------------------------
    # Text utilities
    # ------------------------------------------------------------------

    @staticmethod
    def truncate_to_length(text: str, max_length: int) -> str:
        """
        Truncate *text* to at most *max_length* characters, preserving words.

        Args:
            text:       Input string.
            max_length: Maximum number of characters in the result.

        Returns:
            Possibly truncated string (never longer than max_length).
        """
        if len(text) <= max_length:
            return text
        truncated = text[:max_length].rsplit(" ", 1)[0]
        return truncated + "…"
