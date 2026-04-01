"""
API Key Tester — run this before launching the bot to verify all keys work.

Usage:  python test_keys.py
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv()

try:
    import httpx
except ImportError:
    print("httpx not installed. Run:  pip install -r requirements.txt")
    sys.exit(1)


PASS = "  [PASS]"
FAIL = "  [FAIL]"
SKIP = "  [SKIP]"


async def test_kalshi(key: str) -> tuple[bool, str]:
    if not key:
        return False, "No key set"
    try:
        async with httpx.AsyncClient(timeout=8) as c:
            r = await c.get(
                "https://demo-api.kalshi.com/trade-api/rest/v2/markets",
                headers={"Authorization": f"Bearer {key}"},
                params={"limit": 1},
            )
        if r.status_code == 200:
            return True, f"Connected — HTTP {r.status_code}"
        if r.status_code == 401:
            return False, "Invalid key (401 Unauthorized)"
        return False, f"Unexpected HTTP {r.status_code}"
    except Exception as e:
        return False, str(e)


async def test_xai(key: str) -> tuple[bool, str]:
    if not key:
        return False, "No key set"
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.post(
                "https://api.x.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "grok-3",
                    "messages": [{"role": "user", "content": "Reply with one word: ok"}],
                    "max_tokens": 5,
                },
            )
        if r.status_code == 200:
            return True, "Connected — Grok-3 responded"
        if r.status_code == 401:
            return False, "Invalid key (401 Unauthorized)"
        return False, f"HTTP {r.status_code}: {r.text[:100]}"
    except Exception as e:
        return False, str(e)


async def test_anthropic(key: str) -> tuple[bool, str]:
    if not key:
        return None, "Not set (optional — enables dual-model consensus)"
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5",
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": "Reply: ok"}],
                },
            )
        if r.status_code == 200:
            return True, "Connected — Claude Haiku responded"
        if r.status_code == 401:
            return False, "Invalid key (401 Unauthorized)"
        return False, f"HTTP {r.status_code}: {r.text[:100]}"
    except Exception as e:
        return False, str(e)


async def test_newsapi(key: str) -> tuple[bool, str]:
    if not key:
        return None, "Not set (optional)"
    try:
        async with httpx.AsyncClient(timeout=8) as c:
            r = await c.get(
                "https://newsapi.org/v2/everything",
                params={"q": "market", "pageSize": 1, "apiKey": key},
            )
        if r.status_code == 200:
            return True, "Connected"
        if r.status_code == 401:
            return False, "Invalid key"
        return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, str(e)


async def test_fred(key: str) -> tuple[bool, str]:
    if not key:
        return None, "Not set (optional)"
    try:
        async with httpx.AsyncClient(timeout=8) as c:
            r = await c.get(
                "https://api.stlouisfed.org/fred/series",
                params={"series_id": "GDP", "api_key": key, "file_type": "json"},
            )
        if r.status_code == 200:
            return True, "Connected"
        if r.status_code == 400:
            return False, "Invalid key"
        return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, str(e)


async def test_odds(key: str) -> tuple[bool, str]:
    if not key:
        return None, "Not set (optional)"
    try:
        async with httpx.AsyncClient(timeout=8) as c:
            r = await c.get(
                "https://api.the-odds-api.com/v4/sports",
                params={"apiKey": key},
            )
        if r.status_code == 200:
            return True, f"Connected — {r.headers.get('x-requests-remaining', '?')} requests remaining"
        if r.status_code == 401:
            return False, "Invalid key"
        return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, str(e)


async def test_bls(key: str) -> tuple[bool, str]:
    if not key:
        return None, "Not set (optional)"
    try:
        async with httpx.AsyncClient(timeout=8) as c:
            r = await c.post(
                "https://api.bls.gov/publicAPI/v2/timeseries/data/",
                json={"seriesid": ["CUUR0000SA0"], "registrationkey": key},
            )
        if r.status_code == 200:
            data = r.json()
            status = data.get("status", "")
            if status == "REQUEST_SUCCEEDED":
                return True, "Connected"
            return False, f"API returned: {status}"
        return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, str(e)


async def test_metaculus(key: str) -> tuple[bool, str]:
    if not key:
        return None, "Not set (optional)"
    try:
        async with httpx.AsyncClient(timeout=8) as c:
            r = await c.get(
                "https://www.metaculus.com/api2/questions/",
                headers={"Authorization": f"Token {key}"},
                params={"limit": 1},
            )
        if r.status_code == 200:
            return True, "Connected"
        if r.status_code == 401:
            return False, "Invalid key"
        return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, str(e)


async def test_no_auth(name: str, url: str) -> tuple[bool, str]:
    try:
        async with httpx.AsyncClient(timeout=8) as c:
            r = await c.get(url)
        if r.status_code == 200:
            return True, "Reachable (no key required)"
        return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, str(e)


async def main():
    print()
    print("=" * 52)
    print("  KALSHI AI BOT — API KEY TEST")
    print("=" * 52)

    results = []

    # ── Required keys ────────────────────────────────────────
    print("\n  REQUIRED")
    print("  " + "-" * 40)

    ok, msg = await test_kalshi(os.getenv("KALSHI_API_KEY", ""))
    tag = PASS if ok else FAIL
    print(f"{tag}  Kalshi        {msg}")
    results.append(("Kalshi", ok))

    ok, msg = await test_xai(os.getenv("XAI_API_KEY", ""))
    tag = PASS if ok else FAIL
    print(f"{tag}  xAI / Grok    {msg}")
    results.append(("xAI", ok))

    # ── Optional AI key ──────────────────────────────────────
    print("\n  OPTIONAL AI (dual-model consensus)")
    print("  " + "-" * 40)

    ok, msg = await test_anthropic(os.getenv("ANTHROPIC_API_KEY", ""))
    tag = PASS if ok is True else (SKIP if ok is None else FAIL)
    print(f"{tag}  Anthropic / Claude  {msg}")

    # ── Optional keys ─────────────────────────────────────────
    print("\n  OPTIONAL (lean bot data enrichment)")
    print("  " + "-" * 40)

    for label, coro in [
        ("NewsAPI  ", test_newsapi(os.getenv("NEWSAPI_KEY", ""))),
        ("FRED     ", test_fred(os.getenv("FRED_API_KEY", ""))),
        ("Odds API ", test_odds(os.getenv("ODDS_API_KEY", ""))),
        ("BLS      ", test_bls(os.getenv("BLS_API_KEY", ""))),
        ("Metaculus", test_metaculus(os.getenv("METACULUS_API_KEY", ""))),
    ]:
        ok, msg = await coro
        tag = PASS if ok is True else (SKIP if ok is None else FAIL)
        print(f"{tag}  {label}  {msg}")

    # ── No-auth sources ───────────────────────────────────────
    print("\n  NO KEY REQUIRED")
    print("  " + "-" * 40)

    for label, url in [
        ("Jolpica F1  ", "https://api.jolpi.ca/ergast/f1/current.json"),
        ("Manifold    ", "https://api.manifold.markets/v0/markets?limit=1"),
        ("PredictIt   ", "https://www.predictit.org/api/marketdata/all/"),
        ("NWS Weather ", "https://api.weather.gov/"),
        ("MLB Stats   ", "https://statsapi.mlb.com/api/v1/sports"),
    ]:
        ok, msg = await test_no_auth(label, url)
        tag = PASS if ok else FAIL
        print(f"{tag}  {label}  {msg}")

    # ── Summary ───────────────────────────────────────────────
    print()
    print("=" * 52)
    required_ok = all(ok for _, ok in results)
    if required_ok:
        print("  All required keys working — bot is ready to run.")
    else:
        failed = [name for name, ok in results if not ok]
        print(f"  REQUIRED keys failing: {', '.join(failed)}")
        print("  Check your .env file before launching the bot.")
    print("=" * 52)
    print()


if __name__ == "__main__":
    asyncio.run(main())
