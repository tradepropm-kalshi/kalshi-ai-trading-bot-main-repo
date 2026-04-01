"""
Price normalisation utilities for Kalshi prediction markets.

Kalshi uses **integer cents** (1–99) internally for yes/no prices, but many
parts of the bot work in **decimal dollars** (0.01–0.99).  Mixing the two
representations was the source of several critical bugs (quantities sized at
1/100th the intended value, spread calculations being off by 100×, etc.).

Use these helpers everywhere a price conversion is needed so that the unit
is always explicit at the call site.
"""

from typing import Union


# ---------------------------------------------------------------------------
# Primary converters
# ---------------------------------------------------------------------------

def cents_to_dollars(cents: Union[int, float]) -> float:
    """
    Convert a Kalshi cent price (1–99) to a decimal dollar price (0.01–0.99).

    Args:
        cents: Price in cents, e.g. ``45`` means 45¢.

    Returns:
        Equivalent dollar price, e.g. ``0.45``.

    Examples::

        >>> cents_to_dollars(45)
        0.45
        >>> cents_to_dollars(0)
        0.0
    """
    return float(cents) / 100.0


def dollars_to_cents(dollars: float) -> int:
    """
    Convert a decimal dollar price (0.01–0.99) to an integer cent price (1–99).

    The result is clamped to the valid Kalshi range [1, 99].

    Args:
        dollars: Price in dollars, e.g. ``0.45`` means 45¢.

    Returns:
        Equivalent cent price as an integer, e.g. ``45``.

    Examples::

        >>> dollars_to_cents(0.45)
        45
        >>> dollars_to_cents(1.5)   # clamps to 99
        99
    """
    return int(max(1, min(99, round(dollars * 100))))


# ---------------------------------------------------------------------------
# Probability / price validation helpers
# ---------------------------------------------------------------------------

def clamp_probability(p: float) -> float:
    """
    Clamp *p* to the valid probability range [0.01, 0.99].

    Prices of 0 or 100 are never valid on Kalshi (a resolved market is
    settled, not tradeable), so this range prevents impossible orders.

    Args:
        p: Probability as a decimal (e.g. ``0.55`` for 55 %).

    Returns:
        Clamped value in [0.01, 0.99].
    """
    return max(0.01, min(0.99, p))


def is_dollar_price(value: float) -> bool:
    """
    Heuristic check: return ``True`` if *value* looks like a dollar price.

    Values in (0, 1) are treated as dollars; values >= 1 as cents.
    Note: this is a *hint*, not a guarantee.  Always prefer explicit
    conversion over this function where the unit is known.

    Args:
        value: Numeric price to inspect.

    Returns:
        ``True`` if value is in (0.0, 1.0), ``False`` otherwise.
    """
    return 0.0 < value < 1.0


# ---------------------------------------------------------------------------
# Position-sizing helpers
# ---------------------------------------------------------------------------

def contract_cost_dollars(quantity: int, price_dollars: float) -> float:
    """
    Total cost in dollars to buy *quantity* contracts at *price_dollars* each.

    Args:
        quantity:      Number of contracts.
        price_dollars: Price per contract in dollars (e.g. ``0.45`` for 45¢).

    Returns:
        Total cost in dollars.
    """
    return quantity * price_dollars


def expected_pnl_dollars(
    quantity: int,
    entry_price_dollars: float,
    exit_price_dollars: float,
) -> float:
    """
    Expected profit or loss in dollars for a position.

    Args:
        quantity:            Number of contracts held.
        entry_price_dollars: Price paid per contract (dollars).
        exit_price_dollars:  Price received per contract (dollars).

    Returns:
        Signed P&L in dollars (positive = profit, negative = loss).
    """
    return quantity * (exit_price_dollars - entry_price_dollars)


# ---------------------------------------------------------------------------
# Spread helpers (market making)
# ---------------------------------------------------------------------------

def synthetic_spread_profit(yes_bid_dollars: float, no_bid_dollars: float) -> float:
    """
    Locked profit per contract pair from a synthetic market-making spread.

    On Kalshi, placing a YES BID and a NO BID whose prices sum to less than
    $1.00 creates a locked profit if both sides fill, because the winning
    side pays out $1.00.

    Args:
        yes_bid_dollars: Price offered for YES contracts (dollars).
        no_bid_dollars:  Price offered for NO contracts (dollars).

    Returns:
        Locked profit in dollars per contract pair.  Positive means the
        spread is profitable; zero or negative means it should not be placed.

    Examples::

        >>> synthetic_spread_profit(0.45, 0.50)
        0.05
        >>> synthetic_spread_profit(0.51, 0.50)   # no edge — negative
        -0.01
    """
    return 1.0 - yes_bid_dollars - no_bid_dollars


def mid_price_dollars(bid_dollars: float, ask_dollars: float) -> float:
    """
    Compute the mid-price between bid and ask (both in dollars).

    Args:
        bid_dollars: Best bid price in dollars.
        ask_dollars: Best ask price in dollars.

    Returns:
        Mid-price in dollars.
    """
    return (bid_dollars + ask_dollars) / 2.0
