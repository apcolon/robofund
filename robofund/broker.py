"""
broker.py — the Broker (a.k.a. the portfolio + execution layer).

The strategy decides *what fraction* it wants invested. The broker turns that
wish into reality: it knows how many shares you currently hold, how much cash
you have, charges you a commission on every trade, and records what happened.

Keeping this separate from the strategy is the key separation of concerns.
A strategy should never do arithmetic with share counts or cash — that's the
broker's job. Swap the broker (add slippage, support short selling, charge a
different fee) and every strategy keeps working untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Trade:
    """A single fill. Positive `shares` = a buy, negative = a sell."""

    day: date
    shares: float
    price: float
    cost: float  # commission paid on this trade, in dollars


class Broker:
    """Holds cash + shares of one asset and executes rebalancing orders."""

    def __init__(
        self,
        starting_cash: float,
        commission_bps: float = 0.0,
        allow_fractional: bool = True,
    ) -> None:
        """
        Args:
            starting_cash:   the stake this robot begins with.
            commission_bps:  fee per trade in basis points (1 bp = 0.01%). A
                             realistic retail cost is ~1-5 bps; set it to 0 for
                             an idealized run. Watch what it does to a strategy
                             that trades a lot — that lesson is the whole point.
            allow_fractional: if True you can buy 3.27 shares; if False, whole
                             shares only (more realistic for most brokers).
        """
        self.cash = starting_cash
        self.shares = 0.0
        self.commission_bps = commission_bps
        self.allow_fractional = allow_fractional
        self.trades: list[Trade] = []

    def equity(self, price: float) -> float:
        """Total account value if the asset is worth `price` right now:
        cash on hand plus the market value of the shares held."""
        return self.cash + self.shares * price

    def rebalance_to(self, target_weight: float, price: float, day: date) -> None:
        """Buy or sell just enough to make the asset `target_weight` of equity.

        Example: equity is $10,000, target_weight is 1.0, price is $100. Target
        is $10,000 of stock = 100 shares. If you hold 0, the broker buys 100. If
        the strategy later asks for 0.0, it sells all 100 back to cash.
        """
        target_weight = max(0.0, min(1.0, target_weight))  # clamp for safety

        equity = self.equity(price)
        target_shares = (target_weight * equity) / price
        if not self.allow_fractional:
            target_shares = float(int(target_shares))  # round down to whole shares

        delta = target_shares - self.shares  # +buy / -sell

        # Skip trades too small to matter — avoids churning on rounding dust and
        # racking up commissions for nothing.
        if abs(delta) * price < 0.01:
            return

        cost = abs(delta) * price * (self.commission_bps / 10_000)

        # A buy (delta > 0) spends cash; a sell (delta < 0) returns it. The
        # commission is always a cost, so it always subtracts.
        self.cash -= delta * price + cost
        self.shares += delta
        self.trades.append(Trade(day=day, shares=delta, price=price, cost=cost))
