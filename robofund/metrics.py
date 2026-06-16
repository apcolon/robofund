"""
metrics.py — how we score a robot.

The final dollar amount is the obvious number, but it hides *how* the robot got
there. Two robots can finish at the same $15k — one on a calm glide, the other on
a terrifying rollercoaster. These metrics expose that difference, so "winning"
can mean "best ride for the risk," not just "biggest number."

Everything here is computed from one input: the equity curve (the list of daily
account values the engine produced). No market knowledge needed — just the
shape of the money over time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Markets trade ~252 days a year. We use this to turn daily figures into the
# annual ones the finance world quotes.
TRADING_DAYS_PER_YEAR = 252


@dataclass
class Metrics:
    total_return: float    # overall gain, e.g. 0.50 = +50%
    cagr: float            # annualized growth rate (smooths the whole run to a yearly %)
    volatility: float      # annualized "jumpiness" of returns
    sharpe: float          # return earned per unit of jumpiness — higher is better
    max_drawdown: float    # worst peak-to-trough fall, e.g. -0.30 = dropped 30% from a high


def _daily_returns(equity: list[float]) -> list[float]:
    """Day-over-day percent changes: how much the account moved each day."""
    return [equity[i] / equity[i - 1] - 1 for i in range(1, len(equity))]


def _max_drawdown(equity: list[float]) -> float:
    """The worst drop from any past peak — the "how bad did it ever hurt" number.

    Walk the curve tracking the highest value seen so far (the peak). At each
    point, how far below that peak are we? The deepest such dip is the max
    drawdown. This is the single best gut-check for pain: a -55% drawdown means
    that at some point your account had lost more than half its peak value.
    """
    peak = equity[0]
    worst = 0.0
    for value in equity:
        peak = max(peak, value)
        dip = value / peak - 1.0  # 0 at a new high, negative below it
        worst = min(worst, dip)
    return worst


def compute_metrics(equity: list[float]) -> Metrics:
    """Turn an equity curve into the scoreboard numbers."""
    returns = _daily_returns(equity)
    total_return = equity[-1] / equity[0] - 1.0

    # Annualized growth rate (CAGR): the steady yearly % that would take you from
    # start to finish over the same span. Lets us compare runs of different lengths.
    years = len(returns) / TRADING_DAYS_PER_YEAR
    cagr = (equity[-1] / equity[0]) ** (1 / years) - 1.0 if years > 0 else 0.0

    # Volatility = standard deviation of daily returns, scaled up to a year.
    # Standard deviation measures how much the daily numbers scatter around their
    # average — i.e. how bumpy the ride is.
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    daily_std = math.sqrt(variance)
    volatility = daily_std * math.sqrt(TRADING_DAYS_PER_YEAR)

    # Sharpe ratio = average return divided by its volatility (annualized). It
    # answers: "for every unit of stomach-churn I endured, how much return did I
    # get?" A robot with a higher Sharpe earned its returns more efficiently —
    # this is the number that lets a smoother robot beat a flashier one.
    sharpe = (mean / daily_std) * math.sqrt(TRADING_DAYS_PER_YEAR) if daily_std > 0 else 0.0

    return Metrics(
        total_return=total_return,
        cagr=cagr,
        volatility=volatility,
        sharpe=sharpe,
        max_drawdown=_max_drawdown(equity),
    )
