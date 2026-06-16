"""
engine.py — the event-driven simulator. This is the core of Robofund.

It replays history one bar at a time, the way reality delivers it: you never
get tomorrow's bar before you've acted on today's. That single discipline is
what makes this engine special — because it processes bars in real-world order,
the *exact same loop* can run on historical data (backtest) or on this morning's
bar (live paper trading). You write it once.

The daily cycle, in order — and the order is everything:

    1. EXECUTE  yesterday's decision, filling at today's OPEN price.
    2. MARK     the account to market at today's CLOSE, and record equity.
    3. DECIDE   tomorrow's target, using data only through today's close.

Why decide today but execute tomorrow? Because in real life you can't see a
day's closing price and also trade at that same close — by the time you know
the close, the day is over. So a decision made from today's close can only be
acted on at the next day's open. Encoding that one-day gap is how the engine
makes look-ahead bias structurally impossible instead of a mistake you have to
remember not to make.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from robofund.broker import Broker, Trade
from robofund.data import Bar
from robofund.strategy import Strategy


@dataclass
class BacktestResult:
    """Everything that happened in one run, ready to inspect or plot."""

    name: str
    days: list[date]
    equity: list[float]  # account value at each day's close, parallel to `days`
    trades: list[Trade]
    starting_cash: float

    @property
    def final_equity(self) -> float:
        return self.equity[-1]

    @property
    def total_return(self) -> float:
        """Overall gain as a fraction: 0.25 means +25%."""
        return self.final_equity / self.starting_cash - 1.0


def run_backtest(
    bars: list[Bar],
    strategy: Strategy,
    starting_cash: float = 10_000.0,
    commission_bps: float = 0.0,
) -> BacktestResult:
    """Replay `bars` through `strategy` and report what happened."""
    broker = Broker(starting_cash, commission_bps=commission_bps)

    days: list[date] = []
    equity: list[float] = []
    pending_target: float | None = None  # what we decided to do "tomorrow"

    for i, bar in enumerate(bars):
        # 1. EXECUTE the decision made on the previous bar, at today's open.
        if pending_target is not None:
            broker.rebalance_to(pending_target, price=bar.open, day=bar.day)

        # 2. MARK to market at today's close and record the day's equity.
        days.append(bar.day)
        equity.append(broker.equity(bar.close))

        # 3. DECIDE tomorrow's target. The slice `bars[: i + 1]` is the guard
        #    rail: the strategy physically cannot receive a future bar.
        pending_target = strategy.on_bar(bars[: i + 1])

    return BacktestResult(
        name=strategy.name,
        days=days,
        equity=equity,
        trades=broker.trades,
        starting_cash=starting_cash,
    )
