"""
strategy.py — the Strategy interface.

This is the plug. Every robot in the fund — buy-and-hold, moving-average
crossover, an ML model later — is the same shape: given the history it's
allowed to see, return how much of the portfolio it wants in the asset right
now. Because they all share this interface, the engine can run any of them
without knowing or caring which one it is. That's the Strategy design pattern,
and it's what makes adding robot #7 a 10-line file instead of a rewrite.

The single most important rule lives here:

    on_bar() receives ONLY the bars up to and including today. It can never see
    tomorrow.

The engine enforces this by slicing history before each call. If a strategy
can't peek at the future, it can't accidentally cheat — and "accidentally
cheating" (look-ahead bias) is the bug that makes a useless strategy look
brilliant in a backtest and then lose money for real.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from robofund.data import Bar


class Strategy(ABC):
    """Base class every strategy inherits from.

    A strategy answers one question each day: "what fraction of my money do I
    want in this asset?" — a target *weight* between 0.0 and 1.0.

        0.0 = hold all cash, own none of the asset
        1.0 = fully invested, every dollar in the asset
        0.5 = half and half

    Returning a weight (instead of raw "BUY"/"SELL") is a deliberate choice: it
    decouples the *decision* (how much exposure I want) from the *execution*
    (how many shares that means today), which the broker handles. The strategy
    never has to think about share counts or cash — just conviction.
    """

    #: Human-readable label used on the scoreboard. Override in subclasses.
    name: str = "unnamed"

    @abstractmethod
    def on_bar(self, history: list[Bar]) -> float:
        """Decide today's target weight from the bars seen so far.

        Args:
            history: every Bar up to and including today, oldest-first.
                     `history[-1]` is today. There is no `history[len]` —
                     tomorrow doesn't exist yet.

        Returns:
            Target weight in [0.0, 1.0]. The engine clamps it for safety, but a
            well-behaved strategy stays in range.
        """
        ...


class BuyAndHold(Strategy):
    """The baseline every other robot must beat.

    Buy on day one, never sell. It sounds trivial, but over long horizons it
    quietly crushes most "clever" strategies — which is exactly why it's the
    benchmark. If your fancy robot can't beat just holding, it isn't earning
    its complexity.
    """

    name = "Buy & Hold"

    def on_bar(self, history: list[Bar]) -> float:
        # Always want to be 100% invested. The engine buys on the first bar and
        # then there's nothing left to do — the weight never changes.
        return 1.0


class MovingAverageCrossover(Strategy):
    """Robot #2 — the classic trend-follower.

    It tracks two moving averages of the price: a *fast* one (recent, jumpy) and
    a *slow* one (long-term, steady). The idea:

        fast average ABOVE slow average  ->  price is trending up, be invested (1.0)
        fast average BELOW slow average  ->  trend has rolled over, go to cash (0.0)

    A "moving average" is just the average closing price over the last N days,
    recomputed each day — it smooths out the daily noise so you can see the
    underlying direction. When the fast line crosses above the slow line, that's
    the "golden cross" traders talk about; crossing below is the "death cross."

    The bet this robot makes: by bailing to cash when the trend breaks, it dodges
    the worst of the crashes — giving up some upside in exchange for a smoother
    ride and smaller drawdowns. Whether that trade-off actually pays is exactly
    what the scoreboard will tell us.
    """

    def __init__(self, fast: int = 50, slow: int = 200) -> None:
        if fast >= slow:
            raise ValueError("fast window must be shorter than slow window")
        self.fast = fast
        self.slow = slow
        self.name = f"MA Crossover ({fast}/{slow})"

    def on_bar(self, history: list[Bar]) -> float:
        # Need at least `slow` days of history before the slow average exists.
        # Until then we can't form an opinion, so we sit safely in cash.
        if len(history) < self.slow:
            return 0.0

        closes = [bar.close for bar in history]
        fast_avg = sum(closes[-self.fast:]) / self.fast
        slow_avg = sum(closes[-self.slow:]) / self.slow

        return 1.0 if fast_avg > slow_avg else 0.0
