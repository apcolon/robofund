"""
features.py — turning raw prices into something a model can learn from.

A machine-learning model can't do anything useful with a raw price like "$187.23"
— that number means nothing on its own. What carries information is *context*:
how fast has it been rising, how jumpy has it been, is it stretched far above its
recent average? Each of those is a **feature** — a single number, computed from
the price history, that describes the current situation.

The cardinal rule, same as the engine's: **every feature looks only backward.**
A feature for a given day uses only that day and the days before it — never the
future. If a feature peeked ahead, the model would look brilliant in testing and
fail completely in real life. So each function below is computed from
`closes[: i + 1]` — the history up to and including day i, and nothing after.

We then attach a **label** to each day: did the price go UP the next day (1) or
not (0)? That's what the model tries to predict. The label is the *only* place
the future is allowed in — and only during training, never during prediction.
"""

from __future__ import annotations

import statistics
from datetime import date

from robofund.data import Bar

# Names line up, in order, with the numbers returned by `_features_at`. Handy for
# printing which inputs the model found important.
FEATURE_NAMES = [
    "ret_1d",        # yesterday's 1-day return
    "ret_5d",        # return over the last 5 days
    "ret_10d",       # return over the last 10 days
    "vol_10d",       # volatility (std of daily returns) over the last 10 days
    "sma20_ratio",   # how far price is above/below its 20-day average
    "rsi_14",        # Relative Strength Index — a classic 0-100 momentum gauge
    "mom_20d",       # 20-day momentum (return over ~a trading month)
    "vol_ratio_20d", # today's volume vs its 20-day average (unusual activity?)
]

# Need at least this many prior bars before all features are well-defined.
WARMUP = 30


def _rsi(closes: list[float], n: int = 14) -> float:
    """Relative Strength Index over the last `n` days, scaled 0-100.

    It compares the size of recent up-moves to recent down-moves: near 100 means
    the stock has been almost all gains lately (possibly "overbought"), near 0
    means almost all losses ("oversold"). A staple of technical analysis.
    """
    gains = 0.0
    losses = 0.0
    for k in range(len(closes) - n, len(closes)):
        change = closes[k] - closes[k - 1]
        if change >= 0:
            gains += change
        else:
            losses -= change  # make it positive
    avg_gain = gains / n
    avg_loss = losses / n
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def _features_at(closes: list[float], volumes: list[float]) -> list[float]:
    """Compute the full feature vector for the latest day in the given history."""
    c = closes
    daily_returns = [c[k] / c[k - 1] - 1 for k in range(len(c) - 10, len(c))]
    sma20 = sum(c[-20:]) / 20
    vol_sma20 = sum(volumes[-20:]) / 20

    return [
        c[-1] / c[-2] - 1,                              # ret_1d
        c[-1] / c[-6] - 1,                              # ret_5d
        c[-1] / c[-11] - 1,                             # ret_10d
        statistics.pstdev(daily_returns),               # vol_10d
        c[-1] / sma20 - 1,                              # sma20_ratio
        _rsi(c, 14),                                    # rsi_14
        c[-1] / c[-21] - 1,                             # mom_20d
        (volumes[-1] / vol_sma20 - 1) if vol_sma20 else 0.0,  # vol_ratio_20d
    ]


def build_dataset(bars: list[Bar]) -> tuple[list[date], list[list[float]], list[int]]:
    """Build (dates, X, y) for training.

    X[i] is the feature vector on dates[i]; y[i] is 1 if the *next* day closed
    higher, else 0. The last bar is dropped because it has no "next day" to label.
    """
    closes = [b.close for b in bars]
    volumes = [float(b.volume) for b in bars]

    dates: list[date] = []
    X: list[list[float]] = []
    y: list[int] = []
    for i in range(WARMUP, len(bars) - 1):  # -1: need a next-day price for the label
        dates.append(bars[i].day)
        X.append(_features_at(closes[: i + 1], volumes[: i + 1]))
        y.append(1 if closes[i + 1] > closes[i] else 0)
    return dates, X, y


def build_feature_map(bars: list[Bar]) -> dict[date, list[float]]:
    """Map every usable date -> its feature vector (no labels).

    The live strategy uses this to look up "what did the world look like today?"
    and ask the model for a prediction. Because features are computed from the
    full series here, a date near the start of a test period still gets the
    proper lookback it needs — and still only sees its own past.
    """
    closes = [b.close for b in bars]
    volumes = [float(b.volume) for b in bars]
    return {
        bars[i].day: _features_at(closes[: i + 1], volumes[: i + 1])
        for i in range(WARMUP, len(bars))
    }
