"""
data.py — the DataFeed layer.

This is the *only* part of Robofund that knows where prices come from.
Everything downstream (strategies, the engine, metrics) just receives clean
`Bar` objects and never touches yfinance or the network directly. That
separation is the whole point: later you can swap yfinance for a live API or a
CSV file without changing a single line of strategy code.

A "bar" is one row of market history: the open/high/low/close prices and volume
for a single day. The name comes from candlestick charts, where each day is
drawn as one bar.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class Bar:
    """One day of market data for one ticker.

    `frozen=True` makes it immutable — once a bar exists it can't be edited.
    That matters for backtesting: a strategy should never be able to reach back
    and rewrite history. Immutability makes that class of bug impossible.
    """

    day: date
    open: float
    high: float
    low: float
    close: float
    volume: int


def load_bars(ticker: str, start: str, end: str | None = None) -> list[Bar]:
    """Pull daily history for `ticker` and return it as a list of Bars.

    Args:
        ticker: e.g. "SPY" or "AAPL".
        start:  earliest date, "YYYY-MM-DD".
        end:    latest date, "YYYY-MM-DD". None means "up to today".

    Returns a list of Bars sorted oldest-first — the order reality delivers
    them, which is exactly how the engine will replay them later.
    """
    frame = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,   # adjust for splits/dividends so the series is continuous
        progress=False,
    )

    if frame.empty:
        raise ValueError(f"No data returned for {ticker!r}. Check the ticker and dates.")

    # yfinance hands back a pandas DataFrame. When you ask for a single ticker it
    # still uses a 2-level column index like ("Close", "SPY") — flatten it so we
    # can grab columns by plain name.
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)

    bars: list[Bar] = []
    for ts, row in frame.iterrows():
        bars.append(
            Bar(
                day=ts.date(),
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=int(row["Volume"]),
            )
        )
    return bars


if __name__ == "__main__":
    # Quick smoke test: run `python -m robofund.data` to confirm the feed works.
    sample = load_bars("SPY", start="2024-01-01", end="2024-02-01")
    print(f"Loaded {len(sample)} bars for SPY")
    print("first:", sample[0])
    print("last: ", sample[-1])
