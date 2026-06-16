"""
plotting.py — Robofund's chart toolkit.

Phase 4 is about *seeing* a robot's behavior, not just reading numbers. Each
function here draws onto a matplotlib Axes you pass in, so they compose: the
tearsheet stacks several of them into one figure. Keeping "what to draw" (here)
separate from "go run a backtest" (the scripts) is the same separation-of-concerns
habit as the rest of the codebase.

The three views:
  * equity curves   — who ended up with more money
  * underwater plot  — how much it hurt to get there (drawdown over time)
  * candles + trades — exactly when each buy/sell happened, on the price itself
"""

from __future__ import annotations

from datetime import date

from matplotlib.axes import Axes

from robofund.broker import Trade
from robofund.data import Bar
from robofund.engine import BacktestResult
from robofund.metrics import Metrics

# A small consistent palette so every chart looks like it belongs to one app.
UP = "#16a34a"      # green — an up day / gains
DOWN = "#dc2626"    # red — a down day / losses
INK = "#0f172a"     # near-black for text
MUTED = "#94a3b8"   # grey for reference lines


def drawdown_series(equity: list[float]) -> list[float]:
    """For each day, how far below the highest-value-seen-so-far we are.

    Returns a list of fractions <= 0 (e.g. -0.30 = down 30% from the peak). This
    is the same idea as max drawdown in metrics.py, but kept as the full
    day-by-day series so we can draw the whole "underwater" shape.
    """
    peak = equity[0]
    series = []
    for value in equity:
        peak = max(peak, value)
        series.append(value / peak - 1.0)
    return series


def plot_equity_curves(ax: Axes, results: list[BacktestResult], starting_cash: float) -> None:
    """Each robot's account value over time, all on one Axes."""
    for result in results:
        ax.plot(result.days, result.equity, linewidth=1.7, label=result.name)
    ax.axhline(starting_cash, color=MUTED, linestyle="--", linewidth=1)
    ax.set_ylabel("Account value ($)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")


def plot_underwater(ax: Axes, result: BacktestResult) -> None:
    """The "underwater" plot: drawdown over time, shaded below the waterline.

    The surface (0%) is "at an all-time high." Every dip below it is a stretch
    where the robot was underwater, waiting to recover. Deep, wide pools = long,
    painful drawdowns. This is the single most honest picture of risk there is —
    it shows the suffering the equity curve glosses over.
    """
    dd = [d * 100 for d in drawdown_series(result.equity)]  # to percent
    ax.fill_between(result.days, dd, 0, color=DOWN, alpha=0.25)
    ax.plot(result.days, dd, color=DOWN, linewidth=1.0)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.3)
    worst = min(dd) if dd else 0.0
    ax.set_title(f"Underwater plot — worst drawdown {worst:.1f}%", fontsize=11, loc="left")


def plot_candles_with_trades(
    ax: Axes,
    bars: list[Bar],
    trades: list[Trade],
    max_candles: int = 120,
) -> None:
    """Candlestick price chart with the robot's buy/sell markers laid on top.

    A candlestick packs four prices into one shape: the thin "wick" spans the
    day's low-to-high range, and the fat "body" spans open-to-close. Green when
    the close finished above the open (an up day), red when it closed lower.

    We only draw the most recent `max_candles` days — candlesticks get unreadable
    over years, and the recent window is where you actually want to inspect the
    robot's trades. The x-axis uses evenly-spaced candle slots (not calendar
    dates) so weekends and holidays don't leave ugly gaps.
    """
    window = bars[-max_candles:]
    day_to_x = {bar.day: i for i, bar in enumerate(window)}

    for i, bar in enumerate(window):
        is_up = bar.close >= bar.open
        color = UP if is_up else DOWN
        # Wick: a vertical line across the day's full range.
        ax.vlines(i, bar.low, bar.high, color=color, linewidth=0.8)
        # Body: a bar from open to close. height is the open-close distance.
        ax.bar(i, abs(bar.close - bar.open), bottom=min(bar.open, bar.close),
               width=0.6, color=color)

    # Overlay trades that fall inside the visible window.
    drew_buy = drew_sell = False
    for t in trades:
        x = day_to_x.get(t.day)
        if x is None:
            continue
        if t.shares > 0:  # a buy
            ax.scatter(x, t.price, marker="^", s=130, color=UP, edgecolor=INK,
                       zorder=3, label=None if drew_buy else "buy")
            drew_buy = True
        else:             # a sell
            ax.scatter(x, t.price, marker="v", s=130, color=DOWN, edgecolor=INK,
                       zorder=3, label=None if drew_sell else "sell")
            drew_sell = True

    # Label a handful of x-ticks with real dates so the window has context.
    n = len(window)
    tick_positions = list(range(0, n, max(1, n // 6)))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([window[i].day.strftime("%b %d") for i in tick_positions],
                       rotation=0, fontsize=9)
    ax.set_ylabel("Price ($)")
    ax.set_title(f"Last {n} days — candles + trades", fontsize=11, loc="left")
    if drew_buy or drew_sell:
        ax.legend(loc="upper left", fontsize=9)


def metrics_textbox(ax: Axes, name: str, m: Metrics, final_equity: float, n_trades: int) -> None:
    """Render a robot's key numbers as a clean text panel (no axes)."""
    ax.axis("off")
    lines = [
        ("Final equity", f"${final_equity:,.0f}"),
        ("Total return", f"{m.total_return:+.1%}"),
        ("CAGR", f"{m.cagr:+.1%}"),
        ("Volatility", f"{m.volatility:.1%}"),
        ("Sharpe", f"{m.sharpe:.2f}"),
        ("Max drawdown", f"{m.max_drawdown:.1%}"),
        ("Trades", f"{n_trades}"),
    ]
    ax.text(0, 1.0, name, fontsize=13, fontweight="bold", va="top", color=INK)
    y = 0.82
    for label, value in lines:
        ax.text(0.0, y, label, fontsize=10, va="top", color=MUTED)
        ax.text(1.0, y, value, fontsize=10, va="top", ha="right", color=INK,
                fontweight="bold")
        y -= 0.12
