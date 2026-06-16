"""
tearsheet.py — Phase 4: the one-page summary card.

Runs one robot (plus a Buy & Hold benchmark line) and assembles a single figure
with four panels: equity curves, the underwater drawdown plot, a recent
candlestick chart with the robot's trades marked, and a stats box. This is the
kind of "tearsheet" real funds produce for a strategy — and it screenshots well
for a README.

Run from the project root:
    python scripts/tearsheet.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

from robofund.data import load_bars
from robofund.engine import run_backtest
from robofund.metrics import compute_metrics
from robofund.plotting import (
    metrics_textbox,
    plot_candles_with_trades,
    plot_equity_curves,
    plot_underwater,
)
from robofund.strategy import BuyAndHold, MovingAverageCrossover

TICKER = "AAPL"
START = "2018-01-01"
END = "2024-01-01"
STARTING_CASH = 10_000.0

# The robot this tearsheet profiles, plus the benchmark it's measured against.
ROBOT = MovingAverageCrossover(fast=50, slow=200)
BENCHMARK = BuyAndHold()


def main() -> None:
    bars = load_bars(TICKER, start=START, end=END)

    robot_result = run_backtest(bars, ROBOT, starting_cash=STARTING_CASH)
    bench_result = run_backtest(bars, BENCHMARK, starting_cash=STARTING_CASH)
    m = compute_metrics(robot_result.equity)

    # --- Assemble the figure: top row splits equity curve + stats box, then two
    #     full-width rows below for the underwater plot and the candlesticks. ---
    fig = plt.figure(figsize=(13, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 0.8, 1.1], width_ratios=[3, 1],
                          hspace=0.45, wspace=0.2)

    ax_equity = fig.add_subplot(gs[0, 0])
    ax_stats = fig.add_subplot(gs[0, 1])
    ax_under = fig.add_subplot(gs[1, :])    # full width
    ax_candles = fig.add_subplot(gs[2, :])  # full width

    plot_equity_curves(ax_equity, [robot_result, bench_result], STARTING_CASH)
    ax_equity.set_title(f"{ROBOT.name} vs {BENCHMARK.name}", fontsize=12, loc="left")

    metrics_textbox(ax_stats, ROBOT.name, m, robot_result.final_equity,
                    len(robot_result.trades))

    plot_underwater(ax_under, robot_result)
    plot_candles_with_trades(ax_candles, bars, robot_result.trades, max_candles=120)

    fig.suptitle(f"Robofund tearsheet — {ROBOT.name} on {TICKER}, {START[:4]}–{END[:4]}",
                 fontsize=15, fontweight="bold")

    out = "tearsheet.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved {out}")

    plt.show()


if __name__ == "__main__":
    main()
