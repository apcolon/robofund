"""
compare.py — Phase 3: the first head-to-head.

Runs two robots over the same history, prints a scoreboard comparing them on
return AND risk, and plots both equity curves on one chart so you can see who
took the smoother path.

Run from the project root:
    python scripts/compare.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

from robofund.data import load_bars
from robofund.engine import run_backtest
from robofund.metrics import compute_metrics
from robofund.strategy import BuyAndHold, MovingAverageCrossover

TICKER = "AAPL"
START = "2018-01-01"   # a 6-year span covering booms, the 2020 crash, and 2022's slump
END = "2024-01-01"
STARTING_CASH = 10_000.0

# The robots competing this run. Add another line here and it joins the board —
# that's the Strategy pattern paying off.
ROBOTS = [
    BuyAndHold(),
    MovingAverageCrossover(fast=50, slow=200),
]


def main() -> None:
    bars = load_bars(TICKER, start=START, end=END)
    print(f"\nLoaded {len(bars)} days of {TICKER}, {bars[0].day} -> {bars[-1].day}\n")

    results = [run_backtest(bars, robot, starting_cash=STARTING_CASH) for robot in ROBOTS]

    # --- Scoreboard ---
    header = f"{'Robot':<24}{'Final $':>12}{'Return':>10}{'CAGR':>9}{'Sharpe':>9}{'Max DD':>10}{'Trades':>8}"
    print(header)
    print("-" * len(header))
    for result in results:
        m = compute_metrics(result.equity)
        print(
            f"{result.name:<24}"
            f"{result.final_equity:>12,.0f}"
            f"{m.total_return:>10.1%}"
            f"{m.cagr:>9.1%}"
            f"{m.sharpe:>9.2f}"
            f"{m.max_drawdown:>10.1%}"
            f"{len(result.trades):>8}"
        )
    print()

    # --- Both equity curves on one chart ---
    fig, ax = plt.subplots(figsize=(11, 6))
    for result in results:
        ax.plot(result.days, result.equity, linewidth=1.7, label=result.name)
    ax.axhline(STARTING_CASH, color="#94a3b8", linestyle="--", linewidth=1)

    ax.set_title(f"Robofund scoreboard — {TICKER}, {START[:4]}–{END[:4]}",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Account value ($)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    out = "scoreboard.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")

    plt.show()


if __name__ == "__main__":
    main()
