"""
run_backtest.py — Phase 2: drive the engine.

Loads a ticker, runs the Buy & Hold robot through the simulator, prints a
summary, and plots the equity curve — your $10,000 stake's value over time.

Run from the project root:
    python scripts/run_backtest.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

from robofund.data import load_bars
from robofund.engine import run_backtest
from robofund.strategy import BuyAndHold

TICKER = "AAPL"
STARTING_CASH = 10_000.0


def main() -> None:
    bars = load_bars(TICKER, start="2023-01-01", end="2024-01-01")

    result = run_backtest(bars, BuyAndHold(), starting_cash=STARTING_CASH)

    # --- Summary to the console ---
    print(f"\n  Robot:          {result.name}")
    print(f"  Ticker:         {TICKER}")
    print(f"  Days simulated: {len(result.days)}")
    print(f"  Trades made:    {len(result.trades)}")
    print(f"  Starting cash:  ${result.starting_cash:,.2f}")
    print(f"  Final equity:   ${result.final_equity:,.2f}")
    print(f"  Total return:   {result.total_return:+.2%}\n")

    # --- Equity curve ---
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(result.days, result.equity, linewidth=1.8, color="#16a34a")
    ax.axhline(result.starting_cash, color="#94a3b8", linestyle="--", linewidth=1,
               label="starting stake")

    ax.set_title(f"Robofund — {result.name} on {TICKER}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Account value ($)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    out = "equity_curve.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")

    plt.show()


if __name__ == "__main__":
    main()
