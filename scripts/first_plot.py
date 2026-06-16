"""
first_plot.py — Phase 1: your first visible win.

Pull a year of SPY closing prices and draw them. That's it. The goal here is
not the chart itself — it's proving the whole toolchain works end to end:
yfinance can reach the network, your DataFeed parses it, and matplotlib can
draw it. Once this picture appears, you're officially building Robofund.

Run it from the project root with:
    python scripts/first_plot.py
"""

import sys
from pathlib import Path

# Make the project root importable so `from robofund...` works no matter what
# directory you launch this script from. (Python only auto-adds the script's
# own folder to the import path — not its parent — so we add the parent here.)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

from robofund.data import load_bars

TICKER = "AAPL"


def main() -> None:
    bars = load_bars(TICKER, start="2023-01-01", end="2024-01-01")

    # Pull the two parallel lists we want to plot: x = dates, y = closing prices.
    days = [bar.day for bar in bars]
    closes = [bar.close for bar in bars]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(days, closes, linewidth=1.6, color="#2563eb")

    ax.set_title(f"{TICKER} — daily close, 2023", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()  # angle the date labels so they don't overlap
    fig.tight_layout()

    out = "spy_2023.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")

    # Also pop up an interactive window. Close it to end the program.
    plt.show()


if __name__ == "__main__":
    main()
