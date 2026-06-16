"""
daily.py — Phase 6: the daily heartbeat.

Run this each morning. For every ticker in your watchlist it pulls fresh data
through today, runs the whole fund of robots, tells you what each one wants to do
next, scores them, saves a scoreboard chart, and appends a dated snapshot to a
running ledger.

This is the same engine from phase 2 — the only change is that the data now ends
*today* instead of in the past. A backtest that ends today IS a live paper-trade.

Run from the project root:
    python scripts/daily.py

To track different stocks, just edit the WATCHLIST below. Each ticker is run
independently — the robots compete on each stock separately (this is NOT one
shared basket of money; that would be a bigger change to the engine).
"""

import sys
from dataclasses import replace
from datetime import date
from math import ceil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from robofund.data import load_bars
from robofund.engine import run_backtest
from robofund.features import build_dataset, build_feature_map
from robofund.ml_strategy import MLStrategy
from robofund.report import (
    clip_to_live,
    orders_block,
    score,
    scoreboard_table,
    update_ledger,
)
from robofund.strategy import BuyAndHold, MovingAverageCrossover

# 👇 Your watchlist. Add or remove symbols freely — the fund runs on each one.
WATCHLIST = ["AAPL", "MSFT", "NVDA", "SPY"]

DATA_START = "2018-01-01"          # far enough back for MA-200 + ML training
ML_TRAIN_END = date(2022, 1, 1)    # ML trains only on data before this -> live period is out-of-sample
LIVE_SINCE = ML_TRAIN_END          # score everyone only on the out-of-sample window, starting equal
STARTING_CASH = 10_000.0
REPORTS = Path("reports")


def build_fund(bars):
    """Assemble the competing robots for one ticker. The ML one is trained here,
    on history strictly before ML_TRAIN_END, so every signal it issues for the
    live period is on data it never trained on. Each ticker gets its own model."""
    dates, X, y = build_dataset(bars)
    train_X = [x for d, x in zip(dates, X) if d < ML_TRAIN_END]
    train_y = [t for d, t in zip(dates, y) if d < ML_TRAIN_END]
    model = RandomForestClassifier(
        n_estimators=200, max_depth=4, min_samples_leaf=20, random_state=42
    ).fit(train_X, train_y)

    return [
        BuyAndHold(),
        MovingAverageCrossover(fast=50, slow=200),
        MLStrategy(model, build_feature_map(bars), name="ML (RandomForest)"),
    ]


def run_ticker(ticker: str):
    """Run the whole fund on one ticker. Returns (as_of_date, scoreboard rows)."""
    bars = load_bars(ticker, start=DATA_START, end=None)  # end=None -> through today
    robots = build_fund(bars)

    rows = []
    for robot in robots:
        # Run over full history (for warmup/training), then score only the live,
        # out-of-sample window with every robot restarted at the same $10k.
        full = run_backtest(bars, robot, starting_cash=STARTING_CASH)
        result = clip_to_live(full, LIVE_SINCE, STARTING_CASH)
        # The strategy's decision from the latest close = its order for the next open.
        todays_signal = robot.on_bar(bars)
        rows.append(score(robot.name, todays_signal, result))

    return bars[-1].day, rows


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    print(f"\n=== Robofund daily — watchlist: {', '.join(WATCHLIST)} ===")
    print(f"Live scoreboard since {LIVE_SINCE} (out-of-sample for the ML robot), "
          f"each robot started at ${STARTING_CASH:,.0f}.\n")

    results: dict[str, tuple] = {}
    ledger_rows = []  # all (ticker, robot) rows flattened, for one combined ledger row
    note_sections = []

    for ticker in WATCHLIST:
        as_of, rows = run_ticker(ticker)
        results[ticker] = (as_of, rows)

        print(f"--- {ticker}  (through {as_of}) ---")
        print(orders_block(rows, as_of))
        print()
        print(scoreboard_table(rows))
        print()

        # Prefix each row's name with the ticker so the ledger columns stay unique.
        ledger_rows += [replace(r, name=f"{ticker} {r.name}") for r in rows]
        note_sections.append(
            f"## {ticker}  (through {as_of})\n\n"
            f"```\n{orders_block(rows, as_of)}\n```\n\n"
            f"```\n{scoreboard_table(rows)}\n```\n"
        )

    # --- One combined ledger row per day, covering every ticker + robot. ---
    as_of = max(a for a, _ in results.values())
    days_logged = update_ledger(REPORTS / "ledger.csv", as_of, ledger_rows)
    print(f"Ledger now holds {days_logged} day(s) of live history -> {REPORTS / 'ledger.csv'}")

    # --- Scoreboard chart: one small panel per ticker. ---
    ncols = 2 if len(WATCHLIST) > 1 else 1
    nrows = ceil(len(WATCHLIST) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows), squeeze=False)
    for i, ticker in enumerate(WATCHLIST):
        ax = axes[i // ncols][i % ncols]
        _, rows = results[ticker]
        for r in rows:
            ax.plot(r.result.days, r.result.equity, linewidth=1.4, label=r.name)
        ax.axhline(STARTING_CASH, color="#94a3b8", linestyle="--", linewidth=1)
        ax.set_title(ticker, fontsize=12, fontweight="bold")
        ax.set_ylabel("Account value ($)")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper left", fontsize=8)
    # Hide any empty panels in the grid.
    for j in range(len(WATCHLIST), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(f"Robofund scoreboard — through {as_of}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(REPORTS / "scoreboard.png", dpi=140)
    print(f"Saved {REPORTS / 'scoreboard.png'}")

    # --- A human-readable daily note covering the whole watchlist. ---
    note = REPORTS / "today.md"
    note.write_text(
        f"# Robofund — {as_of}\n\n"
        f"**Watchlist:** {', '.join(WATCHLIST)}  \n"
        f"**Days of live history logged:** {days_logged}\n\n"
        + "\n".join(note_sections)
    )
    print(f"Wrote {note}")

    plt.show()


if __name__ == "__main__":
    main()
