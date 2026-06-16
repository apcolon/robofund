"""
daily.py — Phase 6: the daily heartbeat.

Run this each morning. It pulls fresh market data through today, runs the whole
fund of robots, tells you what each one wants to do next, scores them, saves a
scoreboard chart, and appends a dated snapshot to a running ledger.

This is the same engine from phase 2 — the only change is that the data now ends
*today* instead of in the past. A backtest that ends today IS a live paper-trade.

Run from the project root:
    python scripts/daily.py
"""

import sys
from datetime import date
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

TICKER = "AAPL"
DATA_START = "2018-01-01"          # far enough back for MA-200 + ML training
ML_TRAIN_END = date(2022, 1, 1)    # ML trains only on data before this -> live period is out-of-sample
LIVE_SINCE = ML_TRAIN_END          # score everyone only on the out-of-sample window, starting equal
STARTING_CASH = 10_000.0
REPORTS = Path("reports")


def build_fund(bars):
    """Assemble the competing robots. The ML one is trained here, on history
    strictly before ML_TRAIN_END, so every signal it issues for the live period
    is on data it never trained on."""
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


def main() -> None:
    # end=None -> through the latest available trading day (today, live).
    bars = load_bars(TICKER, start=DATA_START, end=None)
    as_of = bars[-1].day
    print(f"\n=== Robofund daily — {TICKER} — data through {as_of} ===\n")

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

    print(f"Live scoreboard since {LIVE_SINCE} (out-of-sample for the ML robot), "
          f"each robot started at ${STARTING_CASH:,.0f}.\n")
    print(orders_block(rows, as_of))
    print()
    print(scoreboard_table(rows))
    print()

    # --- Persist a dated snapshot so a live record accumulates day by day. ---
    days_logged = update_ledger(REPORTS / "ledger.csv", as_of, rows)
    print(f"Ledger now holds {days_logged} day(s) of live history -> {REPORTS / 'ledger.csv'}")

    # --- Scoreboard chart. ---
    REPORTS.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    for r in rows:
        ax.plot(r.result.days, r.result.equity, linewidth=1.6, label=r.name)
    ax.axhline(STARTING_CASH, color="#94a3b8", linestyle="--", linewidth=1)
    ax.set_title(f"Robofund scoreboard — {TICKER}, through {as_of}",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Account value ($)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(REPORTS / "scoreboard.png", dpi=140)
    print(f"Saved {REPORTS / 'scoreboard.png'}")

    # --- A human-readable daily note. ---
    note = REPORTS / "today.md"
    note.write_text(
        f"# Robofund — {as_of}\n\n"
        f"**Ticker:** {TICKER}  \n**Days of live history logged:** {days_logged}\n\n"
        f"## Orders for the next trading day\n\n```\n{orders_block(rows, as_of)}\n```\n\n"
        f"## Scoreboard\n\n```\n{scoreboard_table(rows)}\n```\n"
    )
    print(f"Wrote {note}")

    plt.show()


if __name__ == "__main__":
    main()
