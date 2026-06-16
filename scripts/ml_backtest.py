"""
ml_backtest.py — Phase 5: can a model beat buy-and-hold?

The honest experiment:
  1. Engineer features from years of price history.
  2. TRAIN a classifier on the early years to predict next-day up/down.
  3. TEST it on later years it has never seen (out-of-sample) — first by raw
     prediction accuracy, then by actually trading on it in the engine.
  4. Compare to Buy & Hold over the same out-of-sample window.

Watch two numbers in particular: the model's accuracy on the *training* data vs.
on the *test* data. A big gap is overfitting — the model memorized noise in the
past that doesn't repeat. That gap is the whole lesson of this phase.

Run from the project root:
    python scripts/ml_backtest.py
"""

import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from robofund.data import load_bars
from robofund.engine import run_backtest
from robofund.features import FEATURE_NAMES, build_dataset, build_feature_map
from robofund.metrics import compute_metrics
from robofund.ml_strategy import MLStrategy
from robofund.strategy import BuyAndHold

TICKER = "AAPL"
START = "2015-01-01"
END = "2024-01-01"
# Everything before this date trains the model; everything on/after it is the
# out-of-sample test the model never saw.
TEST_START = date(2021, 1, 1)
STARTING_CASH = 10_000.0


def main() -> None:
    bars = load_bars(TICKER, start=START, end=END)
    print(f"\nLoaded {len(bars)} days of {TICKER}, {bars[0].day} -> {bars[-1].day}")

    # --- Build the dataset and split it by time (never randomly — that would
    #     leak future days into training). ---
    dates, X, y = build_dataset(bars)
    train_X = [x for d, x in zip(dates, X) if d < TEST_START]
    train_y = [t for d, t in zip(dates, y) if d < TEST_START]
    test_X = [x for d, x in zip(dates, X) if d >= TEST_START]
    test_y = [t for d, t in zip(dates, y) if d >= TEST_START]
    print(f"Training rows: {len(train_X)}  (through {TEST_START - timedelta(days=1)})")
    print(f"Test rows:     {len(test_X)}  (from {TEST_START})\n")

    # --- Train. random_state makes the run reproducible — same data, same model,
    #     same backtest, every time. ---
    model = RandomForestClassifier(
        n_estimators=200, max_depth=4, min_samples_leaf=20, random_state=42
    )
    model.fit(train_X, train_y)

    # --- The honest accuracy report. ---
    train_acc = accuracy_score(train_y, model.predict(train_X))
    test_acc = accuracy_score(test_y, model.predict(test_X))
    baseline = max(sum(test_y) / len(test_y), 1 - sum(test_y) / len(test_y))
    print("  Prediction accuracy (next-day up/down):")
    print(f"    on training data: {train_acc:.1%}")
    print(f"    on test data:     {test_acc:.1%}")
    print(f"    always-guess-majority baseline: {baseline:.1%}")
    print("    (50% is a coin flip. Beating it consistently out-of-sample is the hard part.)\n")

    print("  What the model leaned on most:")
    importances = sorted(zip(FEATURE_NAMES, model.feature_importances_),
                         key=lambda p: p[1], reverse=True)
    for fname, imp in importances:
        print(f"    {fname:<16}{imp:.1%}")
    print()

    # --- Now trade on it, out-of-sample, through the real engine. ---
    feature_map = build_feature_map(bars)
    test_bars = [b for b in bars if b.day >= TEST_START]

    ml = MLStrategy(model, feature_map, threshold=0.5, name="ML (RandomForest)")
    ml_result = run_backtest(test_bars, ml, starting_cash=STARTING_CASH)
    bh_result = run_backtest(test_bars, BuyAndHold(), starting_cash=STARTING_CASH)

    header = f"{'Robot':<22}{'Final $':>12}{'Return':>10}{'Sharpe':>9}{'Max DD':>10}{'Trades':>8}"
    print(header)
    print("-" * len(header))
    for result in (ml_result, bh_result):
        m = compute_metrics(result.equity)
        print(
            f"{result.name:<22}{result.final_equity:>12,.0f}{m.total_return:>10.1%}"
            f"{m.sharpe:>9.2f}{m.max_drawdown:>10.1%}{len(result.trades):>8}"
        )
    print()

    # --- Plot the out-of-sample equity curves. ---
    fig, ax = plt.subplots(figsize=(11, 6))
    for result in (ml_result, bh_result):
        ax.plot(result.days, result.equity, linewidth=1.7, label=result.name)
    ax.axhline(STARTING_CASH, color="#94a3b8", linestyle="--", linewidth=1)
    ax.set_title(f"Out-of-sample: ML vs Buy & Hold — {TICKER}, {TEST_START.year}+",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Account value ($)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    out = "ml_backtest.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.show()


if __name__ == "__main__":
    main()
