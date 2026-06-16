"""
report.py — assemble the daily scoreboard and keep a running ledger.

Phase 6 turns the backtester into a daily habit. Nothing about the engine
changes — we just run it with data ending *today* instead of in the past, then:

  * read off each robot's order for tomorrow (its latest target weight),
  * score every robot on the run so far, and
  * append a dated snapshot to a CSV ledger so a genuine forward record grows by
    one row each day you actually run this.

The ledger is what makes it feel alive: over weeks it becomes a real, honest
track record of robots trading on days none of them had ever seen when the code
was written.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from robofund.engine import BacktestResult
from robofund.metrics import Metrics, compute_metrics


def clip_to_live(result: BacktestResult, since: date, starting_cash: float) -> BacktestResult:
    """Trim an equity curve to the live window and restart it at `starting_cash`.

    This is what keeps the daily scoreboard honest. A robot may need years of
    earlier data for warmup (the MA-200) or training (the ML model), so we run it
    over full history — but we only *score* the period from `since` onward, and we
    renormalize so every robot starts the live window with the same $10k. That way
    the numbers reflect only out-of-sample days no robot was fit to, and the
    in-sample stretch can't flatter anyone.
    """
    idx = next((i for i, d in enumerate(result.days) if d >= since), None)
    if idx is None:
        return result  # nothing in the window; leave as-is
    factor = starting_cash / result.equity[idx]
    return BacktestResult(
        name=result.name,
        days=result.days[idx:],
        equity=[e * factor for e in result.equity[idx:]],
        trades=[t for t in result.trades if t.day >= since],
        starting_cash=starting_cash,
    )


def signal_label(weight: float) -> str:
    """Turn a target weight into a plain order word."""
    if weight >= 0.99:
        return "IN"
    if weight <= 0.01:
        return "CASH"
    return f"{weight:.0%}"


@dataclass
class ScoreRow:
    """One robot's line on the scoreboard."""

    name: str
    signal: float            # target weight decided on the latest bar (tomorrow's order)
    result: BacktestResult
    metrics: Metrics


def score(name: str, signal: float, result: BacktestResult) -> ScoreRow:
    return ScoreRow(name=name, signal=signal, result=result,
                    metrics=compute_metrics(result.equity))


def scoreboard_table(rows: list[ScoreRow]) -> str:
    """Render the scoreboard as a fixed-width text table."""
    header = (f"{'Robot':<22}{'Order':>7}{'Equity':>12}"
              f"{'Return':>10}{'Sharpe':>9}{'MaxDD':>9}{'Trades':>8}")
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r.name:<22}{signal_label(r.signal):>7}{r.result.final_equity:>12,.0f}"
            f"{r.metrics.total_return:>10.1%}{r.metrics.sharpe:>9.2f}"
            f"{r.metrics.max_drawdown:>9.1%}{len(r.result.trades):>8}"
        )
    return "\n".join(lines)


def orders_block(rows: list[ScoreRow], as_of: date) -> str:
    """The actionable bit: what each robot wants to do at the next open."""
    lines = [f"Orders for the next trading day (decided from {as_of}'s close):"]
    for r in rows:
        order = signal_label(r.signal)
        verb = "stay/go INTO the stock" if order == "IN" else (
            "move/stay in CASH" if order == "CASH" else f"hold {order} in the stock")
        lines.append(f"  • {r.name:<22} -> {order:<5}  ({verb})")
    return "\n".join(lines)


def update_ledger(path: Path, as_of: date, rows: list[ScoreRow]) -> int:
    """Append today's snapshot to the CSV ledger (replacing today's row if rerun).

    Returns the total number of dated rows in the ledger after the update — i.e.
    how many days of live history have accumulated.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["date"]
    for r in rows:
        fieldnames += [f"{r.name} equity", f"{r.name} signal"]

    # Keep any prior rows except one for today (so re-running today just updates it).
    existing: list[dict] = []
    if path.exists():
        with path.open(newline="") as f:
            existing = [row for row in csv.DictReader(f)
                        if row.get("date") != as_of.isoformat()]

    record = {"date": as_of.isoformat()}
    for r in rows:
        record[f"{r.name} equity"] = f"{r.result.final_equity:.2f}"
        record[f"{r.name} signal"] = signal_label(r.signal)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in existing:
            writer.writerow(row)
        writer.writerow(record)

    return len(existing) + 1
