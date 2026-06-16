# Robofund

A lab where robot trading strategies compete on live market data.

Each strategy starts with a hypothetical stake and trades on its own rules.
Robofund replays historical prices to see how each one *would* have done
(backtesting), then keeps running them forward on new, unseen days
(paper trading) and tracks the results on a scoreboard. Same engine, same
strategy code — the only thing that changes is whether the data is from the
past or from this morning.

## Status

**Phase 4 — Polish the visuals.** Underwater plot, candlesticks, tearsheet. ✅

Roadmap:
1. **Data + first plot** — prove the toolchain works. ✅
2. **The engine** — strategy interface + a simulator that replays bars. ✅
3. **First real strategy** — moving-average crossover + metrics + equity curve. ✅
4. **Polish the visuals** — drawdowns, candlesticks with trade markers, tearsheet. ✅ *(you are here)*
5. **ML strategy** — engineer features, train a classifier, backtest it, get humbled.
6. **Live scoreboard** — run every strategy on fresh daily data automatically.

## Setup

```bash
# from the project root
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python scripts/first_plot.py     # saves spy_2023.png and opens a window
```

## Layout

```
robofund/          # the package — importable engine code
  data.py          # DataFeed: the only place that knows about yfinance
scripts/           # runnable entry points, one per phase
  first_plot.py    # phase 1
```
