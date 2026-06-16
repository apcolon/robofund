"""
ml_strategy.py — a robot whose trading rule is a trained model.

Everything about this strategy is the same shape as Buy & Hold or the MA
crossover: it implements `on_bar` and returns a target weight. The only
difference is *how* it decides — instead of a hand-written rule, it asks a
machine-learning model "is tomorrow more likely up or down?" and invests when
the answer is up.

That's the payoff of the Strategy pattern: a scikit-learn model drops into the
exact same engine, gets scored by the exact same metrics, and competes on the
exact same scoreboard as the simple robots. No special cases anywhere.
"""

from __future__ import annotations

from datetime import date

from robofund.data import Bar
from robofund.strategy import Strategy


class MLStrategy(Strategy):
    """Invests when a trained classifier predicts the next day is likely up.

    The model and the feature lookup are handed in already prepared — this class
    only does inference, one prediction per bar. It stays in cash on any day it
    has no features for (e.g. before the warmup period).
    """

    def __init__(
        self,
        model,                          # a fitted scikit-learn classifier/pipeline
        feature_map: dict[date, list[float]],
        threshold: float = 0.5,
        name: str = "ML Classifier",
    ) -> None:
        self.model = model
        self.feature_map = feature_map
        # Only go invested when the model's probability of "up" clears this bar.
        # 0.5 = "more likely than not." Raise it to trade only on high conviction.
        self.threshold = threshold
        self.name = name

    def on_bar(self, history: list[Bar]) -> float:
        features = self.feature_map.get(history[-1].day)
        if features is None:
            return 0.0  # no data to judge -> sit safely in cash

        # predict_proba returns [P(down), P(up)]; we want the probability of up.
        prob_up = self.model.predict_proba([features])[0][1]
        return 1.0 if prob_up >= self.threshold else 0.0
