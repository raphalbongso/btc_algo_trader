"""Ensemble strategy combining multiple sub-strategies via voting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.base import StrategyBase


class EnsembleStrategy(StrategyBase):
    """Combine multiple strategies using voting.

    Parameters
    ----------
    strategies : list[StrategyBase]
        Sub-strategies to combine.
    mode : str
        "majority" — sign of sum of signals.
        "unanimous" — only trade when all agree.
        "weighted" — weighted sum of signals.
    weights : list[float] | None
        Weights for "weighted" mode. Must match len(strategies).
    """

    def __init__(
        self,
        strategies: list[StrategyBase],
        mode: str = "majority",
        weights: list[float] | None = None,
    ):
        if not strategies:
            raise ValueError("Must provide at least one strategy")
        self.strategies = strategies
        self.mode = mode
        if weights is not None and len(weights) != len(strategies):
            raise ValueError("weights must match number of strategies")
        self.weights = weights or [1.0] * len(strategies)

    @property
    def required_history(self) -> int:
        return max(s.required_history for s in self.strategies)

    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        signals = pd.DataFrame(
            {f"s{i}": s.generate_signal(data) for i, s in enumerate(self.strategies)},
            index=data.index,
        )

        if self.mode == "unanimous":
            # Only trade when all strategies agree on direction
            unanimous_long = (signals == 1).all(axis=1)
            unanimous_short = (signals == -1).all(axis=1)
            combined = pd.Series(0, index=data.index, dtype=int)
            combined[unanimous_long] = 1
            combined[unanimous_short] = -1
            return combined

        if self.mode == "weighted":
            weighted_sum = sum(
                signals[f"s{i}"] * w for i, w in enumerate(self.weights)
            )
            return np.sign(weighted_sum).astype(int)

        # Default: majority vote
        vote_sum = signals.sum(axis=1)
        return np.sign(vote_sum).astype(int)
