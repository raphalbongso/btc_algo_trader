"""Tests for strategies.ensemble_strategy."""

import numpy as np
import pandas as pd
import pytest

from strategies.sma_strategy import SMAStrategy
from strategies.momentum_strategy import MomentumStrategy
from strategies.mean_reversion_strategy import MeanReversionStrategy
from strategies.ensemble_strategy import EnsembleStrategy


class TestEnsembleStrategy:
    @pytest.fixture
    def sub_strategies(self):
        return [
            SMAStrategy(sma_short=10, sma_long=30),
            MomentumStrategy(window=15),
        ]

    def test_majority_vote(self, btc_data, sub_strategies):
        ensemble = EnsembleStrategy(sub_strategies, mode="majority")
        signals = ensemble.generate_signal(btc_data)
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(btc_data)
        assert signals.isin([-1, 0, 1]).all()

    def test_unanimous_vote(self, btc_data, sub_strategies):
        ensemble = EnsembleStrategy(sub_strategies, mode="unanimous")
        signals = ensemble.generate_signal(btc_data)
        # Unanimous should have fewer non-zero signals than majority
        majority = EnsembleStrategy(sub_strategies, mode="majority")
        maj_signals = majority.generate_signal(btc_data)
        assert (signals != 0).sum() <= (maj_signals != 0).sum()

    def test_weighted_vote(self, btc_data, sub_strategies):
        ensemble = EnsembleStrategy(
            sub_strategies, mode="weighted", weights=[2.0, 1.0]
        )
        signals = ensemble.generate_signal(btc_data)
        assert signals.isin([-1, 0, 1]).all()

    def test_required_history(self, sub_strategies):
        ensemble = EnsembleStrategy(sub_strategies)
        assert ensemble.required_history == max(s.required_history for s in sub_strategies)

    def test_empty_strategies_raises(self):
        with pytest.raises(ValueError):
            EnsembleStrategy([], mode="majority")

    def test_wrong_weights_length_raises(self, sub_strategies):
        with pytest.raises(ValueError):
            EnsembleStrategy(sub_strategies, weights=[1.0])

    def test_three_strategies(self, btc_data):
        strategies = [
            SMAStrategy(10, 30),
            MomentumStrategy(15),
            MeanReversionStrategy(20, 1.0),
        ]
        ensemble = EnsembleStrategy(strategies, mode="majority")
        signals = ensemble.generate_signal(btc_data)
        assert len(signals) == len(btc_data)
