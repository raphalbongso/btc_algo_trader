"""Tests for Kelly criterion simulation."""

import numpy as np
import pytest

from backtesting.performance import kelly_simulation


class TestKellySimulation:
    def test_returns_dict(self):
        results = kelly_simulation(n_trials=10, n_steps=20)
        assert isinstance(results, dict)
        assert len(results) > 0

    def test_shape(self):
        n_trials, n_steps = 10, 20
        results = kelly_simulation(n_trials=n_trials, n_steps=n_steps)
        for key, wealth in results.items():
            assert wealth.shape == (n_trials, n_steps + 1)

    def test_initial_capital(self):
        results = kelly_simulation(initial_capital=1000, n_trials=5, n_steps=10)
        for key, wealth in results.items():
            np.testing.assert_array_equal(wealth[:, 0], 1000)

    def test_reproducibility(self):
        r1 = kelly_simulation(seed=42, n_trials=5, n_steps=10)
        r2 = kelly_simulation(seed=42, n_trials=5, n_steps=10)
        for key in r1:
            np.testing.assert_array_equal(r1[key], r2[key])

    def test_fair_game_kelly_is_zero(self):
        # For p=0.5 (fair game), optimal Kelly is 0 → f*=0
        # All fractions > 0 should lose in expectation over long run
        results = kelly_simulation(p=0.5, n_trials=100, n_steps=500, seed=42)
        # With f=1.0, ruin is guaranteed
        full_kelly = results.get("1.0000")
        if full_kelly is not None:
            ruin_rate = (full_kelly[:, -1] < 1.0).mean()
            assert ruin_rate > 0.9

    def test_favorable_game(self):
        # p=0.6, f*=0.2 should grow
        results = kelly_simulation(p=0.6, f_values=[0.2], n_trials=50, n_steps=200, seed=42)
        wealth = results["0.2000"]
        median_final = np.median(wealth[:, -1])
        assert median_final > 100  # Should grow from 100

    def test_custom_f_values(self):
        f_vals = [0.1, 0.3]
        results = kelly_simulation(f_values=f_vals, n_trials=5, n_steps=10)
        assert set(results.keys()) == {"0.1000", "0.3000"}
