"""Tests for backtesting.performance."""

import numpy as np
import pandas as pd
import pytest

from backtesting.performance import compute_performance_metrics, optimal_leverage


class TestComputePerformanceMetrics:
    def test_returns_dict(self, log_returns):
        metrics = compute_performance_metrics(log_returns)
        assert isinstance(metrics, dict)

    def test_required_keys(self, log_returns):
        metrics = compute_performance_metrics(log_returns)
        expected_keys = [
            "total_return", "annualized_return", "annualized_volatility",
            "sharpe_ratio", "sortino_ratio", "max_drawdown",
            "var_95", "cvar_95", "win_rate", "profit_factor",
            "kelly_fraction", "n_observations",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"

    def test_win_rate_bounded(self, log_returns):
        metrics = compute_performance_metrics(log_returns)
        assert 0 <= metrics["win_rate"] <= 1

    def test_max_drawdown_negative(self, log_returns):
        metrics = compute_performance_metrics(log_returns)
        assert metrics["max_drawdown"] <= 0

    def test_var_is_negative(self, log_returns):
        metrics = compute_performance_metrics(log_returns)
        assert metrics["var_95"] < 0

    def test_cvar_worse_than_var(self, log_returns):
        metrics = compute_performance_metrics(log_returns)
        assert metrics["cvar_95"] <= metrics["var_95"]

    def test_insufficient_data(self):
        metrics = compute_performance_metrics(np.array([0.01]))
        assert "error" in metrics

    def test_all_positive_returns(self):
        returns = np.array([0.01] * 100)
        metrics = compute_performance_metrics(returns)
        assert metrics["win_rate"] == 1.0
        assert metrics["total_return"] > 0

    def test_numpy_and_series_input(self, log_returns):
        m1 = compute_performance_metrics(log_returns.values)
        m2 = compute_performance_metrics(log_returns)
        assert abs(m1["sharpe_ratio"] - m2["sharpe_ratio"]) < 1e-10


class TestOptimalLeverage:
    def test_basic(self):
        f = optimal_leverage(mu=0.10, sigma=0.20)
        # f* = (0.10 - 0) / 0.04 = 2.5
        assert abs(f - 2.5) < 1e-10

    def test_with_risk_free(self):
        f = optimal_leverage(mu=0.10, sigma=0.20, r=0.02)
        # f* = (0.10 - 0.02) / 0.04 = 2.0
        assert abs(f - 2.0) < 1e-10

    def test_zero_volatility(self):
        assert optimal_leverage(mu=0.10, sigma=0.0) == 0.0

    def test_negative_excess_return(self):
        f = optimal_leverage(mu=0.01, sigma=0.20, r=0.05)
        assert f < 0  # Negative leverage = don't trade
