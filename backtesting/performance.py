"""Performance metrics: Sharpe, drawdown, VaR, CVaR, Kelly, win rate."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_performance_metrics(
    returns: pd.Series | np.ndarray,
    trading_days: int = 365,
    risk_free_rate: float = 0.0,
    confidence: float = 0.05,
) -> dict:
    """Compute comprehensive performance metrics from a return series.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Log returns (daily).
    trading_days : int
        Annualization factor (365 for crypto).
    risk_free_rate : float
        Annualized risk-free rate.
    confidence : float
        VaR/CVaR confidence level (0.05 = 95%).

    Returns
    -------
    dict
        Dictionary of performance metrics.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]

    if len(r) < 2:
        return {"error": "insufficient data"}

    # Basic stats
    total_return = np.exp(r.sum()) - 1
    n_years = len(r) / trading_days
    ann_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol = r.std() * np.sqrt(trading_days)
    daily_rf = risk_free_rate / trading_days

    # Sharpe ratio
    excess = r.mean() - daily_rf
    sharpe = (excess / r.std()) * np.sqrt(trading_days) if r.std() > 0 else 0

    # Sortino ratio (downside deviation)
    downside = r[r < daily_rf]
    downside_std = downside.std() * np.sqrt(trading_days) if len(downside) > 1 else 0
    sortino = (ann_return - risk_free_rate) / downside_std if downside_std > 0 else 0

    # Max drawdown
    cum_returns = np.exp(np.cumsum(r))
    peak = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - peak) / peak
    max_drawdown = drawdown.min()

    # VaR and CVaR (Historical)
    var = np.percentile(r, confidence * 100)
    cvar = r[r <= var].mean() if len(r[r <= var]) > 0 else var

    # Win rate
    wins = (r > 0).sum()
    win_rate = wins / len(r)

    # Profit factor
    gross_profit = r[r > 0].sum() if (r > 0).any() else 0
    gross_loss = abs(r[r < 0].sum()) if (r < 0).any() else 1e-10
    profit_factor = gross_profit / gross_loss

    # Kelly criterion (from returns)
    if win_rate > 0 and win_rate < 1:
        avg_win = r[r > 0].mean() if (r > 0).any() else 0
        avg_loss = abs(r[r < 0].mean()) if (r < 0).any() else 1e-10
        win_loss_ratio = avg_win / avg_loss
        kelly = win_rate - (1 - win_rate) / win_loss_ratio
    else:
        kelly = 0.0

    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "var_95": var,
        "cvar_95": cvar,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "kelly_fraction": kelly,
        "n_observations": len(r),
        "n_years": n_years,
    }


def kelly_simulation(
    p: float = 0.55,
    f_values: list[float] | None = None,
    n_trials: int = 50,
    n_steps: int = 100,
    initial_capital: float = 100.0,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Simulate Kelly criterion with different bet fractions.

    Fix for book typo: correctly handles the binomial outcome
    where win pays +f and loss pays -f (not the wealth formula).

    Parameters
    ----------
    p : float
        Win probability.
    f_values : list[float]
        Bet fractions to simulate.
    n_trials : int
        Number of simulation paths per fraction.
    n_steps : int
        Number of bets per path.
    initial_capital : float
        Starting capital.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Keys are f values (as str), values are (n_trials, n_steps+1) arrays.
    """
    if f_values is None:
        f_star = 2 * p - 1  # Optimal Kelly fraction for even-odds
        f_values = [0.05, 0.1, 0.25, f_star, 0.5, 0.75, 1.0]

    rng = np.random.default_rng(seed)
    results = {}

    for f in f_values:
        wealth = np.zeros((n_trials, n_steps + 1))
        wealth[:, 0] = initial_capital

        for step in range(n_steps):
            wins = rng.random((n_trials,)) < p
            # Multiplicative growth: W_{t+1} = W_t * (1 + f) if win, W_t * (1 - f) if loss
            multiplier = np.where(wins, 1 + f, 1 - f)
            wealth[:, step + 1] = wealth[:, step] * multiplier

        results[f"{f:.4f}"] = wealth

    return results


def optimal_leverage(
    mu: float,
    sigma: float,
    r: float = 0.0,
) -> float:
    """Compute optimal Kelly leverage f* = (mu - r) / sigma^2.

    Parameters
    ----------
    mu : float
        Expected return (annualized).
    sigma : float
        Volatility (annualized).
    r : float
        Risk-free rate (annualized).

    Returns
    -------
    float
        Optimal leverage ratio.
    """
    if sigma <= 0:
        return 0.0
    return (mu - r) / (sigma ** 2)
