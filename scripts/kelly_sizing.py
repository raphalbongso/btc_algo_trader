"""CLI for Kelly criterion analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from data.data_loader import load_btc_data
from backtesting.performance import (
    compute_performance_metrics,
    kelly_simulation,
    optimal_leverage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kelly Criterion Analysis for BTC")
    parser.add_argument("--start", type=str, default="2020-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--risk-free", type=float, default=0.0, help="Risk-free rate")
    parser.add_argument("--simulate", action="store_true", help="Run Kelly simulation")
    parser.add_argument("--n-trials", type=int, default=50, help="Simulation trials")
    parser.add_argument("--n-steps", type=int, default=100, help="Simulation steps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading BTC data ({args.start} to {args.end})...")
    data = load_btc_data(start=args.start, end=args.end)

    price_col = "Close" if "Close" in data.columns else "price"
    log_returns = np.log(data[price_col] / data[price_col].shift(1)).dropna()
    print(f"Computed {len(log_returns)} daily log returns\n")

    # Performance metrics
    metrics = compute_performance_metrics(
        log_returns, risk_free_rate=args.risk_free,
    )

    print("=" * 50)
    print("PERFORMANCE METRICS")
    print("=" * 50)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:30s}: {value:>12.4f}")
        else:
            print(f"  {key:30s}: {value!s:>12}")

    # Optimal leverage
    mu = metrics["annualized_return"]
    sigma = metrics["annualized_volatility"]
    f_star = optimal_leverage(mu, sigma, args.risk_free)

    print(f"\n{'=' * 50}")
    print("KELLY CRITERION")
    print("=" * 50)
    print(f"  {'Annualized return (mu)':30s}: {mu:>12.4f}")
    print(f"  {'Annualized volatility (sigma)':30s}: {sigma:>12.4f}")
    print(f"  {'Risk-free rate':30s}: {args.risk_free:>12.4f}")
    print(f"  {'Optimal leverage (f*)':30s}: {f_star:>12.4f}")
    print(f"  {'Half-Kelly (conservative)':30s}: {f_star / 2:>12.4f}")

    # Kelly simulation
    if args.simulate:
        print(f"\n{'=' * 50}")
        print(f"KELLY SIMULATION ({args.n_trials} trials, {args.n_steps} steps)")
        print("=" * 50)

        win_rate = metrics["win_rate"]
        results = kelly_simulation(
            p=win_rate, n_trials=args.n_trials, n_steps=args.n_steps,
        )

        for f_key, wealth_paths in results.items():
            final_wealth = wealth_paths[:, -1]
            median_wealth = np.median(final_wealth)
            mean_wealth = np.mean(final_wealth)
            ruin_pct = (final_wealth < 1.0).mean()
            print(
                f"  f={f_key}: median={median_wealth:>10.2f}, "
                f"mean={mean_wealth:>10.2f}, ruin={ruin_pct:>6.1%}"
            )


if __name__ == "__main__":
    main()
