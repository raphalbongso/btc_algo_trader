"""CLI for running vectorized and event-based backtests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from data.data_loader import load_btc_data
from backtesting.vectorized.sma_backtester import SMAVectorBacktester
from backtesting.vectorized.mom_backtester import MomVectorBacktester
from backtesting.vectorized.mr_backtester import MRVectorBacktester
from backtesting.vectorized.lr_backtester import LRVectorBacktester
from backtesting.vectorized.scikit_backtester import ScikitVectorBacktester
from backtesting.event_based.backtest_long_only import BacktestLongOnly
from backtesting.event_based.backtest_long_short import BacktestLongShort
from backtesting.performance import compute_performance_metrics
from strategies.sma_strategy import SMAStrategy
from strategies.momentum_strategy import MomentumStrategy
from strategies.mean_reversion_strategy import MeanReversionStrategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BTC Algorithmic Trading Backtester")
    parser.add_argument(
        "--strategy", type=str, required=True,
        choices=["sma", "momentum", "mr", "lr", "scikit", "event-long", "event-short"],
        help="Strategy to backtest",
    )
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start date")
    parser.add_argument("--end", type=str, default="2024-12-31", help="End date")
    parser.add_argument("--ptc", type=float, default=0.0005, help="Proportional TC (MEXC spot taker: 0.05%%)")
    parser.add_argument("--ftc", type=float, default=0.0, help="Fixed TC (event-based)")

    # SMA params
    parser.add_argument("--sma-short", type=int, default=20, help="Short SMA window")
    parser.add_argument("--sma-long", type=int, default=50, help="Long SMA window")

    # Momentum params
    parser.add_argument("--momentum", type=int, default=20, help="Momentum window")

    # Mean reversion params
    parser.add_argument("--mr-window", type=int, default=25, help="MR window")
    parser.add_argument("--mr-threshold", type=float, default=1.0, help="MR threshold")

    # ML params
    parser.add_argument("--lags", type=int, default=5, help="Number of lags")
    parser.add_argument("--model", type=str, default="logistic",
                        choices=["logistic", "adaboost"], help="ML model type")

    # Event-based params
    parser.add_argument("--capital", type=float, default=100000, help="Initial capital")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading BTC data ({args.start} to {args.end})...")
    data = load_btc_data(start=args.start, end=args.end)
    print(f"Loaded {len(data)} bars\n")

    if args.strategy == "sma":
        print(f"Running SMA backtest (short={args.sma_short}, long={args.sma_long})...")
        bt = SMAVectorBacktester(
            data, sma_short=args.sma_short, sma_long=args.sma_long, ptc=args.ptc,
        )
        results = bt.run()
        summary = bt.summary()

    elif args.strategy == "momentum":
        print(f"Running Momentum backtest (window={args.momentum})...")
        bt = MomVectorBacktester(data, momentum=args.momentum, ptc=args.ptc)
        results = bt.run()
        summary = bt.summary()

    elif args.strategy == "mr":
        print(f"Running Mean Reversion backtest (window={args.mr_window}, threshold={args.mr_threshold})...")
        bt = MRVectorBacktester(
            data, window=args.mr_window, threshold=args.mr_threshold, ptc=args.ptc,
        )
        results = bt.run()
        summary = bt.summary()

    elif args.strategy == "lr":
        print(f"Running Linear Regression backtest (lags={args.lags})...")
        bt = LRVectorBacktester(data, lags=args.lags, ptc=args.ptc)
        results = bt.run()
        summary = bt.summary()

    elif args.strategy == "scikit":
        print(f"Running Scikit backtest (model={args.model}, lags={args.lags})...")
        bt = ScikitVectorBacktester(
            data, model_type=args.model, lags=args.lags, ptc=args.ptc,
        )
        results = bt.run()
        summary = bt.summary()

    elif args.strategy == "event-long":
        print(f"Running Event-Based Long-Only backtest (SMA {args.sma_short}/{args.sma_long})...")
        strategy = SMAStrategy(sma_short=args.sma_short, sma_long=args.sma_long)
        bt = BacktestLongOnly(
            data, strategy=strategy,
            initial_capital=args.capital, ftc=args.ftc, ptc=args.ptc,
        )
        summary = bt.run()

    elif args.strategy == "event-short":
        print(f"Running Event-Based Long/Short backtest (Momentum {args.momentum})...")
        strategy = MomentumStrategy(window=args.momentum)
        bt = BacktestLongShort(
            data, strategy=strategy,
            initial_capital=args.capital, ftc=args.ftc, ptc=args.ptc,
        )
        summary = bt.run()

    else:
        print(f"Unknown strategy: {args.strategy}")
        sys.exit(1)

    # Print results
    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key:30s}: {value:>12.4f}")
        else:
            print(f"  {key:30s}: {value!s:>12}")

    # Compute detailed performance metrics if we have vectorized results
    if hasattr(bt, "results") and "strategy_net" in bt.results.columns:
        print("\n" + "-" * 50)
        print("DETAILED PERFORMANCE METRICS")
        print("-" * 50)
        metrics = compute_performance_metrics(bt.results["strategy_net"].dropna())
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key:30s}: {value:>12.4f}")
            else:
                print(f"  {key:30s}: {value!s:>12}")


if __name__ == "__main__":
    main()
