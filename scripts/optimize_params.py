"""CLI for brute-force parameter optimization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.data_loader import load_btc_data
from backtesting.vectorized.sma_backtester import SMAVectorBacktester
from backtesting.vectorized.mom_backtester import MomVectorBacktester
from backtesting.vectorized.mr_backtester import MRVectorBacktester
from backtesting.vectorized.lr_backtester import LRVectorBacktester
from backtesting.vectorized.scikit_backtester import ScikitVectorBacktester


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Brute-force parameter optimization")
    parser.add_argument(
        "--strategy", type=str, required=True,
        choices=["sma", "momentum", "mr", "lr", "scikit"],
    )
    parser.add_argument("--start", type=str, default="2020-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--ptc", type=float, default=0.001)
    parser.add_argument("--model", type=str, default="logistic",
                        choices=["logistic", "adaboost"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading BTC data ({args.start} to {args.end})...")
    data = load_btc_data(start=args.start, end=args.end)
    print(f"Loaded {len(data)} bars\n")

    if args.strategy == "sma":
        print("Optimizing SMA crossover...")
        bt = SMAVectorBacktester(data, ptc=args.ptc)
        best_short, best_long, perf = bt.optimize(
            short_range=range(10, 51, 5),
            long_range=range(30, 201, 10),
        )
        print(f"\nBest: SMA({best_short}, {best_long}) → cumulative return = {perf:.4f}")

    elif args.strategy == "momentum":
        print("Optimizing Momentum window...")
        bt = MomVectorBacktester(data, ptc=args.ptc)
        best_mom, perf = bt.optimize(momentum_range=range(5, 101, 5))
        print(f"\nBest: Momentum({best_mom}) → cumulative return = {perf:.4f}")

    elif args.strategy == "mr":
        print("Optimizing Mean Reversion...")
        bt = MRVectorBacktester(data, ptc=args.ptc)
        best_w, best_t, perf = bt.optimize(window_range=range(10, 61, 5))
        print(f"\nBest: MR(window={best_w}, threshold={best_t}) → cumulative return = {perf:.4f}")

    elif args.strategy == "lr":
        print("Optimizing Linear Regression lags...")
        bt = LRVectorBacktester(data, ptc=args.ptc)
        best_lags, perf = bt.optimize(lag_range=range(2, 16))
        print(f"\nBest: LR(lags={best_lags}) → test log return = {perf:.4f}")

    elif args.strategy == "scikit":
        print(f"Optimizing Scikit ({args.model}) lags...")
        bt = ScikitVectorBacktester(data, model_type=args.model, ptc=args.ptc)
        best_lags, perf = bt.optimize(lag_range=range(2, 16))
        print(f"\nBest: Scikit(model={args.model}, lags={best_lags}) → test log return = {perf:.4f}")


if __name__ == "__main__":
    main()
