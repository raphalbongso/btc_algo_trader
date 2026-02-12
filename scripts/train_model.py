"""CLI for training sklearn/DNN models and saving to pickle."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.data_loader import load_btc_data
from strategies.ml_strategy import MLStrategy
from strategies.dnn_strategy import DNNStrategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ML/DNN models for BTC trading")
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["logistic", "adaboost", "dnn"],
        help="Model type to train",
    )
    parser.add_argument("--start", type=str, default="2020-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--lags", type=int, default=5)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=50, help="DNN epochs")
    parser.add_argument("--output", type=str, default=None, help="Output path for model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading BTC data ({args.start} to {args.end})...")
    data = load_btc_data(start=args.start, end=args.end)
    print(f"Loaded {len(data)} bars\n")

    if args.model == "dnn":
        print(f"Training DNN (lags={args.lags}, epochs={args.epochs})...")
        strategy = DNNStrategy(
            lags=args.lags, epochs=args.epochs, train_ratio=args.train_ratio,
        )
        strategy.fit(data)
        output = args.output or "btc_dnn_model.pkl"
    else:
        print(f"Training {args.model} (lags={args.lags})...")
        strategy = MLStrategy(
            model_type=args.model, lags=args.lags, train_ratio=args.train_ratio,
        )
        strategy.fit(data)
        output = args.output or f"btc_{args.model}_model.pkl"

    # Save the strategy (includes the fitted model + normalization params)
    with open(output, "wb") as f:
        pickle.dump(strategy, f)
    print(f"\nModel saved to {output}")

    # Quick eval
    signals = strategy.generate_signal(data)
    long_pct = (signals == 1).mean()
    short_pct = (signals == -1).mean()
    flat_pct = (signals == 0).mean()
    print(f"Signal distribution: Long={long_pct:.1%}, Short={short_pct:.1%}, Flat={flat_pct:.1%}")


if __name__ == "__main__":
    main()
