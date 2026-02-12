"""CLI: Start live trading on MEXC (paper, spot, or futures)."""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import load_config, MEXCConfig, TradingConfig
from config.logging_config import setup_logging
from strategies.sma_strategy import SMAStrategy
from strategies.momentum_strategy import MomentumStrategy
from strategies.mean_reversion_strategy import MeanReversionStrategy
from strategies.ensemble_strategy import EnsembleStrategy
from live.btc_trader import BTCTrader

_trader = None


def _shutdown(signum, frame):
    if _trader:
        _trader.close_out()
    sys.exit(0)


def parse_args():
    parser = argparse.ArgumentParser(description="BTC/USDT Live Trader for MEXC")
    parser.add_argument(
        "--mode", choices=["paper", "spot", "futures"], default="paper",
        help="Trading mode (default: paper)",
    )
    parser.add_argument(
        "--strategy", choices=["sma", "momentum", "mr", "ensemble"], default="momentum",
    )
    parser.add_argument("--intervals", type=int, default=500)
    parser.add_argument("--sleep", type=float, default=5.0, help="Seconds between polls")
    parser.add_argument("--leverage", type=int, default=1, help="Futures leverage")
    parser.add_argument("--units", type=float, default=0.001, help="BTC per trade")
    parser.add_argument("--capital", type=float, default=1000.0, help="Initial USDT capital")
    parser.add_argument("--zmq", action="store_true", help="Enable ZMQ publishing")
    parser.add_argument("--sma-short", type=int, default=42)
    parser.add_argument("--sma-long", type=int, default=252)
    parser.add_argument("--momentum", type=int, default=6)
    parser.add_argument("--bar-length", default="1m", help="OHLCV timeframe")
    return parser.parse_args()


def main():
    global _trader

    args = parse_args()
    setup_logging(level="INFO")

    # Build strategy
    if args.strategy == "sma":
        strategy = SMAStrategy(sma_short=args.sma_short, sma_long=args.sma_long)
    elif args.strategy == "mr":
        strategy = MeanReversionStrategy(window=25, threshold=1.0)
    elif args.strategy == "ensemble":
        strategy = EnsembleStrategy(
            strategies=[
                SMAStrategy(sma_short=args.sma_short, sma_long=args.sma_long),
                MomentumStrategy(window=args.momentum),
            ],
            mode="majority",
        )
    else:
        strategy = MomentumStrategy(window=args.momentum)

    trading_config = TradingConfig(
        units=args.units,
        initial_capital=args.capital,
        bar_length=args.bar_length,
        sma_short=args.sma_short,
        sma_long=args.sma_long,
        momentum_window=args.momentum,
    )

    mexc_config = MEXCConfig(
        leverage=args.leverage,
        trading_type="swap" if args.mode == "futures" else "spot",
        sandbox=(args.mode == "paper"),
    )

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(f"Starting BTC Trader: mode={args.mode}, strategy={args.strategy}")
    if args.mode != "paper":
        print("WARNING: This will trade REAL money on MEXC!")
        print(f"  Units: {args.units} BTC, Leverage: {args.leverage}x")

    _trader = BTCTrader(
        strategy=strategy,
        mode=args.mode,
        trading_config=trading_config,
        mexc_config=mexc_config,
        zmq_publish=args.zmq,
    )

    _trader.run_polling(intervals=args.intervals, sleep_sec=args.sleep)

    summary = _trader.get_summary()
    print("\n" + "=" * 50)
    print("TRADING SUMMARY")
    print("=" * 50)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:>12.4f}")
        else:
            print(f"  {k:30s}: {v!s:>12}")


if __name__ == "__main__":
    main()
