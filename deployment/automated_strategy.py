"""Full deployment script for automated BTC trading on OKX (Ch.10).

Orchestrates:
1. Load config
2. Initialize strategy
3. Start BTCTrader in chosen mode
4. Run until interrupted
5. Clean shutdown
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import load_config
from config.logging_config import setup_logging
from strategies.momentum_strategy import MomentumStrategy
from strategies.sma_strategy import SMAStrategy
from strategies.ensemble_strategy import EnsembleStrategy
from live.btc_trader import BTCTrader

logger = logging.getLogger(__name__)

_trader: BTCTrader | None = None


def _shutdown_handler(signum, frame):
    """Graceful shutdown on SIGINT/SIGTERM."""
    logger.info("Shutdown signal received")
    if _trader:
        _trader.close_out()
    sys.exit(0)


def main():
    global _trader

    parser = argparse.ArgumentParser(description="Automated BTC Strategy Deployment")
    parser.add_argument("--mode", choices=["paper", "spot", "futures"], default="paper")
    parser.add_argument("--strategy", choices=["sma", "momentum", "ensemble"], default="momentum")
    parser.add_argument("--intervals", type=int, default=1000)
    parser.add_argument("--sleep", type=float, default=5.0)
    parser.add_argument("--zmq", action="store_true", help="Enable ZMQ publishing")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--json-log", action="store_true")
    args = parser.parse_args()

    setup_logging(level=args.log_level, json_output=args.json_log)

    config = load_config()

    # Build strategy
    if args.strategy == "sma":
        strategy = SMAStrategy(
            sma_short=config.trading.sma_short,
            sma_long=config.trading.sma_long,
        )
    elif args.strategy == "ensemble":
        strategy = EnsembleStrategy(
            strategies=[
                SMAStrategy(sma_short=config.trading.sma_short, sma_long=config.trading.sma_long),
                MomentumStrategy(window=config.trading.momentum_window),
            ],
            mode="majority",
        )
    else:
        strategy = MomentumStrategy(window=config.trading.momentum_window)

    # Register shutdown handlers
    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    _trader = BTCTrader(
        strategy=strategy,
        mode=args.mode,
        trading_config=config.trading,
        okx_config=config.okx,
        zmq_publish=args.zmq,
    )

    logger.info(f"Starting automated trading: mode={args.mode}, strategy={args.strategy}")
    _trader.run_polling(intervals=args.intervals, sleep_sec=args.sleep)

    summary = _trader.get_summary()
    logger.info(f"Final summary: {summary}")


if __name__ == "__main__":
    main()
