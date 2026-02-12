"""Remote monitoring script (Ch.10).

Connects to a running BTCTrader's ZMQ PUB socket and
prints trade events, metrics, and alerts in real time.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from monitoring.strategy_monitor import StrategyMonitor


def main():
    parser = argparse.ArgumentParser(description="Remote Strategy Monitor")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--topics", nargs="*", default=None,
                        help="Topics to subscribe to (default: all)")
    args = parser.parse_args()

    monitor = StrategyMonitor(
        host=args.host,
        port=args.port,
        topics=args.topics,
    )

    print(f"Monitoring tcp://{args.host}:{args.port} ...")
    print("Press Ctrl+C to stop\n")

    try:
        monitor.run()
    except KeyboardInterrupt:
        print("\nStopping monitor")
    finally:
        monitor.close()


if __name__ == "__main__":
    main()
