"""Remote ZeroMQ subscriber for strategy monitoring (Ch.10)."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Callable, Optional

import zmq

logger = logging.getLogger(__name__)


class StrategyMonitor:
    """ZeroMQ SUB client that receives trade/metric/alert events.

    Connect to a BTCTrader or LoggerMonitor's PUB socket
    to monitor strategy performance remotely.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5555,
        topics: list[str] | None = None,
    ):
        self._ctx = zmq.Context()
        self._sub = self._ctx.socket(zmq.SUB)
        self._sub.connect(f"tcp://{host}:{port}")

        for topic in (topics or [""]):  # "" subscribes to all
            self._sub.setsockopt_string(zmq.SUBSCRIBE, topic)

        self._running = False
        self._callbacks: dict[str, list[Callable]] = {}

        logger.info(f"StrategyMonitor connected to tcp://{host}:{port}")

    def on(self, topic: str, callback: Callable) -> None:
        """Register a callback for a specific topic."""
        self._callbacks.setdefault(topic, []).append(callback)

    def run(self, timeout_ms: int = 5000) -> None:
        """Start the monitoring loop.

        Parameters
        ----------
        timeout_ms : int
            Poll timeout in milliseconds.
        """
        self._running = True
        self._sub.setsockopt(zmq.RCVTIMEO, timeout_ms)

        logger.info("StrategyMonitor listening...")
        while self._running:
            try:
                raw = self._sub.recv_string()
                parts = raw.split(" ", 1)
                topic = parts[0] if len(parts) > 1 else "UNKNOWN"
                payload = parts[1] if len(parts) > 1 else parts[0]

                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    data = {"raw": payload}

                # Call registered callbacks
                for cb in self._callbacks.get(topic, []):
                    cb(data)
                for cb in self._callbacks.get("*", []):
                    cb(topic, data)

                # Default: print
                if topic not in self._callbacks and "*" not in self._callbacks:
                    ts = data.get("timestamp", datetime.now().isoformat())
                    print(f"[{ts}] {topic}: {data}")

            except zmq.Again:
                continue  # Timeout, keep polling

    def stop(self) -> None:
        self._running = False

    def close(self) -> None:
        self.stop()
        self._sub.close()
        self._ctx.term()
