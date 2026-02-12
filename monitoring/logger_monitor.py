"""ZeroMQ + file logging monitor (Ch.10).

Publishes trade events and metrics on a ZMQ PUB socket
for remote subscribers. Also logs to file.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import zmq

logger = logging.getLogger(__name__)


class LoggerMonitor:
    """Dual ZeroMQ + file trade logger."""

    def __init__(
        self,
        zmq_port: int = 5556,
        bind_address: str = "127.0.0.1",
        log_file: Optional[str] = None,
    ):
        self._ctx = zmq.Context()
        self._pub = self._ctx.socket(zmq.PUB)
        self._address = f"tcp://{bind_address}:{zmq_port}"
        self._pub.bind(self._address)

        self._log_file = None
        if log_file:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = open(path, "a")

        logger.info(f"LoggerMonitor bound to {self._address}")

    def log_trade(self, trade: dict) -> None:
        """Log a trade event."""
        trade["logged_at"] = datetime.now().isoformat()
        msg = json.dumps(trade)

        self._pub.send_string(f"TRADE {msg}")

        if self._log_file:
            self._log_file.write(msg + "\n")
            self._log_file.flush()

    def log_metric(self, name: str, value: float) -> None:
        """Log a performance metric."""
        msg = json.dumps({
            "metric": name,
            "value": value,
            "timestamp": datetime.now().isoformat(),
        })
        self._pub.send_string(f"METRIC {msg}")

    def log_alert(self, level: str, message: str) -> None:
        """Log an alert."""
        msg = json.dumps({
            "level": level,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        })
        self._pub.send_string(f"ALERT {msg}")

    def close(self) -> None:
        if self._log_file:
            self._log_file.close()
        self._pub.close()
        self._ctx.term()
