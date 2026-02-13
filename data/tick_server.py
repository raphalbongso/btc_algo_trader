"""ZeroMQ tick server for simulation and distribution (Ch.7).

Publishes OHLCV bars on a PUB socket for strategy subscribers.
Can replay historical data or forward live OKX ticks.
"""

from __future__ import annotations

import json
import time
import logging
from typing import Optional

import pandas as pd
import zmq

logger = logging.getLogger(__name__)


class TickServer:
    """ZeroMQ PUB server that streams price ticks."""

    def __init__(self, port: int = 5555, bind_address: str = "127.0.0.1"):
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.PUB)
        self._address = f"tcp://{bind_address}:{port}"
        self._socket.bind(self._address)
        logger.info(f"TickServer bound to {self._address}")

    def replay_historical(
        self,
        data: pd.DataFrame,
        delay: float = 0.01,
        topic: str = "TICK",
    ) -> None:
        """Replay historical data bar by bar.

        Parameters
        ----------
        data : pd.DataFrame
            Must have DatetimeIndex and 'Close'/'price' column.
        delay : float
            Seconds between ticks.
        topic : str
            ZMQ topic prefix.
        """
        price_col = "Close" if "Close" in data.columns else "price"

        for ts, row in data.iterrows():
            msg = {
                "timestamp": str(ts),
                "price": float(row[price_col]),
                "volume": float(row.get("Volume", 0)),
            }
            self._socket.send_string(f"{topic} {json.dumps(msg)}")
            time.sleep(delay)

        logger.info(f"Replay complete: {len(data)} bars")

    def publish_tick(self, price: float, volume: float = 0, topic: str = "TICK") -> None:
        """Publish a single tick."""
        msg = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "price": price,
            "volume": volume,
        }
        self._socket.send_string(f"{topic} {json.dumps(msg)}")

    def close(self) -> None:
        """Shutdown the server."""
        self._socket.close()
        self._ctx.term()
        logger.info("TickServer closed")


class TickClient:
    """ZeroMQ SUB client that receives ticks."""

    def __init__(self, port: int = 5555, host: str = "127.0.0.1", topic: str = "TICK"):
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.SUB)
        self._socket.connect(f"tcp://{host}:{port}")
        self._socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        logger.info(f"TickClient connected to tcp://{host}:{port}")

    def receive(self, timeout: Optional[int] = None) -> Optional[dict]:
        """Receive a single tick. Returns None on timeout.

        Parameters
        ----------
        timeout : int, optional
            Timeout in milliseconds.
        """
        if timeout is not None:
            self._socket.setsockopt(zmq.RCVTIMEO, timeout)
        try:
            raw = self._socket.recv_string()
            _, payload = raw.split(" ", 1)
            return json.loads(payload)
        except zmq.Again:
            return None

    def close(self) -> None:
        self._socket.close()
        self._ctx.term()
