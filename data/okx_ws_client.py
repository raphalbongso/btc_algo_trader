"""OKX WebSocket client for real-time tick data.

Uses raw websockets for streaming BTC/USDT trades.
Preferred over REST polling for live trading (lower latency).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

OKX_WS_URL = "wss://ws.okx.com:8443/ws/v5/public"


class OKXWebSocketClient:
    """WebSocket client for OKX real-time data."""

    def __init__(self, on_tick: Optional[Callable] = None):
        self._on_tick = on_tick
        self._running = False

    async def subscribe_trades(self, symbol: str = "BTC-USDT") -> None:
        """Subscribe to real-time trade stream."""
        import websockets

        self._running = True

        async with websockets.connect(OKX_WS_URL) as ws:
            sub_msg = {
                "op": "subscribe",
                "args": [{"channel": "trades", "instId": symbol}],
            }
            await ws.send(json.dumps(sub_msg))
            logger.info(f"Subscribed to {symbol} trades")

            while self._running:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=30)
                    data = json.loads(msg)

                    if "data" in data:
                        for trade in data["data"]:
                            tick = {
                                "price": float(trade["px"]),
                                "quantity": float(trade["sz"]),
                                "timestamp": int(trade["ts"]),
                                "side": trade["side"],
                            }
                            if self._on_tick:
                                self._on_tick(tick)

                except asyncio.TimeoutError:
                    await ws.send("ping")
                except Exception as e:
                    logger.error(f"WebSocket error: {e}")
                    break

    def stop(self) -> None:
        """Stop the WebSocket client."""
        self._running = False


def run_ws_client(on_tick: Callable, symbol: str = "BTC-USDT") -> None:
    """Convenience function to run WebSocket client."""
    client = OKXWebSocketClient(on_tick=on_tick)
    asyncio.run(client.subscribe_trades(symbol))
