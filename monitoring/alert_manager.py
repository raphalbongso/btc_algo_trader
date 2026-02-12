"""Threshold-based alerting for trading metrics."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class AlertRule:
    """A single alert rule."""
    name: str
    metric: str
    threshold: float
    comparator: str = "gt"  # 'gt', 'lt', 'gte', 'lte'
    cooldown_ticks: int = 10  # Minimum ticks between repeated alerts
    _last_fired: int = field(default=0, repr=False)

    def check(self, value: float, tick: int) -> bool:
        """Check if alert should fire."""
        if tick - self._last_fired < self.cooldown_ticks:
            return False

        triggered = False
        if self.comparator == "gt" and value > self.threshold:
            triggered = True
        elif self.comparator == "lt" and value < self.threshold:
            triggered = True
        elif self.comparator == "gte" and value >= self.threshold:
            triggered = True
        elif self.comparator == "lte" and value <= self.threshold:
            triggered = True

        if triggered:
            self._last_fired = tick
        return triggered


class AlertManager:
    """Manages alert rules and notifications.

    Parameters
    ----------
    on_alert : callable, optional
        Callback(rule_name, metric, value, message) when alert fires.
    """

    def __init__(self, on_alert: Callable | None = None):
        self.rules: List[AlertRule] = []
        self._on_alert = on_alert or self._default_alert
        self._tick = 0

    def add_rule(self, rule: AlertRule) -> None:
        self.rules.append(rule)

    def add_drawdown_alert(self, threshold: float = -0.10) -> None:
        """Convenience: alert when drawdown exceeds threshold."""
        self.add_rule(AlertRule(
            name="max_drawdown",
            metric="drawdown",
            threshold=threshold,
            comparator="lt",
        ))

    def add_equity_alert(self, min_equity: float = 500.0) -> None:
        """Convenience: alert when equity drops below threshold."""
        self.add_rule(AlertRule(
            name="low_equity",
            metric="equity",
            threshold=min_equity,
            comparator="lt",
        ))

    def check(self, metrics: Dict[str, float]) -> List[str]:
        """Check all rules against current metrics.

        Returns list of triggered alert names.
        """
        self._tick += 1
        triggered = []

        for rule in self.rules:
            value = metrics.get(rule.metric)
            if value is not None and rule.check(value, self._tick):
                msg = (
                    f"ALERT [{rule.name}]: {rule.metric}={value:.4f} "
                    f"{rule.comparator} {rule.threshold}"
                )
                self._on_alert(rule.name, rule.metric, value, msg)
                triggered.append(rule.name)

        return triggered

    @staticmethod
    def _default_alert(name: str, metric: str, value: float, message: str) -> None:
        logger.warning(message)
