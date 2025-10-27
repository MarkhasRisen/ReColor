"""Utilities for monitoring device resources and performance budgets."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque


class ResourceMonitorProtocol:
    """Protocol-like base to avoid typing dependency."""

    def should_offload(self) -> bool:  # pragma: no cover - interface method
        raise NotImplementedError


@dataclass
class PerformanceBudgetMonitor(ResourceMonitorProtocol):
    """Simple latency-based monitor for deciding when to offload inference."""

    latency_threshold: float = 0.05
    window_size: int = 10
    latencies: Deque[float] = field(default_factory=deque)

    def record_latency(self, latency: float) -> None:
        self.latencies.append(latency)
        if len(self.latencies) > self.window_size:
            self.latencies.popleft()

    def should_offload(self) -> bool:
        if not self.latencies:
            return False
        avg_latency = sum(self.latencies) / len(self.latencies)
        return avg_latency > self.latency_threshold
