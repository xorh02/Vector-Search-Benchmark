"""Recall@k and latency statistics."""

from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import BaseModel


def recall_at_k(exact: Sequence[int], approx: Sequence[int], k: int) -> float:
    """|exact[:k] ∩ approx[:k]| / k.

    Matching is by id. If approximate search returns fewer than k rows, the missing
    rows count as misses. Duplicate ids in `approx` are counted once.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if len(exact) < k:
        raise ValueError(f"exact result has {len(exact)} ids, need at least k={k}")
    return len(set(exact[:k]) & set(approx[:k])) / k


def percentile(sorted_values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile: the smallest value with at least p% of samples <= it."""
    if not sorted_values:
        raise ValueError("no samples")
    if not 0 < p <= 100:
        raise ValueError("p must be in (0, 100]")
    rank = math.ceil(p / 100 * len(sorted_values))
    return sorted_values[rank - 1]


class LatencyStats(BaseModel):
    samples: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    throughput_qps: float  # samples / total time, single connection, sequential


def latency_stats(latencies_ns: Sequence[int]) -> LatencyStats:
    if not latencies_ns:
        raise ValueError("no latency samples")
    ms = sorted(x / 1e6 for x in latencies_ns)
    total_s = sum(latencies_ns) / 1e9
    return LatencyStats(
        samples=len(ms),
        mean_ms=sum(ms) / len(ms),
        median_ms=percentile(ms, 50),
        p95_ms=percentile(ms, 95),
        p99_ms=percentile(ms, 99),
        min_ms=ms[0],
        max_ms=ms[-1],
        throughput_qps=len(ms) / total_s if total_s > 0 else 0.0,
    )
