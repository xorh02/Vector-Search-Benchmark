import pytest

from vsbench.metrics import latency_stats, percentile, recall_at_k


def test_recall_known_answers():
    assert recall_at_k([1, 2, 3, 4], [4, 3, 2, 1], 4) == 1.0
    assert recall_at_k([1, 2, 3, 4], [1, 2, 9, 8], 4) == 0.5
    assert recall_at_k([1, 2, 3, 4], [9, 8, 7, 6], 4) == 0.0


def test_recall_only_uses_first_k():
    assert recall_at_k([1, 2, 3, 4], [1, 9, 2, 3], 2) == 0.5


def test_recall_short_approx_counts_missing_as_miss():
    assert recall_at_k([1, 2, 3, 4], [1, 2], 4) == 0.5
    assert recall_at_k([1, 2], [], 2) == 0.0


def test_recall_duplicate_ids_counted_once():
    assert recall_at_k([1, 2], [1, 1], 2) == 0.5


@pytest.mark.parametrize("k", [0, -3])
def test_recall_invalid_k(k):
    with pytest.raises(ValueError, match="k must be"):
        recall_at_k([1], [1], k)


def test_recall_exact_shorter_than_k():
    with pytest.raises(ValueError, match="need at least"):
        recall_at_k([1, 2], [1, 2], 3)


def test_percentile_nearest_rank():
    data = [float(i) for i in range(1, 101)]  # 1..100
    assert percentile(data, 50) == 50
    assert percentile(data, 95) == 95
    assert percentile(data, 99) == 99
    assert percentile(data, 100) == 100
    assert percentile([5.0], 99) == 5.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.0  # lower middle for even n


def test_percentile_errors():
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1.0], 0)


def test_latency_stats_units_and_values():
    ns = [1_000_000, 2_000_000, 3_000_000, 4_000_000]  # 1..4 ms
    s = latency_stats(ns)
    assert s.samples == 4
    assert s.mean_ms == pytest.approx(2.5)
    assert s.median_ms == 2.0
    assert s.min_ms == 1.0 and s.max_ms == 4.0
    assert s.p95_ms == 4.0 and s.p99_ms == 4.0
    assert s.throughput_qps == pytest.approx(4 / 0.010)


def test_latency_stats_empty():
    with pytest.raises(ValueError):
        latency_stats([])
