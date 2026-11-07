import os

import pytest

from vsbench.benchmark import RunResult, SearchResult
from vsbench.config import BenchmarkConfig
from vsbench.metrics import latency_stats


@pytest.fixture
def run_result() -> RunResult:
    """A structurally complete result with arbitrary test values (not a measurement)."""
    cfg = BenchmarkConfig.model_validate({"hnsw": [{"ef_search": [10, 40]}]})

    def sr(method, ef, recall, lat_ms):
        return SearchResult(
            method=method,
            m=16 if ef else None,
            ef_construction=64 if ef else None,
            ef_search=ef,
            index_build_seconds=1.5 if ef else None,
            index_size_bytes=8_388_608 if ef else None,
            latency=latency_stats([int(x * 1e6) for x in lat_ms]),
            server_median_ms=lat_ms[0] / 2,
            server_samples=3,
            recall_mean=recall,
            recall_min=recall - 0.1 if ef else 1.0,
            plan_node_types=["Limit", "Index Scan"] if ef else ["Limit", "Sort", "Seq Scan"],
            plan_indexes=["vsbench_default_hnsw"] if ef else [],
            plan_verified=True,
        )

    return RunResult(
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:01:00+00:00",
        config=cfg.model_dump(mode="json"),
        dataset={
            "source": "synthetic: test fixture",
            "size": 10_000,
            "dimensions": 128,
            "queries": 200,
            "seed": 42,
            "sha256": "0" * 64,
            "numpy_version": "x",
        },
        environment={
            "python_version": "3.12.0",
            "os": "TestOS",
            "machine": "x86_64",
            "processor": None,
            "cpu_count": 4,
            "psycopg_version": "3",
            "server": {
                "postgres_version_full": "PostgreSQL 17",
                "postgres_version": "17.0",
                "pgvector_version": "0.8.0",
                "shared_buffers": "128MB",
                "work_mem": "4MB",
                "max_parallel_workers_per_gather": "2",
                "max_parallel_maintenance_workers": "2",
            },
        },
        load_seconds=0.5,
        table_rows=cfg.dataset.size,
        table_size_bytes=6_000_000,
        exact_crosscheck={"queries_checked": 10, "matching": 10},
        results=[
            sr("exact", None, 1.0, [12.0, 13.0, 14.0]),
            sr("hnsw", 10, 0.81, [0.5, 0.6, 0.7]),
            sr("hnsw", 40, 0.97, [0.9, 1.0, 1.1]),
        ],
    )


@pytest.fixture
def database_url() -> str:
    url = os.environ.get("VSBENCH_DATABASE_URL")
    if not url:
        pytest.skip("set VSBENCH_DATABASE_URL to run PostgreSQL integration tests")
    return url
