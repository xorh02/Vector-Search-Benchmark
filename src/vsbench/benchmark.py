"""Benchmark orchestration: dataset -> load -> exact baseline -> per-index HNSW runs."""

from __future__ import annotations

import os
import platform
import statistics
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Literal

import psycopg
from psycopg import sql
from pydantic import BaseModel

from vsbench import __version__, db
from vsbench.config import MAX_HNSW_DIMENSIONS, BenchmarkConfig
from vsbench.dataset import Dataset, brute_force_knn, generate, validate_vectors
from vsbench.metrics import LatencyStats, latency_stats, recall_at_k

SCHEMA_VERSION = 1
CROSSCHECK_QUERIES = 10
SERVER_TIMING_QUERIES = 20


class SearchResult(BaseModel):
    method: Literal["exact", "hnsw"]
    m: int | None = None
    ef_construction: int | None = None
    ef_search: int | None = None
    index_build_seconds: float | None = None
    index_size_bytes: int | None = None
    latency: LatencyStats
    # Median "Execution Time" from EXPLAIN ANALYZE over the first queries: no network/driver cost.
    server_median_ms: float | None = None
    server_samples: int = 0
    recall_mean: float
    recall_min: float
    plan_node_types: list[str]
    plan_indexes: list[str]
    plan_verified: bool  # exact: sequential scan only; hnsw: plan scans the HNSW index

    @property
    def label(self) -> str:
        if self.method == "exact":
            return "exact"
        return f"hnsw m={self.m} efc={self.ef_construction} efs={self.ef_search}"


class RunResult(BaseModel):
    schema_version: int = SCHEMA_VERSION
    tool_version: str = __version__
    started_at: str
    finished_at: str
    config: dict[str, Any]
    dataset: dict[str, Any]
    environment: dict[str, Any]
    load_seconds: float
    table_rows: int
    table_size_bytes: int
    exact_crosscheck: dict[str, int]
    results: list[SearchResult]

    def best(self, min_recall: float) -> SearchResult | None:
        """Fastest HNSW configuration (by median latency) whose mean recall >= min_recall."""
        ok = [r for r in self.results if r.method == "hnsw" and r.recall_mean >= min_recall]
        return min(ok, key=lambda r: r.latency.median_ms, default=None)


def client_environment() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "psycopg_version": psycopg.__version__,
    }


Log = Callable[[str], None]


def _measure(
    conn: psycopg.Connection[Any], query: sql.Composed, data: Dataset, config: BenchmarkConfig
) -> tuple[dict[str, Any], list[list[int]], dict[str, Any]]:
    """EXPLAIN, warm up, run timed trials, then sample server execution time.

    Returns (plan summary, ids per query from the first trial, measurement fields).
    """
    k = config.top_k
    params = [(db.vector_literal(q), k) for q in data.queries]
    plan = db.plan_summary(db.explain(conn, query, params[0]))
    latencies: list[int] = []
    ids: list[list[int]] = []
    with conn.cursor() as cur:
        for i in range(config.warmup_queries):
            cur.execute(query, params[i % len(params)], prepare=True)
            cur.fetchall()
        for trial in range(config.trials):
            for p in params:
                start = time.perf_counter_ns()
                cur.execute(query, p, prepare=True)
                rows = cur.fetchall()
                latencies.append(time.perf_counter_ns() - start)
                if trial == 0:
                    ids.append([r[0] for r in rows])
    server = [db.server_execution_ms(conn, query, p) for p in params[:SERVER_TIMING_QUERIES]]
    fields = {
        "latency": latency_stats(latencies),
        "server_median_ms": statistics.median(server),
        "server_samples": len(server),
        "plan_node_types": plan["node_types"],
        "plan_indexes": plan["indexes"],
    }
    return plan, ids, fields


def run(
    config: BenchmarkConfig,
    database_url: str,
    *,
    data: Dataset | None = None,
    log: Log | None = None,
) -> RunResult:
    """Run the benchmark described by `config`.

    `data` replaces the synthetic dataset (see `vsbench.from_arrays`); `config.dataset` is then
    ignored. The benchmark table is dropped afterwards unless `config.keep_table` is set.
    """
    log = log or (lambda _: None)
    started = datetime.now(UTC)
    k = config.top_k
    table = config.table_name

    if data is None:
        log(
            f"generating dataset: {config.dataset.size} x {config.dataset.dimensions}, "
            f"{config.dataset.queries} queries, seed {config.dataset.seed}"
        )
        data = generate(config.dataset)
    n, dims = data.vectors.shape
    if k > n:
        raise ValueError(f"top_k ({k}) exceeds the number of stored vectors ({n})")
    if dims > MAX_HNSW_DIMENSIONS:
        raise ValueError(f"pgvector HNSW supports at most {MAX_HNSW_DIMENSIONS} dimensions")
    validate_vectors(data.vectors, config.metric)
    validate_vectors(data.queries, config.metric)

    with db.connect(database_url) as conn:
        db.ensure_pgvector(conn)
        env = client_environment() | {"server": db.server_info(conn)}
        try:
            log(f"loading {n} vectors ({dims} dims) into {table}")
            db.create_table(conn, table, dims)
            load_seconds = db.load_vectors(conn, table, data.vectors)
            rows = db.count_rows(conn, table)
            if rows != n:
                raise db.DatabaseError(f"expected {n} rows, found {rows}")

            # Exact baseline runs before any vector index exists, with index scans disabled.
            log("exact search (sequential scan)")
            db.set_search_mode(conn, hnsw_ef_search=None)
            plan, truth, fields = _measure(
                conn, db.knn_query(table, config.metric, exact=True), data, config
            )
            checked = min(CROSSCHECK_QUERIES, len(truth))
            matching = sum(
                set(truth[i])
                == set(brute_force_knn(data.vectors, data.queries[i], k, config.metric))
                for i in range(checked)
            )
            results = [
                SearchResult(
                    method="exact",
                    recall_mean=1.0,
                    recall_min=1.0,
                    plan_verified=db.plan_is_sequential(plan),
                    **fields,
                )
            ]

            index = f"{table}_hnsw"
            approx_q = db.knn_query(table, config.metric, exact=False)
            for h in config.hnsw:
                log(f"building HNSW index m={h.m} ef_construction={h.ef_construction}")
                build_s = db.create_hnsw_index(
                    conn,
                    table,
                    index,
                    config.metric,
                    h.m,
                    h.ef_construction,
                    config.maintenance_work_mem,
                )
                if not db.index_is_valid(conn, index):
                    raise db.DatabaseError(f"HNSW index {index} was not built correctly")
                size = db.index_size_bytes(conn, index)
                for ef in h.ef_search:
                    log(f"  hnsw search ef_search={ef}")
                    db.set_search_mode(conn, hnsw_ef_search=ef)
                    plan, ids, fields = _measure(conn, approx_q, data, config)
                    recalls = [recall_at_k(t, a, k) for t, a in zip(truth, ids, strict=True)]
                    results.append(
                        SearchResult(
                            method="hnsw",
                            m=h.m,
                            ef_construction=h.ef_construction,
                            ef_search=ef,
                            index_build_seconds=build_s,
                            index_size_bytes=size,
                            recall_mean=sum(recalls) / len(recalls),
                            recall_min=min(recalls),
                            plan_verified=db.plan_uses_index(plan, index),
                            **fields,
                        )
                    )
                db.drop_index(conn, index)
            table_size = db.table_size_bytes(conn, table)
        finally:
            if not config.keep_table:
                db.drop_table(conn, table)

    return RunResult(
        started_at=started.isoformat(timespec="seconds"),
        finished_at=datetime.now(UTC).isoformat(timespec="seconds"),
        config=config.model_dump(mode="json"),
        dataset=data.metadata,
        environment=env,
        load_seconds=load_seconds,
        table_rows=rows,
        table_size_bytes=table_size,
        exact_crosscheck={"queries_checked": checked, "matching": matching},
        results=results,
    )
