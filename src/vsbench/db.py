"""PostgreSQL / pgvector access. All SQL lives here."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import psycopg
from numpy.typing import NDArray
from psycopg import sql

from vsbench.config import Metric

OPERATORS = {Metric.COSINE: "<=>", Metric.EUCLIDEAN: "<->"}
OPCLASSES = {Metric.COSINE: "vector_cosine_ops", Metric.EUCLIDEAN: "vector_l2_ops"}


class DatabaseError(RuntimeError):
    """Actionable database failure (connection, missing extension, ...)."""


@contextmanager
def connect(url: str) -> Iterator[psycopg.Connection[Any]]:
    try:
        conn = psycopg.connect(url, autocommit=True, connect_timeout=5)
    except psycopg.OperationalError as e:
        raise DatabaseError(
            f"cannot connect to PostgreSQL: {str(e).strip()}\n"
            "Start the bundled database with `docker compose up -d --wait`, or set "
            "VSBENCH_DATABASE_URL / --database-url to your server."
        ) from e
    try:
        yield conn
    finally:
        conn.close()


def ensure_pgvector(conn: psycopg.Connection[Any]) -> None:
    row = conn.execute("SELECT 1 FROM pg_available_extensions WHERE name = 'vector'").fetchone()
    if row is None:
        raise DatabaseError(
            "the pgvector extension is not installed on this server "
            "(use the pgvector/pgvector Docker image or install pgvector)"
        )
    try:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    except psycopg.errors.InsufficientPrivilege as e:
        raise DatabaseError(
            "pgvector is available but not enabled, and this user may not run "
            "CREATE EXTENSION vector; ask a superuser to enable it"
        ) from e


def server_info(conn: psycopg.Connection[Any]) -> dict[str, Any]:
    row = conn.execute(
        """SELECT version(), current_setting('server_version'),
                  (SELECT extversion FROM pg_extension WHERE extname = 'vector'),
                  current_setting('shared_buffers'), current_setting('work_mem'),
                  current_setting('max_parallel_workers_per_gather'),
                  current_setting('max_parallel_maintenance_workers')"""
    ).fetchone()
    assert row is not None
    keys = [
        "postgres_version_full",
        "postgres_version",
        "pgvector_version",
        "shared_buffers",
        "work_mem",
        "max_parallel_workers_per_gather",
        "max_parallel_maintenance_workers",
    ]
    return dict(zip(keys, row, strict=True))


def vector_literal(v: NDArray[Any]) -> str:
    # str() of a float32 widened to Python float round-trips exactly back to float32.
    return "[" + ",".join(str(x) for x in v.tolist()) + "]"


def _require_benchmark_table(table: str) -> None:
    # Everything this tool creates or drops is named vsbench_*; never touch other tables.
    if not table.startswith("vsbench_"):
        raise ValueError(f"refusing to modify non-benchmark table {table!r}")


def create_table(conn: psycopg.Connection[Any], table: str, dimensions: int) -> None:
    _require_benchmark_table(table)
    conn.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(table)))
    conn.execute(
        sql.SQL("CREATE TABLE {} (id integer PRIMARY KEY, embedding vector({}) NOT NULL)").format(
            sql.Identifier(table), sql.Literal(dimensions)
        )
    )


def load_vectors(conn: psycopg.Connection[Any], table: str, vectors: NDArray[Any]) -> float:
    """Bulk-load with COPY; ids are 1-based row numbers. Returns seconds taken."""
    start = time.perf_counter_ns()
    with conn.cursor() as cur:
        with cur.copy(
            sql.SQL("COPY {} (id, embedding) FROM STDIN").format(sql.Identifier(table))
        ) as copy:
            for i, v in enumerate(vectors, start=1):
                copy.write_row((i, vector_literal(v)))
        cur.execute(sql.SQL("VACUUM (ANALYZE) {}").format(sql.Identifier(table)))
    return (time.perf_counter_ns() - start) / 1e9


def count_rows(conn: psycopg.Connection[Any], table: str) -> int:
    row = conn.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table))).fetchone()
    assert row is not None
    return int(row[0])


def table_size_bytes(conn: psycopg.Connection[Any], table: str) -> int:
    row = conn.execute("SELECT pg_table_size(%s::regclass)", (table,)).fetchone()
    assert row is not None
    return int(row[0])


def index_size_bytes(conn: psycopg.Connection[Any], index: str) -> int:
    row = conn.execute("SELECT pg_relation_size(%s::regclass)", (index,)).fetchone()
    assert row is not None
    return int(row[0])


def create_hnsw_index(
    conn: psycopg.Connection[Any],
    table: str,
    index: str,
    metric: Metric,
    m: int,
    ef_construction: int,
    maintenance_work_mem: str,
) -> float:
    """Build the HNSW index and return client-observed build time in seconds."""
    conn.execute(sql.SQL("SET maintenance_work_mem = {}").format(sql.Literal(maintenance_work_mem)))
    stmt = sql.SQL(
        "CREATE INDEX {} ON {} USING hnsw (embedding {}) WITH (m = {}, ef_construction = {})"
    ).format(
        sql.Identifier(index),
        sql.Identifier(table),
        sql.SQL(OPCLASSES[metric]),
        sql.Literal(m),
        sql.Literal(ef_construction),
    )
    start = time.perf_counter_ns()
    conn.execute(stmt)
    elapsed = (time.perf_counter_ns() - start) / 1e9
    conn.execute("RESET maintenance_work_mem")
    return elapsed


def index_is_valid(conn: psycopg.Connection[Any], index: str) -> bool:
    row = conn.execute(
        """SELECT i.indisvalid AND am.amname = 'hnsw'
           FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid
           JOIN pg_am am ON am.oid = c.relam
           WHERE i.indexrelid = %s::regclass""",
        (index,),
    ).fetchone()
    return bool(row and row[0])


def drop_index(conn: psycopg.Connection[Any], index: str) -> None:
    conn.execute(sql.SQL("DROP INDEX IF EXISTS {}").format(sql.Identifier(index)))


def drop_table(conn: psycopg.Connection[Any], table: str) -> None:
    _require_benchmark_table(table)
    conn.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(table)))


def set_search_mode(conn: psycopg.Connection[Any], *, hnsw_ef_search: int | None) -> None:
    """Exact mode: index scans disabled. HNSW mode: sequential scans disabled, ef_search set."""
    if hnsw_ef_search is None:
        conn.execute("SET enable_indexscan = off")
        conn.execute("SET enable_bitmapscan = off")
        conn.execute("RESET enable_seqscan")
    else:
        conn.execute("RESET enable_indexscan")
        conn.execute("RESET enable_bitmapscan")
        # Small tables can make a seq scan look cheaper; force the index so we measure it.
        conn.execute("SET enable_seqscan = off")
        conn.execute(sql.SQL("SET hnsw.ef_search = {}").format(sql.Literal(hnsw_ef_search)))


def knn_query(table: str, metric: Metric, *, exact: bool) -> sql.Composed:
    # The exact query adds `id` as a tie-breaker; that extra sort key would prevent
    # pgvector from using the HNSW index, so the approximate query orders by distance only.
    order = sql.SQL("embedding {} %s::vector").format(sql.SQL(OPERATORS[metric]))
    if exact:
        order = sql.SQL("{}, id").format(order)
    return sql.SQL("SELECT id FROM {} ORDER BY {} LIMIT %s").format(sql.Identifier(table), order)


def explain(
    conn: psycopg.Connection[Any], query: sql.Composed, params: tuple[Any, ...]
) -> dict[str, Any]:
    row = conn.execute(sql.SQL("EXPLAIN (FORMAT JSON) {}").format(query), params).fetchone()
    assert row is not None
    plan: dict[str, Any] = row[0][0]["Plan"]
    return plan


def server_execution_ms(
    conn: psycopg.Connection[Any], query: sql.Composed, params: tuple[Any, ...]
) -> float:
    """Server-side execution time from EXPLAIN ANALYZE (per-node timing off to limit overhead)."""
    row = conn.execute(
        sql.SQL("EXPLAIN (ANALYZE, TIMING OFF, FORMAT JSON) {}").format(query), params
    ).fetchone()
    assert row is not None
    return float(row[0][0]["Execution Time"])


def plan_summary(plan: dict[str, Any]) -> dict[str, Any]:
    """Flatten a JSON plan into node types and the indexes it scans."""
    nodes: list[str] = []
    indexes: list[str] = []

    def walk(node: dict[str, Any]) -> None:
        nodes.append(node["Node Type"])
        if "Index Name" in node:
            indexes.append(node["Index Name"])
        for child in node.get("Plans", []):
            walk(child)

    walk(plan)
    return {"node_types": nodes, "indexes": indexes}


def plan_uses_index(summary: dict[str, Any], index: str) -> bool:
    return index in summary["indexes"]


def plan_is_sequential(summary: dict[str, Any]) -> bool:
    return "Seq Scan" in summary["node_types"] and not summary["indexes"]
