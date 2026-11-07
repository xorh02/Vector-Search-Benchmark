"""Real PostgreSQL + pgvector tests; they run only when VSBENCH_DATABASE_URL is set."""

import numpy as np
import pytest
from typer.testing import CliRunner

import vsbench
from vsbench import benchmark, db
from vsbench.cli import app
from vsbench.config import BenchmarkConfig, DatasetConfig, Metric
from vsbench.dataset import brute_force_knn, generate

pytestmark = pytest.mark.integration
TABLE = "vsbench_pytest"


@pytest.fixture
def conn(database_url):
    with db.connect(database_url) as c:
        db.ensure_pgvector(c)
        yield c
        db.drop_table(c, TABLE)


def knn(conn, metric, q, k, *, exact):
    return [r[0] for r in conn.execute(db.knn_query(TABLE, metric, exact=exact),
                                       (db.vector_literal(q), k)).fetchall()]  # fmt: skip


def test_server_info(conn):
    info = db.server_info(conn)
    assert info["pgvector_version"]
    assert int(info["postgres_version"].split(".")[0]) >= 13


def test_bulk_insert_roundtrips_vectors(conn):
    data = generate(DatasetConfig(size=300, dimensions=16, queries=1, seed=1))
    db.create_table(conn, TABLE, 16)
    db.load_vectors(conn, TABLE, data.vectors)
    assert db.count_rows(conn, TABLE) == 300
    row = conn.execute(f"SELECT embedding::text FROM {TABLE} WHERE id = 7").fetchone()
    stored = np.array([float(x) for x in row[0][1:-1].split(",")], dtype=np.float32)
    assert np.array_equal(stored, data.vectors[6])
    assert db.table_size_bytes(conn, TABLE) > 0


@pytest.mark.parametrize("metric", list(Metric))
def test_exact_sql_matches_numpy(conn, metric):
    data = generate(DatasetConfig(size=500, dimensions=8, queries=20, seed=5))
    db.create_table(conn, TABLE, 8)
    db.load_vectors(conn, TABLE, data.vectors)
    db.set_search_mode(conn, hnsw_ef_search=None)
    for q in data.queries:
        assert knn(conn, metric, q, 10, exact=True) == brute_force_knn(data.vectors, q, 10, metric)


def test_exact_sql_breaks_ties_by_id(conn):
    # ids 2 and 4 are duplicates; 1 and 3 are equidistant from the origin
    vecs = np.array([[1, 0], [5, 5], [-1, 0], [5, 5]], dtype=np.float32)
    db.create_table(conn, TABLE, 2)
    db.load_vectors(conn, TABLE, vecs)
    db.set_search_mode(conn, hnsw_ef_search=None)
    q = np.array([0, 0], dtype=np.float32)
    assert knn(conn, Metric.EUCLIDEAN, q, 4, exact=True) == [1, 3, 2, 4]


@pytest.mark.parametrize("metric", list(Metric))
def test_hnsw_index_and_plans(conn, metric):
    data = generate(DatasetConfig(size=1000, dimensions=16, queries=20, seed=3))
    db.create_table(conn, TABLE, 16)
    db.load_vectors(conn, TABLE, data.vectors)
    idx = f"{TABLE}_hnsw"
    q = data.queries[0]
    params = (db.vector_literal(q), 10)

    db.set_search_mode(conn, hnsw_ef_search=None)
    exact_plan = db.plan_summary(db.explain(conn, db.knn_query(TABLE, metric, exact=True), params))
    assert db.plan_is_sequential(exact_plan)

    seconds = db.create_hnsw_index(conn, TABLE, idx, metric, 8, 32, "64MB")
    assert seconds > 0
    assert db.index_is_valid(conn, idx)
    assert db.index_size_bytes(conn, idx) > 0

    # exact mode still avoids the index once it exists
    db.set_search_mode(conn, hnsw_ef_search=None)
    plan = db.plan_summary(db.explain(conn, db.knn_query(TABLE, metric, exact=True), params))
    assert not db.plan_uses_index(plan, idx)

    db.set_search_mode(conn, hnsw_ef_search=100)
    approx_q = db.knn_query(TABLE, metric, exact=False)
    assert db.plan_uses_index(db.plan_summary(db.explain(conn, approx_q, params)), idx)
    got = knn(conn, metric, q, 10, exact=False)
    assert len(got) == 10
    # high ef_search on a tiny index should recover nearly all true neighbours
    truth = brute_force_knn(data.vectors, q, 10, metric)
    assert len(set(got) & set(truth)) >= 8

    db.drop_index(conn, idx)
    assert conn.execute("SELECT to_regclass(%s)", (idx,)).fetchone()[0] is None


def test_full_benchmark_run(database_url):
    cfg = BenchmarkConfig.model_validate({
        "name": "pytest_run", "top_k": 5, "warmup_queries": 2, "trials": 2,
        "dataset": {"size": 800, "dimensions": 12, "queries": 15, "seed": 11},
        "hnsw": [{"m": 8, "ef_construction": 32, "ef_search": [5, 64]}],
    })  # fmt: skip
    result = benchmark.run(cfg, database_url)
    assert [r.method for r in result.results] == ["exact", "hnsw", "hnsw"]
    assert all(r.plan_verified for r in result.results)
    assert result.exact_crosscheck == {"queries_checked": 10, "matching": 10}
    assert result.table_rows == 800
    assert result.results[0].latency.samples == 30
    assert 0 <= result.results[1].recall_mean <= result.results[2].recall_mean <= 1
    assert result.results[2].index_size_bytes and result.results[2].index_build_seconds
    for r in result.results:
        assert r.server_samples == 15 and 0 < r.server_median_ms < r.latency.max_ms
    with db.connect(database_url) as c:  # table cleaned up
        assert c.execute("SELECT to_regclass('vsbench_pytest_run')").fetchone()[0] is None


def test_run_with_user_arrays(database_url):
    rng = np.random.default_rng(0)
    embeddings = rng.standard_normal((600, 24)).astype(np.float64)  # dtype converted to float32
    data = vsbench.from_arrays(embeddings, holdout=12, seed=3)
    cfg = vsbench.BenchmarkConfig(
        name="pytest_arrays",
        metric="euclidean",
        top_k=5,
        warmup_queries=0,
        trials=1,
        hnsw=[vsbench.HnswConfig(m=8, ef_construction=32, ef_search=[40])],
    )
    result = vsbench.run(cfg, database_url, data=data)
    assert result.table_rows == 588 and result.dataset["queries"] == 12
    assert result.dataset["source"] == "user-supplied arrays"
    assert result.exact_crosscheck["matching"] == 10
    assert all(r.plan_verified for r in result.results)


def test_cli_run_with_npy_files(database_url, tmp_path):
    rng = np.random.default_rng(1)
    np.save(tmp_path / "v.npy", rng.standard_normal((400, 8)).astype(np.float32))
    np.save(tmp_path / "q.npy", rng.standard_normal((7, 8)).astype(np.float32))
    cfg = tmp_path / "c.toml"
    cfg.write_text(
        """
name = "pytest_cli"
top_k = 3
trials = 1
[[hnsw]]
m = 8
ef_construction = 32
ef_search = [16]
"""
    )
    out = tmp_path / "out"
    r = CliRunner().invoke(
        app,
        ["run", str(cfg), "--vectors", str(tmp_path / "v.npy"), "--queries",
         str(tmp_path / "q.npy"), "-o", str(out), "--database-url", database_url],
    )  # fmt: skip
    assert r.exit_code == 0, r.output
    loaded = vsbench.load_results(out / "results.json")
    assert loaded.dataset["source"] == "file: v.npy + q.npy"
    assert loaded.table_rows == 400 and loaded.results[0].latency.samples == 7
    assert "file: v.npy" in (out / "report.html").read_text(encoding="utf-8")


def test_example_choose_ef_search(database_url, tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).parents[1] / "examples" / "choose_ef_search.py"
    out = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, str(script), "--size", "3000", "--target", "0.9",
         "--out", str(out), "--database-url", database_url],
        capture_output=True, text=True, timeout=300, cwd=tmp_path,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "recommendation: SET hnsw.ef_search =" in proc.stdout
    assert (out / "report.html").exists()
