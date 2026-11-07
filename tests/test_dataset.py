import numpy as np
import pytest
from pydantic import ValidationError

from vsbench import dataset as dataset_module
from vsbench.config import DatasetConfig, Metric
from vsbench.dataset import (
    brute_force_knn,
    distances,
    from_arrays,
    generate,
    load_npy,
    validate_vectors,
)


def test_same_seed_is_bit_identical():
    cfg = DatasetConfig(size=100, dimensions=8, queries=5, seed=7)
    a, b = generate(cfg), generate(cfg)
    assert np.array_equal(a.vectors, b.vectors)
    assert np.array_equal(a.queries, b.queries)
    assert a.vectors.dtype == np.float32
    assert a.vectors.shape == (100, 8) and a.queries.shape == (5, 8)


def test_clustered_dataset_is_reproducible_and_clustered():
    cfg = DatasetConfig(size=400, dimensions=16, queries=10, seed=4, clusters=4, cluster_std=0.1)
    a, b = generate(cfg), generate(cfg)
    assert np.array_equal(a.vectors, b.vectors) and np.array_equal(a.queries, b.queries)
    assert a.vectors.dtype == np.float32 and a.vectors.shape == (400, 16)
    # tight clusters: every point lies close to some other point
    gaps = [np.sort(np.linalg.norm(a.vectors - v, axis=1))[1] for v in a.vectors[:20]]
    assert max(gaps) < 1.0 < float(np.linalg.norm(a.vectors.std(axis=0)))
    assert "mixture" in a.metadata["source"]


def test_different_seed_differs():
    a = generate(DatasetConfig(size=50, dimensions=4, queries=2, seed=1))
    b = generate(DatasetConfig(size=50, dimensions=4, queries=2, seed=2))
    assert not np.array_equal(a.vectors, b.vectors)


def test_queries_are_not_copies_of_vectors():
    d = generate(DatasetConfig(size=50, dimensions=4, queries=5, seed=3))
    assert not any((d.vectors == q).all(axis=1).any() for q in d.queries)


def test_metadata_records_config():
    md = generate(DatasetConfig(size=10, dimensions=3, queries=1, seed=9)).metadata
    assert md["size"] == 10 and md["dimensions"] == 3 and md["seed"] == 9
    assert "numpy_version" in md and len(md["sha256"]) == 64


@pytest.mark.parametrize(
    "kwargs",
    [
        {"size": 0},
        {"dimensions": 0},
        {"dimensions": 2001},
        {"queries": 0},
        {"seed": -1},
        {"size": 10_000_000, "dimensions": 2000},
    ],
)
def test_invalid_dataset_config(kwargs):
    with pytest.raises(ValidationError):
        DatasetConfig(**kwargs)


def test_rejects_non_finite():
    with pytest.raises(ValueError, match="NaN or infinite"):
        validate_vectors(np.array([[1.0, np.nan]]))
    with pytest.raises(ValueError, match="NaN or infinite"):
        validate_vectors(np.array([[np.inf, 1.0]]))


def test_rejects_bad_shape():
    with pytest.raises(ValueError, match="2-D"):
        validate_vectors(np.array([1.0, 2.0]))


def test_cosine_rejects_zero_vector():
    with pytest.raises(ValueError, match="non-zero"):
        validate_vectors(np.array([[0.0, 0.0], [1.0, 0.0]]), Metric.COSINE)
    validate_vectors(np.array([[0.0, 0.0]]), Metric.EUCLIDEAN)  # fine for L2


def test_distances_known_values():
    v = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [3.0, 4.0]])
    q = np.array([1.0, 0.0])
    assert np.allclose(distances(v, q, Metric.COSINE), [0.0, 1.0, 2.0, 0.4])
    assert np.allclose(distances(v, q, Metric.EUCLIDEAN), [0.0, np.sqrt(2), 2.0, np.sqrt(20)])


def test_brute_force_knn_known_answer():
    v = np.array([[0.0, 0.0], [5.0, 5.0], [1.0, 0.0], [0.0, 2.0]])
    assert brute_force_knn(v, np.array([0.0, 0.0]), 3, Metric.EUCLIDEAN) == [1, 3, 4]


def test_ties_and_duplicates_break_by_lowest_id():
    # ids 2 and 4 are duplicates; ids 1 and 3 are equidistant from the query.
    v = np.array([[1.0, 0.0], [5.0, 5.0], [-1.0, 0.0], [5.0, 5.0]])
    q = np.array([0.0, 0.0])
    assert brute_force_knn(v, q, 4, Metric.EUCLIDEAN) == [1, 3, 2, 4]
    # cosine: duplicates of the same direction tie at distance 0
    v2 = np.array([[2.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    assert brute_force_knn(v2, np.array([1.0, 0.0]), 2, Metric.COSINE) == [1, 3]


@pytest.mark.parametrize("k", [0, -1, 5])
def test_invalid_k(k):
    v = np.ones((4, 2))
    with pytest.raises(ValueError, match="k must be"):
        brute_force_knn(v, np.ones(2), k, Metric.EUCLIDEAN)


def test_from_arrays_with_queries():
    v = np.arange(12, dtype=np.float64).reshape(4, 3) + 1
    q = np.ones((2, 3))
    d = from_arrays(v, q)
    assert d.vectors.dtype == np.float32 and d.vectors.shape == (4, 3)
    assert d.queries.shape == (2, 3)
    assert d.metadata["seed"] is None and d.metadata["source"] == "user-supplied arrays"


def test_from_arrays_holdout_is_disjoint_and_deterministic():
    v = np.random.default_rng(0).standard_normal((50, 4))
    a, b = from_arrays(v, holdout=10, seed=1), from_arrays(v, holdout=10, seed=1)
    assert a.vectors.shape == (40, 4) and a.queries.shape == (10, 4)
    assert np.array_equal(a.queries, b.queries)
    assert a.metadata["sha256"] == b.metadata["sha256"]
    stored = {r.tobytes() for r in a.vectors}
    assert not any(r.tobytes() in stored for r in a.queries)
    assert from_arrays(v, holdout=10, seed=2).metadata["sha256"] != a.metadata["sha256"]


@pytest.mark.parametrize(
    ("vectors", "queries", "kwargs", "match"),
    [
        (np.ones((5, 3)), np.ones((2, 4)), {}, "dimensions"),
        (np.ones((5, 3)), None, {"holdout": 5}, "holdout"),
        (np.ones((5, 3)), None, {"holdout": 0}, "holdout"),
        (np.ones((5, 2001)), np.ones((1, 2001)), {}, "at most 2000"),
        (np.array([[1.0, np.nan]]), None, {}, "NaN"),
    ],
)
def test_from_arrays_rejects_invalid(vectors, queries, kwargs, match):
    with pytest.raises(ValueError, match=match):
        from_arrays(vectors, queries, **kwargs)


def test_load_npy(tmp_path):
    p = tmp_path / "v.npy"
    np.save(p, np.ones((3, 2), dtype=np.float64))
    a = load_npy(p)
    assert a.dtype == np.float32 and a.shape == (3, 2)
    with pytest.raises(ValueError, match="not found"):
        load_npy(tmp_path / "missing.npy")
    np.save(p, np.ones(3))
    with pytest.raises(ValueError, match="shape"):
        load_npy(p)
    np.save(p, np.array([{"a": 1}], dtype=object), allow_pickle=True)
    with pytest.raises(ValueError, match="numeric"):
        load_npy(p)


def test_chunked_distances_match_unchunked(monkeypatch):
    rng = np.random.default_rng(3)
    v, q = rng.standard_normal((100, 8)), rng.standard_normal(8)
    full = {m: distances(v, q, m) for m in Metric}
    monkeypatch.setattr(dataset_module, "_CHUNK_ROWS", 7)
    for m in Metric:
        assert np.allclose(distances(v, q, m), full[m])
