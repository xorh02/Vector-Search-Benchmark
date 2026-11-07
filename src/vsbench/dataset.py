"""Datasets: deterministic synthetic vectors, user-supplied arrays, and a NumPy reference search."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from vsbench.config import MAX_HNSW_DIMENSIONS, DatasetConfig, Metric

Matrix = NDArray[np.float32]
_CHUNK_ROWS = 65_536  # bounds the float64 working set of the reference search


@dataclass(frozen=True)
class Dataset:
    vectors: Matrix  # (n, dimensions); row i is stored with id i + 1
    queries: Matrix  # (q, dimensions); never stored in the table
    metadata: dict[str, Any]


def _fingerprint(vectors: Matrix, queries: Matrix) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(vectors).tobytes())
    h.update(np.ascontiguousarray(queries).tobytes())
    return h.hexdigest()


def _metadata(vectors: Matrix, queries: Matrix, source: str, seed: int | None) -> dict[str, Any]:
    return {
        "source": source,
        "size": int(vectors.shape[0]),
        "dimensions": int(vectors.shape[1]),
        "queries": int(queries.shape[0]),
        "seed": seed,
        "sha256": _fingerprint(vectors, queries),
        "numpy_version": np.__version__,
    }


def generate(config: DatasetConfig) -> Dataset:
    """Same config (including seed) -> bit-identical vectors and queries."""
    rng = np.random.default_rng(config.seed)
    n, q, d = config.size, config.queries, config.dimensions
    if config.clusters == 0:
        vectors = rng.standard_normal((n, d), dtype=np.float32)
        queries = rng.standard_normal((q, d), dtype=np.float32)
        source = "synthetic: isotropic Gaussian"
    else:
        # Queries come from the same mixture as the stored vectors but are separate draws.
        centres = rng.standard_normal((config.clusters, d), dtype=np.float32)
        points = centres[rng.integers(config.clusters, size=n + q)]
        points += np.float32(config.cluster_std) * rng.standard_normal((n + q, d), dtype=np.float32)
        vectors, queries = points[:n], points[n:]
        source = (
            f"synthetic: Gaussian mixture ({config.clusters} clusters, std {config.cluster_std})"
        )
    validate_vectors(vectors)
    validate_vectors(queries)
    return Dataset(vectors, queries, _metadata(vectors, queries, source, config.seed))


def from_arrays(
    vectors: ArrayLike,
    queries: ArrayLike | None = None,
    *,
    holdout: int = 200,
    seed: int = 42,
    source: str = "user-supplied arrays",
) -> Dataset:
    """Wrap your own embeddings. Values are converted to float32 (pgvector's storage type).

    Without `queries`, `holdout` rows are chosen at random (seeded) from `vectors`, removed
    from the stored set and used as queries, so no query is its own nearest neighbour.
    """
    v = np.asarray(vectors, dtype=np.float32)
    validate_vectors(v)
    if queries is not None:
        q = np.asarray(queries, dtype=np.float32)
        validate_vectors(q)
        if q.shape[1] != v.shape[1]:
            raise ValueError(f"queries have {q.shape[1]} dimensions, vectors have {v.shape[1]}")
        used_seed: int | None = None
    else:
        if not 1 <= holdout < len(v):
            raise ValueError(f"holdout must be between 1 and {len(v) - 1}, got {holdout}")
        order = np.random.default_rng(seed).permutation(len(v))
        q, v = v[np.sort(order[:holdout])], v[np.sort(order[holdout:])]
        used_seed = seed
    if v.shape[1] > MAX_HNSW_DIMENSIONS:
        raise ValueError(f"pgvector HNSW supports at most {MAX_HNSW_DIMENSIONS} dimensions")
    return Dataset(v, q, _metadata(v, q, source, used_seed))


def load_npy(path: Path) -> Matrix:
    """Load a 2-D numeric .npy file (pickles are refused)."""
    try:
        a = np.load(path, allow_pickle=False)
    except FileNotFoundError as e:
        raise ValueError(f"file not found: {path}") from e
    except ValueError as e:
        raise ValueError(f"{path}: not a plain numeric .npy array ({e})") from e
    if not isinstance(a, np.ndarray) or not np.issubdtype(a.dtype, np.number):
        raise ValueError(f"{path}: expected a numeric array")
    if a.ndim != 2:
        raise ValueError(f"{path}: expected shape (rows, dimensions), got {a.shape}")
    return a.astype(np.float32, copy=False)


def validate_vectors(a: NDArray[Any], metric: Metric | None = None) -> None:
    if a.ndim != 2 or a.shape[0] == 0 or a.shape[1] == 0:
        raise ValueError(f"expected a non-empty 2-D array, got shape {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError("vectors contain NaN or infinite values")
    # Cosine distance is undefined for zero vectors (pgvector returns NaN).
    if metric is Metric.COSINE and (np.linalg.norm(a, axis=1) == 0).any():
        raise ValueError("cosine metric requires non-zero vectors")


def distances(vectors: NDArray[Any], query: NDArray[Any], metric: Metric) -> NDArray[np.float64]:
    """Distances from `query` to every row, computed in float64 one chunk at a time."""
    q = query.astype(np.float64)
    out = np.empty(len(vectors), dtype=np.float64)
    for start in range(0, len(vectors), _CHUNK_ROWS):
        v = vectors[start : start + _CHUNK_ROWS].astype(np.float64)
        if metric is Metric.EUCLIDEAN:
            d = np.linalg.norm(v - q, axis=1)
        else:
            d = 1.0 - (v @ q) / (np.linalg.norm(v, axis=1) * np.linalg.norm(q))
        out[start : start + len(v)] = d
    return out


def brute_force_knn(
    vectors: NDArray[Any], query: NDArray[Any], k: int, metric: Metric
) -> list[int]:
    """Exact top-k ids (1-based). Ties are broken by lowest id, matching the SQL baseline."""
    if not 1 <= k <= len(vectors):
        raise ValueError(f"k must be between 1 and {len(vectors)}, got {k}")
    validate_vectors(query.reshape(1, -1), metric)
    d = distances(vectors, query, metric)
    order = np.lexsort((np.arange(len(d)), d))  # primary key: distance, secondary: index
    return [int(i) + 1 for i in order[:k]]
