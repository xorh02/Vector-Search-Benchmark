"""Vector Search Bench: exact vs HNSW search benchmarks for PostgreSQL pgvector.

Public API::

    import vsbench

    config = vsbench.BenchmarkConfig(top_k=10, hnsw=[vsbench.HnswConfig(ef_search=[20, 40, 80])])
    data = vsbench.from_arrays(my_embeddings, holdout=200)        # or omit for synthetic data
    result = vsbench.run(config, "postgresql://...", data=data)
    vsbench.write_outputs(result, "results/")                     # report.html, .json, .csv
"""

__version__ = "0.1.0"

from vsbench.benchmark import RunResult, SearchResult, run
from vsbench.config import (
    BenchmarkConfig,
    ConfigError,
    DatasetConfig,
    HnswConfig,
    Metric,
    load_config,
)
from vsbench.dataset import Dataset, from_arrays, generate, load_npy
from vsbench.db import DatabaseError
from vsbench.metrics import LatencyStats, recall_at_k
from vsbench.report import load_results, write_outputs

__all__ = [
    "BenchmarkConfig",
    "ConfigError",
    "DatabaseError",
    "Dataset",
    "DatasetConfig",
    "HnswConfig",
    "LatencyStats",
    "Metric",
    "RunResult",
    "SearchResult",
    "__version__",
    "from_arrays",
    "generate",
    "load_config",
    "load_npy",
    "load_results",
    "recall_at_k",
    "run",
    "write_outputs",
]
