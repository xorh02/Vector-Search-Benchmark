"""Benchmark configuration: TOML file -> validated Pydantic models."""

from __future__ import annotations

import os
import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

DEFAULT_DATABASE_URL = "postgresql://vsbench:vsbench@127.0.0.1:5433/vsbench"
DATABASE_URL_ENV = "VSBENCH_DATABASE_URL"

# Upper bound on generated floats (vectors + queries) to keep memory bounded (~1.6 GB float32).
MAX_TOTAL_FLOATS = 400_000_000
# pgvector cannot build an HNSW index on `vector` columns wider than this.
MAX_HNSW_DIMENSIONS = 2000


class Metric(StrEnum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"


class ConfigError(ValueError):
    """Raised when a configuration file is missing or invalid."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DatasetConfig(_Strict):
    size: int = Field(10_000, ge=1, description="number of stored vectors")
    dimensions: int = Field(128, ge=1, le=MAX_HNSW_DIMENSIONS)
    queries: int = Field(200, ge=1, description="number of query vectors")
    seed: int = Field(42, ge=0)
    # 0 = isotropic Gaussian; >0 = mixture of N(0, 1) centres plus N(0, cluster_std) noise.
    clusters: int = Field(0, ge=0, le=100_000)
    cluster_std: float = Field(0.5, gt=0, le=10)

    @model_validator(mode="after")
    def _bounded_memory(self) -> Self:
        if (self.size + self.queries) * self.dimensions > MAX_TOTAL_FLOATS:
            raise ValueError(
                f"(size + queries) * dimensions must be <= {MAX_TOTAL_FLOATS:,} "
                "to keep the generated dataset in memory"
            )
        return self


class HnswConfig(_Strict):
    m: int = Field(16, ge=2, le=100)
    ef_construction: int = Field(64, ge=4, le=1000)
    ef_search: list[int] = Field(default_factory=lambda: [40], min_length=1)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.ef_construction < 2 * self.m:
            raise ValueError("ef_construction must be >= 2 * m (pgvector requirement)")
        if any(not 1 <= ef <= 1000 for ef in self.ef_search):
            raise ValueError("every ef_search value must be between 1 and 1000")
        if len(set(self.ef_search)) != len(self.ef_search):
            raise ValueError("ef_search values must be unique")
        return self


class BenchmarkConfig(_Strict):
    name: str = Field("default", pattern=r"^[a-z0-9_]{1,40}$")
    metric: Metric = Metric.COSINE
    top_k: int = Field(10, ge=1, le=1000)
    warmup_queries: int = Field(20, ge=0)
    trials: int = Field(3, ge=1, le=100)
    maintenance_work_mem: str = Field("512MB", pattern=r"^[1-9][0-9]{0,6}(kB|MB|GB)$")
    keep_table: bool = False
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    hnsw: list[HnswConfig] = Field(default_factory=lambda: [HnswConfig()], min_length=1)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.top_k > self.dataset.size:
            raise ValueError("top_k must not exceed dataset.size")
        for h in self.hnsw:
            # pgvector's HNSW scan returns at most ef_search rows (iterative scans are off).
            if min(h.ef_search) < self.top_k:
                raise ValueError("every ef_search value must be >= top_k")
        if len({(h.m, h.ef_construction) for h in self.hnsw}) != len(self.hnsw):
            raise ValueError("each [[hnsw]] entry needs a unique (m, ef_construction) pair")
        return self

    @property
    def table_name(self) -> str:
        # Every object the tool creates carries this prefix; cleanup never touches anything else.
        return f"vsbench_{self.name}"


def load_config(path: Path) -> BenchmarkConfig:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise ConfigError(f"config file not found: {path}") from e
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: invalid TOML: {e}") from e
    try:
        return BenchmarkConfig.model_validate(data)
    except ValidationError as e:
        raise ConfigError(f"{path}: invalid configuration:\n{e}") from e


def database_url(override: str | None = None) -> str:
    return override or os.environ.get(DATABASE_URL_ENV) or DEFAULT_DATABASE_URL


EXAMPLE_CONFIG = """\
# Vector Search Bench configuration.
# The database URL is not stored here: set VSBENCH_DATABASE_URL or pass --database-url.

name = "default"          # table is created as vsbench_<name>; [a-z0-9_]
metric = "cosine"         # "cosine" (<=>) or "euclidean" (<->)
top_k = 10                # neighbours returned per query; Recall@k uses this k
warmup_queries = 20       # untimed queries before each measured configuration
trials = 3                # times the full query set is replayed per configuration
maintenance_work_mem = "512MB"   # session setting used while building HNSW indexes
keep_table = false        # drop the benchmark table when the run finishes

[dataset]
size = 20000              # stored vectors
dimensions = 128
queries = 200             # query vectors (generated separately from stored vectors)
seed = 42
clusters = 100            # 0 = structureless Gaussian; >0 = Gaussian mixture
cluster_std = 0.5         # per-dimension noise around each cluster centre

# One [[hnsw]] block per index build. Each ef_search value is measured on that index.
[[hnsw]]
m = 16
ef_construction = 64
ef_search = [10, 20, 40, 80, 160]
"""
