import tomllib

import pytest

from vsbench.config import (
    DEFAULT_DATABASE_URL,
    EXAMPLE_CONFIG,
    BenchmarkConfig,
    ConfigError,
    Metric,
    database_url,
    load_config,
)


def write(tmp_path, text):
    p = tmp_path / "c.toml"
    p.write_text(text)
    return p


def test_example_config_is_valid():
    cfg = BenchmarkConfig.model_validate(tomllib.loads(EXAMPLE_CONFIG))
    assert cfg.metric is Metric.COSINE
    assert cfg.hnsw[0].ef_search == [10, 20, 40, 80, 160]
    assert cfg.table_name == "vsbench_default"


def test_defaults_are_valid():
    cfg = BenchmarkConfig()
    assert cfg.top_k == 10 and cfg.dataset.seed == 42


def test_load_minimal(tmp_path):
    cfg = load_config(write(tmp_path, 'metric = "euclidean"\n[dataset]\nsize = 50\n'))
    assert cfg.metric is Metric.EUCLIDEAN and cfg.dataset.size == 50


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ('metric = "dot"', "metric"),
        ("unknown_key = 1", "unknown_key"),
        ('name = "Bad-Name; DROP"', "name"),
        ("top_k = 0", "top_k"),
        ("top_k = 20\n[dataset]\nsize = 10", "top_k must not exceed"),
        ("top_k = 50\n[[hnsw]]\nef_search = [10]", "ef_search value must be >= top_k"),
        ("[[hnsw]]\nm = 16\nef_construction = 20", "ef_construction must be >= 2"),
        ("[[hnsw]]\nef_search = [40, 40]", "unique"),
        ("[[hnsw]]\nef_search = [2000]", "between 1 and 1000"),
        ("[[hnsw]]\nm=8\n[[hnsw]]\nm=8", "unique \\(m, ef_construction\\)"),
        ("hnsw = []", "hnsw"),
        ('maintenance_work_mem = "1; DROP TABLE x"', "maintenance_work_mem"),
        ("trials = 0", "trials"),
    ],
)
def test_invalid_configs(tmp_path, text, match):
    with pytest.raises(ConfigError, match=match):
        load_config(write(tmp_path, text))


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")


def test_bad_toml(tmp_path):
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(write(tmp_path, "top_k = = 3"))


def test_database_url_precedence(monkeypatch):
    monkeypatch.delenv("VSBENCH_DATABASE_URL", raising=False)
    assert database_url() == DEFAULT_DATABASE_URL
    monkeypatch.setenv("VSBENCH_DATABASE_URL", "postgresql://env/db")
    assert database_url() == "postgresql://env/db"
    assert database_url("postgresql://cli/db") == "postgresql://cli/db"
