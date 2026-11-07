"""Database helpers that do not need a server."""

import numpy as np
import pytest

from vsbench import db
from vsbench.config import Metric

EXACT_PLAN = {
    "Node Type": "Limit",
    "Plans": [{"Node Type": "Sort", "Plans": [{"Node Type": "Seq Scan"}]}],
}
HNSW_PLAN = {"Node Type": "Limit", "Plans": [{"Node Type": "Index Scan", "Index Name": "t_hnsw"}]}


def test_plan_summary_and_checks():
    s = db.plan_summary(EXACT_PLAN)
    assert s == {"node_types": ["Limit", "Sort", "Seq Scan"], "indexes": []}
    assert db.plan_is_sequential(s) and not db.plan_uses_index(s, "t_hnsw")
    h = db.plan_summary(HNSW_PLAN)
    assert db.plan_uses_index(h, "t_hnsw") and not db.plan_is_sequential(h)
    assert not db.plan_uses_index(h, "other_index")


def test_vector_literal_roundtrips_float32():
    v = np.array([0.1, -2.5, 3e-8], dtype=np.float32)
    lit = db.vector_literal(v)
    assert lit.startswith("[") and lit.endswith("]")
    parsed = np.array([float(x) for x in lit[1:-1].split(",")], dtype=np.float32)
    assert np.array_equal(parsed, v)


def test_connect_error_is_actionable():
    with (
        pytest.raises(db.DatabaseError, match="docker compose up"),
        db.connect("postgresql://nobody:x@127.0.0.1:1/none"),
    ):
        pass


def test_drop_table_refuses_unprefixed_names():
    with pytest.raises(ValueError, match="refusing to modify"):
        db.drop_table(None, "users")  # type: ignore[arg-type]


def test_operators():
    assert db.OPERATORS[Metric.COSINE] == "<=>"
    assert db.OPERATORS[Metric.EUCLIDEAN] == "<->"
