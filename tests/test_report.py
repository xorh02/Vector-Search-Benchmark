import csv
import json

import pytest

from vsbench.report import (
    CSV_FIELDS,
    latency_chart,
    load_results,
    recall_chart,
    render_html,
    write_csv,
    write_json,
    write_outputs,
)


def test_json_roundtrip(tmp_path, run_result):
    p = tmp_path / "r.json"
    write_json(run_result, p)
    assert load_results(p) == run_result
    data = json.loads(p.read_text())
    assert data["schema_version"] == 1
    assert data["results"][1]["latency"]["median_ms"] == pytest.approx(0.6)


def test_load_results_rejects_bad_files(tmp_path):
    p = tmp_path / "r.json"
    with pytest.raises(ValueError, match="not found"):
        load_results(p)
    p.write_text("{nope")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_results(p)
    p.write_text('{"schema_version": 99}')
    with pytest.raises(ValueError, match="unsupported schema_version"):
        load_results(p)
    p.write_text('{"schema_version": 1}')
    with pytest.raises(ValueError, match="not a vsbench results file"):
        load_results(p)


def test_csv(tmp_path, run_result):
    p = tmp_path / "r.csv"
    write_csv(run_result, p)
    rows = list(csv.DictReader(p.open()))
    assert list(rows[0]) == CSV_FIELDS
    assert [r["method"] for r in rows] == ["exact", "hnsw", "hnsw"]
    assert rows[0]["ef_search"] == "" and rows[2]["ef_search"] == "40"
    assert float(rows[2]["recall_mean"]) == 0.97
    assert float(rows[1]["latency_median_ms"]) == 0.6
    assert rows[1]["index_size_bytes"] == "8388608"


def test_html_contains_measured_values(run_result):
    html = render_html(run_result)
    assert "<script" not in html  # self-contained, no external resources
    assert "http" not in html.split("<main>")[0]
    assert "13.000" in html  # exact median
    assert "6.000" in html  # exact server median
    assert "0.9700" in html  # recall
    assert "8.0 MiB" in html  # index size
    assert "PostgreSQL 17" in html and "0.8.0" in html
    assert "Recall vs latency" in html
    assert "Query-plan verification" in html


def test_html_flags_unverified_plan(run_result):
    run_result.results[1].plan_verified = False
    html = render_html(run_result)
    assert "Query-plan check failed" in html and "unexpected plan" in html


def test_html_empty_results(run_result):
    run_result.results = []
    html = render_html(run_result)
    assert "contains no measurements" in html


def test_charts(run_result):
    assert latency_chart([]) == ""
    assert "<rect" in latency_chart(run_result.results)
    assert recall_chart(run_result.results[:2]) == ""  # needs >= 2 HNSW points
    svg = recall_chart(run_result.results)
    assert svg.count("<circle") == 2 and "efs=40" in svg


def test_write_outputs(tmp_path, run_result):
    paths = write_outputs(run_result, tmp_path / "out")
    assert all(p.exists() for p in paths.values())


def test_recall_chart_colours_each_index_build(run_result):
    second = [
        r.model_copy(update={"m": 32, "ef_construction": 128}) for r in run_result.results[1:]
    ]
    results = run_result.results + second
    svg = recall_chart(results)
    assert 'class="line s1"' in svg and 'class="line s3"' in svg
    run_result.results = results
    html = render_html(run_result)
    assert "HNSW m=16, ef_construction=64" in html and "HNSW m=32, ef_construction=128" in html


def test_recall_chart_moves_slow_exact_off_scale(run_result):
    # fixture: exact median 13 ms vs HNSW <= 1 ms -> exact shown as a note, not a point
    svg = recall_chart(run_result.results)
    assert "off scale" in svg and 'class="dot s2"' not in svg
    run_result.results[0].latency.median_ms = 1.5
    assert 'class="dot s2"' in recall_chart(run_result.results)
