from typer.testing import CliRunner

from vsbench import __version__
from vsbench.cli import app
from vsbench.report import write_json

runner = CliRunner()


def test_help():
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0
    for cmd in ("init", "check", "run", "report"):
        assert cmd in r.output


def test_version():
    r = runner.invoke(app, ["--version"])
    assert r.exit_code == 0 and __version__ in r.output


def test_init_writes_and_refuses_overwrite(tmp_path):
    p = tmp_path / "b.toml"
    assert runner.invoke(app, ["init", str(p)]).exit_code == 0
    assert "[[hnsw]]" in p.read_text()
    r = runner.invoke(app, ["init", str(p)])
    assert r.exit_code == 1 and "already exists" in r.output
    assert runner.invoke(app, ["init", str(p), "--force"]).exit_code == 0


def test_run_invalid_config(tmp_path):
    p = tmp_path / "b.toml"
    p.write_text("top_k = 0")
    r = runner.invoke(app, ["run", str(p)])
    assert r.exit_code == 1 and "invalid configuration" in r.output


def test_run_missing_config(tmp_path):
    r = runner.invoke(app, ["run", str(tmp_path / "missing.toml")])
    assert r.exit_code == 1 and "not found" in r.output


def test_run_unreachable_database(tmp_path):
    p = tmp_path / "b.toml"
    p.write_text("[dataset]\nsize = 20\nqueries = 2\ndimensions = 4\n")
    r = runner.invoke(
        app,
        ["run", str(p), "--database-url", "postgresql://x:y@127.0.0.1:1/z", "-o", str(tmp_path)],
    )
    assert r.exit_code == 1 and "cannot connect" in r.output
    assert not (tmp_path / "results.json").exists()


def test_check_unreachable_database():
    r = runner.invoke(app, ["check", "--database-url", "postgresql://x:y@127.0.0.1:1/z"])
    assert r.exit_code == 1


def test_report_regenerates(tmp_path, run_result):
    src = tmp_path / "results.json"
    write_json(run_result, src)
    out = tmp_path / "out"
    r = runner.invoke(app, ["report", str(src), "-o", str(out)])
    assert r.exit_code == 0, r.output
    assert (out / "report.html").exists() and (out / "results.csv").exists()


def test_report_bad_file(tmp_path):
    p = tmp_path / "x.json"
    p.write_text("[]")
    r = runner.invoke(app, ["report", str(p)])
    assert r.exit_code == 1


def test_run_queries_requires_vectors(tmp_path):
    r = runner.invoke(app, ["run", "--queries", str(tmp_path / "q.npy")])
    assert r.exit_code == 1 and "--queries requires --vectors" in r.output


def test_run_vectors_file_missing(tmp_path):
    cfg = tmp_path / "b.toml"
    cfg.write_text("")
    r = runner.invoke(app, ["run", str(cfg), "--vectors", str(tmp_path / "nope.npy")])
    assert r.exit_code == 1 and "not found" in r.output
