"""Command-line interface: `vsbench init | check | run | report`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import psycopg
import typer

from vsbench import __version__, benchmark, db, report
from vsbench.config import EXAMPLE_CONFIG, ConfigError, database_url, load_config
from vsbench.dataset import from_arrays, load_npy

app = typer.Typer(
    help="Benchmark exact vs HNSW nearest-neighbour search in PostgreSQL pgvector.",
    no_args_is_help=True,
    add_completion=False,
)

DbUrl = Annotated[
    str | None,
    typer.Option(
        "--database-url",
        help="PostgreSQL URL. Defaults to $VSBENCH_DATABASE_URL, then the bundled Docker database.",
        show_default=False,
    ),
]


def _fail(msg: str) -> typer.Exit:
    typer.secho(f"error: {msg}", fg=typer.colors.RED, err=True)
    return typer.Exit(code=1)


def _version(value: bool) -> None:
    if value:
        typer.echo(f"vsbench {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Show version.")
    ] = False,
) -> None:
    pass


@app.command()
def init(
    path: Annotated[Path, typer.Argument(help="Where to write the config.")] = Path("vsbench.toml"),
    force: Annotated[bool, typer.Option("--force", help="Overwrite an existing file.")] = False,
) -> None:
    """Write an example benchmark configuration."""
    if path.exists() and not force:
        raise _fail(f"{path} already exists (use --force to overwrite)")
    path.write_text(EXAMPLE_CONFIG, encoding="utf-8")
    typer.echo(f"wrote {path}")


@app.command()
def check(url: DbUrl = None) -> None:
    """Check the database connection and pgvector availability."""
    try:
        with db.connect(database_url(url)) as conn:
            db.ensure_pgvector(conn)
            info = db.server_info(conn)
    except db.DatabaseError as e:
        raise _fail(str(e)) from None
    typer.echo(f"PostgreSQL {info['postgres_version']}, pgvector {info['pgvector_version']}: OK")


@app.command()
def run(
    config: Annotated[Path, typer.Argument(help="Benchmark config (TOML).")] = Path("vsbench.toml"),
    out: Annotated[Path, typer.Option("--out", "-o", help="Output directory.")] = Path("results"),
    url: DbUrl = None,
    vectors: Annotated[
        Path | None,
        typer.Option(
            "--vectors",
            help="Benchmark your own embeddings from a 2-D .npy file instead of synthetic data.",
        ),
    ] = None,
    queries: Annotated[
        Path | None,
        typer.Option(
            "--queries",
            help="Query vectors (.npy). Without it, [dataset].queries rows are held out of "
            "--vectors using [dataset].seed.",
        ),
    ] = None,
) -> None:
    """Run a benchmark and write results.json, results.csv and report.html."""
    if queries is not None and vectors is None:
        raise _fail("--queries requires --vectors")
    try:
        cfg = load_config(config)
        data = None
        if vectors is not None:
            data = from_arrays(
                load_npy(vectors),
                None if queries is None else load_npy(queries),
                holdout=cfg.dataset.queries,
                seed=cfg.dataset.seed,
                source=f"file: {vectors.name}" + (f" + {queries.name}" if queries else ""),
            )
        result = benchmark.run(
            cfg, database_url(url), data=data, log=lambda m: typer.echo(m, err=True)
        )
    except (ConfigError, db.DatabaseError, ValueError) as e:
        raise _fail(str(e)) from None
    except psycopg.Error as e:
        raise _fail(f"PostgreSQL error: {str(e).strip()}") from None
    paths = report.write_outputs(result, out)
    for r in result.results:
        typer.echo(
            f"{r.label:<36} recall@{cfg.top_k}={r.recall_mean:.4f}  "
            f"median={r.latency.median_ms:.3f} ms  p95={r.latency.p95_ms:.3f} ms  "
            f"server={r.server_median_ms:.3f} ms"
            + ("" if r.plan_verified else "  [PLAN NOT VERIFIED]")
        )
    for kind, p in paths.items():
        typer.echo(f"{kind}: {p}")
    if not all(r.plan_verified for r in result.results):
        raise _fail("one or more configurations did not use the expected query plan")


@app.command("report")
def report_cmd(
    results: Annotated[Path, typer.Argument(help="A results.json written by `vsbench run`.")],
    out: Annotated[Path | None, typer.Option("--out", "-o", help="Output directory.")] = None,
) -> None:
    """Regenerate report.html and results.csv from a results.json."""
    try:
        result = report.load_results(results)
    except ValueError as e:
        raise _fail(str(e)) from None
    paths = report.write_outputs(result, out or results.parent)
    for kind, p in paths.items():
        typer.echo(f"{kind}: {p}")
