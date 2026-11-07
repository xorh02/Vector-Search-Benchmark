# Vector Search Bench

Measure what pgvector's HNSW index costs you in recall and buys you in latency, on your own PostgreSQL server and your own embeddings.

HNSW search is approximate: it trades some correct neighbours for speed, and how much it trades depends on your data and on `m`, `ef_construction` and `ef_search`. `vsbench` loads vectors into PostgreSQL and runs an exact sequential scan as ground truth. It then builds HNSW indexes and measures every `ef_search` you ask for. The output is a self-contained HTML report plus JSON and CSV. Use it to choose index parameters before they reach production, or to check an existing setup against a recall target.

![Report overview: exact vs HNSW latency, Recall@10, index build time and size](assets/report-overview.png)

*A real report from `vsbench run` with the default configuration: 20,000 synthetic vectors, 128 dimensions, cosine distance, PostgreSQL 17.6 / pgvector 0.8.0 in Docker Desktop on Windows 11. The numbers are from one machine and are not general performance claims.*

## Install

Requirements:

- Python 3.11 or newer
- A PostgreSQL server with pgvector 0.5 or newer. The repository's Docker Compose file starts one.

The package is not published on PyPI. Install it from source:

```bash
git clone https://github.com/xorh02/Vector-Search-Benchmark.git
cd Vector-Search-Benchmark
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install .
vsbench --version                  # vsbench 0.1.0
```

Other ways to install:

- Straight from GitHub, into an existing environment: `pip install git+https://github.com/xorh02/Vector-Search-Benchmark.git`
- As a wheel: run `python -m build`, then `pip install dist/vector_search_bench-0.1.0-py3-none-any.whl` wherever you need it.

Start the database:

```bash
docker compose up -d --wait        # pgvector/pgvector:0.8.0-pg17 on 127.0.0.1:5433
vsbench check
```

```
PostgreSQL 17.6 (Debian 17.6-1.pgdg12+1), pgvector 0.8.0: OK
```

The container only listens on localhost. `docker compose down -v` stops it and deletes its data.

## Quick start

```bash
vsbench init        # writes vsbench.toml
vsbench run         # 10–20 s with the default config on the machine above
```

```
exact                                recall@10=1.0000  median=3.212 ms  p95=4.556 ms  server=2.405 ms
hnsw m=16 efc=64 efs=10              recall@10=0.9590  median=0.536 ms  p95=0.713 ms  server=0.104 ms
hnsw m=16 efc=64 efs=20              recall@10=0.9935  median=0.571 ms  p95=0.791 ms  server=0.141 ms
hnsw m=16 efc=64 efs=40              recall@10=1.0000  median=0.583 ms  p95=0.769 ms  server=0.160 ms
hnsw m=16 efc=64 efs=80              recall@10=1.0000  median=0.629 ms  p95=0.829 ms  server=0.186 ms
hnsw m=16 efc=64 efs=160             recall@10=1.0000  median=0.687 ms  p95=0.906 ms  server=0.253 ms
```

`run` writes `report.html`, `results.json` and `results.csv` to `results/`. Open `results/report.html` in any browser; it is a single file with no external resources. Each line in the output is one search configuration:

- **recall@10** is the fraction of the true 10 nearest neighbours that HNSW returned.
- **median** and **p95** are the latencies the client saw.
- **server** is PostgreSQL's own execution time.

In this run, HNSW at `ef_search=40` found every true neighbour for all 200 queries, and the server did the work about 15× faster than the exact scan (0.16 ms vs 2.41 ms). The client-side gap is smaller: the Docker Desktop network round-trip adds about 0.4 ms to every query.

## Use it with your own embeddings

Synthetic data only tells you so much. The number you care about is recall on your vectors.

### From the command line

Export your vectors to a 2-D `.npy` file and pass it to `run`:

```bash
python -c "import numpy as np; np.save('embeddings.npy', my_vectors)"   # shape (rows, dims)
vsbench run --vectors embeddings.npy
```

By default the tool holds out 200 random rows (`[dataset].queries`, seeded by `[dataset].seed`) as queries. Those rows are not stored, so a query is never its own nearest neighbour. To use real queries, such as embedded search strings from your logs, pass `--queries queries.npy`. Values are converted to float32, pgvector's storage type, and the report records the file name and a SHA-256 of the data.

### From Python

`vsbench` is also a library. This is the complete script, tested against the installed wheel:

```python
import numpy as np
import vsbench

# Your embeddings: any (rows, dimensions) float array, e.g. exported from your vector store.
embeddings = np.load("embeddings.npy")

config = vsbench.BenchmarkConfig(
    metric="cosine",
    top_k=10,
    hnsw=[vsbench.HnswConfig(m=16, ef_construction=64, ef_search=[20, 40, 80])],
)
data = vsbench.from_arrays(embeddings, holdout=200)  # 200 random rows become queries
result = vsbench.run(config, "postgresql://vsbench:vsbench@127.0.0.1:5433/vsbench", data=data)

for r in result.results:
    print(
        f"{r.label:<28} recall={r.recall_mean:.3f}  "
        f"median={r.latency.median_ms:.2f} ms  server={r.server_median_ms:.2f} ms"
    )

best = result.best(min_recall=0.95)
print("fastest config with recall >= 0.95:", best.label if best else "none")
vsbench.write_outputs(result, "my_results")  # report.html, results.json, results.csv
```

```
exact                        recall=1.000  median=3.73 ms  server=2.79 ms
hnsw m=16 efc=64 efs=20      recall=1.000  median=0.68 ms  server=0.16 ms
hnsw m=16 efc=64 efs=40      recall=1.000  median=0.72 ms  server=0.17 ms
hnsw m=16 efc=64 efs=80      recall=1.000  median=0.91 ms  server=0.35 ms
fastest config with recall >= 0.95: hnsw m=16 efc=64 efs=20
```

The public API:

| Name | What it does |
|---|---|
| `BenchmarkConfig`, `HnswConfig`, `DatasetConfig`, `Metric` | Configuration objects. They are validated on construction and are the same as the TOML file. |
| `load_config(path)` | Read a `vsbench.toml`. |
| `generate(DatasetConfig)` | Generate a seeded synthetic dataset. |
| `from_arrays(vectors, queries=None, *, holdout=200, seed=42)` | Wrap your own arrays. |
| `load_npy(path)` | Load a 2-D `.npy` file. Pickled data is refused. |
| `run(config, database_url, *, data=None, log=None)` | Run the benchmark and return a `RunResult`. Pass `log=print` to see progress. |
| `RunResult.results` | A list of `SearchResult`, one per configuration, with `latency`, `server_median_ms`, `recall_mean`, `recall_min`, index metrics and plan checks. |
| `RunResult.best(min_recall)` | The fastest HNSW configuration that meets a recall target. |
| `write_outputs(result, dir)`, `load_results(path)` | Write or reload the HTML, JSON and CSV outputs. |
| `ConfigError`, `DatabaseError` | Raised for invalid configuration and for database problems. Each message says what to fix. |

### Example: choose `ef_search` for a recall target

[`examples/choose_ef_search.py`](examples/choose_ef_search.py) is a standalone script that uses only the installed package. It answers a common tuning question: what is the smallest `ef_search` that meets my recall target? Copy it anywhere and run it against your file, or without arguments for demo data. The demo data is 384-dimensional, unit-length synthetic vectors with low intrinsic dimension, shaped like sentence embeddings.

```bash
python choose_ef_search.py --vectors embeddings.npy --target 0.95
python choose_ef_search.py                                    # demo data
```

```
config                          recall  median ms  server ms
exact                           1.0000      6.507      5.483
hnsw m=16 efc=64 efs=10         0.9960      0.637      0.126
hnsw m=16 efc=64 efs=20         1.0000      0.648      0.149
hnsw m=16 efc=64 efs=40         1.0000      0.671      0.162
hnsw m=16 efc=64 efs=80         1.0000      0.742      0.224
hnsw m=16 efc=64 efs=160        1.0000      1.091      0.527

recommendation: SET hnsw.ef_search = 10;  recall@10 0.996, server median 0.126 ms vs 5.483 ms exact
```

The script writes its full report to `ef_search_results/`. It exits with code 0 when some setting meets the target and 2 when none does, so it can gate a CI job. The output above comes from a clean Python 3.11 virtual environment outside the repository, with only the wheel installed. The CI workflow repeats this check against a freshly built wheel.

## What the report shows

### Latency and throughput

![Latency chart: median and p95 per configuration](assets/report-latency.png)

Each configuration replays the full query set `trials` times on one connection, after untimed warm-up queries. Latency is measured with `time.perf_counter_ns()` around execute + fetch of a prepared statement. The table reports:

- **mean**
- **median**
- **p95 and p99**: 95% or 99% of queries finished within this time
- **min and max**
- **QPS**: sequential single-connection throughput, not server capacity under concurrency

Percentiles use the nearest-rank method.

**server ms** is a separate measurement. After the timed trials, `EXPLAIN (ANALYZE, TIMING OFF)` runs the first 20 queries, and the report shows the median execution time PostgreSQL reports. It is never mixed into client latency. The gap between client and server times is network, protocol and driver cost. If the gap is large, as with Docker Desktop, compare the configurations on server time.

### Recall@k

For one query, Recall@k is `|exact top-k ∩ HNSW top-k| / k`. The report shows the mean over all queries and **Recall min**, the worst single query.

Watch Recall min. Mean recall can look excellent while a few queries get almost nothing back. In the 100k comparison below, every configuration has a minimum of 0.00, including ones with mean recall above 0.99. On that strongly clustered synthetic data, the HNSW search for some queries never reaches the cluster the query belongs to. Inspecting one such query showed HNSW returning 4 points from the right cluster and filling the remaining 6 slots from a different cluster.

How edge cases are handled:

- **Ties.** The exact query orders by `distance, id`, so tied distances go to the lowest id. HNSW orders by distance alone, because a second sort key would stop pgvector from using the index. An equally close neighbour with a different id therefore counts as a miss. This only matters for data with many exact duplicates.
- **Short results.** If HNSW returns fewer than k rows, the missing rows count as misses.

### Comparing index builds

To compare index builds, add more `[[hnsw]]` blocks to the config. Each block is built, measured at every `ef_search` value and dropped before the next. This run used 100,000 vectors with `m=16, ef_construction=64` against `m=32, ef_construction=128`:

![Results table for two index builds on 100,000 vectors](assets/compare-table.png)

![Recall vs latency for two index builds](assets/compare-recall.png)

At equal `ef_search`, the larger graph gives clearly higher recall: 0.96 vs 0.87 at `ef_search=20`. It costs a slower build (19.7 s vs 6.8 s) and a larger index (98.1 MiB vs 79.4 MiB). When an exact scan is far slower than every HNSW point, the chart labels it off-scale instead of squeezing the HNSW curves against the axis.

### Query-plan verification and environment

![Configuration, environment and query-plan verification](assets/report-plans.png)

A benchmark that labels a sequential scan as an HNSW result is worse than no benchmark. Before measuring each configuration, the tool runs `EXPLAIN (FORMAT JSON)` and checks the plan:

- **Exact** must be a `Seq Scan` with no index. The exact baseline runs before any vector index exists, with index scans disabled. The first 10 queries are also compared with a NumPy brute-force search.
- **HNSW** must scan the HNSW index. `enable_seqscan` is turned off for these runs, because on small tables the planner may prefer a sequential scan and you would be timing the wrong thing.

If a check fails, the report flags it in red and `vsbench run` exits with code 1. The results are still written so you can inspect them.

The report also records everything needed to reproduce or question the run:

- **Dataset:** source and SHA-256, seed and sizes
- **Search settings:** metric, k, every HNSW parameter, warm-up and trial counts
- **Database:** PostgreSQL and pgvector versions, and the relevant server settings
- **Client:** Python version, OS and CPU

## Configuration

`vsbench init` writes this file, with every option shown:

```toml
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
```

Every key is optional, and the defaults are the values shown. When you use `--vectors` or `from_arrays`, the `[dataset]` section is ignored except for `queries` (the holdout count) and `seed`.

Validation runs before anything touches the database. These are the errors you are most likely to hit:

| Error | Cause |
|---|---|
| `every ef_search value must be >= top_k` | pgvector's HNSW scan returns at most `ef_search` rows, so a smaller value cannot return k results. |
| `ef_construction must be >= 2 * m` | This is a pgvector requirement. |
| `dimensions` > 2000 | HNSW on the `vector` type supports at most 2000 dimensions. |
| `cosine metric requires non-zero vectors` | Cosine distance is undefined for zero vectors. |
| Unknown keys or bad values | The file is rejected with the offending field named. |
| `HNSW index build ran out of shared memory` | Parallel index builds use about `maintenance_work_mem` of shared memory. Docker containers get only 64 MB by default. Start the container with `--shm-size=1g` (the bundled Compose file already does), or lower `maintenance_work_mem`. |

The database URL comes from `--database-url`, then `$VSBENCH_DATABASE_URL`, then `postgresql://vsbench:vsbench@127.0.0.1:5433/vsbench`, which matches the Compose file. The URL is never written to any output. Compose reads `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` and `VSBENCH_DB_PORT` if you want different local settings.

The tool only creates and drops objects named `vsbench_<name>`. The `vsbench_` prefix is enforced in the code. `vsbench check` and `vsbench run` run `CREATE EXTENSION IF NOT EXISTS vector`. To remove everything: `pip uninstall vector-search-bench` and `docker compose down -v`.

### CLI reference

| Command | Description |
|---|---|
| `vsbench init [PATH] [--force]` | Write the example config (default `vsbench.toml`). Refuses to overwrite an existing file without `--force`. |
| `vsbench check [--database-url URL]` | Connect, enable pgvector and print the server versions. |
| `vsbench run [CONFIG] [-o DIR] [--vectors F.npy] [--queries F.npy] [--database-url URL]` | Run the benchmark and write `report.html`, `results.json` and `results.csv` to `DIR` (default `results/`). |
| `vsbench report RESULTS_JSON [-o DIR]` | Validate a saved `results.json` and regenerate the HTML and CSV from it. |

Exit codes: 0 means success. 1 means invalid input, a database error or a failed plan check. In every error case, the message on stderr says what to fix.

### Output files

`results.json` (schema version 1) holds everything listed above:

- the full validated configuration, dataset metadata and the environment
- load time and table size
- the exact cross-check counts
- one entry per configuration, with all latency fields in `*_ms`, `server_median_ms`, `recall_mean`, `recall_min`, `index_build_seconds`, `index_size_bytes` and the plan check

`results.csv` has one row per configuration, with units in the column names: `latency_p95_ms`, `server_median_ms`, `index_build_seconds`, `index_size_bytes` and so on.

## How it works

```
config / your .npy ──► dataset ──► COPY into vsbench_<name> ──► exact baseline (seq scan) = ground truth
                                                                       │
             for each [[hnsw]]: CREATE INDEX (timed) ──► for each ef_search:
                 EXPLAIN check ─► warm-up ─► timed trials ─► EXPLAIN ANALYZE sample ─► recall vs truth
                                                                       │
                                               RunResult ──► report.html / results.json / results.csv
```

| Module | Responsibility |
|---|---|
| `config.py` | TOML → validated Pydantic models |
| `dataset.py` | Seeded generation, `from_arrays` / `load_npy`, NumPy reference search |
| `db.py` | All SQL: connection, pgvector checks, `COPY`, index build and size, `EXPLAIN` parsing |
| `benchmark.py` | Runs the phases and defines the result models (the JSON schema) |
| `metrics.py` | Recall@k, nearest-rank percentiles |
| `report.py` | JSON and CSV export, inline-SVG charts, HTML rendering |
| `cli.py` | Typer commands |

Design decisions:

- **Ground truth comes from PostgreSQL.** The exact results are produced by the same SQL engine and distance operators that HNSW uses. NumPy is an independent cross-check.
- **One phase at a time.** Loading, index build, warm-up, timed queries and server-time sampling never overlap, so build time is never counted as query latency.
- **Safe SQL.** All SQL uses `psycopg.sql.Identifier` for names, and bind parameters or `sql.Literal` for values.
- **No pgvector Python adapter.** Vectors are sent as text literals, and float32 values round-trip exactly. A test checks this.
- **Self-contained report.** Charts are inline SVG generated in Python, so the HTML has no JavaScript and opens offline.
- **Few dependencies.** The runtime dependencies are NumPy, psycopg 3, Pydantic, Typer and Jinja2.
- **Bounded memory.** The NumPy reference search works in chunks of rows, so its float64 working set stays small even for large datasets.

## Tests and reproducibility

```bash
pip install -e ".[dev]"
docker compose up -d --wait
ruff check . && ruff format --check . && mypy        # lint, format, strict type check
pytest                                                # unit tests; integration tests skip
VSBENCH_DATABASE_URL=postgresql://vsbench:vsbench@127.0.0.1:5433/vsbench pytest   # everything
```

Integration tests run against a real pgvector server, with no mocks. They check:

- COPY round-trips
- exact SQL against NumPy for both metrics
- tie-breaking
- index build, size and validity
- plan detection for exact and HNSW queries
- full runs with synthetic data, with your own arrays and with `.npy` files through the CLI
- the example script

GitHub Actions (`.github/workflows/ci.yml`) runs:

- lint and type checks
- the full suite on Python 3.11, 3.12 and 3.13 against a pgvector service container
- a wheel build and a clean install, followed by a smoke run of the installed CLI and the example script from outside the repository

To reproduce the screenshots:

1. Run `vsbench init` and `vsbench run`. That gives the overview, latency and plan images.
2. Copy the config, set `size = 100000` and add a second `[[hnsw]]` block with `m = 32`, `ef_construction = 128`. That gives the comparison images.
3. Open each `report.html` in a browser.

The synthetic dataset is reproducible bit for bit; check the SHA-256 in the report. Timings are not, so compare runs from the same machine. HNSW recall varies slightly between index builds of identical data, because pgvector assigns graph levels at random and may build in parallel.

## Limitations

- **One machine.** Results describe one machine, one PostgreSQL configuration and one dataset.
- **Single client.** Queries run sequentially on one connection, so concurrent load is not measured.
- **Sampled server time.** Server time is the median over 20 queries per configuration, not over every query.
- **Cache state is not controlled** beyond the warm-up queries, and the dataset is assumed to fit in memory. Your vectors are loaded fully into memory to compute recall.
- **Supported methods.** Only the `vector` type with HNSW and the cosine and L2 metrics are supported. IVFFlat, `halfvec`, `sparsevec`, inner product and iterative index scans are not.
- **Synthetic data.** The bundled datasets only approximate real embeddings. Use `--vectors` for numbers that matter.

## Contributing

Issues and pull requests are welcome. Run the lint, type and full test commands above against the Docker database before opening a pull request, and add tests for new behaviour. Reports must only ever show measured numbers.

## License

[MIT](LICENSE)
