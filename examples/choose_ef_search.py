"""Pick the cheapest HNSW ef_search that meets a recall target for your embeddings.

Uses only the installed `vsbench` package (pip install vector-search-bench / the wheel):

    python choose_ef_search.py --vectors my_embeddings.npy --target 0.95
    python choose_ef_search.py                     # demo embeddings, no file needed

Exits 0 and prints the recommended `SET hnsw.ef_search` when some setting reaches the target,
exits 2 when none does. Full results (HTML/JSON/CSV) are written to --out.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import vsbench


def demo_embeddings(n: int, dims: int = 384, seed: int = 0) -> np.ndarray:
    """Unit-length vectors with low intrinsic dimension, like sentence embeddings.

    Topic centres in a 48-d latent space are projected to `dims` dimensions; each point is a
    noisy copy of a centre. This is synthetic data that only stands in for real embeddings.
    """
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((200, 48))
    latent = centres[rng.integers(200, size=n)] + 0.6 * rng.standard_normal((n, 48))
    x = latent @ rng.standard_normal((48, dims)) + 0.5 * rng.standard_normal((n, dims))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--vectors", help=".npy file of shape (rows, dimensions)")
    p.add_argument("--size", type=int, default=20_000, help="demo embeddings to generate")
    p.add_argument("--target", type=float, default=0.95, help="minimum mean Recall@k")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--out", default="ef_search_results")
    p.add_argument("--database-url", default="postgresql://vsbench:vsbench@127.0.0.1:5433/vsbench")
    args = p.parse_args()

    try:
        if args.vectors:
            data = vsbench.from_arrays(
                vsbench.load_npy(args.vectors), holdout=100, source=Path(args.vectors).name
            )
        else:
            data = vsbench.from_arrays(
                demo_embeddings(args.size), holdout=100, source="demo sentence-like embeddings"
            )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    config = vsbench.BenchmarkConfig(
        name="ef_search_tuner",
        metric=vsbench.Metric.COSINE,
        top_k=args.k,
        trials=2,
        hnsw=[vsbench.HnswConfig(m=16, ef_construction=64, ef_search=[10, 20, 40, 80, 160])],
    )

    try:
        result = vsbench.run(config, args.database_url, data=data, log=print)
    except (vsbench.DatabaseError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    paths = vsbench.write_outputs(result, args.out)

    exact = result.results[0]
    print(f"\n{'config':<30}{'recall':>8}{'median ms':>11}{'server ms':>11}")
    for r in result.results:
        print(
            f"{r.label:<30}{r.recall_mean:>8.4f}{r.latency.median_ms:>11.3f}"
            f"{r.server_median_ms or 0:>11.3f}"
        )
    print(f"\nreport: {paths['html']}")

    best = result.best(args.target)
    if best is None:
        print(f"no ef_search reached recall {args.target}; raise ef_search or m")
        return 2
    print(
        f"recommendation: SET hnsw.ef_search = {best.ef_search};  "
        f"recall@{args.k} {best.recall_mean:.3f}, server median "
        f"{best.server_median_ms:.3f} ms vs {exact.server_median_ms:.3f} ms exact"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
