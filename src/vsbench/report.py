"""JSON / CSV export and the self-contained HTML report."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader, select_autoescape
from markupsafe import Markup, escape
from pydantic import ValidationError

from vsbench.benchmark import SCHEMA_VERSION, RunResult, SearchResult

CSV_FIELDS = [
    "method", "m", "ef_construction", "ef_search", "metric", "top_k",
    "recall_mean", "recall_min",
    "latency_mean_ms", "latency_median_ms", "latency_p95_ms", "latency_p99_ms",
    "latency_min_ms", "latency_max_ms", "throughput_qps", "latency_samples",
    "server_median_ms",
    "index_build_seconds", "index_size_bytes", "plan_verified",
]  # fmt: skip


def write_json(result: RunResult, path: Path) -> None:
    path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_results(path: str | Path) -> RunResult:
    """Load and validate a results.json written by `vsbench run` / `write_outputs`."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise ValueError(f"results file not found: {path}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: not valid JSON: {e}") from e
    if isinstance(data, dict) and data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported schema_version {data.get('schema_version')!r}")
    try:
        return RunResult.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"{path}: not a vsbench results file:\n{e}") from e


def csv_rows(result: RunResult) -> list[dict[str, Any]]:
    rows = []
    for r in result.results:
        lat = r.latency
        rows.append({
            "method": r.method, "m": r.m, "ef_construction": r.ef_construction,
            "ef_search": r.ef_search, "metric": result.config["metric"],
            "top_k": result.config["top_k"],
            "recall_mean": round(r.recall_mean, 6), "recall_min": round(r.recall_min, 6),
            "latency_mean_ms": round(lat.mean_ms, 4), "latency_median_ms": round(lat.median_ms, 4),
            "latency_p95_ms": round(lat.p95_ms, 4), "latency_p99_ms": round(lat.p99_ms, 4),
            "latency_min_ms": round(lat.min_ms, 4), "latency_max_ms": round(lat.max_ms, 4),
            "throughput_qps": round(lat.throughput_qps, 1), "latency_samples": lat.samples,
            "server_median_ms": (
                None if r.server_median_ms is None else round(r.server_median_ms, 4)
            ),
            "index_build_seconds": (
                None if r.index_build_seconds is None else round(r.index_build_seconds, 4)
            ),
            "index_size_bytes": r.index_size_bytes, "plan_verified": r.plan_verified,
        })  # fmt: skip
    return rows


def write_csv(result: RunResult, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(csv_rows(result))


# ---------------------------------------------------------------- charts (inline SVG)


def _nice_max(v: float) -> float:
    """Smallest "nice" bound >= v whose quarters are round numbers (axis upper bound)."""
    if v <= 0:
        return 1.0
    step = 10.0 ** math.floor(math.log10(v))
    return next(m * step for m in (1, 1.2, 1.6, 2, 3, 4, 6, 8, 10) if v <= m * step)


def _ticks(vmax: float, n: int = 4) -> list[float]:
    return [vmax * i / n for i in range(n + 1)]


def _fmt(v: float) -> str:
    return f"{v:.3g}" if v < 10 else f"{v:,.0f}"


def latency_chart(results: list[SearchResult]) -> Markup:
    """Horizontal grouped bars: median and p95 latency per configuration."""
    if not results:
        return Markup("")
    label_w, bar_h, gap, group_gap, top = 210, 10, 2, 14, 8
    width = 720
    plot_w = width - label_w - 24
    vmax = _nice_max(max(r.latency.p95_ms for r in results))
    group_h = 2 * bar_h + gap
    height = top + len(results) * (group_h + group_gap) + 28

    def x(v: float) -> float:
        return label_w + v / vmax * plot_w

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" class="chart" '
        f'aria-label="Median and p95 query latency per configuration">'
    ]
    for t in _ticks(vmax):
        parts.append(
            f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top - 4}" y2="{height - 24}" '
            f'class="grid"/><text x="{x(t):.1f}" y="{height - 8}" class="tick" '
            f'text-anchor="middle">{_fmt(t)}</text>'
        )
    for i, r in enumerate(results):
        y0 = top + i * (group_h + group_gap)
        parts.append(
            f'<text x="{label_w - 8}" y="{y0 + group_h / 2 + 4:.1f}" class="label" '
            f'text-anchor="end">{escape(r.label)}</text>'
        )
        for j, (name, v) in enumerate((("median", r.latency.median_ms), ("p95", r.latency.p95_ms))):
            y = y0 + j * (bar_h + gap)
            w = max(v / vmax * plot_w, 1)
            parts.append(
                f'<rect x="{label_w}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="2" '
                f'class="s{j + 1}"><title>{escape(r.label)} — {name}: {v:.3f} ms</title></rect>'
            )
        parts.append(
            f'<text x="{x(r.latency.p95_ms) + 6:.1f}" y="{y0 + bar_h * 2 + gap - 1}" '
            f'class="value">{r.latency.p95_ms:.2f}</text>'
        )
    parts.append("</svg>")
    return Markup("".join(parts))  # noqa: S704 - all interpolated text is escaped


# Fixed colour slots per index build (exact search owns s2). Builds past the fourth share a grey
# instead of cycling hues, so colours never repeat for different builds.
BUILD_CLASSES = ["s1", "s3", "s4", "s5"]


def index_builds(results: list[SearchResult]) -> list[tuple[str, str]]:
    """(legend label, colour class) for each distinct HNSW build, in run order."""
    keys = list(dict.fromkeys((r.m, r.ef_construction) for r in results if r.method == "hnsw"))
    return [
        (f"HNSW m={m}, ef_construction={efc}", BUILD_CLASSES[i] if i < 4 else "other")
        for i, (m, efc) in enumerate(keys)
    ]


def recall_chart(results: list[SearchResult]) -> Markup:
    """Recall@k vs median latency; HNSW points joined in ef_search order per index."""
    hnsw = [r for r in results if r.method == "hnsw"]
    if len(hnsw) < 2:
        return Markup("")
    width, height, left, right, top, bottom = 720, 300, 56, 70, 12, 40
    pw, ph = width - left - right, height - top - bottom
    # Keep the HNSW curves readable: a much slower exact point is shown as an off-scale note.
    hnsw_max = max(r.latency.median_ms for r in hnsw)
    in_range = [r for r in results if r.method == "hnsw" or r.latency.median_ms <= 3 * hnsw_max]
    xmax = _nice_max(max(r.latency.median_ms for r in in_range))
    rmin = min(r.recall_mean for r in results)
    ymin = min(math.floor(rmin * 10) / 10, 0.9)  # at least a 0.1-wide recall axis

    def x(v: float) -> float:
        return left + v / xmax * pw

    def y(v: float) -> float:
        return top + (1 - (v - ymin) / (1 - ymin)) * ph

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" class="chart" '
        f'aria-label="Recall at k versus median latency">'
    ]
    for t in _ticks(xmax):
        parts.append(
            f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top}" y2="{top + ph}" class="grid"/>'
            f'<text x="{x(t):.1f}" y="{top + ph + 16}" class="tick" '
            f'text-anchor="middle">{_fmt(t)}</text>'
        )
    for i in range(6):
        t = ymin + (1 - ymin) * i / 5
        parts.append(
            f'<line x1="{left}" x2="{left + pw}" y1="{y(t):.1f}" y2="{y(t):.1f}" class="grid"/>'
            f'<text x="{left - 6}" y="{y(t) + 4:.1f}" class="tick" text-anchor="end">{t:.2f}</text>'
        )
    parts.append(
        f'<text x="{left + pw / 2}" y="{height - 4}" class="tick" text-anchor="middle">'
        "median latency (ms)</text>"
        f'<text x="12" y="{top + ph / 2}" class="tick" text-anchor="middle" '
        f'transform="rotate(-90 12 {top + ph / 2})">Recall@k</text>'
    )
    groups: dict[tuple[int | None, int | None], list[SearchResult]] = {}
    for r in hnsw:
        groups.setdefault((r.m, r.ef_construction), []).append(r)
    labelled: list[tuple[float, float]] = []
    for pts, (_, cls) in zip(groups.values(), index_builds(results), strict=True):
        pts.sort(key=lambda r: r.ef_search or 0)
        path = " ".join(f"{x(r.latency.median_ms):.1f},{y(r.recall_mean):.1f}" for r in pts)
        parts.append(f'<polyline points="{path}" class="line {cls}"/>')
        for r in pts:
            cx, cy = x(r.latency.median_ms), y(r.recall_mean)
            parts.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" class="dot {cls}"><title>'
                f"{escape(r.label)}: recall {r.recall_mean:.4f}, "
                f"median {r.latency.median_ms:.3f} ms</title></circle>"
            )
            # Skip labels that would overlap one already drawn; the tooltip and table keep them.
            if all(abs(cx - lx) > 52 or abs(cy - ly) > 14 for lx, ly in labelled):
                labelled.append((cx, cy))
                parts.append(
                    f'<text x="{cx + 8:.1f}" y="{cy + 14:.1f}" class="value">'
                    f"efs={r.ef_search}</text>"
                )
    for r in results:
        if r.method == "exact" and r not in in_range:
            parts.append(
                f'<text x="{left + pw:.1f}" y="{height - 4}" class="value" '
                f'text-anchor="end">exact (recall 1.0): {r.latency.median_ms:.2f} ms, '
                "off scale →</text>"
            )
        elif r.method == "exact":
            cx, cy = x(r.latency.median_ms), y(1.0)
            parts.append(
                f'<rect x="{cx - 5:.1f}" y="{cy - 5:.1f}" width="10" height="10" class="dot s2">'
                f"<title>exact: recall 1.0, median {r.latency.median_ms:.3f} ms</title></rect>"
                f'<text x="{cx:.1f}" y="{cy + 20:.1f}" class="value" '
                f'text-anchor="middle">exact</text>'
            )
    parts.append("</svg>")
    return Markup("".join(parts))  # noqa: S704 - all interpolated text is escaped


def render_html(result: RunResult) -> str:
    env = Environment(
        loader=PackageLoader("vsbench", "templates"), autoescape=select_autoescape(["html"])
    )
    env.filters["thousands"] = lambda n: f"{n:,}"
    env.filters["mib"] = lambda b: f"{b / 2**20:.1f} MiB"
    exact = next((r for r in result.results if r.method == "exact"), None)
    return env.get_template("report.html.j2").render(
        run=result,
        exact=exact,
        hnsw=[r for r in result.results if r.method == "hnsw"],
        latency_svg=latency_chart(result.results),
        recall_svg=recall_chart(result.results),
        builds=index_builds(result.results),
    )


def write_outputs(result: RunResult, out_dir: str | Path) -> dict[str, Path]:
    """Write report.html, results.json and results.csv into `out_dir` (created if missing)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": out_dir / "results.json",
        "csv": out_dir / "results.csv",
        "html": out_dir / "report.html",
    }
    write_json(result, paths["json"])
    write_csv(result, paths["csv"])
    paths["html"].write_text(render_html(result), encoding="utf-8")
    return paths
