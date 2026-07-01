"""Phase 5.1 scientific long-horizon evaluation runner."""

from __future__ import annotations

import json
import logging
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evaluation.metrics.phase5_scientific_metrics import (
    emotion_continuity,
    graph_growth,
    identity_drift,
    identity_graph,
    long_horizon_failures,
    memory_evolution_validation,
    memory_lifetime,
    phase5_summary,
    preference_overwrite,
    reflection_utility,
    retrieval_scaling,
)
from evaluation.synthetic.user_generator import UserGenerator

logger = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = Path("evaluation/results")

try:
    import matplotlib  # type: ignore[import-not-found]

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore[import-not-found]

    _MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MATPLOTLIB_AVAILABLE = False


def _basic_png_line_chart(
    path: Path,
    series: list[tuple[str, list[float], tuple[int, int, int]]],
    *,
    width: int = 800,
    height: int = 480,
) -> None:
    """Write a minimal RGB PNG line chart without optional plotting dependencies."""
    margin = 48
    canvas = bytearray([255, 255, 255] * width * height)

    def set_px(x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < width and 0 <= y < height:
            idx = (y * width + x) * 3
            canvas[idx : idx + 3] = bytes(color)

    def line(x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            for ox in (-1, 0, 1):
                for oy in (-1, 0, 1):
                    set_px(x0 + ox, y0 + oy, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    all_values = [value for _, values, _ in series for value in values]
    max_value = max(all_values) if all_values else 1.0
    min_value = min(all_values) if all_values else 0.0
    span = max(1.0, max_value - min_value)
    plot_w = width - margin * 2
    plot_h = height - margin * 2

    line(margin, height - margin, width - margin, height - margin, (40, 40, 40))
    line(margin, margin, margin, height - margin, (40, 40, 40))

    for _, values, color in series:
        if not values:
            continue
        points: list[tuple[int, int]] = []
        for idx, value in enumerate(values):
            x = margin + round((idx / max(1, len(values) - 1)) * plot_w)
            y = height - margin - round(((value - min_value) / span) * plot_h)
            points.append((x, y))
        for first, second in zip(points, points[1:], strict=False):
            line(first[0], first[1], second[0], second[1], color)

    raw_rows = bytearray()
    row_bytes = width * 3
    for y in range(height):
        raw_rows.append(0)
        start = y * row_bytes
        raw_rows.extend(canvas[start : start + row_bytes])

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(raw_rows), 9))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, default=str)
    logger.info("Wrote %s", path)


def _style_plots() -> None:
    if not _MATPLOTLIB_AVAILABLE:
        return
    plt.rcParams.update({
        "figure.dpi": 150,
        "figure.figsize": (8, 4.8),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.family": "DejaVu Sans",
        "font.size": 10,
    })


def _plot_graph_growth(graph_data: dict[str, Any], path: Path) -> None:
    rows = graph_data.get("per_user", [])
    nodes = [row["nodes"] for row in rows]
    edges = [row["edges"] for row in rows]
    if not _MATPLOTLIB_AVAILABLE:
        _basic_png_line_chart(path, [("Nodes", nodes, (45, 108, 223)), ("Edges", edges, (217, 95, 2))])
        return
    _style_plots()
    xs = list(range(len(rows)))
    fig, ax = plt.subplots()
    ax.plot(xs, nodes, marker="o", label="Nodes", color="#2d6cdf")
    ax.plot(xs, edges, marker="s", label="Edges", color="#d95f02")
    ax.set_title("Identity Graph Growth")
    ax.set_xlabel("Synthetic user index")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_retrieval_scaling(scaling_data: dict[str, Any], path: Path) -> None:
    rows = scaling_data.get("checkpoints", [])
    ys = [row["average_retrieval_latency_ms"] for row in rows]
    if not _MATPLOTLIB_AVAILABLE:
        _basic_png_line_chart(path, [("Latency", ys, (45, 154, 39))])
        return
    _style_plots()
    xs = [row["memory_count"] for row in rows]
    fig, ax = plt.subplots()
    ax.plot(xs, ys, marker="o", color="#2d9a27")
    ax.set_title("Retrieval Scaling")
    ax.set_xlabel("Memory count")
    ax.set_ylabel("Average retrieval latency (ms)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


@dataclass(slots=True)
class Phase5ScientificRunner:
    """Generate Phase 5.1 scientific evaluation artifacts."""

    n_users: int = 100
    min_turns: int = 500
    max_turns: int = 1000
    seed_base: int = UserGenerator.DEFAULT_SEED_BASE
    results_dir: Path = field(default_factory=lambda: DEFAULT_RESULTS_DIR)

    def run(self) -> dict[str, Any]:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Generating Phase 5.1 synthetic cohort: %d users", self.n_users)
        users = UserGenerator(
            n_users=self.n_users,
            min_turns=self.min_turns,
            max_turns=self.max_turns,
            seed_base=self.seed_base,
        ).generate_all()

        identity = identity_drift(users)
        preference = preference_overwrite(users)
        evolution = memory_evolution_validation(users)
        emotion = emotion_continuity(users)
        reflection = reflection_utility(users)
        graph_model = identity_graph(users)
        lifetime = memory_lifetime(users)
        graph_stats = graph_growth(graph_model)
        scaling = retrieval_scaling(users)
        failures = long_horizon_failures(users, identity, preference, emotion, evolution, reflection)
        summary = phase5_summary(
            identity=identity,
            preference=preference,
            emotion=emotion,
            reflection=reflection,
            evolution=evolution,
            lifetime=lifetime,
            graph=graph_stats,
            scaling=scaling,
            failures=failures,
        )

        outputs = {
            "identity_drift.json": identity,
            "preference_overwrite.json": preference,
            "memory_evolution_validation.json": evolution,
            "emotion_continuity.json": emotion,
            "reflection_utility.json": reflection,
            "identity_graph.json": graph_model,
            "memory_lifetime.json": lifetime,
            "graph_growth.json": graph_stats,
            "retrieval_scaling.json": scaling,
            "long_horizon_failures.json": failures,
            "phase5_validation_summary.json": summary,
        }
        for filename, data in outputs.items():
            _write_json(self.results_dir / filename, data)

        _plot_graph_growth(graph_stats, self.results_dir / "graph_growth.png")
        _plot_retrieval_scaling(scaling, self.results_dir / "retrieval_scaling.png")

        return {
            "n_users": len(users),
            "results_dir": str(self.results_dir),
            "outputs": sorted(outputs),
            "plots_generated": _MATPLOTLIB_AVAILABLE,
            "summary": summary,
        }


def run_phase5_scientific_evaluation() -> dict[str, Any]:
    """Run Phase 5.1 scientific evaluation with default settings."""
    return Phase5ScientificRunner().run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_phase5_scientific_evaluation()
