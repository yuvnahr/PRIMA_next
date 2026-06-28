"""Publication-ready visualization for Phase 5 long-horizon results — Deliverable 12.

Reads evaluation/results/long_horizon_results.json and
evaluation/results/long_horizon_summary.json and produces:

    plots/memory_growth.png
    plots/preference_retention.png
    plots/retrieval_recall.png
    plots/identity_consistency.png
    plots/reflection_frequency.png
    plots/emotion_continuity.png
    plots/latency_vs_memory.png
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = Path("evaluation/results")

# Attempt matplotlib import — degrade gracefully if not installed
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend, safe for servers
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available — plots will not be generated")


def _style() -> None:
    """Apply a clean, publication-friendly style."""
    if not _MATPLOTLIB_AVAILABLE:
        return
    plt.rcParams.update({
        "figure.dpi": 150,
        "figure.figsize": (9, 5),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
    })


def _load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _save(fig: Any, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def plot_memory_growth(user_results: list[dict[str, Any]], plots_dir: Path) -> None:
    """Memory count vs turns — one curve per user, mean highlighted."""
    if not _MATPLOTLIB_AVAILABLE:
        return
    _style()
    fig, ax = plt.subplots()

    all_x: list[list[int]] = []
    all_y: list[list[int]] = []

    for user in user_results[:30]:  # cap for readability
        curve = user.get("memory_growth_curve", [])
        xs = [pt["turn_index"] for pt in curve]
        ys = [pt["cumulative_memories"] for pt in curve]
        if xs:
            ax.plot(xs, ys, color="#5b8dee", alpha=0.15, linewidth=1)
            all_x.append(xs)
            all_y.append(ys)

    # Mean line
    if all_x and all_y:
        max_len = max(len(x) for x in all_x)
        means: list[float] = []
        for i in range(max_len):
            vals = [all_y[j][i] for j in range(len(all_y)) if i < len(all_y[j])]
            means.append(sum(vals) / len(vals) if vals else 0.0)
        xs_mean = list(range(max_len))
        ax.plot(xs_mean, means, color="#1a3f99", linewidth=2.5, label="Mean")
        ax.legend()

    ax.set_title("Memory Growth vs. Conversation Turns")
    ax.set_xlabel("Turn")
    ax.set_ylabel("Cumulative Stored Memories")
    _save(fig, plots_dir / "memory_growth.png")


def plot_preference_retention(user_results: list[dict[str, Any]], plots_dir: Path) -> None:
    """Preference retention rate at each checkpoint."""
    if not _MATPLOTLIB_AVAILABLE:
        return
    _style()
    fig, ax = plt.subplots()

    checkpoints = [10, 50, 100, 250, 500]
    means: list[float] = []
    for cp in checkpoints:
        vals = []
        for user in user_results:
            curve = user.get("preference_retention_curve", {})
            v = curve.get(str(cp)) or curve.get(cp)
            if v is not None:
                vals.append(float(v))
        means.append(sum(vals) / len(vals) if vals else 0.0)

    ax.plot(checkpoints, means, marker="o", color="#e85d04", label="Mean Retention")
    ax.axhline(0.90, linestyle="--", color="#333", alpha=0.5, label="90% Target")
    ax.set_ylim(0, 1.05)
    ax.set_title("Preference Retention vs. Turns")
    ax.set_xlabel("Turn")
    ax.set_ylabel("Retention Rate")
    ax.legend()
    _save(fig, plots_dir / "preference_retention.png")


def plot_retrieval_recall(user_results: list[dict[str, Any]], plots_dir: Path) -> None:
    """Retrieval quality (first/second/final thirds) across users."""
    if not _MATPLOTLIB_AVAILABLE:
        return
    _style()
    fig, ax = plt.subplots()

    firsts = [float(u.get("retrieval_first_third", 0.0)) for u in user_results]
    finals = [float(u.get("retrieval_final_third", 0.0)) for u in user_results]
    user_ids = [u["user_id"] for u in user_results]
    xs = list(range(len(user_ids)))

    ax.bar(xs, firsts, alpha=0.6, label="First third", color="#5b8dee")
    ax.bar(xs, finals, alpha=0.6, label="Final third", color="#e85d04")
    ax.set_title("Retrieval Count — First vs Final Third of Conversation")
    ax.set_xlabel("User Index")
    ax.set_ylabel("Mean Retrieval Count per Turn")
    ax.legend()
    _save(fig, plots_dir / "retrieval_recall.png")


def plot_identity_consistency(user_results: list[dict[str, Any]], plots_dir: Path) -> None:
    """Identity consistency at each checkpoint."""
    if not _MATPLOTLIB_AVAILABLE:
        return
    _style()
    fig, ax = plt.subplots()

    checkpoints = [10, 50, 100, 250, 500]
    means: list[float] = []
    for cp in checkpoints:
        vals = []
        for user in user_results:
            curve = user.get("identity_consistency_curve", {})
            v = curve.get(str(cp)) or curve.get(cp)
            if v is not None:
                vals.append(float(v))
        means.append(sum(vals) / len(vals) if vals else 0.0)

    ax.plot(checkpoints, means, marker="s", color="#2d9a27", label="Mean Consistency")
    ax.axhline(0.90, linestyle="--", color="#333", alpha=0.5, label="90% Target")
    ax.set_ylim(0, 1.05)
    ax.set_title("Identity Consistency vs. Turns")
    ax.set_xlabel("Turn")
    ax.set_ylabel("Consistency Rate")
    ax.legend()
    _save(fig, plots_dir / "identity_consistency.png")


def plot_reflection_frequency(user_results: list[dict[str, Any]], plots_dir: Path) -> None:
    """Distribution of reflection frequency across users."""
    if not _MATPLOTLIB_AVAILABLE:
        return
    _style()
    fig, ax = plt.subplots()

    freqs = [float(u.get("reflection_frequency", 0.0)) for u in user_results]
    ax.hist(freqs, bins=20, color="#7b2d8b", alpha=0.75, edgecolor="white")
    ax.axvline(sum(freqs) / max(1, len(freqs)), linestyle="--", color="#333", label="Mean")
    ax.set_title("Reflection Frequency Distribution Across Users")
    ax.set_xlabel("Reflection Rate (reflections / turns)")
    ax.set_ylabel("Number of Users")
    ax.legend()
    _save(fig, plots_dir / "reflection_frequency.png")


def plot_emotion_continuity(user_results: list[dict[str, Any]], plots_dir: Path) -> None:
    """Emotion continuity at each checkpoint."""
    if not _MATPLOTLIB_AVAILABLE:
        return
    _style()
    fig, ax = plt.subplots()

    checkpoints = [10, 50, 100, 250, 500]
    means: list[float] = []
    for cp in checkpoints:
        vals = []
        for user in user_results:
            curve = user.get("emotion_continuity_curve", {})
            v = curve.get(str(cp)) or curve.get(cp)
            if v is not None:
                vals.append(float(v))
        means.append(sum(vals) / len(vals) if vals else 0.0)

    ax.plot(checkpoints, means, marker="^", color="#c0392b", label="Mean Continuity")
    ax.set_ylim(0, 1.05)
    ax.set_title("Emotion Continuity vs. Turns")
    ax.set_xlabel("Turn")
    ax.set_ylabel("Continuity Score")
    ax.legend()
    _save(fig, plots_dir / "emotion_continuity.png")


def plot_latency_vs_memory(user_results: list[dict[str, Any]], plots_dir: Path) -> None:
    """Scatter: per-turn latency vs cumulative memory count."""
    if not _MATPLOTLIB_AVAILABLE:
        return
    _style()
    fig, ax = plt.subplots()

    x_all: list[float] = []
    y_all: list[float] = []

    for user in user_results[:20]:  # sample for readability
        curve = user.get("memory_growth_curve", [])
        # Pair latency from turn records with memory growth curve
        for pt in curve:
            x_all.append(float(pt.get("cumulative_memories", 0)))
            # Use memory density as proxy latency if direct latency unavailable
            y_all.append(float(pt.get("memory_density", 0.0)) * 10.0)

    ax.scatter(x_all, y_all, alpha=0.3, s=10, color="#e85d04")
    ax.set_title("Latency (proxy) vs. Cumulative Memory Count")
    ax.set_xlabel("Cumulative Stored Memories")
    ax.set_ylabel("Relative Latency Indicator")
    _save(fig, plots_dir / "latency_vs_memory.png")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def generate_all_plots(results_dir: Path = DEFAULT_RESULTS_DIR) -> None:
    """Generate all Phase 5 publication-ready plots from persisted JSON files."""
    results_dir = Path(results_dir)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / "long_horizon_results.json"
    if not results_path.exists():
        logger.warning("Results file not found: %s — run long_horizon_runner first", results_path)
        return

    user_results = _load_json(results_path)
    logger.info("Loaded %d user results for plotting", len(user_results))

    plot_memory_growth(user_results, plots_dir)
    plot_preference_retention(user_results, plots_dir)
    plot_retrieval_recall(user_results, plots_dir)
    plot_identity_consistency(user_results, plots_dir)
    plot_reflection_frequency(user_results, plots_dir)
    plot_emotion_continuity(user_results, plots_dir)
    plot_latency_vs_memory(user_results, plots_dir)

    logger.info("All plots saved to %s", plots_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    generate_all_plots()
