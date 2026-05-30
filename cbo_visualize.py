"""
cbo_visualize.py — CBO Diagnostic Visualisations
===================================================
Generates five analysis plots for the Contextual Bandit Optimizer benchmark:

1. **Crossover Convergence** — How the crossover estimate evolves over time.
2. **Final Crossover Estimates** — Bar chart comparing final estimates.
3. **Cumulative Regret** — Online learning regret analysis.
4. **Recall vs Latency Scatter** — SLA-aware quality map per strategy.
5. **Optimizer Overhead Histogram** — Decision-time distribution.

Can be used as a module (imported from ``cbo_benchmark``) or run standalone
to regenerate plots from previously exported JSON files.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Lazy matplotlib import (avoid crashing if not installed) ──────────────────
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for server/CI use
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    _HAS_MPL = True
except ImportError:  # pragma: no cover
    _HAS_MPL = False

# ── Colors ────────────────────────────────────────────────────────────────────
STRATEGY_COLORS: Dict[str, str] = {
    "tier_based": "#2196F3",       # Blue
    "exponential_decay": "#FF9800", # Orange
    "softmax": "#4CAF50",           # Green
}

FALLBACK_COLOR = "#9E9E9E"

_SAVE_KW = {"dpi": 150, "bbox_inches": "tight", "facecolor": "white"}


def _get_color(name: str) -> str:
    """Return the colour assigned to an exploration strategy name."""
    return STRATEGY_COLORS.get(name, FALLBACK_COLOR)


def _rolling_mean(values: List[float], window: int = 50) -> np.ndarray:
    """Compute a rolling average, padding the beginning with NaN."""
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    smoothed = np.convolve(arr, kernel, mode="valid")
    # Pad the start so the array length matches the original
    pad = np.full(window - 1, np.nan)
    return np.concatenate([pad, smoothed])


# ═══════════════════════════════════════════════════════════════════════════════
#  Data extraction helpers (from MetricsTracker or raw JSON)
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_records(tracker_or_data: Any) -> List[Dict[str, Any]]:
    """Accept either a MetricsTracker instance or a list of dicts (from JSON).

    Returns a plain list of record dicts.
    """
    if isinstance(tracker_or_data, list):
        return tracker_or_data
    # Assume MetricsTracker – access its internal records
    if hasattr(tracker_or_data, "records"):
        return tracker_or_data.records
    if hasattr(tracker_or_data, "_records"):
        return tracker_or_data._records
    raise TypeError(
        f"Cannot extract records from {type(tracker_or_data).__name__}"
    )


def _field(records: List[Any], key: str) -> List[Any]:
    """Extract a single field from every record (handles dicts and dataclasses/objects)."""
    if not records:
        return []
    if isinstance(records[0], dict):
        return [r.get(key) for r in records]
    return [getattr(r, key, None) for r in records]


# ═══════════════════════════════════════════════════════════════════════════════
#  Plot 1: Crossover Convergence
# ═══════════════════════════════════════════════════════════════════════════════

def plot_crossover_convergence(
    trackers: Dict[str, Any],
    plots_dir: Path,
    *,
    window: int = 50,
    ground_truth_crossover: Optional[float] = None,
) -> None:
    """Plot the crossover estimate over global steps for each strategy.

    Parameters
    ----------
    trackers : dict
        ``{strategy_name: MetricsTracker | list[dict]}``
    plots_dir : Path
        Directory to save the figure.
    window : int
        Rolling-average window size.
    ground_truth_crossover : float, optional
        If known, draw a horizontal reference line.
    """
    n_strats = len(trackers)
    fig, axes = plt.subplots(n_strats, 1, figsize=(10, 3 * n_strats), sharex=True, squeeze=False)
    axes = axes.flatten()

    for ax, (name, trk) in zip(axes, trackers.items()):
        records = _extract_records(trk)
        crossovers = _field(records, "crossover_estimate")

        # Filter out None/null values but keep index for x-axis
        steps: List[int] = []
        vals: List[float] = []
        for i, c in enumerate(crossovers):
            if c is not None:
                steps.append(i)
                vals.append(float(c))

        if not vals:
            ax.set_title(f"{name} (No Data)")
            continue

        smoothed = _rolling_mean(vals, window=window)
        ax.plot(steps, smoothed, label=name, color=_get_color(name), linewidth=2.0)

        if ground_truth_crossover is not None:
            ax.axhline(
                ground_truth_crossover, color="red", linestyle="--",
                linewidth=1.2, alpha=0.7, label=f"Ground truth ({ground_truth_crossover:.2f})",
            )

        ax.set_ylabel("Crossover Selectivity")
        ax.set_title(f"Convergence: {name}", fontsize=11, fontweight="bold")
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Global Step (query iteration)")
    fig.suptitle("Crossover Point Convergence per Strategy", fontsize=14, y=1.02)
    fig.tight_layout()

    path = plots_dir / "01_crossover_convergence.png"
    fig.savefig(path, **_SAVE_KW)
    plt.close(fig)
    logger.info("Saved plot → %s", path)


# ═══════════════════════════════════════════════════════════════════════════════
#  Plot 2: Final Crossover Estimates (bar chart)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_ground_truth_accuracy(
    trackers: Dict[str, Any],
    plots_dir: Path,
    *,
    tail_n: int = 500,
) -> None:
    """Bar chart of final crossover estimate per exploration strategy.

    Parameters
    ----------
    trackers : dict
        ``{strategy_name: MetricsTracker | list[dict]}``
    plots_dir : Path
        Directory to save the figure.
    tail_n : int
        Number of last records used for std-dev error bars.
    """
    names: List[str] = []
    means: List[float] = []
    stds: List[float] = []
    colors: List[str] = []

    for name, trk in trackers.items():
        records = _extract_records(trk)
        crossovers = [
            float(c)
            for c in _field(records, "crossover_estimate")
            if c is not None
        ]
        if not crossovers:
            continue

        tail = crossovers[-tail_n:]
        names.append(name)
        means.append(np.mean(tail))
        stds.append(np.std(tail))
        colors.append(_get_color(name))

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, capsize=6, color=colors, edgecolor="white",
                  linewidth=0.8, alpha=0.85)

    # Annotate bar values
    for bar, m, s in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.005,
            f"{m:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Crossover Selectivity")
    ax.set_title("Final Crossover Estimates")
    ax.grid(axis="y", alpha=0.3)

    path = plots_dir / "02_final_crossover_estimates.png"
    fig.savefig(path, **_SAVE_KW)
    plt.close(fig)
    logger.info("Saved plot → %s", path)


# ═══════════════════════════════════════════════════════════════════════════════
#  Plot 3: Cumulative Regret
# ═══════════════════════════════════════════════════════════════════════════════

def plot_cumulative_regret(
    trackers: Dict[str, Any],
    plots_dir: Path,
) -> None:
    """Plot cumulative regret over global steps.

    Regret at step *t* = oracle_reward_t − agent_reward_t.

    Parameters
    ----------
    trackers : dict
        ``{strategy_name: MetricsTracker | list[dict]}``
    plots_dir : Path
        Directory to save the figure.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    for name, trk in trackers.items():
        records = _extract_records(trk)

        # Compute per-step regret
        cumulative: List[float] = []
        running = 0.0
        for r in records:
            if isinstance(r, dict):
                r_rew = float(r.get("reward", 0.0))
                o_rew = float(r.get("oracle_reward", 0.0))
            else:
                r_rew = float(getattr(r, "reward", 0.0))
                o_rew = float(getattr(r, "oracle_reward", 0.0))
            
            regret = o_rew - r_rew
            running += max(regret, 0.0)  # Clamp negative regret to 0
            cumulative.append(running)

        final_regret = cumulative[-1] if cumulative else 0.0
        label = f"{name}  (final={final_regret:.1f})"
        ax.plot(
            cumulative, label=label, color=_get_color(name), linewidth=1.5,
        )

    ax.set_xlabel("Global Step (query iteration)")
    ax.set_ylabel("Cumulative Regret")
    ax.set_title("Cumulative Regret Analysis")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    path = plots_dir / "03_cumulative_regret.png"
    fig.savefig(path, **_SAVE_KW)
    plt.close(fig)
    logger.info("Saved plot → %s", path)


# ═══════════════════════════════════════════════════════════════════════════════
#  Plot 4: Recall vs Latency Scatter
# ═══════════════════════════════════════════════════════════════════════════════

def plot_recall_latency_scatter(
    trackers: Dict[str, Any],
    r_target: float,
    l_max: float,
    plots_dir: Path,
) -> None:
    """Scatter plot of recall vs latency, coloured by reward.

    One subplot per exploration strategy (horizontal layout).

    Parameters
    ----------
    trackers : dict
        ``{strategy_name: MetricsTracker | list[dict]}``
    r_target : float
        Recall SLA target (horizontal dashed line).
    l_max : float
        Maximum acceptable latency (ms).
    plots_dir : Path
        Directory to save the figure.
    """
    n_strats = len(trackers)
    fig, axes = plt.subplots(1, n_strats, figsize=(6 * n_strats, 5), squeeze=False)
    axes = axes[0]

    for ax, (name, trk) in zip(axes, trackers.items()):
        records = _extract_records(trk)

        latencies = np.array([float(r["latency_ms"]) for r in records])
        recalls = np.array([float(r["recall"]) for r in records])
        rewards = np.array([float(r["reward"]) for r in records])

        sc = ax.scatter(
            latencies, recalls, c=rewards, cmap="RdYlGn",
            s=8, alpha=0.5, edgecolors="none",
            vmin=0.0, vmax=1.0,
        )

        # SLA line
        ax.axhline(r_target, color="red", linestyle="--", linewidth=1.0,
                    alpha=0.7, label=f"R_target={r_target}")

        # Region annotations
        ax.text(
            0.97, 0.97, "Safe Zone", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color="green",
            fontweight="bold", alpha=0.6,
        )
        near_miss_y = r_target - 0.04
        if near_miss_y > 0:
            ax.text(
                0.97, max(0.3, near_miss_y / 1.1), "Near Miss",
                transform=ax.transAxes, ha="right", va="center",
                fontsize=9, color="orange", fontweight="bold", alpha=0.6,
            )
        ax.text(
            0.97, 0.05, "Dangerous Slide", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, color="red",
            fontweight="bold", alpha=0.6,
        )

        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("Recall")
        ax.set_title(name, fontsize=11)
        ax.set_ylim(-0.02, 1.05)
        ax.legend(loc="lower left", fontsize=8, framealpha=0.8)
        ax.grid(True, alpha=0.2)

    # Shared colour bar
    cbar = fig.colorbar(sc, ax=axes.tolist(), shrink=0.8, pad=0.02)
    cbar.set_label("Reward")

    fig.suptitle("Recall vs. Latency (Soft Cliff SLA)", fontsize=13, y=1.02)
    fig.tight_layout()

    path = plots_dir / "04_recall_latency_scatter.png"
    fig.savefig(path, **_SAVE_KW)
    plt.close(fig)
    logger.info("Saved plot → %s", path)


# ═══════════════════════════════════════════════════════════════════════════════
#  Plot 5: Optimizer Overhead Histogram
# ═══════════════════════════════════════════════════════════════════════════════

def plot_optimizer_overhead(
    trackers: Dict[str, Any],
    plots_dir: Path,
) -> None:
    """Histogram of CBO decision overhead (µs) across all strategies.

    Parameters
    ----------
    trackers : dict
        ``{strategy_name: MetricsTracker | list[dict]}``
    plots_dir : Path
        Directory to save the figure.
    """
    # Aggregate overhead values from all strategies
    all_overhead: List[float] = []
    for name, trk in trackers.items():
        records = _extract_records(trk)
        all_overhead.extend(float(r["decision_overhead_us"]) for r in records)

    if not all_overhead:
        logger.warning("No overhead data — skipping overhead histogram.")
        return

    arr = np.array(all_overhead)
    mean_val = np.mean(arr)
    p50 = np.percentile(arr, 50)
    p95 = np.percentile(arr, 95)
    p99 = np.percentile(arr, 99)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Use automatic bin count capped at 100
    n_bins = min(100, max(20, len(arr) // 50))
    ax.hist(arr, bins=n_bins, color="#42A5F5", edgecolor="white",
            linewidth=0.5, alpha=0.85)

    # Vertical percentile lines
    line_kw = {"linewidth": 1.5, "alpha": 0.85}
    ax.axvline(mean_val, color="#E53935", linestyle="-", label=f"Mean = {mean_val:.1f} µs", **line_kw)
    ax.axvline(p50, color="#FB8C00", linestyle="--", label=f"p50 = {p50:.1f} µs", **line_kw)
    ax.axvline(p95, color="#8E24AA", linestyle="--", label=f"p95 = {p95:.1f} µs", **line_kw)
    ax.axvline(p99, color="#D32F2F", linestyle=":", label=f"p99 = {p99:.1f} µs", **line_kw)

    ax.set_xlabel("Overhead (µs)")
    ax.set_ylabel("Count")
    ax.set_title("CBO Decision Overhead")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    path = plots_dir / "05_optimizer_overhead.png"
    fig.savefig(path, **_SAVE_KW)
    plt.close(fig)
    logger.info("Saved plot → %s", path)


# ═══════════════════════════════════════════════════════════════════════════════
#  Public entry point
# ═══════════════════════════════════════════════════════════════════════════════

def generate_all_plots(
    cbo_dir: Path,
    trackers: Dict[str, Any],
    r_target: float,
    l_max: float,
) -> None:
    """Generate all 5 CBO analysis plots.

    Parameters
    ----------
    cbo_dir : Path
        Root output directory for this CBO run.
    trackers : dict
        ``{strategy_name: MetricsTracker | list[dict]}``
    r_target : float
        Recall SLA target.
    l_max : float
        Maximum acceptable latency (ms).
    """
    if not _HAS_MPL:
        logger.error(
            "matplotlib is not installed — cannot generate plots. "
            "Run: pip install 'matplotlib>=3.8.0'"
        )
        return

    # Apply a clean professional style
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            pass  # Fall back to default matplotlib style

    plots_dir = cbo_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print(f"\n  Generating CBO visualisations → {plots_dir}")

    plot_crossover_convergence(trackers, plots_dir)
    plot_ground_truth_accuracy(trackers, plots_dir)
    plot_cumulative_regret(trackers, plots_dir)
    plot_recall_latency_scatter(trackers, r_target, l_max, plots_dir)
    plot_optimizer_overhead(trackers, plots_dir)

    print(f"  ✓ 5 plots saved to {plots_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Standalone mode: regenerate plots from saved JSON
# ═══════════════════════════════════════════════════════════════════════════════

def _load_trackers_from_dir(cbo_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load raw JSON files from a CBO results directory.

    Returns
    -------
    trackers : dict
        ``{strategy_name: list_of_record_dicts}``
    """
    trackers: Dict[str, List[Dict[str, Any]]] = {}
    for json_path in sorted(cbo_dir.glob("cbo_*_raw.json")):
        # Extract strategy name from filename: cbo_<name>_raw.json
        stem = json_path.stem  # e.g. "cbo_tier_based_raw"
        # Remove prefix "cbo_" and suffix "_raw"
        name = stem.removeprefix("cbo_").removesuffix("_raw")
        with open(json_path) as f:
            data = json.load(f)
        if isinstance(data, list):
            trackers[name] = data
        elif isinstance(data, dict) and "records" in data:
            trackers[name] = data["records"]
        else:
            logger.warning("Unexpected JSON structure in %s — skipping.", json_path)
    return trackers


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Regenerate CBO visualisation plots from saved JSON data.",
    )
    parser.add_argument(
        "cbo_dir",
        type=Path,
        help="Path to CBO results directory (e.g. results/cbo_20260529_120000).",
    )
    parser.add_argument(
        "--r-target",
        type=float,
        default=None,
        help="Recall SLA target (default: read from experiment_meta.json or 0.93).",
    )
    parser.add_argument(
        "--l-max",
        type=float,
        default=None,
        help="Max acceptable latency in ms (default: read from experiment_meta.json or 20.0).",
    )
    args = parser.parse_args()

    if not args.cbo_dir.is_dir():
        print(f"Error: '{args.cbo_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    # Try to load metadata for defaults
    r_target = args.r_target
    l_max = args.l_max
    meta_path = args.cbo_dir / "experiment_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        if r_target is None:
            r_target = meta.get("r_target", 0.93)
        if l_max is None:
            l_max = meta.get("l_max", 20.0)
    else:
        if r_target is None:
            r_target = 0.93
        if l_max is None:
            l_max = 20.0

    trackers = _load_trackers_from_dir(args.cbo_dir)
    if not trackers:
        print("No CBO raw JSON files found in the directory.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(trackers)} strategy results: {list(trackers.keys())}")
    generate_all_plots(args.cbo_dir, trackers, r_target, l_max)
