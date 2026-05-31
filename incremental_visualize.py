"""
incremental_visualize.py — Visualization for Incremental Data Growth Experiment
==================================================================================
Generates 4 diagnostic plots showing how the CBO adapts as the dataset grows.

Usage:
    Imported by incremental_benchmark.py automatically.
    Can also be called standalone:
        python incremental_visualize.py results/incremental_YYYYMMDD_HHMMSS
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

logger = logging.getLogger(__name__)

# ── Plot defaults ─────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.grid": True,
    "axes.grid.which": "both",
    "grid.alpha": 0.3,
    "font.size": 10,
})

_SAVE_KW = dict(dpi=150, bbox_inches="tight", facecolor="white")

COLORS = {
    "cbo": "#2196F3",
    "bitmap": "#FF5722",
    "post": "#4CAF50",
    "gain_bitmap": "#FF9800",
    "gain_post": "#9C27B0",
    "heatmap_bitmap": "#1565C0",
    "heatmap_post": "#C62828",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Plot 1: Crossover Adaptation
# ═══════════════════════════════════════════════════════════════════════════════

def plot_crossover_adaptation(phases_data: List[Dict], plots_dir: Path) -> None:
    """Show how the crossover point shifts as data grows."""
    fig, ax = plt.subplots(figsize=(10, 5))

    n_docs = [p["n_docs"] for p in phases_data]
    crossovers = [p.get("cbo_final_crossover") for p in phases_data]

    # Filter out None values
    valid = [(d, c) for d, c in zip(n_docs, crossovers) if c is not None]
    if not valid:
        logger.warning("No valid crossover data to plot.")
        plt.close(fig)
        return

    x_vals, y_vals = zip(*valid)
    x_labels = [f"{x // 1000}k" for x in x_vals]

    ax.plot(
        range(len(x_vals)), y_vals,
        "o-", color=COLORS["cbo"], linewidth=2.5, markersize=10,
        label="CBO Crossover", zorder=5,
    )

    # Annotate each point
    for i, (xl, yv) in enumerate(zip(x_labels, y_vals)):
        ax.annotate(
            f"{yv:.3f}",
            (i, yv),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=9,
            fontweight="bold",
            color=COLORS["cbo"],
        )

    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Dataset Size", fontsize=12)
    ax.set_ylabel("Crossover Selectivity", fontsize=12)
    ax.set_title("Crossover Point Adaptation Across Data Growth", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=11)
    ax.set_ylim(0, 1)

    path = plots_dir / "01_crossover_adaptation.png"
    fig.savefig(path, **_SAVE_KW)
    plt.close(fig)
    logger.info("Saved plot → %s", path)


# ═══════════════════════════════════════════════════════════════════════════════
#  Plot 2: Reward Comparison (3 Lines)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_reward_comparison(phases_data: List[Dict], plots_dir: Path) -> None:
    """Compare CBO vs always-bitmap vs always-postfilter rewards."""
    fig, ax = plt.subplots(figsize=(10, 5))

    n_docs = [p["n_docs"] for p in phases_data]
    x_labels = [f"{d // 1000}k" for d in n_docs]
    x_pos = range(len(n_docs))

    cbo_rewards = [p["cbo_mean_reward"] for p in phases_data]
    bitmap_rewards = [p["baseline_bitmap_mean_reward"] for p in phases_data]
    post_rewards = [p["baseline_post_mean_reward"] for p in phases_data]

    ax.plot(
        x_pos, cbo_rewards, "o-",
        color=COLORS["cbo"], linewidth=2.5, markersize=10,
        label="CBO (Softmax)", zorder=5,
    )
    ax.plot(
        x_pos, bitmap_rewards, "s--",
        color=COLORS["bitmap"], linewidth=2, markersize=8,
        label="Always BitmapPreFilter", alpha=0.85,
    )
    ax.plot(
        x_pos, post_rewards, "^--",
        color=COLORS["post"], linewidth=2, markersize=8,
        label="Always PostFilter", alpha=0.85,
    )

    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Dataset Size", fontsize=12)
    ax.set_ylabel("Mean Reward", fontsize=12)
    ax.set_title("Reward Comparison: CBO vs Static Baselines", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=11)

    path = plots_dir / "02_reward_comparison.png"
    fig.savefig(path, **_SAVE_KW)
    plt.close(fig)
    logger.info("Saved plot → %s", path)


# ═══════════════════════════════════════════════════════════════════════════════
#  Plot 3: Gain Bar Chart
# ═══════════════════════════════════════════════════════════════════════════════

def plot_gain_bars(phases_data: List[Dict], plots_dir: Path) -> None:
    """Show percentage gain of CBO over each baseline at every phase."""
    fig, ax = plt.subplots(figsize=(10, 5))

    n_docs = [p["n_docs"] for p in phases_data]
    x_labels = [f"{d // 1000}k" for d in n_docs]
    x = np.arange(len(n_docs))
    width = 0.35

    gains_bitmap = [p["gain_vs_bitmap_pct"] for p in phases_data]
    gains_post = [p["gain_vs_post_pct"] for p in phases_data]

    bars1 = ax.bar(
        x - width / 2, gains_bitmap, width,
        label="Gain vs Always-Bitmap",
        color=COLORS["gain_bitmap"], edgecolor="white",
    )
    bars2 = ax.bar(
        x + width / 2, gains_post, width,
        label="Gain vs Always-PostFilter",
        color=COLORS["gain_post"], edgecolor="white",
    )

    # Value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            sign = "+" if height >= 0 else ""
            ax.annotate(
                f"{sign}{height:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4 if height >= 0 else -14),
                textcoords="offset points",
                ha="center", va="bottom" if height >= 0 else "top",
                fontsize=9, fontweight="bold",
            )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Dataset Size", fontsize=12)
    ax.set_ylabel("Gain (%)", fontsize=12)
    ax.set_title("CBO Gain Over Static Baselines", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=11)

    path = plots_dir / "03_gain_bars.png"
    fig.savefig(path, **_SAVE_KW)
    plt.close(fig)
    logger.info("Saved plot → %s", path)


# ═══════════════════════════════════════════════════════════════════════════════
#  Plot 4: Q-Table Heatmap Evolution
# ═══════════════════════════════════════════════════════════════════════════════

def plot_qtable_heatmap(phases_data: List[Dict], qtable_dir: Path, plots_dir: Path) -> None:
    """Show Q-value advantage (Q_bitmap - Q_post) as a heatmap per phase."""
    n_phases = len(phases_data)
    fig, axes = plt.subplots(n_phases, 1, figsize=(14, 3 * n_phases), sharex=True)
    if n_phases == 1:
        axes = [axes]

    for ax, phase_info in zip(axes, phases_data):
        phase_num = phase_info["phase"]
        n_docs = phase_info["n_docs"]

        qtable_path = qtable_dir / f"phase_{phase_num}_qtable.json"
        if not qtable_path.exists():
            ax.set_title(f"Phase {phase_num} ({n_docs // 1000}k) — no Q-table found")
            continue

        with open(qtable_path) as f:
            qtable = json.load(f)

        bucket_labels = [b["bucket_id"] for b in qtable]
        diffs = [b["q_bitmap"] - b["q_post"] for b in qtable]

        colors = []
        for d in diffs:
            if d > 0:
                # Bitmap is better → blue
                intensity = min(abs(d) / 0.5, 1.0)
                colors.append((0.1, 0.3 + 0.5 * (1 - intensity), 0.8 + 0.2 * (1 - intensity)))
            else:
                # PostFilter is better → red
                intensity = min(abs(d) / 0.5, 1.0)
                colors.append((0.8 + 0.2 * (1 - intensity), 0.3 + 0.5 * (1 - intensity), 0.1))

        bars = ax.bar(range(len(diffs)), diffs, color=colors, edgecolor="gray", linewidth=0.5)

        ax.axhline(0, color="black", linewidth=1)
        ax.set_ylabel("Q_bitmap − Q_post")
        ax.set_title(
            f"Phase {phase_num}: {n_docs // 1000}k docs  |  "
            f"Crossover ≈ {phase_info.get('cbo_final_crossover', 'N/A')}",
            fontsize=11, fontweight="bold",
        )

        # Show crossover marker
        crossover = phase_info.get("cbo_final_crossover")
        if crossover is not None:
            # Find the bucket closest to the crossover
            for i, b in enumerate(qtable):
                if b["lower"] <= crossover < b["upper"]:
                    ax.axvline(i, color="red", linewidth=2, linestyle="--", alpha=0.8, label="Crossover")
                    ax.legend(loc="upper right", fontsize=9)
                    break

    # X-axis labels on bottom plot only
    if qtable_path.exists():
        axes[-1].set_xticks(range(len(bucket_labels)))
        axes[-1].set_xticklabels(bucket_labels, rotation=45, ha="right", fontsize=7)
    axes[-1].set_xlabel("Selectivity Bucket")

    fig.suptitle("Q-Table Evolution: Bitmap Advantage per Bucket", fontsize=14, y=1.02)
    fig.tight_layout()

    path = plots_dir / "04_qtable_heatmap.png"
    fig.savefig(path, **_SAVE_KW)
    plt.close(fig)
    logger.info("Saved plot → %s", path)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def generate_incremental_plots(out_dir: Path, phase_results=None) -> None:
    """Generate all incremental experiment plots.

    Can accept PhaseResult dataclass list or load from JSON.
    """
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load data from JSON if not passed directly
    summary_path = out_dir / "phases_summary.json"
    if phase_results is not None:
        from dataclasses import asdict
        phases_data = [
            {
                "phase": pr.phase,
                "n_docs": pr.n_docs,
                "cbo_mean_reward": pr.cbo_mean_reward,
                "cbo_final_crossover": pr.cbo_final_crossover,
                "cbo_epoch_rewards": pr.cbo_epoch_rewards,
                "cbo_epoch_crossovers": pr.cbo_epoch_crossovers,
                "baseline_bitmap_mean_reward": pr.baseline_bitmap_mean_reward,
                "baseline_post_mean_reward": pr.baseline_post_mean_reward,
                "gain_vs_bitmap_pct": pr.gain_vs_bitmap_pct,
                "gain_vs_post_pct": pr.gain_vs_post_pct,
            }
            for pr in phase_results
        ]
    elif summary_path.exists():
        with open(summary_path) as f:
            data = json.load(f)
        phases_data = data["results"]
    else:
        logger.error("No data found. Pass phase_results or ensure %s exists.", summary_path)
        return

    print(f"\n  Generating incremental plots → {plots_dir}")

    plot_crossover_adaptation(phases_data, plots_dir)
    plot_reward_comparison(phases_data, plots_dir)
    plot_gain_bars(phases_data, plots_dir)
    plot_qtable_heatmap(phases_data, out_dir, plots_dir)

    print(f"  ✓ 4 plots saved to {plots_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python incremental_visualize.py <results_dir>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist")
        sys.exit(1)

    generate_incremental_plots(results_dir)
