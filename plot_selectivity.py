"""
plot_selectivity.py
===================
Generates summary charts for the Selectivity Benchmark results.
Shows Latency vs Selectivity and Recall vs Selectivity.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import config

def generate_selectivity_plots():
    csv_path = config.RESULTS_DIR / "selectivity_analysis" / "selectivity_metrics.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    df = df.sort_values(by="selectivity_pct")

    # Plot settings
    plt.rcParams.update({
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "axes.grid": True,
        "axes.grid.which": "both",
        "grid.alpha": 0.3,
        "font.size": 11,
    })
    
    colors = {
        "pf": "#4CAF50",    # Green
        "idsel": "#9C27B0", # Purple
        "bf": "#FF5722",    # Deep Orange
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Plot 1: Latency vs Selectivity ---
    ax1.plot(df["selectivity_pct"], df["bitmap_bf_latency_ms"], marker='s', color=colors["bf"], linewidth=2, label="Bitmap (Brute-Force)")
    ax1.plot(df["selectivity_pct"], df["idsel_latency_ms"], marker='D', color=colors["idsel"], linewidth=2, label="IDSelector (HNSW)")
    ax1.plot(df["selectivity_pct"], df["pf_latency_ms"], marker='^', color=colors["pf"], linewidth=2, label="PostFilter (HNSW)")
    
    # Highlight the L_MAX SLA (20ms)
    ax1.axhline(config.CBO_L_MAX, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label=f"L_MAX SLA ({config.CBO_L_MAX}ms)")

    ax1.set_xlabel("Selectivity (%)", fontsize=12)
    ax1.set_ylabel("Latency (ms)", fontsize=12)
    ax1.set_title("Latency vs. Selectivity (388k Docs)", fontsize=14, fontweight="bold")
    ax1.set_ylim(0, 150) # Cap at 150ms for better readability
    ax1.legend()

    # --- Plot 2: Recall vs Selectivity ---
    ax2.plot(df["selectivity_pct"], df["bitmap_bf_recall"] * 100, marker='s', color=colors["bf"], linewidth=2, label="Bitmap (Brute-Force)")
    ax2.plot(df["selectivity_pct"], df["idsel_recall"] * 100, marker='D', color=colors["idsel"], linewidth=2, label="IDSelector (HNSW)")
    ax2.plot(df["selectivity_pct"], df["pf_recall"] * 100, marker='^', color=colors["pf"], linewidth=2, label="PostFilter (HNSW)")

    # Highlight the R_TARGET SLA (90%)
    ax2.axhline(config.CBO_R_TARGET * 100, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label=f"R_TARGET SLA ({int(config.CBO_R_TARGET*100)}%)")

    ax2.set_xlabel("Selectivity (%)", fontsize=12)
    ax2.set_ylabel("Recall (%)", fontsize=12)
    ax2.set_title("Recall vs. Selectivity (388k Docs)", fontsize=14, fontweight="bold")
    ax2.set_ylim(70, 102) # Focus on the 70-100% range
    ax2.legend()

    plt.tight_layout()
    
    out_path = config.RESULTS_DIR / "selectivity_analysis" / "selectivity_summary.png"
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Plot saved to: {out_path}")

if __name__ == "__main__":
    generate_selectivity_plots()
