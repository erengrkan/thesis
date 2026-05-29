"""
optimizer/plot_academic.py — High-Quality Academic Plots
===========================================================
Generates clean, aesthetic, publication-ready plots focusing on the
winning strategy (Softmax) to clearly tell the thesis story.
"""

import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline

# Academic aesthetic settings
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.2

os.makedirs("results/plots", exist_ok=True)

def get_latest_files():
    summaries = glob.glob("results/cbo_summary_*.csv")
    latest_summary = max(summaries, key=os.path.getctime)
    return os.path.basename(latest_summary).replace("cbo_summary_", "").replace(".csv", "")

def plot_routing_decision(ts):
    """Plot 1: Strategy Routing Decision (Area Plot)"""
    df = pd.read_csv(f"results/cbo_decisions_softmax_{ts}.csv")
    
    # Bin selectivities
    bins = np.linspace(0, 1.0, 21) # 5% increments
    df['sel_bin'] = pd.cut(df['selectivity'], bins=bins)
    
    # Calculate % Post-filter chosen in each bin
    bin_stats = df.groupby('sel_bin', observed=False).agg(
        total=('chosen_strategy', 'count'),
        post=('chosen_strategy', lambda x: (x == 'post_filter').sum())
    ).reset_index()
    
    bin_stats = bin_stats[bin_stats['total'] > 0].copy()
    bin_stats['post_rate'] = bin_stats['post'] / bin_stats['total'] * 100
    bin_stats['pre_rate'] = 100 - bin_stats['post_rate']
    bin_stats['sel_mid'] = bin_stats['sel_bin'].apply(lambda x: x.mid * 100)
    
    # Smooth the curves for aesthetic appeal
    x = bin_stats['sel_mid'].values
    y_pre = bin_stats['pre_rate'].values
    y_post = bin_stats['post_rate'].values
    
    x_smooth = np.linspace(x.min(), x.max(), 300)
    spl_pre = make_interp_spline(x, y_pre, k=3)
    spl_post = make_interp_spline(x, y_post, k=3)
    
    y_pre_smooth = np.clip(spl_pre(x_smooth), 0, 100)
    y_post_smooth = np.clip(spl_post(x_smooth), 0, 100)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.stackplot(x_smooth, y_pre_smooth, y_post_smooth, 
                 labels=['Pre-Filter (Bitmap + Brute-Force)', 'Post-Filter (HNSW Oversampling)'],
                 colors=['#3498db', '#e74c3c'], alpha=0.8)
    
    # Find exact crossover
    crossover_idx = np.abs(y_post_smooth - 50).argmin()
    crossover_x = x_smooth[crossover_idx]
    
    ax.axvline(crossover_x, color='black', linestyle='--', linewidth=2, alpha=0.7)
    ax.text(crossover_x + 2, 85, f'Switch Point\n~{crossover_x:.1f}%', 
            fontweight='bold', color='black', fontsize=12,
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.5'))
    
    ax.set_title("CBO Dynamic Routing: Strategy Selection Probability", pad=20, fontweight='bold', fontsize=16)
    ax.set_xlabel("Filter Selectivity (%)", fontweight='bold')
    ax.set_ylabel("Selection Rate (%)", fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.margins(x=0, y=0)
    
    # Legend formatting
    ax.legend(loc='upper right', frameon=True, edgecolor='black', shadow=True)
    
    plt.tight_layout()
    out_path = f"results/plots/academic_decision_{ts}.png"
    plt.savefig(out_path)
    print(f"Saved: {out_path}")
    plt.close()

def plot_qtable_crossover(ts):
    """Plot 2: Learned Q-Values and Crossover Point"""
    df = pd.read_csv(f"results/cbo_qtable_softmax_{ts}.csv")
    df['sel_mid'] = (df['range_lower'] + df['range_upper']) / 2 * 100
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot lines with markers
    ax.plot(df['sel_mid'], df['q_pre'], marker='o', markersize=8, linewidth=3, 
            color='#3498db', label='Pre-Filter Reward Expectation')
    ax.plot(df['sel_mid'], df['q_post'], marker='s', markersize=8, linewidth=3, 
            color='#e74c3c', label='Post-Filter Reward Expectation')
    
    # Highlight the crossover area
    ax.axvspan(10, 20, color='gray', alpha=0.15, label='Crossover Zone')
    
    ax.set_title("Learned Reward Expectations via Soft-Cliff SLA", pad=20, fontweight='bold', fontsize=16)
    ax.set_xlabel("Filter Selectivity (%)", fontweight='bold')
    ax.set_ylabel("Expected Reward (Q-Value)", fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.05)
    
    ax.legend(loc='lower right', frameon=True, edgecolor='black', shadow=True)
    
    plt.tight_layout()
    out_path = f"results/plots/academic_qtable_{ts}.png"
    plt.savefig(out_path)
    print(f"Saved: {out_path}")
    plt.close()

if __name__ == "__main__":
    ts = get_latest_files()
    plot_routing_decision(ts)
    plot_qtable_crossover(ts)
