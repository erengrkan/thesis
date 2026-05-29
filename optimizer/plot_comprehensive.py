"""
optimizer/plot_comprehensive.py — Comprehensive Thesis Plots
============================================================
Generates a complete suite of academic plots for the CBO chapter,
incorporating decision boundaries, Q-values, latency/recall profiles,
and cross-strategy comparisons based on the corrected HNSW Pre-filter data.
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

def get_latest_ts():
    summaries = glob.glob("results/cbo_summary_*.csv")
    latest_summary = max(summaries, key=os.path.getctime)
    return os.path.basename(latest_summary).replace("cbo_summary_", "").replace(".csv", "")

def smooth_curve(x, y, points=300):
    # Ensure x is strictly increasing for spline
    idx = np.argsort(x)
    x_sorted, y_sorted = x[idx], y[idx]
    
    # Remove duplicates
    unique_x, indices = np.unique(x_sorted, return_index=True)
    unique_y = y_sorted[indices]
    
    if len(unique_x) < 4:
        return x_sorted, y_sorted
        
    x_smooth = np.linspace(unique_x.min(), unique_x.max(), points)
    spl = make_interp_spline(unique_x, unique_y, k=3)
    y_smooth = spl(x_smooth)
    return x_smooth, y_smooth

def plot_1_routing(ts):
    """Plot 1: Softmax Routing Decision (Area Plot)"""
    df = pd.read_csv(f"results/cbo_decisions_softmax_{ts}.csv")
    bins = np.linspace(0, 1.0, 21)
    df['sel_bin'] = pd.cut(df['selectivity'], bins=bins)
    
    bin_stats = df.groupby('sel_bin', observed=False).agg(
        total=('chosen_strategy', 'count'),
        post=('chosen_strategy', lambda x: (x == 'post_filter').sum())
    ).reset_index()
    
    bin_stats = bin_stats[bin_stats['total'] > 0].copy()
    bin_stats['post_rate'] = bin_stats['post'] / bin_stats['total'] * 100
    bin_stats['pre_rate'] = 100 - bin_stats['post_rate']
    bin_stats['sel_mid'] = bin_stats['sel_bin'].apply(lambda x: x.mid * 100)
    
    x_smooth, y_pre_smooth = smooth_curve(bin_stats['sel_mid'].values, bin_stats['pre_rate'].values)
    _, y_post_smooth = smooth_curve(bin_stats['sel_mid'].values, bin_stats['post_rate'].values)
    
    y_pre_smooth = np.clip(y_pre_smooth, 0, 100)
    y_post_smooth = np.clip(y_post_smooth, 0, 100)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.stackplot(x_smooth, y_pre_smooth, y_post_smooth, 
                 labels=['Pre-Filter (HNSW + IDSelector)', 'Post-Filter (HNSW Oversampling)'],
                 colors=['#3498db', '#e74c3c'], alpha=0.8)
    
    crossover_idx = np.abs(y_post_smooth - 50).argmin()
    crossover_x = x_smooth[crossover_idx]
    
    ax.axvline(crossover_x, color='black', linestyle='--', linewidth=2, alpha=0.7)
    ax.text(crossover_x + 2, 85, f'Switch Point\n~{crossover_x:.1f}%', 
            fontweight='bold', color='black', fontsize=12,
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.5'))
    
    ax.set_title("Fig 1: CBO Routing Probability (Softmax)", pad=20, fontweight='bold', fontsize=16)
    ax.set_xlabel("Filter Selectivity (%)", fontweight='bold')
    ax.set_ylabel("Selection Rate (%)", fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.margins(x=0, y=0)
    ax.legend(loc='lower right', frameon=True, edgecolor='black', shadow=True)
    
    plt.tight_layout()
    plt.savefig(f"results/plots/fig1_routing_{ts}.png")
    plt.close()

def plot_2_qtable(ts):
    """Plot 2: Learned Q-Values and Crossover (Softmax)"""
    df = pd.read_csv(f"results/cbo_qtable_softmax_{ts}.csv")
    df['sel_mid'] = (df['range_lower'] + df['range_upper']) / 2 * 100
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['sel_mid'], df['q_pre'], marker='o', markersize=6, linewidth=2.5, 
            color='#3498db', label='Q(Pre-Filter)')
    ax.plot(df['sel_mid'], df['q_post'], marker='s', markersize=6, linewidth=2.5, 
            color='#e74c3c', label='Q(Post-Filter)')
    
    # Find approximate crossover bucket
    crossover_zone = df[(df['q_post'] > df['q_pre']) & (df['range_lower'] > 0.05)].head(1)
    if not crossover_zone.empty:
        lo, hi = crossover_zone['range_lower'].values[0]*100, crossover_zone['range_upper'].values[0]*100
        ax.axvspan(lo, hi, color='gray', alpha=0.2, label='Crossover Bucket')
    
    ax.set_title("Fig 2: Learned Reward Expectations (Q-Table)", pad=20, fontweight='bold', fontsize=16)
    ax.set_xlabel("Filter Selectivity (%)", fontweight='bold')
    ax.set_ylabel("Expected Reward (0-1)", fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right', frameon=True, edgecolor='black', shadow=True)
    
    plt.tight_layout()
    plt.savefig(f"results/plots/fig2_qtable_{ts}.png")
    plt.close()

def plot_3_performance(ts):
    """Plot 3: Latency and Recall Profile of the CBO (Softmax)"""
    df = pd.read_csv(f"results/cbo_decisions_softmax_{ts}.csv")
    
    # We want to show moving average of latency and recall
    df_sorted = df.sort_values('selectivity')
    window = max(10, len(df) // 20)
    
    df_sorted['sel_pct'] = df_sorted['selectivity'] * 100
    df_sorted['lat_ma'] = df_sorted['latency_ms'].rolling(window, center=True).mean()
    df_sorted['rec_ma'] = df_sorted['recall'].rolling(window, center=True).mean()
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color1 = '#2c3e50'
    ax1.set_xlabel('Filter Selectivity (%)', fontweight='bold')
    ax1.set_ylabel('End-to-End Latency (ms)', color=color1, fontweight='bold')
    ax1.plot(df_sorted['sel_pct'], df_sorted['lat_ma'], color=color1, linewidth=3, label='Latency (MA)')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xlim(0, 100)
    
    ax2 = ax1.twinx()  
    color2 = '#27ae60'
    ax2.set_ylabel('Recall@K', color=color2, fontweight='bold')
    ax2.plot(df_sorted['sel_pct'], df_sorted['rec_ma'], color=color2, linewidth=3, linestyle='--', label='Recall (MA)')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, 1.05)
    
    # Target SLA line
    ax2.axhline(0.95, color='red', linestyle=':', alpha=0.7, label='SLA Target (95%)')
    
    fig.suptitle("Fig 3: CBO Performance Profile (Latency & Recall)", fontweight='bold', fontsize=16)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', frameon=True, edgecolor='black', shadow=True)
    
    plt.tight_layout()
    plt.savefig(f"results/plots/fig3_performance_{ts}.png")
    plt.close()

def plot_4_strategy_comparison(ts):
    """Plot 4: Cross-Strategy Bar Charts"""
    df = pd.read_csv(f"results/cbo_summary_{ts}.csv")
    
    # Clean strategy names
    df['strategy'] = df['strategy'].str.replace('_', ' ').str.title()
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    sns.barplot(data=df, x='strategy', y='mean_latency_ms', ax=ax1, palette='Blues_d')
    ax1.set_title("Mean Latency (ms)")
    ax1.set_ylabel("")
    ax1.set_xlabel("")
    
    sns.barplot(data=df, x='strategy', y='mean_recall', ax=ax2, palette='Greens_d')
    ax2.set_title("Mean Recall")
    ax2.set_ylim(0.8, 1.0) # Zoom in on interesting range
    ax2.set_ylabel("")
    ax2.set_xlabel("")
    ax2.axhline(0.95, color='red', linestyle='--', alpha=0.5)
    
    sns.barplot(data=df, x='strategy', y='mean_reward', ax=ax3, palette='Oranges_d')
    ax3.set_title("Mean SLA Reward")
    ax3.set_ylim(0, 1.0)
    ax3.set_ylabel("")
    ax3.set_xlabel("")
    
    fig.suptitle("Fig 4: Exploration Strategy Comparison", fontweight='bold', fontsize=16, y=1.05)
    
    plt.tight_layout()
    plt.savefig(f"results/plots/fig4_comparison_{ts}.png")
    plt.close()

if __name__ == "__main__":
    ts = get_latest_ts()
    print(f"Generating comprehensive plots for run: {ts}")
    plot_1_routing(ts)
    plot_2_qtable(ts)
    plot_3_performance(ts)
    plot_4_strategy_comparison(ts)
    print("Done.")
