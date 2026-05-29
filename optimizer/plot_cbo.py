"""
optimizer/plot_cbo.py — CBO Visualization
==========================================
Generates thesis-ready charts showing how the 3 exploration strategies
behave across different selectivities.

Plots generated:
1. Strategy Selection Rate vs Selectivity (for all 3 strategies)
2. Final Q-Table values (Q_pre vs Q_post) vs Selectivity
"""

import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure output dir exists
os.makedirs("results/plots", exist_ok=True)

def get_latest_files():
    """Find the latest CSV files for decisions and Q-tables."""
    strategies = ['tier_based', 'exponential_decay', 'softmax']
    latest_ts = None
    
    # Find latest timestamp from summary files
    summaries = glob.glob("results/cbo_summary_*.csv")
    if not summaries:
        raise FileNotFoundError("No benchmark results found.")
    
    latest_summary = max(summaries, key=os.path.getctime)
    # File is like "results/cbo_summary_20260526_234633.csv"
    latest_ts = os.path.basename(latest_summary).replace("cbo_summary_", "").replace(".csv", "")
    
    return latest_ts

def plot_selection_rate(ts):
    """Plot the % of times Post-filter was chosen vs Selectivity."""
    strategies = ['tier_based', 'exponential_decay', 'softmax']
    plt.figure(figsize=(15, 5))
    
    for i, strat in enumerate(strategies, 1):
        df = pd.read_csv(f"results/cbo_decisions_{strat}_{ts}.csv")
        
        # Bin selectivities into 5% increments
        bins = [x/100.0 for x in range(0, 105, 5)]
        labels = [f"{x}%-{x+5}%" for x in range(0, 100, 5)]
        
        df['sel_bin'] = pd.cut(df['selectivity'], bins=bins, labels=labels, right=False)
        
        # Calculate % Post-filter chosen in each bin
        bin_stats = df.groupby('sel_bin', observed=False).agg(
            total=('chosen_strategy', 'count'),
            post=('chosen_strategy', lambda x: (x == 'post_filter').sum())
        ).reset_index()
        
        # Filter empty bins
        bin_stats = bin_stats[bin_stats['total'] > 0].copy()
        bin_stats['post_rate'] = bin_stats['post'] / bin_stats['total'] * 100
        
        # Plot
        plt.subplot(1, 3, i)
        sns.barplot(data=bin_stats, x='sel_bin', y='post_rate', color='steelblue')
        plt.axhline(50, color='red', linestyle='--', alpha=0.5, label='50% Crossover')
        plt.title(f"{strat.replace('_', ' ').title()}\nStrategy Selection Rate")
        plt.xlabel("Selectivity Range")
        plt.ylabel("% Post-Filter Selected")
        plt.ylim(0, 100)
        plt.xticks(rotation=45, ha='right')
        
    plt.tight_layout()
    out_path = f"results/plots/cbo_selection_rate_{ts}.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()

def plot_qtable(ts):
    """Plot Final Q_pre and Q_post vs Selectivity."""
    strategies = ['tier_based', 'exponential_decay', 'softmax']
    plt.figure(figsize=(15, 5))
    
    for i, strat in enumerate(strategies, 1):
        df = pd.read_csv(f"results/cbo_qtable_{strat}_{ts}.csv")
        
        # Midpoint of bucket for plotting
        df['sel_mid'] = (df['range_lower'] + df['range_upper']) / 2 * 100
        
        plt.subplot(1, 3, i)
        plt.plot(df['sel_mid'], df['q_pre'], 'o-', label='Q (Pre-filter)', color='red')
        plt.plot(df['sel_mid'], df['q_post'], 's-', label='Q (Post-filter)', color='blue')
        
        plt.title(f"{strat.replace('_', ' ').title()}\nLearned Q-Values")
        plt.xlabel("Selectivity (%)")
        plt.ylabel("Expected Reward (Q)")
        plt.ylim(0, 1.1)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
    plt.tight_layout()
    out_path = f"results/plots/cbo_qtable_values_{ts}.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()

if __name__ == "__main__":
    ts = get_latest_files()
    print(f"Generating plots for run timestamp: {ts}")
    plot_selection_rate(ts)
    plot_qtable(ts)
