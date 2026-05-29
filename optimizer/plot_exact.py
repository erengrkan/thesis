"""
optimizer/plot_exact.py
=======================
Generates exact requested plots comparing ALL 3 methods against selectivity:
1. Transition Point (Where each method switches from Pre to Post)
2. Latency Performance (How each method performs across selectivity)
3. Recall Performance (How each method performs across selectivity)
"""

import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
os.makedirs("results/plots", exist_ok=True)

def get_latest_ts():
    summaries = glob.glob("results/cbo_summary_*.csv")
    latest_summary = max(summaries, key=os.path.getctime)
    return os.path.basename(latest_summary).replace("cbo_summary_", "").replace(".csv", "")

def plot_exact_comparison(ts):
    strategies = {
        'tier_based': ('Tier-Based', '#34495e', '--'),
        'exponential_decay': ('Exponential Decay', '#2ecc71', '-.'),
        'softmax': ('Softmax', '#e74c3c', '-')
    }
    
    # Bins for selectivity
    bins = np.linspace(0, 1.0, 21)
    
    all_data = []
    
    for strat_key, (strat_name, color, line) in strategies.items():
        df = pd.read_csv(f"results/cbo_decisions_{strat_key}_{ts}.csv")
        df['sel_bin'] = pd.cut(df['selectivity'], bins=bins)
        
        # We only want to plot the last 1000 queries to see converged behavior
        df = df[df['episode'] > 1000]
        
        bin_stats = df.groupby('sel_bin', observed=False).agg(
            total=('chosen_strategy', 'count'),
            post=('chosen_strategy', lambda x: (x == 'post_filter').sum()),
            avg_lat=('latency_ms', 'mean'),
            avg_rec=('recall', 'mean')
        ).reset_index()
        
        bin_stats = bin_stats[bin_stats['total'] > 0].copy()
        bin_stats['post_rate'] = bin_stats['post'] / bin_stats['total'] * 100
        bin_stats['sel_mid'] = bin_stats['sel_bin'].apply(lambda x: x.mid * 100)
        bin_stats['strategy'] = strat_name
        all_data.append(bin_stats)
    
    # 1. Plot Transition (Crossover)
    plt.figure(figsize=(10, 6))
    for i, (strat_key, (strat_name, color, line)) in enumerate(strategies.items()):
        data = all_data[i]
        plt.plot(data['sel_mid'], data['post_rate'], label=strat_name, 
                 color=color, linestyle=line, linewidth=3, marker='o')
        
    plt.axhline(50, color='gray', linestyle=':', label='50% Transition Boundary')
    plt.title("Where Do They Switch? (Pre-Filter to Post-Filter Transition)", fontweight='bold')
    plt.xlabel("Filter Selectivity (%)", fontweight='bold')
    plt.ylabel("% of Time Post-Filter is Chosen", fontweight='bold')
    plt.ylim(-5, 105)
    plt.xlim(0, 100)
    plt.legend()
    plt.savefig(f"results/plots/exact_transition_{ts}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Plot Latency
    plt.figure(figsize=(10, 6))
    for i, (strat_key, (strat_name, color, line)) in enumerate(strategies.items()):
        data = all_data[i]
        plt.plot(data['sel_mid'], data['avg_lat'], label=strat_name, 
                 color=color, linestyle=line, linewidth=3, marker='s')
        
    plt.title("Latency Performance vs Selectivity", fontweight='bold')
    plt.xlabel("Filter Selectivity (%)", fontweight='bold')
    plt.ylabel("Average Latency (ms)", fontweight='bold')
    plt.xlim(0, 100)
    plt.legend()
    plt.savefig(f"results/plots/exact_latency_{ts}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Plot Recall
    plt.figure(figsize=(10, 6))
    for i, (strat_key, (strat_name, color, line)) in enumerate(strategies.items()):
        data = all_data[i]
        plt.plot(data['sel_mid'], data['avg_rec'], label=strat_name, 
                 color=color, linestyle=line, linewidth=3, marker='^')
        
    plt.axhline(0.95, color='red', linestyle='--', alpha=0.5, label='SLA Target (95%)')
    plt.title("Recall Performance vs Selectivity", fontweight='bold')
    plt.xlabel("Filter Selectivity (%)", fontweight='bold')
    plt.ylabel("Average Recall@K", fontweight='bold')
    plt.ylim(0.8, 1.05)
    plt.xlim(0, 100)
    plt.legend()
    plt.savefig(f"results/plots/exact_recall_{ts}.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    ts = get_latest_ts()
    plot_exact_comparison(ts)
