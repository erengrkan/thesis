"""
selectivity_benchmark.py
========================
Tests all strategies across different selectivity levels using the full
388k dataset. Exports Latency and Recall per selectivity bucket.
"""

import csv
import logging
import random
import time
from pathlib import Path
import numpy as np

import config
from bitmap_index import BitmapIndex
from faiss_index import FAISSIndex
from filters import generate_filters
from strategies import BitmapPreFilter, BitmapHNSWPreFilter, PostFilter, BruteForce

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_selectivity_benchmark():
    out_dir = config.RESULTS_DIR / "selectivity_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    logger.info("Loading full index (388k docs)...")
    faiss_idx = FAISSIndex.load(config.INDEX_DIR)
    bitmap_idx = BitmapIndex.load(config.INDEX_DIR / "bitmap.idx")
    
    import pickle
    with open(config.INDEX_DIR / "all_metadatas.pkl", "rb") as fh:
        all_metadatas = pickle.load(fh)

    # 2. Generate queries
    np.random.seed(42)
    random.seed(42)
    n_queries = 50
    query_indices = random.sample(range(faiss_idx.hnsw_index.ntotal), n_queries)
    query_embeddings = faiss_idx._embeddings[query_indices]

    # 3. Generate filters
    logger.info("Generating filters...")
    filters = generate_filters(all_metadatas, config.SELECTIVITY_TARGETS)

    # 4. Initialize strategies
    strategies = {
        "PostFilter (HNSW)": PostFilter(faiss_idx, all_metadatas),
        "IDSelector (HNSW)": BitmapHNSWPreFilter(faiss_idx, bitmap_idx),
        "Bitmap (BruteForce)": BitmapPreFilter(faiss_idx, bitmap_idx),
    }
    brute_force = BruteForce(faiss_idx, all_metadatas)

    results = []

    print("\n" + "="*80)
    print(f"{'Selectivity':>11} | {'PostFilter (HNSW)':>20} | {'IDSelector (HNSW)':>20} | {'Bitmap (BruteForce)':>20}")
    print(f"{'':>11} | {'Latency':>9} {'Recall':>10} | {'Latency':>9} {'Recall':>10} | {'Latency':>9} {'Recall':>10}")
    print("-" * 80)

    for f_idx, fspec in enumerate(filters):
        # Arrays to store metrics
        metrics = {name: {"lat": [], "rec": []} for name in strategies}

        for qvec in query_embeddings:
            # Ground truth
            bf_res = brute_force.search(qvec, 10, fspec)
            ref_ids = set(bf_res.ids)

            for name, strat in strategies.items():
                res = strat.search(qvec, 10, fspec)
                recall = len(ref_ids & set(res.ids)) / len(ref_ids) if ref_ids else 1.0
                metrics[name]["lat"].append(res.total_time_ms)
                metrics[name]["rec"].append(recall)

        # Averages
        avg_pf_lat = np.mean(metrics["PostFilter (HNSW)"]["lat"])
        avg_pf_rec = np.mean(metrics["PostFilter (HNSW)"]["rec"])
        
        avg_ids_lat = np.mean(metrics["IDSelector (HNSW)"]["lat"])
        avg_ids_rec = np.mean(metrics["IDSelector (HNSW)"]["rec"])

        avg_bf_lat = np.mean(metrics["Bitmap (BruteForce)"]["lat"])
        avg_bf_rec = np.mean(metrics["Bitmap (BruteForce)"]["rec"])

        sel_pct = fspec.actual_selectivity * 100

        print(f"{sel_pct:>10.1f}% | {avg_pf_lat:>6.1f} ms {avg_pf_rec:>9.1%} | {avg_ids_lat:>6.1f} ms {avg_ids_rec:>9.1%} | {avg_bf_lat:>6.1f} ms {avg_bf_rec:>9.1%}")

        results.append({
            "selectivity_pct": round(sel_pct, 2),
            "filter_name": fspec.name,
            "pf_latency_ms": round(avg_pf_lat, 2),
            "pf_recall": round(avg_pf_rec, 4),
            "idsel_latency_ms": round(avg_ids_lat, 2),
            "idsel_recall": round(avg_ids_rec, 4),
            "bitmap_bf_latency_ms": round(avg_bf_lat, 2),
            "bitmap_bf_recall": round(avg_bf_rec, 4),
        })

    # Export to CSV
    csv_path = out_dir / "selectivity_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    logger.info("Saved detailed selectivity metrics to %s", csv_path)

if __name__ == "__main__":
    run_selectivity_benchmark()
