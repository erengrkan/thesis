"""
benchmark.py — FAISS Filtering Strategy Benchmark Suite
=========================================================
Runs filtered KNN queries across four strategies:
  - BruteForce       (exhaustive search — ground truth)
  - NaivePreFilter   (linear metadata scan → subset search)
  - PostFilter       (HNSW oversampling → Python filter)
  - BitmapPreFilter  (Roaring Bitmap → subset search)

For every (filter_spec, top_k, strategy, query) combination, records:
  - filter_time_ms, search_time_ms, total_time_ms
  - candidates_after_filter, results_returned
  - recall@K (vs brute-force ground truth)

Output (3 files per run):
  - benchmark_TIMESTAMP_raw.csv       — every individual query measurement
  - benchmark_TIMESTAMP_summary.csv   — aggregated statistics per combination
  - benchmark_TIMESTAMP.json          — full metadata + summary results
"""

from __future__ import annotations

import csv
import json
import logging
import pickle
import random
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import config
from bitmap_index import BitmapIndex
from faiss_index import FAISSIndex
from filters import generate_filters
from strategies import (
    BitmapPreFilter,
    BruteForce,
    FilterStrategy,
    NaivePreFilter,
    PostFilter,
    SearchResult,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Raw CSV field names ────────────────────────────────────────────────────────
RAW_FIELDS = [
    "strategy",
    "filter_name",
    "target_selectivity",
    "actual_selectivity",
    "top_k",
    "query_idx",
    "filter_time_ms",
    "search_time_ms",
    "total_time_ms",
    "candidates_after_filter",
    "results_returned",
    "recall",
]

# ── Summary CSV field names ────────────────────────────────────────────────────
SUMMARY_FIELDS = [
    "strategy",
    "filter_name",
    "target_selectivity",
    "actual_selectivity",
    "top_k",
    "num_queries",
    "mean_latency_ms",
    "median_latency_ms",
    "std_latency_ms",
    "min_latency_ms",
    "max_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "mean_filter_ms",
    "median_filter_ms",
    "mean_search_ms",
    "median_search_ms",
    "mean_recall",
    "min_recall",
    "max_recall",
    "std_recall",
    "avg_results_returned",
    "avg_candidates_after_filter",
    "qps",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Benchmark Runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_benchmark() -> None:
    """Execute the full benchmark suite."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("  FAISS Filtering Strategy — Benchmark Suite")
    print("=" * 70)

    # ── Load indices ───────────────────────────────────────────────────────
    logger.info("Loading FAISS index from %s …", config.INDEX_DIR)
    faiss_idx = FAISSIndex.load(config.INDEX_DIR)
    total_docs = faiss_idx.hnsw_index.ntotal
    print(f"  FAISS: {total_docs:,} vectors, dim={faiss_idx.dim}")

    logger.info("Loading Bitmap index …")
    bitmap_idx = BitmapIndex.load(config.INDEX_DIR / "bitmap.idx")
    print(f"  Bitmap: {bitmap_idx.total_docs:,} docs, {len(bitmap_idx.field_bitmaps)} fields")

    logger.info("Loading metadata list …")
    with open(config.INDEX_DIR / "all_metadatas.pkl", "rb") as fh:
        all_metadatas: List[Dict[str, Any]] = pickle.load(fh)
    print(f"  Metadata: {len(all_metadatas):,} entries")

    # ── Generate filters ───────────────────────────────────────────────────
    logger.info("Generating filters for targets: %s", config.SELECTIVITY_TARGETS)
    filters = generate_filters(all_metadatas, config.SELECTIVITY_TARGETS)
    print(f"\n  Generated {len(filters)} filters:")
    for f in filters:
        print(f"    target={f.target_selectivity:.0%}  actual={f.actual_selectivity:.1%}  {f.name}")

    # ── Prepare query embeddings ───────────────────────────────────────────
    n_queries = config.NUM_QUERY_SAMPLES
    all_indices = list(range(total_docs))
    random.seed(42)
    query_indices = random.sample(all_indices, min(n_queries, total_docs))
    query_embeddings = faiss_idx._embeddings[query_indices]  # already normalised
    print(f"\n  Query vectors: {len(query_embeddings)} samples")

    # ── Build strategies ───────────────────────────────────────────────────
    # BruteForce MUST be first — it provides the ground truth for recall.
    strategies: List[FilterStrategy] = [
        BruteForce(faiss_idx, all_metadatas),
        NaivePreFilter(faiss_idx, all_metadatas),
        PostFilter(faiss_idx, all_metadatas),
        BitmapPreFilter(faiss_idx, bitmap_idx),
    ]

    # ── Open raw CSV for streaming writes ──────────────────────────────────
    raw_csv_path = config.RESULTS_DIR / f"benchmark_{timestamp}_raw.csv"
    raw_csv_file = open(raw_csv_path, "w", newline="")
    raw_writer = csv.DictWriter(raw_csv_file, fieldnames=RAW_FIELDS)
    raw_writer.writeheader()

    # ── Benchmark Loop ─────────────────────────────────────────────────────
    summary_results: List[Dict[str, Any]] = []

    for fspec in filters:
        print(f"\n{'─' * 70}")
        print(f"  Filter: {fspec.name}")
        print(f"  Target selectivity: {fspec.target_selectivity:.0%}  "
              f"Actual: {fspec.actual_selectivity:.1%}")
        print(f"{'─' * 70}")

        for top_k in config.TOP_K_VALUES:
            print(f"\n  top_k={top_k}")

            # Ground truth: BruteForce (first strategy) gives perfect recall
            reference_ids: List[List[str]] = []

            for strategy in strategies:
                latencies: List[float] = []
                filter_times: List[float] = []
                search_times: List[float] = []
                candidates_list: List[int] = []
                results_returned_list: List[int] = []
                strategy_result_ids: List[List[str]] = []

                # Warmup
                for _ in range(config.WARMUP_QUERIES):
                    strategy.search(query_embeddings[0], top_k, fspec)

                # Measured queries
                for i in range(len(query_embeddings)):
                    result = strategy.search(query_embeddings[i], top_k, fspec)
                    latencies.append(result.total_time_ms)
                    filter_times.append(result.filter_time_ms)
                    search_times.append(result.search_time_ms)
                    candidates_list.append(result.candidates_after_filter)
                    results_returned_list.append(result.result_count)
                    strategy_result_ids.append(result.ids)

                # BruteForce is ground truth (perfect recall)
                if strategy.name == "brute_force":
                    reference_ids = strategy_result_ids

                # Compute per-query recall against BruteForce
                recall_values: List[float] = []
                for ref, actual in zip(reference_ids, strategy_result_ids):
                    if not ref:
                        recall_values.append(1.0)
                        continue
                    overlap = len(set(ref) & set(actual))
                    recall_values.append(overlap / len(ref) if ref else 1.0)

                # ── Write raw per-query rows ──────────────────────────
                for i in range(len(query_embeddings)):
                    raw_writer.writerow({
                        "strategy": strategy.name,
                        "filter_name": fspec.name,
                        "target_selectivity": round(fspec.target_selectivity, 4),
                        "actual_selectivity": round(fspec.actual_selectivity, 4),
                        "top_k": top_k,
                        "query_idx": i,
                        "filter_time_ms": round(filter_times[i], 4),
                        "search_time_ms": round(search_times[i], 4),
                        "total_time_ms": round(latencies[i], 4),
                        "candidates_after_filter": candidates_list[i],
                        "results_returned": results_returned_list[i],
                        "recall": round(recall_values[i], 4) if i < len(recall_values) else "",
                    })

                # ── Compute summary statistics ────────────────────────
                sorted_lat = sorted(latencies)
                n = len(latencies)

                summary_entry = {
                    "strategy": strategy.name,
                    "filter_name": fspec.name,
                    "target_selectivity": round(fspec.target_selectivity, 4),
                    "actual_selectivity": round(fspec.actual_selectivity, 4),
                    "top_k": top_k,
                    "num_queries": n,
                    "mean_latency_ms": round(statistics.mean(latencies), 4),
                    "median_latency_ms": round(statistics.median(latencies), 4),
                    "std_latency_ms": round(statistics.stdev(latencies), 4) if n > 1 else 0.0,
                    "min_latency_ms": round(min(latencies), 4),
                    "max_latency_ms": round(max(latencies), 4),
                    "p95_latency_ms": round(sorted_lat[int(n * 0.95)], 4),
                    "p99_latency_ms": round(sorted_lat[int(n * 0.99)], 4),
                    "mean_filter_ms": round(statistics.mean(filter_times), 4),
                    "median_filter_ms": round(statistics.median(filter_times), 4),
                    "mean_search_ms": round(statistics.mean(search_times), 4),
                    "median_search_ms": round(statistics.median(search_times), 4),
                    "mean_recall": round(statistics.mean(recall_values), 4) if recall_values else 0.0,
                    "min_recall": round(min(recall_values), 4) if recall_values else 0.0,
                    "max_recall": round(max(recall_values), 4) if recall_values else 0.0,
                    "std_recall": round(statistics.stdev(recall_values), 4) if len(recall_values) > 1 else 0.0,
                    "avg_results_returned": round(statistics.mean(results_returned_list), 2),
                    "avg_candidates_after_filter": round(statistics.mean(candidates_list), 1),
                    "qps": round(1000 / statistics.mean(latencies), 2) if statistics.mean(latencies) > 0 else 0,
                }
                summary_results.append(summary_entry)

                print(
                    f"    {strategy.name:18s}  "
                    f"median={summary_entry['median_latency_ms']:7.1f}ms  "
                    f"p95={summary_entry['p95_latency_ms']:7.1f}ms  "
                    f"QPS={summary_entry['qps']:6.1f}  "
                    f"recall={summary_entry['mean_recall']:.4f}  "
                    f"filter={summary_entry['mean_filter_ms']:.2f}ms  "
                    f"search={summary_entry['mean_search_ms']:.2f}ms  "
                    f"cands={summary_entry['avg_candidates_after_filter']:.0f}"
                )

    # ── Close raw CSV ──────────────────────────────────────────────────────
    raw_csv_file.close()

    # ── Save Summary CSV ───────────────────────────────────────────────────
    summary_csv_path = config.RESULTS_DIR / f"benchmark_{timestamp}_summary.csv"
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_results)

    # ── Save JSON (full metadata + summary) ────────────────────────────────
    json_path = config.RESULTS_DIR / f"benchmark_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "total_docs": total_docs,
                "num_queries": len(query_embeddings),
                "hnsw_M": config.HNSW_M,
                "hnsw_efSearch": config.HNSW_EF_SEARCH,
                "embedding_model": config.EMBEDDING_MODEL,
                "selectivity_targets": config.SELECTIVITY_TARGETS,
                "top_k_values": config.TOP_K_VALUES,
                "strategies": [s.name for s in strategies],
                "results": summary_results,
            },
            f,
            indent=2,
        )

    # ── Print output paths ─────────────────────────────────────────────────
    raw_rows = len(query_embeddings) * len(filters) * len(config.TOP_K_VALUES) * len(strategies)
    print(f"\n{'=' * 70}")
    print(f"  Results saved:")
    print(f"    Raw CSV:     {raw_csv_path}  ({raw_rows:,} rows)")
    print(f"    Summary CSV: {summary_csv_path}  ({len(summary_results)} rows)")
    print(f"    JSON:        {json_path}")
    print(f"{'=' * 70}")

    # ── Print Summary Table ────────────────────────────────────────────────
    print(f"\n{'Strategy':18s} {'Filter':<30s} {'Sel':>5s} {'k':>4s} "
          f"{'Med(ms)':>8s} {'p95(ms)':>8s} {'QPS':>7s} {'Recall':>7s} "
          f"{'Filt(ms)':>9s} {'Srch(ms)':>9s} {'Cands':>7s}")
    print("─" * 125)
    for r in summary_results:
        fname = r["filter_name"][:28]
        print(
            f"{r['strategy']:18s} "
            f"{fname:<30s} "
            f"{r['actual_selectivity']:>4.0%} "
            f"{r['top_k']:>4d} "
            f"{r['median_latency_ms']:>8.1f} "
            f"{r['p95_latency_ms']:>8.1f} "
            f"{r['qps']:>7.1f} "
            f"{r['mean_recall']:>7.4f} "
            f"{r['mean_filter_ms']:>9.2f} "
            f"{r['mean_search_ms']:>9.2f} "
            f"{r['avg_candidates_after_filter']:>7.0f}"
        )


if __name__ == "__main__":
    run_benchmark()
