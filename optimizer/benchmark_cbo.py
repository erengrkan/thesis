"""
optimizer/benchmark_cbo.py — CBO Benchmark Runner
====================================================
Runs the Contextual Bandit optimizer through the full selectivity
spectrum and compares the three exploration strategies:

  1. TierBased        — Discrete confidence tiers
  2. ExponentialDecay — Smooth ε reduction
  3. Softmax          — Boltzmann distribution

For each strategy, the benchmark:
  - Generates filters across manually-specified selectivity targets
  - Runs CBO_NUM_QUERIES queries in randomized order
  - Records per-query decision traces (selectivity, chosen strategy,
    latency, recall, reward, Q-table evolution)
  - Exports results for thesis-ready analysis

Output files (in results/):
  - cbo_decisions_{strategy}_{timestamp}.csv   — per-query trace
  - cbo_qtable_{strategy}_{timestamp}.csv      — final Q-table state
  - cbo_summary_{timestamp}.csv                — cross-strategy comparison
"""

from __future__ import annotations

import csv
import json
import logging
import os
import pickle
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Prevent FAISS / HuggingFace segfaults on Mac
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from bitmap_index import BitmapIndex
from faiss_index import FAISSIndex
from filters import generate_filters

from optimizer import config as cbo_config
from optimizer.bandit import ContextualBandit
from optimizer.exploration import ExponentialDecay, Softmax, TierBased
from optimizer.guardrails import Guardrails
from optimizer.qtable import QTable
from optimizer.reward import RewardCalculator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Decision CSV Fields
# ═══════════════════════════════════════════════════════════════════════════════

DECISION_FIELDS = [
    "episode", "selectivity", "bucket_id",
    "q_pre_before", "q_post_before",
    "chosen_strategy", "was_guardrail",
    "latency_ms", "recall", "reward",
    "q_pre_after", "q_post_after",
    "gt_latency_ms", "filter_name",
]

QTABLE_FIELDS = [
    "bucket_id", "range_lower", "range_upper",
    "q_pre", "q_post", "delta",
    "visits_pre", "visits_post", "total_visits", "preferred",
]


def _load_indices():
    """Load FAISS index, Bitmap index, and metadata."""
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

    return faiss_idx, bitmap_idx, all_metadatas


def _build_query_schedule(
    filters,
    n_queries: int,
    total_docs: int,
    faiss_idx: FAISSIndex,
    seed: int = cbo_config.CBO_SEED,
):
    """Build a randomized sequence of (query_embedding, filter_spec) tuples.

    Each query is paired with a randomly selected filter from the generated set,
    creating a realistic workload where the Bandit encounters diverse
    selectivities in unpredictable order.
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    # Pre-sample random query indices
    all_indices = list(range(total_docs))
    query_indices = [rng.choice(all_indices) for _ in range(n_queries)]
    query_embeddings = faiss_idx._embeddings[query_indices]

    # Pair each query with a random filter
    schedule = []
    for i in range(n_queries):
        fspec = rng.choice(filters)
        schedule.append((query_embeddings[i], fspec))

    return schedule


def run_single_strategy(
    strategy_name: str,
    bandit: ContextualBandit,
    schedule: list,
    top_k: int,
    warmup: int = cbo_config.CBO_WARMUP,
) -> List[Dict]:
    """Run the CBO through the full query schedule with one exploration strategy.

    Returns the decision log as a list of dicts.
    """
    print(f"\n{'═' * 70}")
    print(f"  Strategy: {strategy_name}")
    print(f"  Queries: {len(schedule)} (warmup: {warmup})")
    print(f"{'═' * 70}")

    # Warmup (results discarded — just warms CPU caches)
    for i in range(min(warmup, len(schedule))):
        q_emb, fspec = schedule[i]
        bandit.route(q_emb, top_k, fspec)
    # Reset after warmup
    bandit.reset()

    # Measured run
    t0 = time.time()
    for i, (q_emb, fspec) in enumerate(schedule):
        bandit.route(q_emb, top_k, fspec)
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"    Progress: {i+1}/{len(schedule)} queries ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s ({len(schedule)/elapsed:.1f} queries/sec)")

    return bandit.get_decision_log()


def run_benchmark() -> None:
    """Execute the full CBO benchmark across all three exploration strategies."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("  Contextual Bandit CBO — Benchmark Suite")
    print("=" * 70)

    # ── Load indices ──────────────────────────────────────────────────────
    faiss_idx, bitmap_idx, all_metadatas = _load_indices()
    total_docs = faiss_idx.hnsw_index.ntotal

    # ── Generate filters ──────────────────────────────────────────────────
    logger.info("Generating filters for targets: %s", config.SELECTIVITY_TARGETS)
    filters = generate_filters(all_metadatas, config.SELECTIVITY_TARGETS)
    print(f"\n  Generated {len(filters)} filters:")
    for f in filters:
        print(f"    target={f.target_selectivity:.0%}  actual={f.actual_selectivity:.1%}  {f.name}")

    # ── Build query schedule ──────────────────────────────────────────────
    n_queries = cbo_config.CBO_NUM_QUERIES
    top_k = 10  # Use k=10 for faster iteration
    schedule = _build_query_schedule(
        filters, n_queries, total_docs, faiss_idx
    )
    print(f"\n  Query schedule: {n_queries} queries, top_k={top_k}")

    # ── Run all three exploration strategies ───────────────────────────────
    exploration_strategies = [
        ("tier_based", TierBased()),
        ("exponential_decay", ExponentialDecay()),
        ("softmax", Softmax()),
    ]

    all_summaries = []

    for strat_name, exploration in exploration_strategies:
        # Fresh Bandit for each strategy (clean Q-table)
        bandit = ContextualBandit(
            faiss_idx=faiss_idx,
            bitmap_idx=bitmap_idx,
            all_metadatas=all_metadatas,
            exploration=exploration,
        )

        # Run the benchmark
        decisions = run_single_strategy(
            strat_name, bandit, schedule, top_k
        )

        # ── Save decision trace CSV ───────────────────────────────────────
        dec_path = config.RESULTS_DIR / f"cbo_decisions_{strat_name}_{timestamp}.csv"
        with open(dec_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=DECISION_FIELDS)
            writer.writeheader()
            writer.writerows(decisions)
        print(f"    Decisions → {dec_path}")

        # ── Save final Q-table CSV ────────────────────────────────────────
        qt_path = config.RESULTS_DIR / f"cbo_qtable_{strat_name}_{timestamp}.csv"
        qtable_state = bandit.get_qtable_snapshot()
        with open(qt_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=QTABLE_FIELDS)
            writer.writeheader()
            writer.writerows(qtable_state)
        print(f"    Q-Table  → {qt_path}")

        # ── Compute summary statistics ────────────────────────────────────
        if decisions:
            latencies = [d["latency_ms"] for d in decisions]
            recalls = [d["recall"] for d in decisions]
            rewards = [d["reward"] for d in decisions]
            pre_count = sum(1 for d in decisions if d["chosen_strategy"] == cbo_config.PRE_FILTER_NAME)
            post_count = len(decisions) - pre_count
            guardrail_count = sum(1 for d in decisions if d["was_guardrail"])

            summary = {
                "strategy": strat_name,
                "total_queries": len(decisions),
                "pre_filter_chosen": pre_count,
                "post_filter_chosen": post_count,
                "guardrail_decisions": guardrail_count,
                "mean_latency_ms": round(sum(latencies) / len(latencies), 4),
                "mean_recall": round(sum(recalls) / len(recalls), 4),
                "mean_reward": round(sum(rewards) / len(rewards), 4),
                "min_recall": round(min(recalls), 4),
                "max_latency_ms": round(max(latencies), 4),
            }
            all_summaries.append(summary)

            print(f"\n    Summary for {strat_name}:")
            print(f"      Pre-filter chosen:  {pre_count}/{len(decisions)}")
            print(f"      Post-filter chosen: {post_count}/{len(decisions)}")
            print(f"      Guardrail decisions: {guardrail_count}")
            print(f"      Mean latency: {summary['mean_latency_ms']:.1f} ms")
            print(f"      Mean recall:  {summary['mean_recall']:.4f}")
            print(f"      Mean reward:  {summary['mean_reward']:.4f}")

    # ── Save cross-strategy summary CSV ───────────────────────────────────
    summary_path = config.RESULTS_DIR / f"cbo_summary_{timestamp}.csv"
    if all_summaries:
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_summaries[0].keys())
            writer.writeheader()
            writer.writerows(all_summaries)
        print(f"\n  Cross-strategy summary → {summary_path}")

    # ── Final output ──────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  CBO Benchmark Complete")
    print(f"  Results saved to: {config.RESULTS_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_benchmark()
