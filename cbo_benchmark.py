"""
cbo_benchmark.py — Contextual Bandit Optimizer Benchmark Runner
=================================================================
Evaluates the CBO's ability to learn the optimal crossover selectivity
between PostFilter and BitmapPreFilter strategies.

Methodology:
  1. Pre-compute ALL (strategy × filter × query) results into a cache
     so that every exploration strategy sees identical latency/recall data.
  2. Compute the oracle (best strategy per query instance).
  3. Run bandit experiments for each exploration strategy over N epochs.
  4. Export per-strategy CSV/JSON, Q-table snapshots, and experiment metadata.
  5. Generate visualisation plots via ``cbo_visualize``.

Usage:
    python cbo_benchmark.py              # Full benchmark
    python cbo_benchmark.py --smoke-test # Quick 2-epoch / 3-filter run
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import config
from bitmap_index import BitmapIndex
from faiss_index import FAISSIndex
from filters import FilterSpec, generate_filters
from strategies import BitmapPreFilter, BruteForce, PostFilter, SearchResult

from cbo import (
    ContextualBanditOptimizer,
    ExponentialDecay,
    Guardrails,
    MetricsTracker,
    Softmax,
    SoftCliffReward,
    TierBased,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Pre-computation helpers
# ═══════════════════════════════════════════════════════════════════════════════

CacheDict = Dict[str, Dict[int, Dict[int, Tuple[float, float]]]]
"""Type alias: cache[strategy_name][filter_idx][query_idx] = (latency_ms, recall)."""

OracleDict = Dict[int, Dict[int, Tuple[str, float]]]
"""Type alias: oracle[filter_idx][query_idx] = (best_strategy_name, best_reward)."""


def _precompute_results(
    faiss_idx: FAISSIndex,
    bitmap_idx: BitmapIndex,
    all_metadatas: List[Dict[str, Any]],
    filters: List[FilterSpec],
    query_embeddings: np.ndarray,
    top_k: int,
) -> Tuple[CacheDict, Dict[int, Dict[int, List[str]]]]:
    """Run every (strategy, filter, query) combination and cache results.

    Returns
    -------
    cache : CacheDict
        ``cache[strategy_name][f_idx][q_idx] = (latency_ms, recall)``
    ground_truth : dict
        ``ground_truth[f_idx][q_idx] = list_of_reference_ids``
    """
    strategies = {
        "post_filter": PostFilter(faiss_idx, all_metadatas),
        "bitmap_prefilter": BitmapPreFilter(faiss_idx, bitmap_idx),
    }
    brute_force = BruteForce(faiss_idx, all_metadatas)

    cache: CacheDict = {name: {} for name in strategies}
    ground_truth: Dict[int, Dict[int, List[str]]] = {}

    n_queries = len(query_embeddings)
    n_filters = len(filters)

    for f_idx, fspec in enumerate(filters):
        ground_truth[f_idx] = {}
        for name in strategies:
            cache[name][f_idx] = {}

        # ── Warmup runs (excluded from timing) ────────────────────────────
        for _ in range(config.WARMUP_QUERIES):
            brute_force.search(query_embeddings[0], top_k, fspec)
            for strat in strategies.values():
                strat.search(query_embeddings[0], top_k, fspec)

        # ── Measured queries ──────────────────────────────────────────────
        for q_idx in range(n_queries):
            qvec = query_embeddings[q_idx]

            # Ground truth (BruteForce)
            bf_result = brute_force.search(qvec, top_k, fspec)
            ref_ids = bf_result.ids
            ground_truth[f_idx][q_idx] = ref_ids

            # PostFilter
            pf_result = strategies["post_filter"].search(qvec, top_k, fspec)
            pf_recall = (
                len(set(ref_ids) & set(pf_result.ids)) / len(ref_ids)
                if ref_ids
                else 1.0
            )
            cache["post_filter"][f_idx][q_idx] = (pf_result.total_time_ms, pf_recall)

            # BitmapPreFilter
            bp_result = strategies["bitmap_prefilter"].search(qvec, top_k, fspec)
            bp_recall = (
                len(set(ref_ids) & set(bp_result.ids)) / len(ref_ids)
                if ref_ids
                else 1.0
            )
            cache["bitmap_prefilter"][f_idx][q_idx] = (
                bp_result.total_time_ms,
                bp_recall,
            )

        progress = f"[{f_idx + 1}/{n_filters}]"
        print(
            f"  {progress}  Pre-computed filter: {fspec.name}  "
            f"(σ={fspec.actual_selectivity:.1%})"
        )

    return cache, ground_truth


def _compute_oracle(
    cache: CacheDict,
    filters: List[FilterSpec],
    n_queries: int,
    reward_fn: SoftCliffReward,
) -> OracleDict:
    """Determine the best strategy per (filter, query) pair.

    Returns
    -------
    oracle : OracleDict
        ``oracle[f_idx][q_idx] = (strategy_name, reward)``
    """
    oracle: OracleDict = {}
    for f_idx in range(len(filters)):
        oracle[f_idx] = {}
        for q_idx in range(n_queries):
            pf_lat, pf_rec = cache["post_filter"][f_idx][q_idx]
            bp_lat, bp_rec = cache["bitmap_prefilter"][f_idx][q_idx]
            pf_reward = reward_fn.compute(pf_lat, pf_rec)
            bp_reward = reward_fn.compute(bp_lat, bp_rec)
            if bp_reward >= pf_reward:
                oracle[f_idx][q_idx] = ("bitmap_prefilter", bp_reward)
            else:
                oracle[f_idx][q_idx] = ("post_filter", pf_reward)
    return oracle


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def run_cbo_benchmark(*, smoke_test: bool = False) -> None:
    """Run the full CBO benchmark suite.

    Parameters
    ----------
    smoke_test : bool
        If ``True``, run only 2 epochs with 3 filters for quick validation.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cbo_dir = config.RESULTS_DIR / f"cbo_{timestamp}"
    cbo_dir.mkdir(parents=True, exist_ok=True)

    n_epochs = 2 if smoke_test else config.CBO_N_EPOCHS
    selectivity_targets = (
        config.SELECTIVITY_TARGETS[:3] if smoke_test else config.SELECTIVITY_TARGETS
    )

    print("=" * 70)
    print("  CBO — Contextual Bandit Optimizer Benchmark")
    if smoke_test:
        print("  ⚡ SMOKE TEST MODE (2 epochs, 3 filters)")
    print("=" * 70)

    # ── 1. Load indices ───────────────────────────────────────────────────
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

    # ── 2. Generate filters ───────────────────────────────────────────────
    logger.info("Generating filters for targets: %s", selectivity_targets)
    filters = generate_filters(all_metadatas, selectivity_targets)
    print(f"\n  Generated {len(filters)} filters:")
    for f in filters:
        print(
            f"    target={f.target_selectivity:.0%}  "
            f"actual={f.actual_selectivity:.1%}  {f.name}"
        )

    # ── 3. Prepare query embeddings ───────────────────────────────────────
    random.seed(42)
    query_indices = random.sample(
        range(total_docs), min(config.NUM_QUERY_SAMPLES, total_docs)
    )
    query_embeddings = faiss_idx._embeddings[query_indices]
    print(f"\n  Query vectors: {len(query_embeddings)} samples")

    # ── 4. Pre-compute all strategy results ───────────────────────────────
    top_k = config.CBO_TOP_K
    print(f"\n  Pre-computing strategy results (top_k={top_k}) …")
    cache, ground_truth = _precompute_results(
        faiss_idx, bitmap_idx, all_metadatas, filters, query_embeddings, top_k
    )
    print("  ✓ Pre-computation complete.\n")

    # ── 5. Compute oracle ─────────────────────────────────────────────────
    reward_fn = SoftCliffReward()
    oracle = _compute_oracle(cache, filters, len(query_embeddings), reward_fn)

    # ── 6. Run bandit experiments ─────────────────────────────────────────
    exploration_strategies = [
        TierBased(),
        ExponentialDecay(),
        Softmax(),
    ]

    all_trackers: Dict[str, MetricsTracker] = {}

    for exp_strategy in exploration_strategies:
        print(f"\n{'=' * 70}")
        print(f"  Running CBO with exploration: {exp_strategy.name}")
        print(f"{'=' * 70}")

        optimizer = ContextualBanditOptimizer(
            exploration_strategy=exp_strategy,
            alpha=config.CBO_ALPHA,
            crossover_hint=config.CBO_CROSSOVER_HINT,
        )
        tracker = MetricsTracker()

        for epoch in range(n_epochs):
            # Build (filter_idx, query_idx) pairs and shuffle to prevent
            # ordering bias within each epoch.
            query_pairs = [
                (f_idx, q_idx)
                for f_idx in range(len(filters))
                for q_idx in range(len(query_embeddings))
            ]
            random.shuffle(query_pairs)

            epoch_rewards: List[float] = []

            for f_idx, q_idx in query_pairs:
                selectivity = filters[f_idx].actual_selectivity

                # Snapshot Q-values *before* routing (for diagnostics)
                q_vals = optimizer.qtable.get_q_values(selectivity)

                # Route query
                strategy_name, was_guardrail = optimizer.route(selectivity)
                overhead_us = optimizer.decision_overhead_us

                # Look up cached latency & recall
                latency_ms, recall = cache[strategy_name][f_idx][q_idx]

                # Provide feedback → get reward
                reward = optimizer.feedback(
                    selectivity, strategy_name, latency_ms, recall
                )

                # Oracle baseline
                oracle_strat, oracle_rew = oracle[f_idx][q_idx]

                # Record in tracker
                tracker.record(
                    epoch=epoch,
                    query_idx=q_idx,
                    selectivity=selectivity,
                    strategy_chosen=strategy_name,
                    latency_ms=latency_ms,
                    recall=recall,
                    reward=reward,
                    q_bitmap=q_vals["bitmap_prefilter"],
                    q_post=q_vals["post_filter"],
                    crossover_estimate=optimizer.get_crossover_estimate(),
                    decision_overhead_us=overhead_us,
                    was_guardrail=was_guardrail,
                    oracle_strategy=oracle_strat,
                    oracle_reward=oracle_rew,
                )
                epoch_rewards.append(reward)

            mean_rew = (
                sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0.0
            )
            crossover = optimizer.get_crossover_estimate()
            crossover_str = (
                f"{crossover:.4f}" if crossover is not None else "N/A"
            )
            print(
                f"  Epoch {epoch + 1:2d}/{n_epochs}  "
                f"mean_reward={mean_rew:.4f}  "
                f"crossover={crossover_str}"
            )

        # ── Save per-strategy outputs ─────────────────────────────────────
        all_trackers[exp_strategy.name] = tracker

        tracker.export_csv(cbo_dir / f"cbo_{exp_strategy.name}_raw.csv")
        tracker.export_json(cbo_dir / f"cbo_{exp_strategy.name}_raw.json")

        # Final Q-table snapshot
        with open(cbo_dir / f"cbo_{exp_strategy.name}_qtable.json", "w") as f:
            json.dump(optimizer.get_q_snapshot(), f, indent=2)

        logger.info(
            "Saved results for exploration strategy '%s'", exp_strategy.name
        )

    # ── 7. Save experiment metadata ───────────────────────────────────────
    meta: Dict[str, Any] = {
        "timestamp": timestamp,
        "smoke_test": smoke_test,
        "total_docs": total_docs,
        "num_queries": len(query_embeddings),
        "num_filters": len(filters),
        "n_epochs": n_epochs,
        "top_k": top_k,
        "r_target": config.CBO_R_TARGET,
        "l_max": config.CBO_L_MAX,
        "beta": config.CBO_BETA,
        "alpha": config.CBO_ALPHA,
        "sigma_lower": config.CBO_SIGMA_LOWER,
        "sigma_upper": config.CBO_SIGMA_UPPER,
        "exploration_strategies": [s.name for s in exploration_strategies],
        "filters": [
            {
                "name": f.name,
                "target_sel": f.target_selectivity,
                "actual_sel": f.actual_selectivity,
            }
            for f in filters
        ],
    }
    with open(cbo_dir / "experiment_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"  CBO Benchmark Complete")
    print(f"  Results: {cbo_dir}")
    print(f"{'=' * 70}")

    # ── 8. Generate visualisations ────────────────────────────────────────
    try:
        from cbo_visualize import generate_all_plots

        generate_all_plots(
            cbo_dir, all_trackers, config.CBO_R_TARGET, config.CBO_L_MAX
        )
        print(f"  Plots:   {cbo_dir / 'plots'}")
    except ImportError:
        logger.warning(
            "cbo_visualize not available — skipping plot generation. "
            "Install matplotlib to enable visualisations."
        )
    except Exception:
        logger.exception("Plot generation failed (non-fatal)")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CBO — Contextual Bandit Optimizer Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        default=False,
        help="Quick validation run: 2 epochs, 3 filters.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_cbo_benchmark(smoke_test=args.smoke_test)
