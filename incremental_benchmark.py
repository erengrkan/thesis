"""
incremental_benchmark.py — Incremental Data Growth Experiment
================================================================
Simulates a production scenario where the dataset grows over time
(100k → 200k → 300k → 400k) and measures how the Softmax-based CBO
adapts its crossover point across phases.

At each phase:
  1. Build FAISS + Bitmap indexes from the first N documents
  2. Run CBO with Softmax (Q-table persists across phases)
  3. Compute always-bitmap and always-postfilter baselines
  4. Record crossover evolution, reward gains, and detailed metrics

Usage:
    python incremental_benchmark.py
"""

from __future__ import annotations

import json
import logging
import pickle
import random
import csv
from copy import deepcopy
from datetime import datetime
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import config
from bitmap_index import BitmapIndex
from faiss_index import FAISSIndex
from filters import FilterSpec, generate_filters
from strategies import BitmapPreFilter, BruteForce, PostFilter

from cbo import (
    ContextualBanditOptimizer,
    MetricsTracker,
    Softmax,
    SoftCliffReward,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════════

PHASES = [100_000, 200_000, 300_000, 400_000]
N_EPOCHS_PER_PHASE = 10
TOP_K = 10
NUM_QUERIES = 50
RANDOM_SEED = 42


# ═══════════════════════════════════════════════════════════════════════════════
#  Data classes for structured output
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PhaseResult:
    """Summary of a single phase."""
    phase: int
    n_docs: int
    n_docs_actual: int
    n_filters: int
    n_queries: int
    n_epochs: int

    # CBO Softmax results
    cbo_mean_reward: float
    cbo_final_crossover: Optional[float]
    cbo_epoch_rewards: List[float]
    cbo_epoch_crossovers: List[Optional[float]]

    # Baseline results
    baseline_bitmap_mean_reward: float
    baseline_post_mean_reward: float

    # Gains
    gain_vs_bitmap_pct: float
    gain_vs_post_pct: float

    # Q-table snapshot
    qtable_snapshot: List[Dict]

    # Filters used
    filters_info: List[Dict]


@dataclass
class QueryDetail:
    """Per-query detail record for CSV export."""
    phase: int
    n_docs: int
    epoch: int
    query_idx: int
    filter_idx: int
    selectivity: float
    strategy_chosen: str
    latency_ms: float
    recall: float
    reward: float
    q_bitmap: float
    q_post: float
    crossover_estimate: Optional[float]
    was_guardrail: bool
    oracle_strategy: str
    oracle_reward: float


# ═══════════════════════════════════════════════════════════════════════════════
#  Pre-computation (reused from cbo_benchmark pattern)
# ═══════════════════════════════════════════════════════════════════════════════

CacheDict = Dict[str, Dict[int, Dict[int, Tuple[float, float]]]]
OracleDict = Dict[int, Dict[int, Tuple[str, float]]]


def _precompute_results(
    faiss_idx: FAISSIndex,
    bitmap_idx: BitmapIndex,
    all_metadatas: List[Dict[str, Any]],
    filters: List[FilterSpec],
    query_embeddings: np.ndarray,
    top_k: int,
) -> Tuple[CacheDict, OracleDict]:
    """Pre-compute all strategy results and oracle decisions."""

    strategies = {
        "post_filter": PostFilter(faiss_idx, all_metadatas),
        "bitmap_prefilter": BitmapPreFilter(faiss_idx, bitmap_idx),
    }
    brute_force = BruteForce(faiss_idx, all_metadatas)
    reward_fn = SoftCliffReward()

    cache: CacheDict = {name: {} for name in strategies}
    oracle: OracleDict = {}

    n_queries = len(query_embeddings)
    n_filters = len(filters)

    for f_idx, fspec in enumerate(filters):
        oracle[f_idx] = {}
        for name in strategies:
            cache[name][f_idx] = {}

        # Warmup
        for _ in range(config.WARMUP_QUERIES):
            brute_force.search(query_embeddings[0], top_k, fspec)
            for strat in strategies.values():
                strat.search(query_embeddings[0], top_k, fspec)

        # Measured queries
        for q_idx in range(n_queries):
            qvec = query_embeddings[q_idx]

            # Ground truth
            bf_result = brute_force.search(qvec, top_k, fspec)
            ref_ids = bf_result.ids

            # PostFilter
            pf_result = strategies["post_filter"].search(qvec, top_k, fspec)
            pf_recall = (
                len(set(ref_ids) & set(pf_result.ids)) / len(ref_ids)
                if ref_ids else 1.0
            )
            cache["post_filter"][f_idx][q_idx] = (pf_result.total_time_ms, pf_recall)

            # BitmapPreFilter
            bp_result = strategies["bitmap_prefilter"].search(qvec, top_k, fspec)
            bp_recall = (
                len(set(ref_ids) & set(bp_result.ids)) / len(ref_ids)
                if ref_ids else 1.0
            )
            cache["bitmap_prefilter"][f_idx][q_idx] = (bp_result.total_time_ms, bp_recall)

            # Oracle
            pf_rew = reward_fn.compute(pf_result.total_time_ms, pf_recall)
            bp_rew = reward_fn.compute(bp_result.total_time_ms, bp_recall)
            if bp_rew >= pf_rew:
                oracle[f_idx][q_idx] = ("bitmap_prefilter", bp_rew)
            else:
                oracle[f_idx][q_idx] = ("post_filter", pf_rew)

        print(
            f"    [{f_idx + 1}/{n_filters}] Pre-computed: {fspec.name}  "
            f"(σ={fspec.actual_selectivity:.1%})"
        )

    return cache, oracle


def _compute_baseline_reward(
    cache: CacheDict,
    strategy_name: str,
    filters: List[FilterSpec],
    n_queries: int,
) -> float:
    """Compute mean reward for an always-use-this-strategy baseline."""
    reward_fn = SoftCliffReward()
    rewards = []
    for f_idx in range(len(filters)):
        for q_idx in range(n_queries):
            lat, rec = cache[strategy_name][f_idx][q_idx]
            rewards.append(reward_fn.compute(lat, rec))
    return sum(rewards) / len(rewards) if rewards else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Incremental Benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def run_incremental_benchmark() -> None:
    """Run the 4-phase incremental data growth experiment."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = config.RESULTS_DIR / f"incremental_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Incremental Data Growth Experiment")
    print(f"  Phases: {[f'{p // 1000}k' for p in PHASES]}")
    print(f"  Strategy: Softmax CBO")
    print(f"  Epochs per phase: {N_EPOCHS_PER_PHASE}")
    print("=" * 70)

    # ── Load full 400k dataset ────────────────────────────────────────────
    logger.info("Loading full embeddings and metadata from %s …", config.INDEX_DIR)

    full_faiss_idx = FAISSIndex.load(config.INDEX_DIR)
    total_available = full_faiss_idx.hnsw_index.ntotal
    full_embeddings = full_faiss_idx._embeddings  # (N, 768)

    with open(config.INDEX_DIR / "all_metadatas.pkl", "rb") as fh:
        full_metadatas: List[Dict[str, Any]] = pickle.load(fh)

    with open(config.INDEX_DIR / "id_mappings.pkl", "rb") as fh:
        id_mappings = pickle.load(fh)

    print(f"\n  Full dataset loaded: {total_available:,} vectors")
    print(f"  Embedding dim: {full_faiss_idx.dim}")

    # ── Fixed query set (same queries across all phases) ──────────────────
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Use queries from the first 100k so they exist in all phases
    query_indices = random.sample(range(min(100_000, total_available)), NUM_QUERIES)
    query_embeddings = full_embeddings[query_indices]
    print(f"  Query vectors: {len(query_embeddings)} (fixed across all phases)")

    # ── Create persistent CBO optimizer ───────────────────────────────────
    optimizer = ContextualBanditOptimizer(
        exploration_strategy=Softmax(),
        alpha=config.CBO_ALPHA,
        crossover_hint=config.CBO_CROSSOVER_HINT,
    )

    # ── Phase loop ────────────────────────────────────────────────────────
    phase_results: List[PhaseResult] = []
    all_query_details: List[QueryDetail] = []

    for phase_num, n_docs in enumerate(PHASES, start=1):
        if n_docs > total_available:
            logger.warning(
                "Phase %d requests %d docs but only %d available. Using all.",
                phase_num, n_docs, total_available,
            )
            n_docs = total_available

        print(f"\n{'═' * 70}")
        print(f"  Phase {phase_num}/{len(PHASES)}: {n_docs:,} documents")
        print(f"{'═' * 70}")

        # ── Build phase-specific indexes ──────────────────────────────────
        phase_embeddings = full_embeddings[:n_docs]
        phase_metadatas = full_metadatas[:n_docs]

        # Build new FAISS index for this phase
        print("  Building FAISS HNSW index …")
        phase_faiss = FAISSIndex()
        doc_ids = [f"doc_{i}" for i in range(n_docs)]
        phase_faiss.build(phase_embeddings, doc_ids)
        print(f"    FAISS: {phase_faiss.hnsw_index.ntotal:,} vectors")

        # Build new BitmapIndex for this phase
        print("  Building Bitmap index …")
        phase_bitmap = BitmapIndex()
        for idx, meta in enumerate(phase_metadatas):
            phase_bitmap.add_document(idx, f"doc_{idx}", meta)
        print(f"    Bitmap: {phase_bitmap.total_docs:,} docs")

        # ── Generate filters for this phase's data ────────────────────────
        print("  Generating filters …")
        filters = generate_filters(phase_metadatas, config.SELECTIVITY_TARGETS)
        print(f"    Generated {len(filters)} filters")

        # ── Pre-compute all results ───────────────────────────────────────
        print("  Pre-computing strategy results …")
        cache, oracle = _precompute_results(
            phase_faiss, phase_bitmap, phase_metadatas,
            filters, query_embeddings, TOP_K,
        )
        print("  ✓ Pre-computation complete")

        # ── Compute baselines ─────────────────────────────────────────────
        baseline_bitmap = _compute_baseline_reward(
            cache, "bitmap_prefilter", filters, NUM_QUERIES
        )
        baseline_post = _compute_baseline_reward(
            cache, "post_filter", filters, NUM_QUERIES
        )
        print(f"  Baselines: Always-Bitmap={baseline_bitmap:.4f}  Always-Post={baseline_post:.4f}")

        # ── Reset step counter (fresh decay each phase) ───────────────────
        optimizer._step_counter = 0

        # ── Run CBO epochs ────────────────────────────────────────────────
        epoch_rewards: List[float] = []
        epoch_crossovers: List[Optional[float]] = []

        for epoch in range(N_EPOCHS_PER_PHASE):
            query_pairs = [
                (f_idx, q_idx)
                for f_idx in range(len(filters))
                for q_idx in range(NUM_QUERIES)
            ]
            random.shuffle(query_pairs)

            ep_rewards: List[float] = []

            for f_idx, q_idx in query_pairs:
                selectivity = filters[f_idx].actual_selectivity
                q_vals = optimizer.qtable.get_q_values(selectivity)

                strategy_name, was_guardrail = optimizer.route(selectivity)
                latency_ms, recall = cache[strategy_name][f_idx][q_idx]
                reward = optimizer.feedback(selectivity, strategy_name, latency_ms, recall)

                oracle_strat, oracle_rew = oracle[f_idx][q_idx]

                # Record detail
                all_query_details.append(QueryDetail(
                    phase=phase_num,
                    n_docs=n_docs,
                    epoch=epoch,
                    query_idx=q_idx,
                    filter_idx=f_idx,
                    selectivity=selectivity,
                    strategy_chosen=strategy_name,
                    latency_ms=latency_ms,
                    recall=recall,
                    reward=reward,
                    q_bitmap=q_vals["bitmap_prefilter"],
                    q_post=q_vals["post_filter"],
                    crossover_estimate=optimizer.get_crossover_estimate(),
                    was_guardrail=was_guardrail,
                    oracle_strategy=oracle_strat,
                    oracle_reward=oracle_rew,
                ))
                ep_rewards.append(reward)

            mean_rew = sum(ep_rewards) / len(ep_rewards)
            crossover = optimizer.get_crossover_estimate()
            epoch_rewards.append(mean_rew)
            epoch_crossovers.append(crossover)

            crossover_str = f"{crossover:.4f}" if crossover is not None else "N/A"
            print(
                f"    Epoch {epoch + 1:2d}/{N_EPOCHS_PER_PHASE}  "
                f"mean_reward={mean_rew:.4f}  crossover={crossover_str}"
            )

        # ── Compute gains ─────────────────────────────────────────────────
        final_cbo_reward = epoch_rewards[-1]
        gain_bitmap = (
            (final_cbo_reward - baseline_bitmap) / baseline_bitmap * 100
            if baseline_bitmap > 0 else 0.0
        )
        gain_post = (
            (final_cbo_reward - baseline_post) / baseline_post * 100
            if baseline_post > 0 else 0.0
        )

        print(f"\n  Phase {phase_num} Summary:")
        print(f"    CBO Reward (last epoch): {final_cbo_reward:.4f}")
        print(f"    Crossover:               {epoch_crossovers[-1]}")
        print(f"    Gain vs Bitmap:          {gain_bitmap:+.2f}%")
        print(f"    Gain vs PostFilter:      {gain_post:+.2f}%")

        # ── Save phase result ─────────────────────────────────────────────
        phase_results.append(PhaseResult(
            phase=phase_num,
            n_docs=n_docs,
            n_docs_actual=phase_faiss.hnsw_index.ntotal,
            n_filters=len(filters),
            n_queries=NUM_QUERIES,
            n_epochs=N_EPOCHS_PER_PHASE,
            cbo_mean_reward=final_cbo_reward,
            cbo_final_crossover=epoch_crossovers[-1],
            cbo_epoch_rewards=epoch_rewards,
            cbo_epoch_crossovers=epoch_crossovers,
            baseline_bitmap_mean_reward=baseline_bitmap,
            baseline_post_mean_reward=baseline_post,
            gain_vs_bitmap_pct=gain_bitmap,
            gain_vs_post_pct=gain_post,
            qtable_snapshot=optimizer.get_q_snapshot(),
            filters_info=[
                {
                    "name": f.name,
                    "target_sel": f.target_selectivity,
                    "actual_sel": f.actual_selectivity,
                }
                for f in filters
            ],
        ))

        # Save Q-table snapshot for this phase
        with open(out_dir / f"phase_{phase_num}_qtable.json", "w") as f:
            json.dump(optimizer.get_q_snapshot(), f, indent=2)

    # ═══════════════════════════════════════════════════════════════════════
    #  Export Results
    # ═══════════════════════════════════════════════════════════════════════

    # 1. Phase summary JSON
    phases_summary = {
        "timestamp": timestamp,
        "experiment": "incremental_data_growth",
        "phases": PHASES,
        "n_epochs_per_phase": N_EPOCHS_PER_PHASE,
        "exploration_strategy": "softmax",
        "top_k": TOP_K,
        "num_queries": NUM_QUERIES,
        "r_target": config.CBO_R_TARGET,
        "l_max": config.CBO_L_MAX,
        "alpha": config.CBO_ALPHA,
        "results": [
            {
                "phase": pr.phase,
                "n_docs": pr.n_docs,
                "n_docs_actual": pr.n_docs_actual,
                "n_filters": pr.n_filters,
                "cbo_mean_reward": round(pr.cbo_mean_reward, 6),
                "cbo_final_crossover": pr.cbo_final_crossover,
                "cbo_epoch_rewards": [round(r, 6) for r in pr.cbo_epoch_rewards],
                "cbo_epoch_crossovers": pr.cbo_epoch_crossovers,
                "baseline_bitmap_mean_reward": round(pr.baseline_bitmap_mean_reward, 6),
                "baseline_post_mean_reward": round(pr.baseline_post_mean_reward, 6),
                "gain_vs_bitmap_pct": round(pr.gain_vs_bitmap_pct, 4),
                "gain_vs_post_pct": round(pr.gain_vs_post_pct, 4),
                "filters": pr.filters_info,
            }
            for pr in phase_results
        ],
    }
    with open(out_dir / "phases_summary.json", "w") as f:
        json.dump(phases_summary, f, indent=2)
    logger.info("Saved phases_summary.json")

    # 2. Phase summary CSV (compact table)
    with open(out_dir / "phases_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "phase", "n_docs", "cbo_reward", "crossover",
            "bitmap_reward", "post_reward",
            "gain_vs_bitmap_pct", "gain_vs_post_pct",
        ])
        for pr in phase_results:
            writer.writerow([
                pr.phase, pr.n_docs,
                round(pr.cbo_mean_reward, 6),
                pr.cbo_final_crossover,
                round(pr.baseline_bitmap_mean_reward, 6),
                round(pr.baseline_post_mean_reward, 6),
                round(pr.gain_vs_bitmap_pct, 4),
                round(pr.gain_vs_post_pct, 4),
            ])
    logger.info("Saved phases_summary.csv")

    # 3. Full query details CSV
    detail_path = out_dir / "query_details.csv"
    with open(detail_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(all_query_details[0]).keys()))
        writer.writeheader()
        for detail in all_query_details:
            writer.writerow(asdict(detail))
    logger.info("Saved query_details.csv (%d records)", len(all_query_details))

    # 4. Per-phase epoch tracking CSV
    with open(out_dir / "epoch_tracking.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["phase", "n_docs", "epoch", "mean_reward", "crossover"])
        for pr in phase_results:
            for ep_idx, (rew, xo) in enumerate(
                zip(pr.cbo_epoch_rewards, pr.cbo_epoch_crossovers)
            ):
                writer.writerow([pr.phase, pr.n_docs, ep_idx + 1, round(rew, 6), xo])
    logger.info("Saved epoch_tracking.csv")

    print(f"\n{'=' * 70}")
    print(f"  Incremental Benchmark Complete")
    print(f"  Results: {out_dir}")
    print(f"{'=' * 70}")

    # ── Final summary table ───────────────────────────────────────────────
    print(f"\n  {'Phase':>5} {'Docs':>8} {'CBO':>8} {'Bitmap':>8} {'Post':>8} {'Gain/BM':>8} {'Gain/PF':>8} {'Crossover':>10}")
    print(f"  {'─' * 5} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 10}")
    for pr in phase_results:
        xo_str = f"{pr.cbo_final_crossover:.4f}" if pr.cbo_final_crossover else "N/A"
        print(
            f"  {pr.phase:>5} {pr.n_docs:>8,} {pr.cbo_mean_reward:>8.4f} "
            f"{pr.baseline_bitmap_mean_reward:>8.4f} {pr.baseline_post_mean_reward:>8.4f} "
            f"{pr.gain_vs_bitmap_pct:>+7.2f}% {pr.gain_vs_post_pct:>+7.2f}% {xo_str:>10}"
        )

    # ── Generate visualizations ───────────────────────────────────────────
    try:
        from incremental_visualize import generate_incremental_plots
        generate_incremental_plots(out_dir, phase_results)
    except Exception:
        logger.exception("Visualization failed (non-fatal)")


if __name__ == "__main__":
    run_incremental_benchmark()
