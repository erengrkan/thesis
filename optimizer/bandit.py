"""
optimizer/bandit.py — Contextual Bandit Orchestrator
=====================================================
The main CBO entry point. Routes queries through the 4-stage pipeline:

  Stage 1: Check guardrails (hard short-circuit for extremes)
  Stage 2: Lookup Q-values from variable-granularity Q-Table
  Stage 3: Exploit or Explore (via selected exploration strategy)
  Stage 4: Execute, measure, compute reward, update Q-table

Each query produces a DecisionRecord for full traceability.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

import faiss

from bitmap_index import BitmapIndex
from faiss_index import FAISSIndex
from filters import FilterSpec
from strategies import PostFilter, SearchResult

from optimizer import config as cbo_config
from optimizer.exploration import ExplorationStrategy
from optimizer.ground_truth import GroundTruthOracle
from optimizer.guardrails import Guardrails
from optimizer.qtable import QTable
from optimizer.reward import RewardCalculator

log = logging.getLogger(__name__)


@dataclass
class DecisionRecord:
    """Full trace of a single routing decision.

    Attributes
    ----------
    episode : int
        Sequential query number.
    selectivity : float
        Actual selectivity of the filter.
    bucket_id : int
        Q-Table bucket the selectivity mapped to.
    q_pre_before : float
        Q_pre value BEFORE this update.
    q_post_before : float
        Q_post value BEFORE this update.
    chosen_strategy : str
        Strategy selected by the Bandit.
    was_guardrail : bool
        True if the decision was made by guardrails (not the Bandit).
    latency_ms : float
        End-to-end latency of the executed strategy.
    recall : float
        Recall@K against Brute-Force ground truth.
    reward : float
        Computed Soft Cliff SLA reward.
    q_pre_after : float
        Q_pre value AFTER this update.
    q_post_after : float
        Q_post value AFTER this update.
    gt_latency_ms : float
        Brute-Force ground truth latency (for reference).
    filter_name : str
        Human-readable filter description.
    """
    episode: int = 0
    selectivity: float = 0.0
    bucket_id: int = 0
    q_pre_before: float = 0.0
    q_post_before: float = 0.0
    chosen_strategy: str = ""
    was_guardrail: bool = False
    latency_ms: float = 0.0
    recall: float = 0.0
    reward: float = 0.0
    q_pre_after: float = 0.0
    q_post_after: float = 0.0
    gt_latency_ms: float = 0.0
    filter_name: str = ""


class ContextualBandit:
    """The main Cost-Based Optimizer. Routes queries through 4 stages.

    Parameters
    ----------
    faiss_idx : FAISSIndex
        Loaded FAISS index (provides HNSW + raw embeddings).
    bitmap_idx : BitmapIndex
        Loaded Bitmap index (for Pre-filter).
    all_metadatas : list[dict]
        Metadata for all documents (for Post-filter predicate evaluation).
    exploration : ExplorationStrategy
        The exploration strategy to use (Tier/Exponential/Softmax).
    guardrails : Guardrails, optional
        Hard boundaries. Created with defaults if not provided.
    qtable : QTable, optional
        Q-Table state. Created with defaults if not provided.
    reward_calc : RewardCalculator, optional
        Reward function. Created with defaults if not provided.
    """

    def __init__(
        self,
        faiss_idx: FAISSIndex,
        bitmap_idx: BitmapIndex,
        all_metadatas: List[Dict[str, Any]],
        exploration: ExplorationStrategy,
        guardrails: Optional[Guardrails] = None,
        qtable: Optional[QTable] = None,
        reward_calc: Optional[RewardCalculator] = None,
    ) -> None:
        self.faiss_idx = faiss_idx
        self.bitmap_idx = bitmap_idx
        self.all_metadatas = all_metadatas
        self.total_docs = faiss_idx.hnsw_index.ntotal

        # Stage 1: Guardrails
        self.guardrails = guardrails or Guardrails(total_docs=self.total_docs)

        # Stage 2: Q-Table
        self.qtable = qtable or QTable()

        # Stage 3: Exploration strategy
        self.exploration = exploration

        # Stage 4: Reward function
        self.reward_calc = reward_calc or RewardCalculator()

        # Ground truth oracle (runs Brute-Force for recall computation)
        self.oracle = GroundTruthOracle(faiss_idx, all_metadatas)

        # Post-filter strategy (HNSW without filter → Python-side filtering)
        self.post_filter = PostFilter(faiss_idx, all_metadatas)

        # Pre-filter uses HNSW + IDSelector (bitmap → IDSelectorArray → HNSW)
        # This is the TRUE pre-filter: the graph traversal is constrained to
        # only visit nodes that pass the bitmap filter.
        # NOT BitmapPreFilter which uses Brute-Force (IndexFlatIP) on subset.

        # Decision log
        self.decision_log: List[DecisionRecord] = []
        self._episode_counter = 0

        log.info(
            "ContextualBandit initialized: exploration=%s, total_docs=%d",
            exploration.name, self.total_docs,
        )

    def _execute_hnsw_prefilter(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        filter_spec: FilterSpec,
    ) -> SearchResult:
        """Execute Pre-filter: Bitmap → HNSW with IDSelectorArray.

        This is the TRUE pre-filter strategy:
          1. Resolve matching doc IDs via bitmap (µs).
          2. Create a FAISS IDSelectorArray from those IDs.
          3. Search HNSW with the IDSelector constraining graph traversal.

        The HNSW graph only visits nodes that pass the bitmap filter,
        but the graph structure (edges) can cause "rejection overhead"
        when many neighbors are filtered out.
        """
        # Step 1: Bitmap resolution (µs-level)
        t0 = time.perf_counter()
        matching_bitmap = filter_spec.resolve_bitmap(self.bitmap_idx)
        matching_ids = np.array(matching_bitmap.to_array(), dtype=np.int64)
        filter_ms = (time.perf_counter() - t0) * 1000

        n_candidates = len(matching_ids)

        if n_candidates == 0:
            return SearchResult(
                filter_time_ms=filter_ms,
                search_time_ms=0.0,
                total_time_ms=filter_ms,
                candidates_after_filter=0,
            )

        # Step 2: Create IDSelector from bitmap result
        id_selector = faiss.IDSelectorArray(matching_ids)

        # Step 3: HNSW search with IDSelector (graph-constrained pre-filter)
        t1 = time.perf_counter()
        distances, ids = self.faiss_idx.search_hnsw(
            query_embedding, top_k, id_selector=id_selector,
        )
        search_ms = (time.perf_counter() - t1) * 1000

        # Map int IDs back to string IDs
        result_ids = []
        result_dists = []
        for j in range(ids.shape[1]):
            int_id = int(ids[0, j])
            if int_id < 0:
                continue
            str_id = self.faiss_idx.int_to_str.get(int_id, f"unknown_{int_id}")
            result_ids.append(str_id)
            result_dists.append(float(1.0 - distances[0, j]))

        return SearchResult(
            ids=result_ids,
            distances=result_dists,
            filter_time_ms=filter_ms,
            search_time_ms=search_ms,
            total_time_ms=filter_ms + search_ms,
            candidates_after_filter=n_candidates,
        )

    def route(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        filter_spec: FilterSpec,
    ) -> SearchResult:
        """Route a query through the 4-stage pipeline.

        Parameters
        ----------
        query_embedding : np.ndarray
            Pre-computed query vector.
        top_k : int
            Number of results to return.
        filter_spec : FilterSpec
            The metadata filter (with both predicate and bitmap_resolver).

        Returns
        -------
        SearchResult
            The result from whichever strategy was selected.
        """
        self._episode_counter += 1
        episode = self._episode_counter

        # ── Compute selectivity via Bitmap (O(1) popcount) ────────────────
        result_bitmap = filter_spec.resolve_bitmap(self.bitmap_idx)
        selectivity = len(result_bitmap) / self.total_docs
        bucket_id = self.qtable.get_bucket_id(selectivity)

        # Snapshot Q-values BEFORE decision
        state = self.qtable.get_state(selectivity)
        q_pre_before = state.q_pre
        q_post_before = state.q_post

        # ── Stage 1: Check guardrails ─────────────────────────────────────
        guardrail_decision = self.guardrails.check(selectivity)
        was_guardrail = guardrail_decision is not None

        if guardrail_decision is not None:
            chosen = guardrail_decision
        else:
            # ── Stage 2+3: Q-Table lookup + Exploration decision ──────────
            chosen = self.exploration.select_action(q_pre_before, q_post_before)

        # ── Stage 4a: Execute the chosen strategy ─────────────────────────
        if chosen == cbo_config.PRE_FILTER_NAME:
            result = self._execute_hnsw_prefilter(query_embedding, top_k, filter_spec)
        else:
            result = self.post_filter.search(query_embedding, top_k, filter_spec)

        latency_ms = result.total_time_ms

        # ── Stage 4b: Get ground truth for recall computation ─────────────
        gt_result = self.oracle.get_ground_truth(
            query_embedding, top_k, filter_spec
        )
        recall = GroundTruthOracle.compute_recall(gt_result.ids, result.ids)

        # ── Stage 4c: Compute reward (Soft Cliff SLA) ─────────────────────
        reward = self.reward_calc.compute(latency_ms, recall)

        # ── Stage 4d: Update Q-table (TD learning) ────────────────────────
        self.qtable.update(selectivity, chosen, reward)

        # Snapshot Q-values AFTER update
        state = self.qtable.get_state(selectivity)

        # ── Log the decision ──────────────────────────────────────────────
        record = DecisionRecord(
            episode=episode,
            selectivity=round(selectivity, 6),
            bucket_id=bucket_id,
            q_pre_before=round(q_pre_before, 6),
            q_post_before=round(q_post_before, 6),
            chosen_strategy=chosen,
            was_guardrail=was_guardrail,
            latency_ms=round(latency_ms, 4),
            recall=round(recall, 4),
            reward=round(reward, 6),
            q_pre_after=round(state.q_pre, 6),
            q_post_after=round(state.q_post, 6),
            gt_latency_ms=round(gt_result.total_time_ms, 4),
            filter_name=filter_spec.name,
        )
        self.decision_log.append(record)

        if episode % 50 == 0:
            log.info(
                "Episode %d: sel=%.2f%% bucket=%d chose=%s "
                "latency=%.1fms recall=%.4f reward=%.4f",
                episode, selectivity * 100, bucket_id, chosen,
                latency_ms, recall, reward,
            )

        return result

    def get_decision_log(self) -> List[Dict[str, Any]]:
        """Export the full decision log as a list of dicts (for CSV output)."""
        return [
            {
                "episode": r.episode,
                "selectivity": r.selectivity,
                "bucket_id": r.bucket_id,
                "q_pre_before": r.q_pre_before,
                "q_post_before": r.q_post_before,
                "chosen_strategy": r.chosen_strategy,
                "was_guardrail": r.was_guardrail,
                "latency_ms": r.latency_ms,
                "recall": r.recall,
                "reward": r.reward,
                "q_pre_after": r.q_pre_after,
                "q_post_after": r.q_post_after,
                "gt_latency_ms": r.gt_latency_ms,
                "filter_name": r.filter_name,
            }
            for r in self.decision_log
        ]

    def get_qtable_snapshot(self) -> List[Dict]:
        """Export the current Q-table state."""
        return self.qtable.export_state()

    def reset(self) -> None:
        """Reset the Bandit state for a fresh run (new Q-table, clear logs)."""
        self.qtable = QTable()
        self.decision_log.clear()
        self._episode_counter = 0
        log.info("ContextualBandit reset: fresh Q-table, logs cleared.")
