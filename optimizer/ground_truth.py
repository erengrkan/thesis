"""
optimizer/ground_truth.py — Brute-Force Ground Truth for Recall Computation
==============================================================================
Provides the Brute-Force search result for a given query, which serves as
the recall reference. The Bandit needs per-query recall to compute rewards,
and Brute-Force (exhaustive search) is the only strategy guaranteed to
return the mathematically optimal top-K results.

This module wraps the existing BruteForce strategy from strategies.py
for use by the CBO benchmark.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Set

from strategies import BruteForce, SearchResult
from faiss_index import FAISSIndex
from filters import FilterSpec

log = logging.getLogger(__name__)


class GroundTruthOracle:
    """Provides Brute-Force ground truth for recall computation.

    This is intentionally NOT cached — each call runs the full exhaustive
    search, as requested by the user to match real-world behavior.

    Parameters
    ----------
    faiss_idx : FAISSIndex
        The loaded FAISS index.
    all_metadatas : list[dict]
        Metadata for all documents.
    """

    def __init__(
        self,
        faiss_idx: FAISSIndex,
        all_metadatas: List[Dict[str, Any]],
    ) -> None:
        self.brute_force = BruteForce(faiss_idx, all_metadatas)
        log.info("GroundTruthOracle initialized (Brute-Force on %d docs)", faiss_idx.hnsw_index.ntotal)

    def get_ground_truth(
        self,
        query_embedding,
        top_k: int,
        filter_spec: FilterSpec,
    ) -> SearchResult:
        """Run Brute-Force search and return the exact top-K results.

        Parameters
        ----------
        query_embedding : np.ndarray
            Query vector.
        top_k : int
            Number of results.
        filter_spec : FilterSpec
            The metadata filter to apply.

        Returns
        -------
        SearchResult
            The ground-truth search result.
        """
        return self.brute_force.search(query_embedding, top_k, filter_spec)

    @staticmethod
    def compute_recall(
        ground_truth_ids: List[str],
        candidate_ids: List[str],
    ) -> float:
        """Compute Recall@K between ground truth and candidate results.

        Parameters
        ----------
        ground_truth_ids : list[str]
            Document IDs from the Brute-Force result.
        candidate_ids : list[str]
            Document IDs from the strategy being evaluated.

        Returns
        -------
        float
            Recall value in [0.0, 1.0].
        """
        if not ground_truth_ids:
            return 1.0

        gt_set: Set[str] = set(ground_truth_ids)
        overlap = len(gt_set & set(candidate_ids))
        return overlap / len(gt_set)
