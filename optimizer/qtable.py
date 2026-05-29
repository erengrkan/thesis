"""
optimizer/qtable.py — Stage 2: Contextual State Management (Q-Table)
=====================================================================
Maintains a lightweight, in-memory state map with O(1) lookup to track
the expected rewards of Pre-filter and Post-filter strategies across
different selectivity ranges.

Key design decisions:
  1. Variable Granularity Bucketing — tight buckets (2%) in the
     battleground zone where the crossover lives, wide buckets (10%)
     at the predictable extremes.
  2. Optimistic Initialization — all Q-values start at 1.0 (maximum
     possible reward) to force aggressive exploration during cold start.
  3. O(1) bucket lookup via bisect on sorted boundaries.
"""

from __future__ import annotations

import bisect
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from optimizer import config as cbo_config

log = logging.getLogger(__name__)


@dataclass
class BucketState:
    """State for a single selectivity bucket.

    Attributes
    ----------
    q_pre : float
        Expected reward for Pre-filter strategy in this bucket.
    q_post : float
        Expected reward for Post-filter strategy in this bucket.
    visits_pre : int
        Number of times Pre-filter was selected in this bucket.
    visits_post : int
        Number of times Post-filter was selected in this bucket.
    """
    q_pre: float = cbo_config.Q_INIT
    q_post: float = cbo_config.Q_INIT
    visits_pre: int = 0
    visits_post: int = 0

    @property
    def total_visits(self) -> int:
        return self.visits_pre + self.visits_post

    @property
    def delta(self) -> float:
        """Confidence gap: |Q_pre - Q_post|."""
        return abs(self.q_pre - self.q_post)

    @property
    def preferred_strategy(self) -> str:
        """The strategy with the higher Q-value."""
        return (
            cbo_config.PRE_FILTER_NAME
            if self.q_pre >= self.q_post
            else cbo_config.POST_FILTER_NAME
        )


class QTable:
    """Variable-granularity Q-table for contextual bandit state management.

    Parameters
    ----------
    bucket_boundaries : list[float]
        Sorted list of boundary values that define the bucket edges.
        A selectivity of ``s`` falls into bucket ``i`` where
        ``boundaries[i-1] <= s < boundaries[i]``.
    q_init : float
        Initial Q-value for all (strategy, bucket) pairs.
        Must be 1.0 for optimistic initialization.
    """

    def __init__(
        self,
        bucket_boundaries: List[float] = None,
        q_init: float = cbo_config.Q_INIT,
    ) -> None:
        if bucket_boundaries is None:
            bucket_boundaries = list(cbo_config.BUCKET_BOUNDARIES)

        self.boundaries = sorted(bucket_boundaries)
        self.q_init = q_init
        self.n_buckets = len(self.boundaries) + 1  # includes the overflow bucket

        # Initialize all buckets with optimistic Q-values
        self.buckets: Dict[int, BucketState] = {
            i: BucketState(q_pre=q_init, q_post=q_init)
            for i in range(self.n_buckets)
        }

        log.info(
            "QTable initialized: %d buckets, Q_init=%.1f, boundaries=%s",
            self.n_buckets, q_init, self.boundaries,
        )

    def get_bucket_id(self, selectivity: float) -> int:
        """Map a selectivity value to its bucket index. O(log N) via bisect.

        Parameters
        ----------
        selectivity : float
            Fraction of documents matching the filter (0.0 to 1.0).

        Returns
        -------
        int
            Bucket index (0 to n_buckets - 1).
        """
        return bisect.bisect_right(self.boundaries, selectivity)

    def get_bucket_range(self, bucket_id: int) -> Tuple[float, float]:
        """Return the (lower, upper) selectivity range for a bucket.

        Parameters
        ----------
        bucket_id : int
            Bucket index.

        Returns
        -------
        tuple[float, float]
            (lower_bound, upper_bound) — inclusive lower, exclusive upper.
        """
        lower = self.boundaries[bucket_id - 1] if bucket_id > 0 else 0.0
        upper = self.boundaries[bucket_id] if bucket_id < len(self.boundaries) else 1.0
        return (lower, upper)

    def get_q_values(self, selectivity: float) -> Tuple[float, float]:
        """Return (Q_pre, Q_post) for the bucket containing this selectivity.

        Parameters
        ----------
        selectivity : float
            Fraction of documents matching the filter.

        Returns
        -------
        tuple[float, float]
            (Q_pre, Q_post) expected rewards.
        """
        bucket_id = self.get_bucket_id(selectivity)
        state = self.buckets[bucket_id]
        return (state.q_pre, state.q_post)

    def get_state(self, selectivity: float) -> BucketState:
        """Return the full BucketState for a given selectivity."""
        return self.buckets[self.get_bucket_id(selectivity)]

    def update(
        self,
        selectivity: float,
        strategy: str,
        reward: float,
        alpha: float = cbo_config.ALPHA,
    ) -> None:
        """Apply Temporal Difference update to the Q-value.

        Formula: Q_new = Q_old + α * (Reward - Q_old)

        Parameters
        ----------
        selectivity : float
            The selectivity of the query that was just executed.
        strategy : str
            Which strategy was used ('pre_filter' or 'post_filter').
        reward : float
            The computed reward from the Soft Cliff SLA function.
        alpha : float
            Learning rate (default from config).
        """
        bucket_id = self.get_bucket_id(selectivity)
        state = self.buckets[bucket_id]

        if strategy == cbo_config.PRE_FILTER_NAME:
            old_q = state.q_pre
            state.q_pre = old_q + alpha * (reward - old_q)
            state.visits_pre += 1
            log.debug(
                "Q-update [bucket=%d, pre]: Q %.4f → %.4f (reward=%.4f, α=%.2f)",
                bucket_id, old_q, state.q_pre, reward, alpha,
            )
        elif strategy == cbo_config.POST_FILTER_NAME:
            old_q = state.q_post
            state.q_post = old_q + alpha * (reward - old_q)
            state.visits_post += 1
            log.debug(
                "Q-update [bucket=%d, post]: Q %.4f → %.4f (reward=%.4f, α=%.2f)",
                bucket_id, old_q, state.q_post, reward, alpha,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")

    def export_state(self) -> List[Dict]:
        """Export the full Q-table state for analysis and CSV output.

        Returns
        -------
        list[dict]
            One dict per bucket with all state fields.
        """
        rows = []
        for bucket_id in range(self.n_buckets):
            lower, upper = self.get_bucket_range(bucket_id)
            state = self.buckets[bucket_id]
            rows.append({
                "bucket_id": bucket_id,
                "range_lower": round(lower, 4),
                "range_upper": round(upper, 4),
                "q_pre": round(state.q_pre, 6),
                "q_post": round(state.q_post, 6),
                "delta": round(state.delta, 6),
                "visits_pre": state.visits_pre,
                "visits_post": state.visits_post,
                "total_visits": state.total_visits,
                "preferred": state.preferred_strategy,
            })
        return rows

    def __repr__(self) -> str:
        return f"QTable(n_buckets={self.n_buckets}, q_init={self.q_init})"
