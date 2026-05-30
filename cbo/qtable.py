"""
cbo.qtable — Variable-Granularity Q-Table
==========================================
Maps selectivity ranges (buckets) to Q-values for each execution strategy.

Bucket layout (25 buckets total):
  • Extreme low   [0.00, 0.10)  — 10% wide, 1 bucket
  • Inner zone    [0.10, 0.20)  —  2% wide, 5 buckets
  • Battleground  [0.20, 0.36)  —  2% wide, 8 buckets  (finest grain)
  • Mid zone      [0.36, 0.80)  —  5% wide, 9 buckets
  • Extreme high  [0.80, 1.00)  — 10% wide, 2 buckets

Finer granularity around the expected crossover region (≈0.25) gives the
bandit maximum discrimination power where it matters most.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import config

log = logging.getLogger(__name__)


@dataclass
class Bucket:
    """A single selectivity bucket holding Q-values and visit counts.

    Attributes
    ----------
    id : str
        Human-readable label, e.g. ``"0.20-0.22"``.
    lower : float
        Inclusive lower bound of the bucket.
    upper : float
        Exclusive upper bound of the bucket.
    q_bitmap : float
        Current Q-value estimate for ``bitmap_prefilter``.
    q_post : float
        Current Q-value estimate for ``post_filter``.
    visits_bitmap : int
        Number of times ``bitmap_prefilter`` was observed in this bucket.
    visits_post : int
        Number of times ``post_filter`` was observed in this bucket.
    """

    id: str
    lower: float
    upper: float
    q_bitmap: float = 1.0
    q_post: float = 1.0
    visits_bitmap: int = 0
    visits_post: int = 0

    @property
    def midpoint(self) -> float:
        """Return the midpoint of the bucket interval."""
        return (self.lower + self.upper) / 2.0

    @property
    def total_visits(self) -> int:
        """Total observations across both strategies."""
        return self.visits_bitmap + self.visits_post


class QTable:
    """Variable-granularity Q-table indexed by filter selectivity.

    The table stores Q-value estimates for two strategies:
      • ``bitmap_prefilter`` — pre-filter using a bitmap index
      • ``post_filter`` — retrieve more candidates then filter

    All Q-values are *optimistically initialised* to 1.0 so the bandit
    naturally explores unvisited buckets first.

    Parameters
    ----------
    crossover_hint : float
        Estimated selectivity at which the optimal strategy switches.
        Used only for informational purposes (bucket layout is fixed).
    """

    STRATEGIES: List[str] = ["bitmap_prefilter", "post_filter"]

    def __init__(self, crossover_hint: float = config.CBO_CROSSOVER_HINT) -> None:
        self.crossover_hint = crossover_hint
        self.buckets: List[Bucket] = self._build_buckets()
        # Pre-compute sorted bounds for fast lookup
        self._sorted_bounds: List[Tuple[float, float, int]] = [
            (b.lower, b.upper, i) for i, b in enumerate(self.buckets)
        ]
        log.info(
            "QTable initialised: %d buckets, crossover_hint=%.2f",
            len(self.buckets),
            crossover_hint,
        )

    # ── Construction ─────────────────────────────────────────────────────

    @staticmethod
    def _build_buckets() -> List[Bucket]:
        """Build the 25 variable-granularity buckets with optimistic Q=1.0."""
        edges: List[Tuple[float, float]] = []

        # Extreme low: 1 bucket [0.00, 0.10)
        edges.append((0.00, 0.10))

        # Inner zone (2% granularity): 5 buckets [0.10, 0.20)
        for i in range(5):
            lo = round(0.10 + i * 0.02, 2)
            hi = round(lo + 0.02, 2)
            edges.append((lo, hi))

        # Battleground (2% granularity): 8 buckets [0.20, 0.36)
        for i in range(8):
            lo = round(0.20 + i * 0.02, 2)
            hi = round(lo + 0.02, 2)
            edges.append((lo, hi))

        # Mid zone (variable width): 9 buckets [0.36, 0.80)
        mid_edges: List[Tuple[float, float]] = [
            (0.36, 0.41),
            (0.41, 0.46),
            (0.46, 0.51),
            (0.51, 0.56),
            (0.56, 0.61),
            (0.61, 0.66),
            (0.66, 0.71),
            (0.71, 0.76),
            (0.76, 0.80),
        ]
        edges.extend(mid_edges)

        # Extreme high: 2 buckets [0.80, 1.00)
        edges.append((0.80, 0.90))
        edges.append((0.90, 1.00))

        return [
            Bucket(
                id=f"{lo:.2f}-{hi:.2f}",
                lower=lo,
                upper=hi,
                q_bitmap=0.5,
                q_post=0.5,
                visits_bitmap=0,
                visits_post=0,
            )
            for lo, hi in edges
        ]

    # ── Lookup ───────────────────────────────────────────────────────────

    def get_bucket(self, selectivity: float) -> Bucket:
        """Find the bucket containing *selectivity*.

        Uses a linear scan over 25 buckets (fast enough; could use bisect
        for larger tables).  Selectivity is clamped to [0, 0.9999].

        Parameters
        ----------
        selectivity : float
            Filter selectivity in [0, 1].

        Returns
        -------
        Bucket
            The matching bucket.
        """
        s = max(0.0, min(selectivity, 0.9999))
        for b in self.buckets:
            if b.lower <= s < b.upper:
                return b
        # Fallback (should be unreachable with correct bucket edges)
        log.warning(
            "Selectivity %.6f did not match any bucket, falling back to last.",
            selectivity,
        )
        return self.buckets[-1]

    def get_q_values(self, selectivity: float) -> Dict[str, float]:
        """Return Q-values for both strategies at *selectivity*.

        Parameters
        ----------
        selectivity : float
            Filter selectivity in [0, 1].

        Returns
        -------
        dict
            ``{"bitmap_prefilter": float, "post_filter": float}``
        """
        b = self.get_bucket(selectivity)
        return {"bitmap_prefilter": b.q_bitmap, "post_filter": b.q_post}

    # ── Learning ─────────────────────────────────────────────────────────

    def update(
        self,
        selectivity: float,
        strategy: str,
        reward: float,
        alpha: float,
    ) -> None:
        """Apply a TD(0) update to the Q-value of *strategy* in the
        bucket for *selectivity*.

        ``Q_new = Q_old + α · (reward − Q_old)``

        Parameters
        ----------
        selectivity : float
            Filter selectivity that was observed.
        strategy : str
            Strategy that was executed (``"bitmap_prefilter"`` or
            ``"post_filter"``).
        reward : float
            Observed reward from the Soft Cliff SLA function.
        alpha : float
            Learning rate for the TD update.

        Raises
        ------
        ValueError
            If *strategy* is not one of the known strategies.
        """
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Expected one of {self.STRATEGIES}."
            )

        b = self.get_bucket(selectivity)
        if strategy == "bitmap_prefilter":
            b.q_bitmap += alpha * (reward - b.q_bitmap)
            b.visits_bitmap += 1
        else:
            b.q_post += alpha * (reward - b.q_post)
            b.visits_post += 1

        log.debug(
            "Q-update bucket=%s strategy=%s reward=%.4f → q_bitmap=%.4f q_post=%.4f",
            b.id,
            strategy,
            reward,
            b.q_bitmap,
            b.q_post,
        )

    # ── Analytics ────────────────────────────────────────────────────────

    def get_crossover_estimate(self) -> Optional[float]:
        """Estimate the selectivity crossover point from learned Q-values.
        
        Uses trend verification: requires at least 2 consecutive buckets
        where Q_post > Q_bitmap to confirm the crossover.
        
        Returns
        -------
        float or None
            Estimated crossover selectivity, or ``None``.
        """
        for i in range(len(self.buckets) - 1):
            b_curr = self.buckets[i]
            b_next = self.buckets[i+1]
            
            # If Q_post wins in this bucket AND the next one, it's a stable crossover
            if b_curr.q_post > b_curr.q_bitmap and b_next.q_post > b_next.q_bitmap:
                if i > 0:
                    return self.buckets[i].lower
                return b_curr.lower
        return None

    def get_snapshot(self) -> List[Dict]:
        """Return a serialisable snapshot of the entire Q-table.

        Useful for logging, visualisation, and checkpointing.

        Returns
        -------
        list of dict
            One dict per bucket with keys ``bucket_id``, ``lower``,
            ``upper``, ``q_bitmap``, ``q_post``, ``visits_bitmap``,
            ``visits_post``.
        """
        return [
            {
                "bucket_id": b.id,
                "lower": b.lower,
                "upper": b.upper,
                "q_bitmap": round(b.q_bitmap, 6),
                "q_post": round(b.q_post, 6),
                "visits_bitmap": b.visits_bitmap,
                "visits_post": b.visits_post,
            }
            for b in self.buckets
        ]

    def __repr__(self) -> str:
        return f"QTable(buckets={len(self.buckets)}, crossover_hint={self.crossover_hint})"
