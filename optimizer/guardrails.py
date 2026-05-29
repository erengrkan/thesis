"""
optimizer/guardrails.py — Stage 1: Algorithmic & Hardware Boundaries
=====================================================================
Prevents catastrophic latency spikes, recall collapses, and unnecessary
CPU overhead BEFORE the Contextual Bandit is consulted.

Two fences operate as hard short-circuits:

  Lower Fence (σ_lower):
    At extremely low selectivities, HNSW graph connectivity degrades,
    leading to recall collapse and excessive pointer-chasing.
    → Forces Pre-filter (Bitmap + Brute-Force).

  Upper Fence (σ_upper):
    At extremely high selectivities, generating a massive bitset for
    queries that filter out almost nothing wastes CPU cycles.
    → Forces Post-filter (HNSW oversampling).

If selectivity falls between the fences, the Bandit decides.
"""

from __future__ import annotations

import logging
from typing import Optional

from optimizer import config as cbo_config

log = logging.getLogger(__name__)


class Guardrails:
    """Hard boundaries that bypass the Bandit for extreme selectivities.

    Parameters
    ----------
    sigma_lower : float
        Lower selectivity fence. Below this, Pre-filter is forced.
    sigma_upper : float
        Upper selectivity fence. Above this, Post-filter is forced.
    total_docs : int
        Total number of documents in the index (for logging/formula derivation).
    """

    def __init__(
        self,
        sigma_lower: float = cbo_config.SIGMA_LOWER,
        sigma_upper: float = cbo_config.SIGMA_UPPER,
        total_docs: int = 0,
    ) -> None:
        if sigma_lower >= sigma_upper:
            raise ValueError(
                f"σ_lower ({sigma_lower}) must be < σ_upper ({sigma_upper})"
            )
        self.sigma_lower = sigma_lower
        self.sigma_upper = sigma_upper
        self.total_docs = total_docs
        log.info(
            "Guardrails initialized: σ_lower=%.2f, σ_upper=%.2f, N=%d",
            sigma_lower, sigma_upper, total_docs,
        )

    def check(self, selectivity: float) -> Optional[str]:
        """Evaluate guardrails for the given selectivity.

        Parameters
        ----------
        selectivity : float
            Fraction of documents matching the filter (0.0 to 1.0).

        Returns
        -------
        str or None
            - ``'pre_filter'`` if lower fence triggers (force Pre-filter).
            - ``'post_filter'`` if upper fence triggers (force Post-filter).
            - ``None`` if selectivity is in the Bandit's decision zone.
        """
        if selectivity < self.sigma_lower:
            log.debug(
                "Lower fence triggered: selectivity=%.4f < σ_lower=%.2f → Pre-filter",
                selectivity, self.sigma_lower,
            )
            return cbo_config.PRE_FILTER_NAME

        if selectivity > self.sigma_upper:
            log.debug(
                "Upper fence triggered: selectivity=%.4f > σ_upper=%.2f → Post-filter",
                selectivity, self.sigma_upper,
            )
            return cbo_config.POST_FILTER_NAME

        return None

    def __repr__(self) -> str:
        return (
            f"Guardrails(σ_lower={self.sigma_lower:.2f}, "
            f"σ_upper={self.sigma_upper:.2f}, N={self.total_docs:,})"
        )
