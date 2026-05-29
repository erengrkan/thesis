"""
optimizer/reward.py — Stage 4: Soft Cliff SLA Reward Function
================================================================
Replaces rigid Hard SLAs and unstable weighted averages, enabling
smart trade-offs between speed and accuracy.

The reward function:
  1. Normalizes latency into [0, 1] scale based on L_max.
  2. If recall meets the target: pure latency reward.
  3. If recall misses the target: exponential penalty via β exponent.

The β exponent acts as a "shock absorber":
  - Tolerates minor recall misses if latency gains are massive (Near Miss).
  - Ruthlessly penalizes severe recall degradation (Dangerous Slide).
"""

from __future__ import annotations

import logging
import math

from optimizer import config as cbo_config

log = logging.getLogger(__name__)


class RewardCalculator:
    """Soft Cliff SLA reward function.

    Parameters
    ----------
    l_max : float
        Maximum acceptable latency (ms). Used for normalization.
    r_target : float
        Target recall threshold. Queries meeting this get pure latency reward.
    beta : float
        Penalty exponent. Higher values penalize recall drops more harshly.
    """

    def __init__(
        self,
        l_max: float = cbo_config.L_MAX,
        r_target: float = cbo_config.R_TARGET,
        beta: float = cbo_config.BETA,
    ) -> None:
        if l_max <= 0:
            raise ValueError(f"L_max must be positive, got {l_max}")
        if not (0.0 < r_target <= 1.0):
            raise ValueError(f"R_target must be in (0, 1], got {r_target}")
        if beta < 0:
            raise ValueError(f"Beta must be non-negative, got {beta}")

        self.l_max = l_max
        self.r_target = r_target
        self.beta = beta

        log.info(
            "RewardCalculator initialized: L_max=%.1f ms, R_target=%.2f, β=%d",
            l_max, r_target, beta,
        )

    def normalize_latency(self, latency_ms: float) -> float:
        """Compress latency into [0, 1] scale.

        Formula: L_norm = max(0, 1 - L / L_max)

        A latency of 0 ms → 1.0 (perfect).
        A latency of L_max ms → 0.0 (worst acceptable).
        A latency > L_max → 0.0 (clamped).
        """
        return max(0.0, 1.0 - latency_ms / self.l_max)

    def compute(self, latency_ms: float, recall: float) -> float:
        """Compute the Soft Cliff SLA reward.

        Parameters
        ----------
        latency_ms : float
            End-to-end query latency in milliseconds.
        recall : float
            Recall@K of the query result (0.0 to 1.0).

        Returns
        -------
        float
            Reward value in [0, 1]. Higher is better.
        """
        l_norm = self.normalize_latency(latency_ms)

        if recall >= self.r_target:
            # Target met → reward is strictly based on speed
            reward = l_norm
            log.debug(
                "Reward (target met): L=%.2f ms, R=%.4f, L_norm=%.4f → reward=%.4f",
                latency_ms, recall, l_norm, reward,
            )
        else:
            # Target missed → apply exponential penalty
            ratio = recall / self.r_target  # < 1.0
            penalty = math.pow(ratio, self.beta)
            reward = l_norm * penalty
            log.debug(
                "Reward (PENALTY): L=%.2f ms, R=%.4f, L_norm=%.4f, "
                "penalty=(%.4f/%.2f)^%d=%.6f → reward=%.6f",
                latency_ms, recall, l_norm, recall, self.r_target,
                self.beta, penalty, reward,
            )

        return reward

    def __repr__(self) -> str:
        return (
            f"RewardCalculator(L_max={self.l_max}, "
            f"R_target={self.r_target}, β={self.beta})"
        )
