"""
optimizer/exploration.py — Stage 3: Decision Engine (Dynamic Routing)
======================================================================
Determines whether to Exploit the best-known strategy or Explore the
alternative, based on the system's confidence level (Δ = |Q_pre - Q_post|).

Three competing exploration mechanisms are implemented for comparative
benchmarking:

  1. TierBased      — Discrete confidence tiers with fixed ε per tier.
  2. ExponentialDecay — Smooth ε reduction as confidence grows.
  3. Softmax         — Boltzmann distribution over Q-values (no explicit ε).
"""

from __future__ import annotations

import logging
import math
import random
from abc import ABC, abstractmethod
from typing import Tuple

from optimizer import config as cbo_config

log = logging.getLogger(__name__)


class ExplorationStrategy(ABC):
    """Base class for exploration/exploitation decision making.

    All strategies take (Q_pre, Q_post) and return a strategy name.
    """

    name: str = "base"

    @abstractmethod
    def select_action(self, q_pre: float, q_post: float) -> str:
        """Choose between pre_filter and post_filter.

        Parameters
        ----------
        q_pre : float
            Expected reward for Pre-filter.
        q_post : float
            Expected reward for Post-filter.

        Returns
        -------
        str
            ``'pre_filter'`` or ``'post_filter'``.
        """

    def _greedy(self, q_pre: float, q_post: float) -> str:
        """Return the strategy with the higher Q-value (exploit)."""
        if q_pre >= q_post:
            return cbo_config.PRE_FILTER_NAME
        return cbo_config.POST_FILTER_NAME

    def _random(self) -> str:
        """Return a uniformly random strategy (explore)."""
        return random.choice([
            cbo_config.PRE_FILTER_NAME,
            cbo_config.POST_FILTER_NAME,
        ])


class TierBased(ExplorationStrategy):
    """Discrete confidence tiers → fixed epsilon per tier.

    Confidence Gap (Δ = |Q_pre - Q_post|) determines the tier:
      - High Confidence   (Δ > 0.30): ε = 0.05  → mostly exploit
      - Moderate Confidence (Δ > 0.10): ε = 0.20  → cautious exploration
      - Battleground      (Δ ≤ 0.10): ε = 0.50  → aggressive exploration
    """

    name = "tier_based"

    def __init__(
        self,
        high_threshold: float = cbo_config.TIER_HIGH_CONFIDENCE,
        moderate_threshold: float = cbo_config.TIER_MODERATE_CONFIDENCE,
        epsilon_max: float = cbo_config.EPSILON_MAX,
    ) -> None:
        self.high_threshold = high_threshold
        self.moderate_threshold = moderate_threshold
        self.epsilon_max = epsilon_max

    def select_action(self, q_pre: float, q_post: float) -> str:
        delta = abs(q_pre - q_post)

        if delta > self.high_threshold:
            epsilon = 0.05  # High confidence → mostly exploit
        elif delta > self.moderate_threshold:
            epsilon = 0.20  # Moderate confidence → cautious
        else:
            epsilon = self.epsilon_max  # Battleground → aggressive explore

        if random.random() < epsilon:
            action = self._random()
            log.debug("TierBased: EXPLORE (Δ=%.4f, ε=%.2f) → %s", delta, epsilon, action)
        else:
            action = self._greedy(q_pre, q_post)
            log.debug("TierBased: EXPLOIT (Δ=%.4f, ε=%.2f) → %s", delta, epsilon, action)

        return action


class ExponentialDecay(ExplorationStrategy):
    """Smoothly reduces exploration rate as confidence grows.

    Formula: ε = ε_max · e^(-k · Δ)

    As the confidence gap (Δ) widens, ε decays exponentially toward 0,
    smoothly transitioning from exploration to exploitation without
    discrete tier boundaries.
    """

    name = "exponential_decay"

    def __init__(
        self,
        epsilon_max: float = cbo_config.EPSILON_MAX,
        decay_k: float = cbo_config.DECAY_K,
    ) -> None:
        self.epsilon_max = epsilon_max
        self.decay_k = decay_k

    def select_action(self, q_pre: float, q_post: float) -> str:
        delta = abs(q_pre - q_post)
        epsilon = self.epsilon_max * math.exp(-self.decay_k * delta)

        if random.random() < epsilon:
            action = self._random()
            log.debug(
                "ExponentialDecay: EXPLORE (Δ=%.4f, ε=%.4f) → %s",
                delta, epsilon, action,
            )
        else:
            action = self._greedy(q_pre, q_post)
            log.debug(
                "ExponentialDecay: EXPLOIT (Δ=%.4f, ε=%.4f) → %s",
                delta, epsilon, action,
            )

        return action


class Softmax(ExplorationStrategy):
    """Boltzmann distribution over Q-values.

    Translates Q-values directly into selection probabilities,
    removing the need for explicit epsilon calculation.

    Formula:
      P(pre) = e^(Q_pre / τ) / (e^(Q_pre / τ) + e^(Q_post / τ))

    Temperature (τ):
      - Low τ (→ 0): greedy (exploit), one strategy dominates.
      - High τ (→ ∞): uniform (explore), both strategies equally likely.
    """

    name = "softmax"

    def __init__(self, tau: float = cbo_config.BOLTZMANN_TAU) -> None:
        self.tau = tau

    def _compute_probabilities(
        self, q_pre: float, q_post: float
    ) -> Tuple[float, float]:
        """Compute Boltzmann selection probabilities.

        Uses the log-sum-exp trick to prevent numerical overflow.

        Returns
        -------
        tuple[float, float]
            (P_pre, P_post) — probabilities summing to 1.0.
        """
        # Clamp tau to avoid division by zero
        tau = max(self.tau, 1e-8)

        logit_pre = q_pre / tau
        logit_post = q_post / tau

        # Log-sum-exp trick for numerical stability
        max_logit = max(logit_pre, logit_post)
        exp_pre = math.exp(logit_pre - max_logit)
        exp_post = math.exp(logit_post - max_logit)
        total = exp_pre + exp_post

        p_pre = exp_pre / total
        p_post = exp_post / total

        return (p_pre, p_post)

    def select_action(self, q_pre: float, q_post: float) -> str:
        p_pre, p_post = self._compute_probabilities(q_pre, q_post)

        if random.random() < p_pre:
            action = cbo_config.PRE_FILTER_NAME
        else:
            action = cbo_config.POST_FILTER_NAME

        log.debug(
            "Softmax: P(pre)=%.4f, P(post)=%.4f, τ=%.3f → %s",
            p_pre, p_post, self.tau, action,
        )

        return action
