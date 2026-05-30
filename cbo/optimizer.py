"""
cbo.optimizer — Main CBO Orchestrator
=======================================
The :class:`ContextualBanditOptimizer` ties together all CBO components
into a single routing + learning pipeline:

  1. **Guardrails** — Hard boundaries for extreme selectivities.
  2. **Q-Table**    — Lookup expected rewards for the selectivity bucket.
  3. **Decision Engine** — Explore vs. exploit using the chosen strategy.
  4. **Feedback**   — Compute reward and update Q-table via TD(0).
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import config
from cbo.exploration import ExplorationStrategy
from cbo.guardrails import Guardrails
from cbo.qtable import QTable
from cbo.reward import SoftCliffReward

log = logging.getLogger(__name__)


class ContextualBanditOptimizer:
    """Cost-Based Optimizer using a contextual bandit for strategy routing.

    Parameters
    ----------
    exploration_strategy : ExplorationStrategy
        The exploration strategy to use for the bandit.
    guardrails : Guardrails or None
        Hard boundary guardrails.  If ``None``, default thresholds are used.
    reward_fn : SoftCliffReward or None
        Reward function.  If ``None``, default parameters are used.
    alpha : float
        TD(0) learning rate.
    crossover_hint : float
        Initial estimate of the selectivity crossover point.

    Examples
    --------
    >>> from cbo.exploration import TierBased
    >>> opt = ContextualBanditOptimizer(exploration_strategy=TierBased())
    >>> strategy, was_guardrail = opt.route(0.25)
    >>> reward = opt.feedback(0.25, strategy, latency_ms=3.2, recall=0.95)
    """

    def __init__(
        self,
        exploration_strategy: ExplorationStrategy,
        guardrails: Optional[Guardrails] = None,
        reward_fn: Optional[SoftCliffReward] = None,
        alpha: float = config.CBO_ALPHA,
        crossover_hint: float = config.CBO_CROSSOVER_HINT,
    ) -> None:
        self.exploration = exploration_strategy
        self.guardrails = guardrails or Guardrails()
        self.reward_fn = reward_fn or SoftCliffReward()
        self.alpha = alpha
        self.qtable = QTable(crossover_hint)
        self._last_decision_overhead_us: float = 0.0
        self._step_counter: int = 0

        log.info(
            "CBO initialised: strategy=%s alpha=%.3f crossover_hint=%.2f",
            exploration_strategy.name,
            alpha,
            crossover_hint,
        )

    # ── Routing ──────────────────────────────────────────────────────────

    def route(self, selectivity: float) -> Tuple[str, bool]:
        """Decide which execution strategy to use for a query.

        The decision pipeline is:
          1. Check guardrails (hard boundaries).
          2. Lookup Q-values from the Q-table.
          3. Apply the exploration strategy to choose.

        Parameters
        ----------
        selectivity : float
            Filter selectivity in [0, 1].

        Returns
        -------
        tuple of (str, bool)
            ``(strategy_name, was_guardrail)`` where *was_guardrail*
            indicates that the decision was forced by a guardrail.
        """
        t0 = time.perf_counter()
        
        self._step_counter += 1

        # Stage 1: Guardrails
        forced = self.guardrails.check(selectivity)
        if forced is not None:
            self._last_decision_overhead_us = (time.perf_counter() - t0) * 1e6
            log.debug("Route σ=%.4f → %s (guardrail)", selectivity, forced)
            return forced, True

        # Stage 2: Q-Table lookup
        q_vals = self.qtable.get_q_values(selectivity)

        # Stage 3: Decision Engine (explore vs. exploit)
        strategy = self.exploration.choose(
            q_vals["bitmap_prefilter"],
            q_vals["post_filter"],
            step=self._step_counter,
        )

        self._last_decision_overhead_us = (time.perf_counter() - t0) * 1e6
        log.debug(
            "Route σ=%.4f → %s (q_bmp=%.4f q_post=%.4f)",
            selectivity,
            strategy,
            q_vals["bitmap_prefilter"],
            q_vals["post_filter"],
        )
        return strategy, False

    # ── Feedback / Learning ──────────────────────────────────────────────

    def feedback(
        self,
        selectivity: float,
        strategy: str,
        latency_ms: float,
        recall: float,
    ) -> float:
        """Compute the reward and update the Q-table.

        Parameters
        ----------
        selectivity : float
            Filter selectivity that was observed.
        strategy : str
            Strategy that was executed.
        latency_ms : float
            Observed query latency in milliseconds.
        recall : float
            Observed recall@K.

        Returns
        -------
        float
            The computed reward.
        """
        reward = self.reward_fn.compute(latency_ms, recall)
        self.qtable.update(selectivity, strategy, reward, self.alpha)

        log.debug(
            "Feedback σ=%.4f %s: latency=%.2fms recall=%.4f → reward=%.4f",
            selectivity,
            strategy,
            latency_ms,
            recall,
            reward,
        )
        return reward

    # ── Accessors ────────────────────────────────────────────────────────

    @property
    def decision_overhead_us(self) -> float:
        """Time spent on the last routing decision, in microseconds."""
        return self._last_decision_overhead_us

    def get_crossover_estimate(self) -> Optional[float]:
        """Return the current crossover estimate from the Q-table."""
        return self.qtable.get_crossover_estimate()

    def get_q_snapshot(self) -> List[Dict[str, Any]]:
        """Return a serialisable snapshot of the Q-table state."""
        return self.qtable.get_snapshot()

    def __repr__(self) -> str:
        return (
            f"ContextualBanditOptimizer("
            f"strategy={self.exploration.name}, "
            f"alpha={self.alpha}, "
            f"buckets={len(self.qtable.buckets)})"
        )
