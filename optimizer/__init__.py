"""
optimizer — Contextual Bandit-Based Cost-Based Optimizer (CBO)
================================================================
Routes vector database queries between Pre-filter (Bitmap + Brute-Force)
and Post-filter (HNSW oversampling) strategies using a Contextual Bandit
learning mechanism that autonomously discovers the optimal crossover
point without hardcoded, dataset-dependent thresholds.

Architecture (4 Stages):
  Stage 1: Guardrails      — Algorithmic & hardware boundaries
  Stage 2: Q-Table         — Variable-granularity state management
  Stage 3: Decision Engine — Exploit/Explore via 3 competing strategies
  Stage 4: Feedback Loop   — Soft Cliff SLA reward + TD update
"""

from optimizer.bandit import ContextualBandit
from optimizer.guardrails import Guardrails
from optimizer.qtable import QTable
from optimizer.exploration import TierBased, ExponentialDecay, Softmax
from optimizer.reward import RewardCalculator

__all__ = [
    "ContextualBandit",
    "Guardrails",
    "QTable",
    "TierBased",
    "ExponentialDecay",
    "Softmax",
    "RewardCalculator",
]
