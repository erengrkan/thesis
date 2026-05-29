"""
optimizer/config.py — CBO Hyperparameters
============================================
All tunable parameters for the Contextual Bandit optimizer.
Isolated from the main benchmark config to maintain separation of concerns.

These values are empirically informed by our benchmark results:
  - Brute-Force baseline latency: ~25 ms (stable across all selectivities)
  - Empirical crossover point: ~33-37% selectivity
  - Bitmap Pre-Filter dominates below ~30%
  - Brute-Force dominates above ~40%
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 1 — Guardrail Thresholds (Empirically Derived)
# ═══════════════════════════════════════════════════════════════════════════════

# Lower fence: below this selectivity, HNSW graph rejection is too severe.
# Pre-filter is forced. Derived from benchmark data where bitmap_prefilter
# consistently outperforms all alternatives below 5% selectivity.
SIGMA_LOWER = 0.05

# Upper fence: above this selectivity, the bitmap covers nearly the full
# dataset and the bitset creation overhead exceeds its benefit.
# Post-filter is forced.
SIGMA_UPPER = 0.90

# Maximum acceptable unnecessary HNSW node visits (for formula-based σ_lower)
# Currently unused — kept for future formula-based derivation: σ_lower = K / M_MAX
M_MAX = 10_000

# Maximum acceptable bitset creation time in ms (for formula-based σ_upper)
# Currently unused — kept for future formula-based derivation
W_MAX = 1.0


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 2 — Q-Table Configuration
# ═══════════════════════════════════════════════════════════════════════════════

# Variable-granularity bucket boundaries.
# - Extremes (0-20%, 50-100%): wide 10% buckets (predictable, low learning value)
# - Battleground (20-50%): tight 2% buckets (high precision for crossover detection)
BUCKET_BOUNDARIES = [
    # --- Extreme low selectivity (wide buckets) ---
    0.10, 0.20,
    # --- Battleground zone (tight 2% buckets) ---
    0.22, 0.24, 0.26, 0.28, 0.30,
    0.32, 0.34, 0.36, 0.38, 0.40,
    0.42, 0.44, 0.46, 0.48, 0.50,
    # --- Extreme high selectivity (wide buckets) ---
    0.60, 0.70, 0.80, 0.90,
    # Anything > 0.90 falls into the last bucket
]

# Optimistic initialization: forces aggressive exploration during cold start.
# All Q-values (Q_pre, Q_post) start at 1.0 instead of 0.0.
Q_INIT = 1.0


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 3 — Exploration Strategy Parameters
# ═══════════════════════════════════════════════════════════════════════════════

# Maximum exploration rate (used by Tier-Based and Exponential Decay)
EPSILON_MAX = 0.5

# --- Tier-Based thresholds ---
# Confidence gap (Δ) thresholds for discrete tiers
TIER_HIGH_CONFIDENCE = 0.30      # Δ > 0.30 → ε = 0.05 (mostly exploit)
TIER_MODERATE_CONFIDENCE = 0.10  # Δ > 0.10 → ε = 0.20
# Below 0.10 → Battleground → ε = EPSILON_MAX

# --- Exponential Decay ---
DECAY_K = 10.0  # Controls how fast ε decays as Δ grows
# ε = EPSILON_MAX * e^(-DECAY_K * Δ)

# --- Softmax (Boltzmann) ---
BOLTZMANN_TAU = 0.1  # Temperature parameter
# Lower τ → more greedy (exploit); Higher τ → more uniform (explore)


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 4 — Reward Function Parameters (Soft Cliff SLA)
# ═══════════════════════════════════════════════════════════════════════════════

# Maximum acceptable latency for normalization.
# Derived from Brute-Force baseline (~25 ms) with 20% headroom.
L_MAX = 30.0

# Target recall threshold — queries meeting this get pure latency reward.
R_TARGET = 0.95

# Penalty exponent (β). Controls the "shock absorber" behavior:
#   β=1  → linear penalty (too lenient)
#   β=5  → moderate penalty
#   β=10 → harsh penalty for severe recall degradation (recommended)
BETA = 10

# Learning rate for TD update: Q_new = Q_old + α * (Reward - Q_old)
ALPHA = 0.1


# ═══════════════════════════════════════════════════════════════════════════════
#  Benchmark Configuration
# ═══════════════════════════════════════════════════════════════════════════════

# Number of queries per benchmark run
CBO_NUM_QUERIES = 2000

# Warmup queries (excluded from measurements)
CBO_WARMUP = 10

# Random seed for reproducibility
CBO_SEED = 42

# Strategy names (must match strategy.name in strategies.py)
PRE_FILTER_NAME = "pre_filter"
POST_FILTER_NAME = "post_filter"
