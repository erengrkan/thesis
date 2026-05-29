# Contextual Bandit-Based Cost-Based Optimizer (CBO)

A dynamic query routing system for vector databases that learns the optimal crossover point between **Pre-filter** (Bitmap + Brute-Force) and **Post-filter** (HNSW oversampling) strategies using a Contextual Bandit mechanism.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Incoming Query                             │
│              (query_vector, metadata_filter)                  │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  SELECTIVITY ESTIMATION                                      │
│  len(bitmap) / total_docs  →  O(1) popcount                 │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  STAGE 1: GUARDRAILS                                         │
│                                                              │
│  σ < 0.05  ──→  FORCE Pre-filter  (HNSW would collapse)     │
│  σ > 0.90  ──→  FORCE Post-filter (bitmap overhead wasted)  │
│  else      ──→  Continue to Bandit                           │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  STAGE 2: Q-TABLE LOOKUP                                     │
│                                                              │
│  Variable-granularity buckets:                               │
│    [0-10%] [10-20%] [20-22%] [22-24%] ... [48-50%] [50-60%] │
│     wide     wide    ◀── tight 2% battleground ──▶    wide   │
│                                                              │
│  Returns (Q_pre, Q_post) for the selectivity's bucket        │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  STAGE 3: DECISION ENGINE                                    │
│                                                              │
│  Δ = |Q_pre - Q_post|  (confidence gap)                     │
│                                                              │
│  Three competing strategies:                                 │
│    1. TierBased:        Δ > 0.30 → ε=0.05                   │
│                         Δ > 0.10 → ε=0.20                   │
│                         else     → ε=0.50                    │
│    2. ExponentialDecay: ε = 0.5 · e^(-10·Δ)                 │
│    3. Softmax:          P(pre) = e^(Q_pre/τ) / Σ            │
│                                                              │
│  → Selects Pre-filter or Post-filter                         │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  STAGE 4: EXECUTE & LEARN                                    │
│                                                              │
│  1. Execute chosen strategy → measure latency                │
│  2. Run Brute-Force ground truth → compute recall            │
│  3. Soft Cliff SLA reward:                                   │
│       L_norm = max(0, 1 - L/L_max)                          │
│       if R ≥ R_target: reward = L_norm                      │
│       if R < R_target: reward = L_norm · (R/R_target)^β     │
│  4. TD update: Q_new = Q_old + α · (reward - Q_old)         │
└──────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# From the faiss_bench/ directory
cd /Users/erengurkan/jobs/thesis/faiss_bench

# Ensure indices exist (run ingestion if needed)
# python3 ingestion.py

# Run the CBO benchmark
python3 -m optimizer.benchmark_cbo
```

## Module Overview

| File | Stage | Description |
|---|---|---|
| `config.py` | — | All CBO hyperparameters |
| `guardrails.py` | 1 | Lower/upper fence hard boundaries |
| `qtable.py` | 2 | Variable-granularity Q-table with optimistic init |
| `exploration.py` | 3 | Tier-Based, Exponential Decay, Softmax strategies |
| `reward.py` | 4 | Soft Cliff SLA reward function |
| `bandit.py` | 1-4 | Main orchestrator tying all stages together |
| `ground_truth.py` | 4 | Brute-Force oracle for per-query recall |
| `benchmark_cbo.py` | — | Full benchmark runner with CSV output |

## Output Files

After running the benchmark, results appear in `results/`:

- **`cbo_decisions_{strategy}_{timestamp}.csv`** — Per-query decision trace:
  episode, selectivity, bucket, Q-values (before/after), chosen strategy,
  latency, recall, reward

- **`cbo_qtable_{strategy}_{timestamp}.csv`** — Final learned Q-table:
  bucket ranges, Q_pre, Q_post, confidence gap, visit counts, preferred strategy

- **`cbo_summary_{timestamp}.csv`** — Cross-strategy comparison:
  mean latency, recall, reward, strategy selection counts

## Key Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `SIGMA_LOWER` | 0.05 | Lower guardrail (forces Pre-filter) |
| `SIGMA_UPPER` | 0.90 | Upper guardrail (forces Post-filter) |
| `Q_INIT` | 1.0 | Optimistic initialization |
| `ALPHA` | 0.1 | TD learning rate |
| `L_MAX` | 30.0 ms | Maximum acceptable latency |
| `R_TARGET` | 0.95 | Minimum acceptable recall |
| `BETA` | 10 | Soft cliff penalty exponent |
| `BOLTZMANN_TAU` | 0.1 | Softmax temperature |
| `DECAY_K` | 10.0 | Exponential decay constant |
| `CBO_NUM_QUERIES` | 500 | Queries per benchmark run |

## Mathematical Formulations

### Reward Function (Soft Cliff SLA)

$$L_{norm} = \max(0, 1 - \frac{L}{L_{max}})$$

$$Reward = \begin{cases} L_{norm} & \text{if } R \geq R_{target} \\ L_{norm} \cdot \left(\frac{R}{R_{target}}\right)^{\beta} & \text{if } R < R_{target} \end{cases}$$

### TD Update

$$Q_{new} = Q_{old} + \alpha \cdot (Reward - Q_{old})$$

### Exploration Strategies

**Exponential Decay:** $\epsilon = \epsilon_{max} \cdot e^{-k \cdot \Delta}$

**Softmax:** $P(pre) = \frac{e^{Q_{pre}/\tau}}{e^{Q_{pre}/\tau} + e^{Q_{post}/\tau}}$
