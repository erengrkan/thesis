
# Vector Database Cost-Based Optimizer (CBO) with Contextual Bandits

This repo implements a Contextual Bandit-based Cost-Based Optimizer (CBO) to solve metadata filtering in high-dimensional vector databases. Instead of relying on a high-level abstraction like ChromaDB, this is built directly on top of raw **FAISS (Facebook AI Similarity Search)** so we can completely control the execution path.

## 🎯 The Problem

When you apply metadata filters to a vector search, you generally have two options:
1. **Pre-Filtering:** Find all the matching IDs first (via a Bitmap Index), then restrict the HNSW search to that specific subspace. 
   * *The catch:* If your filter is loose (e.g., 90% of the data matches), you end up brute-forcing the HNSW graph. Latency goes through the roof.
2. **Post-Filtering:** Do a standard HNSW search first (with an oversampling multiplier), then filter the metadata of the results.
   * *The catch:* If your filter is strict (e.g., only 1% of the data matches), none of the nearest neighbors found by HNSW will pass the filter. Your recall completely tanks (the classic "Zero Results Syndrome").

**The core question:** Given an incoming query's selectivity (what percentage of the dataset actually matches the filter), which execution plan should the database pick?

Instead of hardcoding a static threshold, this system uses a Contextual Bandit. It actively learns and adapts, dynamically finding the exact crossover point where it makes sense to switch from Pre-Filter to Post-Filter.

---

## 🏗️ CBO Architecture

The decision engine lives under the `/cbo/` package and operates in 4 main stages:

### Stage 1: Guardrails
No need to waste ML cycles on extreme edge cases. The system enforces hard boundaries for instant routing (`cbo/guardrails.py`):
- **`σ < 0.03 (Lower Bound):`** Force Pre-Filter. Post-filtering here would result in zero recall.
- **`σ > 0.90 (Upper Bound):`** Force Post-Filter. Pre-filtering here would just choke the CPU.

### Stage 2: Variable-Granularity Q-Table
`cbo/qtable.py` maps selectivity ratios into 25 state buckets. 
- The resolution isn't flat. In the critical "battleground" zone (`20% - 36%`) where the crossover actually happens, buckets are narrow (**2%** width) for high precision. At the extremes, we widen them to `10%`.
- **Optimistic Initialization:** All Q-Values start at `0.5` to encourage early exploration.
- **Trend Verification:** To stop the optimizer from flapping back and forth due to noise, a crossover decision must be verified across at least two consecutive buckets before becoming the new baseline.

### Stage 3: Exploration Strategies
This is how the Bandit balances exploiting the best-known route vs. exploring alternative execution plans (`cbo/exploration.py`). All strategies decay over time to prioritize stability as step count increases:
1. **Tier-Based:** Steps down exploration rates (40%, 15%, 5%) based on the delta between Q-values.
2. **Exponential Decay:** Cuts off exploration aggressively as the gap widens, using `ε = ε_max · e^(−k · Δ)`.
3. **Softmax (Temperature):** Probabilistic routing where the `Tau` variable cools down over time, similar to simulated annealing.

### Stage 4: Soft Cliff SLA Reward
This is the penalty function (`cbo/reward.py`). It heavily punishes the system for ruining recall—usually by picking Post-Filter when it shouldn't have.
- **Target Recall (R_target) = 93%**
- If the system hits the target, it gets a latency-based reward: `Reward = max(0, 1 - Latency / 20ms)`
- If recall drops below the target, it falls off a cliff with a heavy logarithmic penalty (β=10): `Reward = Speed_Reward * (Recall / 0.93)^10`

---

## 🚀 Setup & Usage

### Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

```

### 1. Indexing & Ingestion

We're using the Amazon Electronics Reviews dataset. This step writes the FAISS HNSW and Bitmap Indexes to disk:

```bash
python ingestion.py

```

### 2. Static Benchmark

Pre-computes speed and recall metrics for the classic, hardcoded methods (Pre-Filter, Post-Filter, Brute-Force) to serve as a baseline before introducing the Bandit:

```bash
python benchmark.py

```

### 3. Contextual Bandit Optimizer (CBO) Benchmark

Fires up the Bandit simulation to see how quickly it finds the ideal crossover threshold. Tweak `CBO_N_EPOCHS` and `CBO_ALPHA` in `config.py` to play with the hyperparams.

```bash
python cbo_benchmark.py

```

When finished, it automatically dumps telemetry plots into `/results/cbo_YYYYMMDD_HHMMSS/plots/`:

* **Crossover Convergence:** Watch the strategies hunt down and lock onto the ideal threshold.
* **Cumulative Regret:** Proves the system stops making bad routing decisions over time.
* **Recall vs Latency Scatter:** Visualizes how execution plans get penalized and rejected when they hit the Soft Cliff.

---

## 📁 Repo Structure

```text
faiss_bench/
├── cbo/
│   ├── __init__.py
│   ├── exploration.py   # Exploration & decay mechanics
│   ├── guardrails.py    # Hard thresholding bypass
│   ├── metrics.py       # Telemetry and logging
│   ├── optimizer.py     # Main CBO engine
│   ├── qtable.py        # State-bucket & reward mappings
│   └── reward.py        # Soft Cliff SLA logic
├── config.py            # Global hyperparameters
├── ingestion.py         # FAISS/Bitmap dataset processing
├── faiss_index.py       # FAISS wrapper
├── bitmap_index.py      # Roaring Bitmap search logic
├── strategies.py        # DB query execution plans (Exact, Pre, Post)
├── benchmark.py         # Static routing baseline tests
├── cbo_benchmark.py     # Bandit learning simulator
├── cbo_visualize.py     # Matplotlib plot generator
└── requirements.txt

```

```

