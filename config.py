"""
config.py — Merkezi Konfigürasyon
====================================
Cost-Based Optimization in Vector Databases (Senior Thesis)

Tüm yollar, hiper-parametreler ve özellik anahtarları burada
tanımlanır.  Projede kullanılan her modül bu dosyayı referans alır.
"""

from pathlib import Path

# ── Dataset Yolları ────────────────────────────────────────────────────────────
REVIEWS_PATH = Path("/Users/erengurkan/jobs/thesis/dataset/Electronics.json")
META_PATH    = Path("/Users/erengurkan/jobs/thesis/dataset/meta_Electronics.json")

# ── FAISS İndeks Dizini ────────────────────────────────────────────────────────
INDEX_DIR = Path("/Users/erengurkan/jobs/thesis/faiss_bench/index_data")

# ── Çıktı ──────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("/Users/erengurkan/jobs/thesis/faiss_bench/results")

# ── Ingestion ──────────────────────────────────────────────────────────────────
BATCH_SIZE     = 512            # Embedding batch boyutu
MAX_DOCUMENTS  = 200_000        # PoC sınırı  (None → tam dataset)
TEXT_MAX_CHARS = 512            # Metin kırpma (embedding verimliliği)

# ── Embedder ───────────────────────────────────────────────────────────────────
# BERT-base model fine-tuned for sentence embeddings (768-dim)
EMBEDDING_MODEL  = "sentence-transformers/bert-base-nli-mean-tokens"
EMBEDDING_DIM    = 768          # BERT-base output dimension
HF_CACHE_DIR     = Path("/Users/erengurkan/jobs/thesis/faiss_bench/.hf-cache")

# ── FAISS HNSW Parametreleri ───────────────────────────────────────────────────
HNSW_M          = 32            # Her düğüme bağlanan komşu sayısı
HNSW_EF_CONSTR  = 200           # İndeks inşa sırasındaki ef değeri
HNSW_EF_SEARCH  = 128           # Arama sırasındaki ef değeri

# ── Benchmark ──────────────────────────────────────────────────────────────────
TOP_K_VALUES          = [10, 50]
SELECTIVITY_TARGETS   = [0.05, 0.07, 0.09, 0.12, 0.14, 0.16, 0.18, 0.21, 0.23, 0.25, 0.27, 0.30, 0.32, 0.34, 0.39, 0.61, 0.66, 0.68, 0.70, 0.73, 0.75, 0.77, 0.79, 0.82, 0.84, 0.86, 0.88, 0.91, 0.93, 0.95]
NUM_QUERY_SAMPLES     = 50
WARMUP_QUERIES        = 5
POST_FILTER_EXPANSION = 100     # Post-filter: top_k × expansion_factor

# ── Bitmap İndeks Alanları ─────────────────────────────────────────────────────
BITMAP_FIELDS = ["main_cat", "brand", "overall", "verified"]
