"""
ingestion.py — Data Ingestion into FAISS + Bitmap Index
=========================================================
Reads Amazon Electronics reviews from JSON, computes embeddings with
sentence-transformers, and builds:

  1. A FAISS HNSW index   (for approximate nearest-neighbor search)
  2. A Bitmap index       (for microsecond metadata filtering)
  3. A metadata list      (for post-filter predicate evaluation)

All three are persisted to ``config.INDEX_DIR`` for reuse by the
benchmark runner.

Optimisations for large datasets (~10 GB+):
  - JSON-Lines streaming  → constant memory for reading
  - Two-pass metadata loading  → only keeps ASINs we need
  - MPS / CUDA device auto-detection  → GPU-accelerated embedding
  - Chunked embedding  → controlled peak memory
"""

from __future__ import annotations

import os
# Prevent FAISS / HuggingFace segfaults on Mac (OpenMP & Multiprocessing conflicts)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import config
from bitmap_index import BitmapIndex
from faiss_index import FAISSIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Device Detection
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_device() -> str:
    """Pick the best available compute device for embedding.

    Priority: CUDA → MPS (Apple Silicon) → CPU
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    log.info("Compute device: %s", device)
    return device


# ═══════════════════════════════════════════════════════════════════════════════
#  JSON-Lines Reader
# ═══════════════════════════════════════════════════════════════════════════════

def _read_json_lines(path: Path, max_records: Optional[int] = None):
    """Yield one JSON object per line (JSON-Lines format)."""
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
            count += 1
            if max_records and count >= max_records:
                return


def _load_metadata_selective(
    meta_path: Path,
    needed_asins: Set[str],
) -> Dict[str, Dict[str, Any]]:
    """Load product metadata ONLY for the ASINs we actually need.

    Instead of loading the entire ~11 GB meta file into memory,
    this streams through it and only keeps entries whose ASIN
    is in ``needed_asins``.  Once all needed ASINs are found,
    reading stops early.

    Parameters
    ----------
    meta_path : Path
        Path to the meta_Electronics.json file.
    needed_asins : set[str]
        ASINs extracted from the reviews we're ingesting.

    Returns
    -------
    dict[str, dict]
        ASIN → metadata dict (only the fields we care about).
    """
    if not meta_path.exists():
        log.warning("Metadata file not found: %s", meta_path)
        return {}

    log.info("Streaming meta file for %s needed ASINs …", f"{len(needed_asins):,}")
    lookup: Dict[str, Dict[str, Any]] = {}
    scanned = 0

    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            scanned += 1
            if scanned % 500_000 == 0:
                log.info(
                    "  … scanned %s lines, found %s / %s ASINs",
                    f"{scanned:,}",
                    f"{len(lookup):,}",
                    f"{len(needed_asins):,}",
                )

            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            asin = obj.get("asin")
            if asin and asin in needed_asins:
                # Keep only the fields we use → saves RAM
                lookup[asin] = {
                    "brand": obj.get("brand"),
                    "price": obj.get("price"),
                    "main_cat": obj.get("main_cat"),
                }

                # Early exit if we've found everything
                if len(lookup) >= len(needed_asins):
                    log.info("  All needed ASINs found — stopping early.")
                    break

    log.info(
        "  Meta loaded: %s entries (scanned %s lines)",
        f"{len(lookup):,}", f"{scanned:,}",
    )
    return lookup


# ═══════════════════════════════════════════════════════════════════════════════
#  Record Preparation
# ═══════════════════════════════════════════════════════════════════════════════

def _prepare_record(review: dict, meta: dict) -> Optional[Dict[str, Any]]:
    """Build a single record from a review + product metadata."""
    review_id = review.get("reviewerID", "") + "_" + review.get("asin", "")
    text = review.get("reviewText", "") or review.get("summary", "")
    if not text or len(text.strip()) < 10:
        return None

    text = text[:config.TEXT_MAX_CHARS]

    # Merge review fields + product metadata into a flat metadata dict
    metadata: Dict[str, Any] = {}

    # From review
    overall = review.get("overall")
    if overall is not None:
        metadata["overall"] = float(overall)

    verified = review.get("verified")
    if verified is not None:
        metadata["verified"] = bool(verified)

    reviewer_name = review.get("reviewerName")
    if reviewer_name and isinstance(reviewer_name, str):
        metadata["reviewerName"] = reviewer_name[:100]

    asin = review.get("asin")
    if asin:
        metadata["asin"] = str(asin)

    # From product metadata
    brand = meta.get("brand")
    if brand and isinstance(brand, str) and brand.strip():
        metadata["brand"] = brand.strip()[:100]

    price = meta.get("price")
    if price is not None:
        try:
            metadata["price"] = float(str(price).replace("$", "").replace(",", ""))
        except (ValueError, TypeError):
            pass

    main_cat = meta.get("main_cat")
    if main_cat and isinstance(main_cat, str) and main_cat.strip():
        metadata["main_cat"] = main_cat.strip()[:100]

    return {
        "id": review_id,
        "document": text,
        "metadata": metadata,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Ingestion Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def ingest() -> None:
    """Main ingestion pipeline.

    Pipeline steps:
      1. Read reviews (streaming, first MAX_DOCUMENTS lines only).
      2. Collect needed ASINs → selective meta loading (minimal RAM).
      3. Compute embeddings with GPU acceleration (MPS / CUDA).
      4. Build FAISS HNSW index.
      5. Build Bitmap metadata index + save metadata list.
    """
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already ingested
    hnsw_path = config.INDEX_DIR / "hnsw.index"
    if hnsw_path.exists():
        log.info("Index already exists at %s. Skipping ingestion.", config.INDEX_DIR)
        log.info("Delete %s to re-ingest.", config.INDEX_DIR)
        return

    device = _detect_device()

    print(f"{'=' * 70}")
    print(f"  FAISS Ingestion Pipeline")
    print(f"{'=' * 70}")
    print(f"  Data source:  {config.REVIEWS_PATH}")
    print(f"  Meta source:  {config.META_PATH}")
    print(f"  Index dir:    {config.INDEX_DIR}")
    print(f"  Max docs:     {config.MAX_DOCUMENTS or 'unlimited'}")
    print(f"  Embedding:    {config.EMBEDDING_MODEL}")
    print(f"  Device:       {device}")
    print()

    # ── 1. Read reviews (streaming — only first MAX_DOCUMENTS) ─────────────
    log.info("[1/6] Reading reviews (first %s lines) …", f"{config.MAX_DOCUMENTS:,}")
    raw_reviews: List[dict] = []

    for review in tqdm(
        _read_json_lines(config.REVIEWS_PATH, max_records=config.MAX_DOCUMENTS),
        desc="Reading reviews",
        total=config.MAX_DOCUMENTS,
    ):
        raw_reviews.append(review)

    log.info("  Read %s raw reviews.", f"{len(raw_reviews):,}")

    # ── 2. Collect ASINs → selective meta loading ──────────────────────────
    log.info("[2/6] Collecting ASINs for selective meta loading …")
    needed_asins: Set[str] = set()
    for r in raw_reviews:
        a = r.get("asin")
        if a:
            needed_asins.add(a)
    log.info("  Unique ASINs in reviews: %s", f"{len(needed_asins):,}")

    meta_lookup = _load_metadata_selective(config.META_PATH, needed_asins)
    del needed_asins  # free memory

    # ── 3. Prepare records ─────────────────────────────────────────────────
    log.info("[3/6] Preparing records …")
    records: List[Dict[str, Any]] = []
    skipped = 0

    for review in raw_reviews:
        asin = review.get("asin", "")
        meta = meta_lookup.get(asin, {})
        rec = _prepare_record(review, meta)
        if rec is None:
            skipped += 1
            continue
        records.append(rec)

    del raw_reviews, meta_lookup  # free memory
    log.info("  Prepared %s records (skipped %s)", f"{len(records):,}", f"{skipped:,}")

    # ── 4. Compute embeddings (GPU-accelerated) ────────────────────────────
    raw_emb_path = config.INDEX_DIR / "raw_embeddings.npy"
    if raw_emb_path.exists():
        log.info("[4/6] Loading pre-computed embeddings from disk (%s) …", raw_emb_path)
        embeddings = np.load(raw_emb_path)
        embed_time = 0.0
    else:
        log.info("[4/6] Computing embeddings with %s on %s …", config.EMBEDDING_MODEL, device)
        model = SentenceTransformer(
            config.EMBEDDING_MODEL,
            cache_folder=str(config.HF_CACHE_DIR),
            device=device,
        )

        texts = [r["document"] for r in records]
        n_texts = len(texts)

        # Show time estimate
        test_batch = texts[:min(100, n_texts)]
        t_test = time.time()
        model.encode(test_batch, batch_size=config.BATCH_SIZE, normalize_embeddings=True)
        t_per_doc = (time.time() - t_test) / len(test_batch)
        eta_min = (n_texts * t_per_doc) / 60
        log.info(
            "  Speed estimate: %.1f ms/doc → ETA %.1f min for %s docs",
            t_per_doc * 1000, eta_min, f"{n_texts:,}",
        )

        t0 = time.time()
        embeddings = model.encode(
            texts,
            batch_size=config.BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,  # L2-normalise for cosine similarity
            device=device,
        )
        embeddings = np.array(embeddings, dtype=np.float32)
        embed_time = time.time() - t0

        log.info("Saving computed embeddings defensively to %s", raw_emb_path)
        np.save(raw_emb_path, embeddings)

        del texts, model  # free memory
        if device == "mps":
            torch.mps.empty_cache()

        log.info(
            "  Embeddings: shape=%s, time=%.1fs (%.0f docs/sec)",
            embeddings.shape, embed_time, len(records) / embed_time,
        )

    # ── 5. Build FAISS index ───────────────────────────────────────────────
    log.info("[5/6] Building FAISS HNSW index …")
    doc_ids = [r["id"] for r in records]

    faiss_idx = FAISSIndex(dim=embeddings.shape[1])
    faiss_idx.build(embeddings, doc_ids)
    faiss_idx.save(config.INDEX_DIR)

    # ── 6. Build Bitmap index ──────────────────────────────────────────────
    log.info("[6/6] Building Bitmap metadata index …")
    bitmap_idx = BitmapIndex()
    all_metadatas: List[Dict[str, Any]] = []

    for i, rec in enumerate(records):
        bitmap_idx.add_document(i, rec["id"], rec["metadata"])
        all_metadatas.append(rec["metadata"])

    bitmap_path = config.INDEX_DIR / "bitmap.idx"
    bitmap_idx.save(bitmap_path)

    # Save all metadata list (for post-filter predicate evaluation)
    meta_path = config.INDEX_DIR / "all_metadatas.pkl"
    with open(meta_path, "wb") as fh:
        pickle.dump(all_metadatas, fh, protocol=pickle.HIGHEST_PROTOCOL)
    log.info("Metadata list saved → %s", meta_path)

    # ── Summary ────────────────────────────────────────────────────────────
    stats = bitmap_idx.get_field_stats()
    print(f"\n{'=' * 70}")
    print(f"  Ingestion Complete")
    print(f"{'=' * 70}")
    print(f"  Total documents: {len(records):,}  (skipped {skipped:,})")
    print(f"  Embedding time:  {embed_time:.1f}s  ({len(records)/embed_time:.0f} docs/sec)")
    print(f"  Device used:     {device}")
    print(f"  FAISS ntotal:    {faiss_idx.hnsw_index.ntotal:,}")
    print(f"  Bitmap fields:")
    for field, vals in stats.items():
        print(f"    {field}: {len(vals)} unique values")
    print(f"  Index dir:       {config.INDEX_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    ingest()
