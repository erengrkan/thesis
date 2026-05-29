# FAISS Filtering Benchmark

Vector database'lerde **metadata filtreleme stratejilerinin** performans
karşılaştırması.  ChromaDB gibi yüksek seviyeli bir veritabanı yerine,
**FAISS** (Facebook AI Similarity Search) üzerinde düşük seviyeli kontrol
sağlayarak üç farklı filtreleme stratejisini ölçmektedir:

| Strateji | Açıklama |
|---|---|
| **Pre-Filter** | Önce bitmap ile eşleşen ID'ler bulunur, sonra FAISS `IDSelector` ile HNSW aranır |
| **Post-Filter** | Önce filtersiz HNSW aranır (oversampling), sonra Python'da metadata kontrolü yapılır |
| **Bitmap-Exact** | Bitmap ile eşleşen ID'ler bulunur, bu alt küme üzerinde brute-force (FlatIP) yapılır |

## Dataset

Amazon Electronics Reviews (JSON-Lines):
- `Electronics.json` — review metinleri
- `meta_Electronics.json` — ürün metadata'sı

## Kurulum

```bash
cd /Users/erengurkan/jobs/thesis/faiss_bench
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Çalıştırma

```bash
# 1. Veri yükle ve indeks oluştur
python ingestion.py

# 2. Benchmark çalıştır
python benchmark.py
```

## Proje Yapısı

```
faiss_bench/
├── config.py           # Merkezi konfigürasyon
├── ingestion.py        # Dataset → FAISS indeks + Bitmap indeks
├── faiss_index.py      # FAISS indeks yönetimi (HNSW, Flat)
├── bitmap_index.py     # Roaring Bitmap metadata indeksi
├── filters.py          # FilterSpec üretimi (selectivity bazlı)
├── strategies.py       # Pre-Filter, Post-Filter, Bitmap-Exact
├── benchmark.py        # Benchmark runner (QPS, latency, recall)
├── requirements.txt
└── results/            # Benchmark çıktıları (JSON)
```
