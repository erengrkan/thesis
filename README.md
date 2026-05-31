# Vector Database Cost-Based Optimizer (CBO) with Contextual Bandits

Bu proje, yüksek boyutlu vektör veritabanlarında (Vector DB) metadata filtreleme sorgularının optimizasyonu için geliştirilmiş bir **Bağlamsal Haydut (Contextual Bandit)** tabanlı **Cost-Based Optimizer (CBO)** uygulamasıdır. Proje, ChromaDB gibi yüksek seviyeli veritabanları yerine, kontrolün tamamen bizde olması için düşük seviyeli **FAISS (Facebook AI Similarity Search)** üzerine inşa edilmiştir.

## 🎯 Projenin Amacı ve Problemin Tanımı

Vektör aramalarında filtreleme (metadata filtering) yaparken genel olarak iki ana yaklaşım vardır:
1. **Pre-Filter:** Önce filtreye uyan tüm ID'ler (Bitmap Index) bulunur. Sonra sadece bu kısıtlanmış uzayda HNSW araması yapılır.
   - *Dezavantaj:* Filtre çok gevşekse (örn. verinin %90'ı uyuyorsa), HNSW grafı üzerinde gezinmek yerine tüm uzayı filtrelemek maliyetli hale gelir ve latency uçar.
2. **Post-Filter:** Önce filtersiz HNSW araması yapılır (oversampling katsayısı ile), çıkan sonuçların metadata'sı filtrelenir. 
   - *Dezavantaj:* Filtre çok sıkıysa (örn. verinin %1'i uyuyorsa), HNSW'nin bulduğu en yakın komşuların hiçbiri filtreye uymaz ve **Recall çöker (Sıfır Sonuç Sendromu)**.

**Soru şudur:** *Gelen bir sorgunun "Selectivity" (Seçicilik) oranına göre (hedef kitle % kaç), veritabanı hangi yöntemi seçmelidir?*

Sabit bir %Threshold belirlemek yerine, bu projede sistemin hangi noktada Pre-Filter'dan Post-Filter'a geçeceğini **kendi kendine öğrenebildiği** Contextual Bandit tabanlı dinamik bir Optimizer geliştirilmiştir.

---

## 🏗️ CBO Mimarisinin 4 Aşaması

Sistem, `/cbo/` paketi altında modüler olarak tasarlanmıştır ve karar mekanizması 4 ana aşamadan oluşur:

### Stage 1: Guardrails (Sert Sınırlar)
Aşırı uçlarda sistemin makine öğrenmesine ihtiyaç duymadan doğrudan hızlı karar vermesini sağlar (`cbo/guardrails.py`).
- **`σ < 0.03 (Alt Sınır):`** Sadece Pre-Filter (Çünkü Post-Filter'ın doğruluğu tamamen sıfırlanır).
- **`σ > 0.90 (Üst Sınır):`** Sadece Post-Filter (Çünkü Pre-Filter CPU'yu boğup gecikmeye neden olur).

### Stage 2: Variable-Granularity Q-Table (Değişken Hassasiyetli Durum Tablosu)
`cbo/qtable.py` dosyası selectivity oranlarına (durumlara) karşılık gelen 25 adet kova (bucket) oluşturur.
- Kritik geçişin (crossover) yaşandığı **Battleground (Savaş Alanı)** bölgesinde (`%20 - %36`) kovalar **%2** hassasiyete sahipken, uç bölgelerde `%10` genişliğindedir.
- **Optimistic Initialization:** Öğrenmeyi hızlandırmak için başlangıçta tüm Q-Değerleri `0.5` olarak başlatılır.
- **Trend Verification (Trend Doğrulaması):** Sistemin gürültüden (noise) etkilenip geçiş noktasını yanlış tahmin etmesini (flapping) engellemek için, crossover kararının en az iki ardışık bucket'ta doğrulanması şartı getirilmiştir.

### Stage 3: Exploration Strategies (Keşif Stratejileri)
Bandit'in "en iyi bilineni kullanmak (Exploit)" ile "farklı bir yöntem denemek (Explore)" arasındaki dengeyi bulmasını sağlar (`cbo/exploration.py`). Stratejilerin hepsi, zamanla (step sayısı arttıkça) keşif oranını **sıfıra yakınsayacak şekilde söndüren (Decay)** mekanizmalara sahiptir:
1. **Tier-Based (Kademeli):** Q değerleri arasındaki farka (Delta) göre kademeli (%5, %15, %40) keşif oranları belirler.
2. **Exponential Decay (Üssel Düşüş):** `ε = ε_max · e^(−k · Δ)` formülünü kullanarak, fark açıldıkça keşif yapmayı daha hızlı keser.
3. **Softmax (Sıcaklık):** Olasılıksal bir seçim yapar. `Tau` sıcaklık değişkeni zamanla soğutulur (Simulated Annealing).

### Stage 4: Soft Cliff SLA Reward (Cezalandırıcı Ödül Fonksiyonu)
Sistemin hatalı kararlarını (özellikle Post-Filter kullanarak Recall'u düşürmesini) ağır cezalandıran ödül fonksiyonudur (`cbo/reward.py`).
- **Hedef Recall (R_target) = %93**
- Eğer Recall hedefi tutturulursa, sadece hız bazlı ödül verilir: `Reward = max(0, 1 - Latency / 20ms)`
- Eğer Recall hedefin altına düşerse, **ağır bir logaritmik ceza (β=10)** uygulanır: `Reward = Hız_Ödülü * (Recall / 0.93)^10`

---

## 🚀 Kurulum ve Çalıştırma

### Bağımlılıklar
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1. Indexing & Ingestion (Veri Hazırlığı)
Amazon Electronics Reviews veri seti kullanılır. Bu adımda FAISS HNSW ve Bitmap Index'leri diske oluşturulur:
```bash
python ingestion.py
```

### 2. Statik Benchmark
Herhangi bir makine öğrenmesi (Bandit) kullanılmadan klasik yöntemlerin (Pre-Filter, Post-Filter, Brute-Force) hız ve doğruluk verilerini pre-compute eder:
```bash
python benchmark.py
```

### 3. Contextual Bandit Optimizer (CBO) Benchmark
Bandit sistemini çalıştırarak, sistemin geçiş eşiğini (Crossover) nasıl kendi kendine bulduğunu simüle eder. Model `config.py` içerisindeki hiperparametrelere (`CBO_N_EPOCHS`, `CBO_ALPHA`) göre eğitilir.
```bash
python cbo_benchmark.py
```
Bu script bittiğinde, `/results/cbo_YYYYMMDD_HHMMSS/plots/` klasörüne şu analiz grafiklerini otomatik üretir:
- **Crossover Convergence:** Stratejilerin ideal geçiş eşiğine nasıl tutunduğunu (converge ettiğini) gösterir.
- **Cumulative Regret:** Sistemin zamanla hata yapmayı bırakıp oracle(mükemmel) noktasına ulaştığını kanıtlar.
- **Recall vs Latency Scatter:** Soft Cliff ceza bölgesine düşen kararların nasıl reddedildiğini gösterir.

---

## 📁 Proje Yapısı

```
faiss_bench/
├── cbo/
│   ├── __init__.py
│   ├── exploration.py   # Keşif ve Decay Stratejileri
│   ├── guardrails.py    # Hard Thresholding
│   ├── metrics.py       # Telemetri ve Kayıtlar
│   ├── optimizer.py     # Ana CBO Sınıfı
│   ├── qtable.py        # Durum ve Ödül Tablosu
│   └── reward.py        # Soft Cliff Fonksiyonu
├── config.py            # Tüm hiperparametreler
├── ingestion.py         # FAISS indekslerinin oluşturulması
├── faiss_index.py       # FAISS sarmalayıcısı
├── bitmap_index.py      # Roaring Bitmap arama
├── strategies.py        # Veritabanı sorgu stratejileri (Exact, Pre, Post)
├── benchmark.py         # Statik sistem hız testi
├── cbo_benchmark.py     # Bandit öğrenme simülatörü
├── cbo_visualize.py     # Matplotlib grafik oluşturucusu
└── requirements.txt
```
