# VLM-RL-Stress — Proje Durum Raporu

**Öğrenci:** Taha Yasin ER (N25142287) — CMP 620
**Son Güncelleme:** 29 Nisan 2026 (Sohbet 2 sonu)
**Durum:** Faz 0-5 tam tamamlandı — paper-faithful 3-seed inference reproduction başarılı
**Bu doküman:** Faz 6'ya geçişte yeni sohbete kaldığımız yerden devam etmek için

---

## 1. Tek Cümlede Durum

VLM-RL paper Tab 4'ün **inference-only reproduction'ı** 5 town × 3 inference seed × 10 route protokolüyle (N=30/cell) başarıyla tamamlandı; **Town02 (eğitim town'u) point estimate'leri paper toleransı içinde** (RC=0.93 vs 0.97, SR=0.90 vs 0.93), **Town04 paper'dan daha iyi** (SR=0.90 vs 0.70±0.28), **Town03/Town05'te generalization paper'dan zayıf**.

---

## 2. Sistem Konfigürasyonu (Doğrulanmış)

### Donanım
- GPU: NVIDIA RTX 5080 (16 GB) — Blackwell, sm_120
- Driver: 595.71
- CUDA Runtime: 13.2 (driver tarafı)

### Yazılım Yığını
| Bileşen | Versiyon | Durum |
|---|---|---|
| Windows | 11 | ✅ |
| Python | 3.12.12 | ✅ |
| PyTorch | 2.11.0+cu128 | ✅ sm_120 native |
| CARLA Server | 0.9.16 | ✅ Binary installed |
| CARLA Client | 0.9.16 | ✅ pip wheel |
| Conda env | `carla_env` | ✅ |
| NumPy | 1.26.4 | ✅ |
| SB3 | 2.8.0 | ✅ |
| Gymnasium | 1.2.3 | ✅ |
| gym (legacy) | 0.23.0 | ✅ Yan yana |
| OpenCLIP | 3.3.0 | ✅ |
| h5py | latest | ✅ |

### Klasör Yapısı (Güncel)
```
C:\Users\TAHA\Desktop\CARLA_0.9.16\
├── CarlaUE4.exe
├── PythonAPI\
│   └── VLM-RL-Stress\
│       ├── repo\                              # Klonlanmış VLM-RL repo
│       │   ├── carla_env\
│       │   ├── clip\
│       │   ├── config.py
│       │   ├── train.py                       # default seed=100, --seed yok
│       │   ├── eval.py                        # default seed=101
│       │   ├── run_eval.py
│       │   ├── my_eval_minimal.py             ← bizim (Phase 5 tek-ep validator)
│       │   ├── my_eval_full.py                ← bizim (legacy 5-town in-process)
│       │   ├── eval_one_town.py               ← bizim (CURRENT, --seed destekli worker)
│       │   ├── run_full_eval.py               ← bizim (CURRENT, multi-seed orchestrator)
│       │   ├── record_videos.py               ← bizim (1 success + 1 fail episode)
│       │   ├── record_town_routes.py          ← bizim (tek town tüm 10 route tek video)
│       │   ├── train_one_seed.py              ← bizim (Phase 6 hazırlık, henüz kullanılmadı)
│       │   ├── run_training.py                ← bizim (Phase 6 orchestrator, kullanılmadı)
│       │   └── train_smoke_test.py            ← bizim (kullanılmadı, smoke atlandı)
│       ├── checkpoints\
│       │   └── hf_vlmrl\
│       │       └── sac_model.zip              # 32 MB - HF checkpoint (1 seed)
│       └── results\
│           ├── eval_full_episodes.csv         # 150 satır: 5 town × 3 seed × 10 ep
│           ├── eval_full_summary.csv          # 5 town: 3-seed mean ± std
│           ├── eval_5towns_n10_backup.csv     # ilk N=10 single-seed run yedek
│           ├── eval_n30_single_seed_*.csv     # ara N=30 single-seed run yedek
│           └── videos\
│               ├── Town02_all_routes_seed100.avi   # 10 route ardışık video
│               └── (diğer townlar opsiyonel)
```

---

## 3. Tamamlanan Fazlar

### ✅ Faz 0-4: Sistem Kurulumu (Sohbet 1'de)
- Donanım/yazılım yığını kuruldu, CARLA 0.9.13 → 0.9.16 tek API farkı düzeltildi (`CityObjectLabel.Vehicles` → Car+Bus+Truck+Motorcycle+Bicycle)
- Python 3.8 → 3.12 pickle uyumsuzluğu **manuel state_dict yükleme** ile aşıldı
- Detay için Sohbet 1 PROJECT_STATUS yedeği

### ✅ Faz 5: Reproduction (Sohbet 2'de tamamlandı)

**Protokol:**
- 5 town (Town01-05) × 3 inference seed (42, 100, 2024) × 10 predefined route
- N=30 per cell (paper Tab 4 ile aynı yapı)
- max_steps = 15000 per episode (paper'da yok, bizim emniyet cap'i, **kimse cap'e takılmadı**)
- Inference-only: HF checkpoint (paper'ın eğittiği 3 seed'den biri)
- eval=True env (paper protokolü)
- traffic_flow=True (20 NPC autopilot — paper "regular density")

**N=10 → N=30 Evrimi (Önemli ders):**
1. İlk run: 5 town × 10 ep tek koşum (single-seed, single-trial)
2. Ara run: Town01+02 × 30 ep tek seed (route×3 tekrar) — env varyansı kontrolü
3. Final run: 5 town × 3 seed × 10 ep — **paper-faithful**

**Bug'lar Çözüldü:**
- Hız birim akışı **doğrulandı**: `vehicle.get_speed()` zaten km/h, çift çarpım yok (matematiksel cross-check ile)
- Multi-map CARLA bug: per-cell subprocess + fresh CARLA = %100 stable
- Windows cp1254 cmd encoding: `PYTHONIOENCODING=utf-8` ile aşıldı
- Module-level `eval_routes` cycle: per-cell reset
- Resume logic: (town, seed) çifti bazlı, crash → re-run = pick up where left off

---

## 4. Faz 5 Sonuçları — Paper Tab 4 Karşılaştırma

### 4.1 Bizim Tam Sonuçlar (3-seed mean ± std)

| Town | AS (km/h) | RC (clipped) | SR | CR | CS (km/h) |
|---|---|---|---|---|---|
| Town01 | 23.18 ± 0.73 | **1.000 ± 0.000** | 0.900 ± 0.000 | 0.100 ± 0.000 | 2.04 ± 1.23 |
| Town02 | 20.79 ± 0.86 | **0.930 ± 0.034** | 0.900 ± 0.000 | 0.100 ± 0.000 | 9.02 ± 7.78 |
| Town03 | 21.27 ± 0.47 | 0.793 ± 0.026 | 0.667 ± 0.058 | 0.333 ± 0.058 | 2.73 ± 1.78 |
| Town04 | 22.33 ± 0.08 | **0.929 ± 0.000** | **0.900 ± 0.000** | 0.100 ± 0.000 | 0.08 ± 0.02 |
| Town05 | 23.33 ± 0.47 | 0.864 ± 0.070 | 0.633 ± 0.208 | 0.367 ± 0.208 | 2.38 ± 2.74 |

### 4.2 Paper Tab 4 (vlm_rl, 3 trained seed)

| Town | AS | RC | SR | CR | CS |
|---|---|---|---|---|---|
| Town01 | 22.9±0.6 | 1.00±0.00 | 1.00±0.00 | ~0 | 0.03±0.05 |
| Town02 | 19.3±1.29 | 0.97±0.03 | 0.93±0.04 | ~0.07 | 0.02±0.03 |
| Town03 | 21.7±0.6 | 0.91±0.07 | 0.87±0.09 | ~0.13 | 1.14±1.54 |
| Town04 | 22.0±3.7 | 0.80±0.17 | **0.70±0.28** | ~0.30 | 2.15±1.59 |
| Town05 | 22.9±0.9 | 0.93±0.03 | 0.87±0.05 | ~0.13 | 0.46±0.54 |

### 4.3 Karşılaştırma Yorumu

**✅ Eşleşen (paper toleransı içinde):**
- **Town02**: AS Δ%8, RC Δ%4, SR Δ%3 — eğitim town'unda paper neredeyse birebir reproduce edildi
- **Town01 RC**: Paper'la aynı (1.00) ama SR farkı var (paper 1.00, biz 0.90, route 6'da deterministic collision)

**⚠️ Sapma — açıklanabilir:**
- **Town04 SR daha iyi**: Bizim 0.90, paper 0.70±**0.28**. Paper'ın std'si çok yüksek (3 seed arasında varyans), point estimate güvenilirliği düşük. Bizim cap=15000 yetti, hiç max_steps kurbanı olmadı.
- **Town03 SR Δ%23**: 0.67 vs 0.87. Route 1, 2, 7 problematik (3 seed'de de collision)
- **Town05 SR Δ%26**: 0.63 vs 0.87. Yüksek seed varyansı (std=0.208), Town05 environment-stokastiğine duyarlı

### 4.4 Fundamental Bulgu — Std Yapısı

| Town | SR_std | Yorum |
|---|---|---|
| Town01 | 0.000 | Tüm seed'ler aynı |
| Town02 | 0.000 | Tüm seed'ler aynı |
| Town03 | 0.058 | Çok az varyans |
| Town04 | 0.000 | Tüm seed'ler aynı |
| Town05 | **0.208** | Yüksek env-varyans |

**4/5 town'da std=0** → trafik manager seed'i çoğu town'da deterministic davranıyor. Town05 hariç. Collision pattern'ları büyük ölçüde **agent-deterministic** (HF checkpoint'in zayıf noktaları).

### 4.5 Akademik Etiket — Reproduction İddiamızın Sınırı

**Yapılan:**
> "Single-checkpoint inference-only reproduction with N=30 (3 inference seeds × 10 routes per town). Captures environment-level variance from CARLA traffic stochasticity."

**Yapılmayan:**
> "Paper'ın 3 trained agent (policy-level) varyansı reproduce edilmedi — bunun için 3 ayrı eğitim koşumu gerekirdi (HF'da sadece 1 checkpoint var)."

Bu sınırı raporda **dürüstçe belirtmek zorunlu**. İddia ederken: "We reproduce paper's point estimates within tolerance for the training town (Town02), but our std values reflect environment stochasticity rather than seed-level training variance."

---

## 5. Önemli Bulgular ve Sınırlamalar

### 5.1 Deterministic Failure Routes
Tüm 3 seed'de tutarlı şekilde başarısız olan rotalar:
- **Town01 Route 6**: 3/3 collision (low-speed, scrape-like)
- **Town02 Route 4**: 3/3 collision (yüksek hızlı, gerçek çarpışma)
- **Town03 Route 1, 2, 7**: çoğunlukla collision
- **Town04 Route 3**: 3/3 collision (low-speed)
- **Town05 Route 1, 3, 6, 7**: yüksek collision oranı

Bu rotalar HF checkpoint'in **özgün zayıflıkları** — env stokastiğinden bağımsız.

### 5.2 Town04 Ana Bulgusu
Önceki run'larda 5000 step cap kurbanıydı (4/10 max_steps). 15000 cap ile **0 cap kurbanı**. Bu Town04 SR=0.90'ın gerçek olduğu anlamına geliyor — paper'ın 0.70±0.28 sonucundan **daha iyi**.

### 5.3 Hız Birimleri Doğrulandı
`env.vehicle.get_speed()` ve `info['collision_speed']` ikisi de **km/h** döndürüyor (env içinde 3.6 çarpımı var). `eval_one_town.py` çift çarpım yapmıyor. Matematik cross-check ile teyitli (TD ≈ duration × speed/3.6).

### 5.4 CARLA Multi-Map Bug
0.9.16'da ardışık `client.load_world()` çağrıları **MultiStreamState assertion failure** veriyor. Çözüm: per-cell subprocess + fresh CARLA. `run_full_eval.py` bunu otomatik yapıyor.

---

## 6. Faz 5 Çıktıları (Görsel + Veri)

### 6.1 CSV'ler
- **`results/eval_full_episodes.csv`**: 150 satır, her episode için detaylı metrikler
  - Kolonlar: town, seed, ep_idx, route_idx, route_repetition, steps, mean_speed_kmh, routes_completed, collision_state, collision_speed_kmh, success, terminal_reason, vs.
- **`results/eval_full_summary.csv`**: 5 satır, paper Tab 4 formatı (3-seed mean ± std)

### 6.2 Videolar
- **`results/videos/Town02_all_routes_seed100.avi`**: Tek video, 10 route ardışık, route geçişlerinde overlay
- Diğer townlar için aynı script ile koşturulabilir: `python record_town_routes.py --town <Town>`

### 6.3 Yedek Datalar
- `eval_5towns_n10_backup.csv`: İlk N=10 single-seed run (1500/5000 cap karışık)
- `eval_n30_single_seed_*.csv`: Ara N=30 single-seed run (Town01+02)

---

## 7. Sıradaki Sohbet — Faz 6 Plan

### 7.1 İlk 10 Dakika
1. Bu dokümanı yükle
2. CARLA çalışıyor mu kontrol: `python -c "import carla; c=carla.Client('localhost',2000); c.set_timeout(10); print(c.get_server_version())"`
3. Çalışmıyorsa: standart CARLA başlatma prosedürü (Section 8.2)

### 7.2 Faz 6 Deneyleri (Proje Proposal'a Göre)

Plan dokümanı `VLM-RL-Stress_Project_Plan.md` §8'e göre 3 deney:

**Deney 3: Weather Robustness (önerilen ilk — eğitim yok)**
- Eğitim gerektirmez, mevcut altyapı ile koşulabilir
- Paper §5.1.2'de weather sabitlenmiş ("default sunny"), biz farklı CARLA weather presetleri test edeceğiz (rain, fog, night)
- Kullanılacak: `eval_one_town.py` + `--weather` parametresi (eklenmesi gerek)
- N=30 protokolü, Town01+Town02 (en güvenilir reproduction'ımız olduğu için)
- Tahmini süre: ~3-4 saat (3 weather × 2 town × 3 seed × 10 ep)

**Deney 2: CLG Sensitivity (orta zorluk)**
- Alternatif positive/negative language goal pairs
- Paper'ın default CLG: `("Two cars have collided with each other on the road", "The road is clear with no car accidents")`
- Test edilecek alternatifler: tek pair değişikliği (örn "vehicle drives smoothly" vs "vehicle is reckless")
- Eğitim gerektirir (~6-12 saat per varyant)
- Bütçeye göre 2-3 varyant test edilebilir

**Deney 1: CLIP Capacity (en uzun)**
- ViT-bigG-14 (paper'ın seçtiği) vs ViT-H-14 vs ViT-L-14 vs ViT-B-32
- Her biri için 1M step training (~6-12 saat × 4 = 24-48 saat)
- Eğer compute izin verirse 2 varyant (B + bigG karşılaştırması)

### 7.3 Faz 6 İçin Hazırlanmış (Henüz Kullanılmamış) Dosyalar
- `train_one_seed.py`: --seed --resume destekli training worker
- `run_training.py`: 3-seed sequential orchestrator
- `train_smoke_test.py`: throughput ölçer

Bunlar hazır ama 3-seed eğitim yolundan vazgeçilince kullanılmadı. Faz 6 Deney 1/2 için baz olarak kullanılacak.

### 7.4 Önerilen Sıralama
1. **Deney 3 (weather)** önce — eğitim yok, hızlı katma değer
2. Sonra ders bütçesi/zaman müsaitliğine göre Deney 2 ve/veya Deney 1

---

## 8. Quick-Start Komutları (Yeni Sohbet İçin)

### 8.1 Conda Env
```powershell
conda activate carla_env
cd C:\Users\TAHA\Desktop\CARLA_0.9.16\PythonAPI\VLM-RL-Stress\repo
```

### 8.2 CARLA Server Başlat (Standard, eval için RenderOffScreen ile)
```powershell
:: Eskiyi öldür
taskkill /F /IM CarlaUE4-Win64-Shipping.exe /T
taskkill /F /IM CarlaUE4.exe /T
timeout /t 5

:: Headless başlat
start "CARLA Server" /D "C:\Users\TAHA\Desktop\CARLA_0.9.16" CarlaUE4.exe -quality_level=Low -benchmark -fps=15 -RenderOffScreen -prefernvidia -carla-world-port=2000

:: Bekle ve doğrula
timeout /t 25
python -c "import carla; c=carla.Client('localhost',2000); c.set_timeout(10); print('OK -', c.get_server_version())"
```

### 8.3 Video Kaydetmek İçin (RenderOffScreen YOK)
```powershell
:: -RenderOffScreen flag'i olmadan başlat (camera frame'leri için)
start "CARLA Server" /D "C:\Users\TAHA\Desktop\CARLA_0.9.16" CarlaUE4.exe -quality_level=Low -benchmark -fps=15 -prefernvidia -carla-world-port=2000
timeout /t 25
```

### 8.4 Eval Komutları
```powershell
:: Mevcut Faz 5 reproduction (gerekirse rerun)
python run_full_eval.py --towns Town01 Town02 Town03 Town04 Town05 --seeds 100 42 2024 --episodes 10 --max-steps 15000

:: Tek town video
python record_town_routes.py --town Town02

:: Sadece bir town subset koş
python run_full_eval.py --towns Town02 --seeds 100 --episodes 10 --max-steps 15000
```

---

## 9. Kritik Kavramsal Noktalar

### 9.1 "Inference Reproduction" vs "Independent Replication"
- **Inference reproduction** (yaptığımız): paper'ın eğittiği checkpoint'i kullan, kendi ortamında test et. **HF checkpoint = paper agent.**
- **Independent replication** (yapmadığımız): kendin sıfırdan eğit, kendi agent'ını test et. **3 ayrı training koşumu gerekir.**

İlki geçerli akademik iddia, ikincisi daha güçlü ama compute pahalı.

### 9.2 Paper'ın "3 Seed" Anlamı
- Paper "3 seed" = 3 ayrı training run = 3 farklı policy
- Bizim "3 seed" = 1 policy + 3 farklı inference seed = aynı policy farklı env stokastiğinde
- **Std değerleri farklı şeyleri ölçüyor** (policy varyans vs env varyans)

### 9.3 Cap Kararı
Paper'da explicit max_steps yok — episode env auto-terminate ile biter (route end / collision / 3000m TD). Bizim 15000 cap pratik tavan, **bu run'da kimse cap'e takılmadı** → cap bizim sonuçlarımızı bozmadı.

---

## 10. Bağlantılar

- **Paper:** https://arxiv.org/abs/2412.15544
- **GitHub:** https://github.com/zihaosheng/VLM-RL
- **HF Checkpoint:** https://huggingface.co/zihaosheng/VLM-RL (gated)
- **CARLA 0.9.16 docs:** https://carla.readthedocs.io/en/0.9.16/

---

## 11. Faz 5 Sonuç — Akademik Hikaye

Üç ana mesaj raporda yer almalı:

**Mesaj 1 — Training-town reproduction başarılı:**
"VLM-RL paper Tab 4'ün Town02 sonuçları (eğitim ortamı) Windows + RTX 5080 + CARLA 0.9.16 setup'ında inference-only reproduction protokolüyle paper toleransı içinde elde edildi (RC delta %4, SR delta %3)."

**Mesaj 2 — Generalization paper'dan zayıf:**
"Generalization town'larında (Town03, Town05) SR paper'dan %23-26 düşük. Bu sapma muhtemelen iki kaynaklı: (a) HF checkpoint paper'ın 3 trained agent'ından biri, en iyi olmayabilir, (b) CARLA 0.9.13 → 0.9.16 davranış farkları (özellikle traffic manager NPC dynamics)."

**Mesaj 3 — Town04'te paper'dan daha iyi performans:**
"Town04'te SR=0.90 (paper 0.70±0.28). Paper'ın Town04 std değeri çok yüksek (0.28), 3 trained seed arasında büyük varyansa işaret. Bizim ortamda HF checkpoint Town04'te tutarlı performans sergiliyor."

---

## 12. Son Söz

Faz 5 reproduction tamamen kapatıldı. **150 episode evaluation, paper-faithful aggregation, video kanıtı, dürüst akademik çerçeveleme** elde edildi. Yöntemsel sınırlamaları (single checkpoint, env-only varyans) raporda transparently belirtilecek.

Faz 6 üç deney önümüzde — Deney 3 (weather) ile başlayıp gerekirse Deney 2/1'e geçilecek. Mevcut altyapı (`run_full_eval.py`, `eval_one_town.py`) küçük genişletmelerle Faz 6'ya hazır.

**Sonraki sohbet açıldığında bu dokümanı yükleyip Faz 6 Deney 3 ile başlanacak.**
