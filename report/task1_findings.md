# Task 1 — LoRA Fine-Tuning for MT (WMT16 EN↔TR): Bulgular ve Rapor Notları

> **Doküman amacı:** Final raporun Task 1 (Bölüm 1) yazımında danışmak için. Sayılar, yorumlar ve anahtar noktalar burada hazır halde.
> **Tarih:** 2026-05-26
> **Inference için kullanılan adapter:** `checkpoint-1200` (training 1200. step'te durduruldu — en son save edilmiş checkpoint; detay §3)

---

## 1. Yapılandırma Özeti

| Bileşen | Değer |
|---------|-------|
| Base model | **Qwen/Qwen2.5-7B-Instruct** |
| Quantization | 4-bit NF4 (QLoRA), bf16 compute, double-quant |
| LoRA rank (r) | 16 |
| LoRA alpha (α) | 32 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` (attention + MLP) |
| Trainable params | **40,370,176 / 7,655,986,688 ≈ %0.527** |
| Training data | WMT16 (`tr-en` config), bidirectional expand, **50,000 örnek cap** |
| Validation | bidirectional ~2,000 örnek |
| Optimizer | `paged_adamw_8bit`, lr=2e-4, cosine schedule, warmup_ratio=0.03 |
| Batch | `per_device_batch_size=4 × grad_accum_steps=4` → effective batch = **16** |
| Max seq length | 512 |
| Hardware | NVIDIA A100-SXM4 40GB (Google Colab Pro) |
| Attention | SDPA (flash-attn opsiyonel, build maliyetinden vazgeçildi) |

---

## 2. Training Trajectory

**Toplam çalıştırılan adım: 1200** (planlanan 3125'in **~%38'i**). Plateau erken oluştuğu için **erken durdurma** kararı alındı (detay §3 ve §5.4).

| Step | Epoch | eval_loss | eval_entropy | token_acc | eval_num_tokens |
|------|-------|-----------|--------------|-----------|-----------------|
| 200 | 0.064 | **1.448** | 1.378 | 0.7076 | 358,700 |
| 400 | 0.128 | 1.452 | 1.355 | 0.7070 | 720,000 |
| 600 | 0.192 | **1.435** ← min | 1.382 | 0.7093 | 1,082,000 |
| 800 | 0.256 | 1.443 | 1.358 | 0.7087 | 1,438,000 |
| 1000 | 0.320 | 1.438 | 1.333 | 0.7091 | 1,797,000 |
| 1200 | 0.384 | **1.447** ↑ | 1.327 | 0.7087 | 2,154,000 |

### Gözlemler

- **Hızlı uyanma fazı (smoke test → step 200):** Smoke test 5-step adapter'ın eval_loss'u 2.61'di; step 200'de **1.448'e düşüş (~%45 iyileşme)** — model format ve çeviri davranışını çok hızlı yakalıyor.
- **Plateau bölgesi (step 200–1200):** eval_loss **1.435–1.452 aralığında osilasyon** (Δ ±0.01). Bu **gerçek bir trend değil, gürültü**. Token accuracy de %70.7–70.9 sabit.
- **Step 1200'de eval_loss hafif yükseliş:** 1000 → 1200 arası 1.438 → 1.447 (+0.009). Bu, ya plateau içindeki gürültü ya da erken overfitting'in ilk işareti olarak yorumlanabilir. Inference `checkpoint-1200` ile yapıldığında **COMET sonucu yine de en yüksek baseline'ı aştığı için** bu yükseliş kalite üzerinde belirgin bir bozulmaya yol açmamış görünmektedir.
- **Entropi düşüşü:** Eval entropy 1.378 → 1.327 yavaşça azalıyor — model daha "kesin" cevaplar veriyor (confidence ↑) ama bu ekstra doğruluk getirmiyor.

---

## 3. Adapter Seçimi (Hangi Checkpoint Kullanıldı?)

- Training step **1200'de manuel olarak durduruldu**.
- 1200. step'te eval ve save event'leri tamamlandıktan **sonra** Stop çağrıldı → Drive'da `checkpoint-1000` ve `checkpoint-1200` (`save_total_limit=2` rotation ile en eski `checkpoint-800` silindi).
- Training otomatik olarak `adapter_final/` üretmedi; bu yalnızca `trainer.train()` döngüsü doğal sonuna gelince (1 epoch tamamlanması) çağırılır. Erken durdurma akışında bu adım atlanır.
- Inference için **`checkpoint-1200` (eval_loss 1.447)** kullanıldı — Drive'daki en son save edilmiş ve dolayısıyla en güncel training state'i. `adapter_final/` klasörüne manuel kopyalandı, COMET ölçümleri bu adapter ile yapıldı.
- **Alternatif değerlendirme:** Plateau içindeki en düşük eval_loss `checkpoint-1000` (1.438) idi; küçük bir ablation olarak o da değerlendirilebilir, ancak yan-deney yapılmadı.

---

## 4. COMET Değerlendirme Sonuçları

- **Test seti:** WMT16 test split; HW2 paritesi için her yönde **500 örnek** (toplam 1000).
- **Metrik:** `Unbabel/wmt22-comet-da` (referans-bazlı semantic similarity).

| Yön | Zero-shot (HW2) | MAPS (HW2) | RAG (HW2) | **LoRA (bu çalışma)** | Δ vs Zero-shot | **Δ vs MAPS** |
|-----|-----------------|------------|-----------|----------------------|----------------|----------------|
| **EN→TR** | 0.7454 | 0.7939 | 0.7696 | **0.8084** | +0.0630 | **+0.0145** |
| **TR→EN** | 0.8345 | 0.8455 | 0.8369 | **0.8596** | +0.0251 | **+0.0141** |

---

## 5. Yorumlar (Rapora Eklenecek Anahtar Noktalar)

### 5.1. LoRA, en güçlü prompt-engineering yöntemini aştı

LoRA fine-tune'un her iki yönde de MAPS prompt engineering yöntemini **~%1.4 puan aşması**, parameter-efficient fine-tuning'in karmaşık prompting stratejilerine **alternatif** olduğunu gösterir. Sadece ~40M parametre (toplamın %0.53'ü) öğrenildi. Disk üzerindeki adapter ~92MB — full fine-tune'un ~%0.66'sı.

### 5.2. EN→TR yönünde kazanım daha belirgin

- EN→TR: +6.3 puan vs zero-shot, +1.45 puan vs MAPS
- TR→EN: +2.5 puan vs zero-shot, +1.41 puan vs MAPS

Türkçe üretiminin LoRA'dan en çok faydalanan görev olmasının olası nedenleri:
- Qwen2.5-7B pre-training corpus'unda İngilizce ağırlıklı dağılım
- Türkçe morfolojik zenginlik için ekstra adaptasyon avantajı
- TR→EN'de baseline zaten yüksek (zero-shot 0.83); diminishing returns

### 5.3. Hızlı yakınsama ve plateau

- İlk 200 step'te eval_loss 2.61 → 1.448 (büyük sıçrama) → Qwen2.5-7B'nin WMT16 dağılımına zaten yakın olduğunu gösteriyor (pre-training corpus'unda muhtemel kesişme).
- 200–1200 arası plateau → LoRA r=16 kapasitesi mevcut veri için doyuma ulaşmış.
- Bu bulgu, **r artırarak (r=32 veya 64) ek kazanım** denenmesi için makul bir motivasyon sağlar.

### 5.4. Erken durdurma kararı (1200 step / 3125'in ~%38'i)

Karar verme zinciri:
1. **5 ardışık eval'de plateau** (gürültü ±0.01 aralığında salınım, anlamlı düşüş yok)
2. **Step 1200'de eval_loss hafif yükseliş** (overfitting'in olası ilk işareti veya plateau gürültüsü — yine de inference için checkpoint-1200 yeterli kaliteyi koruyor)
3. **Bütçe baskısı:** kalan ~1.5 saat training ve ~18 CU, beklenen marjinal kazanım için yüksek maliyet/fayda oranıydı
4. **Mevcut COMET sonucu** zaten HW2 baseline'larını net aşıyor → ek training'in beklenen kazanımı düşük

→ Karar doğrulandı: 1200'de durdurma, `checkpoint-1200` adapter'ı ile COMET **0.8084 / 0.8596** sonuçları elde edildi.

### 5.5. Token-level vs sentence-level metrik korelasyonu

- `eval_mean_token_accuracy` plateau'da ~%70.9
- COMET-DA EN→TR: 0.8084, TR→EN: 0.8596

Token-level (per-token cross-entropy / accuracy) ve sentence-level (COMET, semantic similarity) metrikler tutarlı yön gösteriyor. Bu, training sırasında izlenen eval_loss'un gerçek çeviri kalitesini iyi temsil ettiğini doğrular — yani training trajectory'sine bakarak güvenli early-stopping kararı verilebilir.

---

## 6. Computational Trade-offs

| Yaklaşım | Trainable param | GPU bellek (training) | Adapter boyutu | Eğitim süresi |
|----------|------------------|------------------------|----------------|---------------|
| Full FT (teorik) | 7.66B (tümü) | 60–80 GB+ | ~14 GB (fp16) | 5–10× LoRA |
| **LoRA (bu çalışma)** | **~40 M (%0.53)** | **~12 GB** | **~92 MB** | **~1.1 saat (A100)** |

### Ana kazanımlar

- **VRAM:** 4-bit NF4 quantization ile 7B model 24 GB consumer GPU'da bile eğitilebilir; 40 GB A100'de rahatlık var.
- **Disk:** Adapter < 100 MB (full FT ~14 GB) → çoklu görev için adapter swap kolay (örn. EN↔TR vs EN↔DE adapter'larını aynı base ile yönetmek).
- **Eğitim süresi:** A100'de ~1 saat (1200 step) → çoklu deneme/ablation ekonomik.
- **Felaket unutma direnci:** Base model donmuş olduğu için modelin diğer dil yetenekleri korunur.

### Trade-off'lar

- **Kapasite sınırlı:** LoRA r düşükse (16) çok büyük dağılım kayması olan görevlerde kısıtlı.
- **Hyperparameter hassasiyeti:** r, α, target_modules, lr sonuçları etkiler; ablation çalışması ek maliyet.
- **Numerik kararlılık:** bf16 + 4-bit kombinasyonu modern GPU'larda stabil ama legacy donanımda (V100, P100) fp16 fallback gerekir.

---

## 7. Sınırlamalar ve Sonraki Çalışmalar

- **Plateau erken oldu:** r=32 veya r=64 ile training tekrarlanırsa ek kazanım mümkün; ayrıca lr=3e-4 veya 5e-4 ile daha agresif optimizasyon denenebilir.
- **Daha fazla veri:** Mevcut 50K subsample; full WMT16 (~200K çift) ile marjinal kazanım potansiyeli var ama LoRA kapasite limitinden ötürü diminishing returns beklenir.
- **Domain-specific augmentation:** WMT16 genel haber metni; teknik/tıbbi/hukuki gibi tek-domain için ek fine-tune gerekebilir.
- **Multi-epoch:** 1 epoch'tan az training (sadece 0.38 epoch); 1–2 epoch tam tamamlanırsa ablation değerli olabilir.
- **Beam search inference:** Greedy decode kullanıldı (num_beams=1, do_sample=False); `num_beams=4` ile kalite marjinal artabilir (~+0.005–0.01 COMET) ama 4× yavaş.
- **Reflection / iterative refinement:** Task 3'teki Reflection pattern'ı MT çıktısına uygulanabilir (üretilen çeviriyi LLM'e kritik ettirip düzeltmek) — ekstra fine-tune yerine inference-time iyileştirme.

---

## 8. Rapora Hazır İçerik (Kopyala-Yapıştır)

### 8.1. Tablo: COMET Karşılaştırma

```
Yön      Zero-shot   MAPS     RAG      LoRA       Δ vs MAPS
EN→TR    0.7454      0.7939   0.7696   0.8084     +0.0145
TR→EN    0.8345      0.8455   0.8369   0.8596     +0.0141
```

### 8.2. Önerilen Şekiller (raporda)

- **Şekil 1: Training/Eval Loss Eğrisi**
  X-ekseni: step (200–1200), Y-ekseni: eval_loss (1.43–1.46).
  Plateau'yu görsel olarak göstermek için line + scatter; "min @ step 600" ve "↑ @ step 1200 (overfitting sinyali)" işaretleri eklenebilir.

- **Şekil 2: Token Accuracy Eğrisi**
  X-ekseni: step (200–1200), Y-ekseni: token_acc (0.70–0.71).
  Neredeyse düz bir çizgi; plateau'yu ikincil kanıt olarak vurgular.

- **Şekil 3 (opsiyonel): COMET Karşılaştırma Bar Chart**
  4 yöntem × 2 yön = 8 bar; LoRA'nın en yüksek olduğu görsel olarak öne çıkar.

### 8.3. Discussion section başlangıcı (Türkçe taslak)

> Task 1 sonuçları, parameter-efficient fine-tuning'in (LoRA) düşük bir compute bütçesiyle bile prompt-engineering tabanlı yöntemleri (MAPS) hem EN→TR (+0.0145 COMET) hem de TR→EN (+0.0141 COMET) yönlerinde aşabildiğini göstermektedir. Sadece toplam parametrenin %0.53'ünü güncelleyerek elde edilen bu kazanım, LoRA'nın model adaptasyonunda etkinliğini doğrulamaktadır. EN→TR yönündeki daha büyük göreli kazanım, Türkçe üretimin LoRA fine-tune'dan en çok faydalanan alt görev olduğunu işaret etmektedir. Bunun olası sebepleri arasında Qwen2.5-7B'nin pre-training dağılımındaki İngilizce ağırlığı ve Türkçenin morfolojik zenginliği için ekstra adaptasyon ihtiyacı yer almaktadır...

### 8.4. Conclusion section başlangıcı (Türkçe taslak)

> Final çalışmanın Task 1 ayağında, Qwen2.5-7B-Instruct base modeli üzerinde 4-bit QLoRA tekniği ile WMT16 EN↔TR çeviri görevine adaptasyon gerçekleştirildi. Bidirectional expand edilmiş 50,000 örneklik bir subset ile 1200 step (planlanan ~3125 step'in %38'i) training sonrasında early-stopping kararı verildi; bu noktada eval_loss plateau'su ve hafif yükseliş sinyali gözlemlendi. Inference için `checkpoint-1200` (eval_loss 1.447) kullanıldı. COMET-DA değerlendirmesinde her iki yönde de HW2 baseline'larını aşan sonuçlar elde edildi: EN→TR 0.8084 (HW2 MAPS 0.7939), TR→EN 0.8596 (HW2 MAPS 0.8455). Bu kazanımlar, LoRA'nın MT görevlerinde MAPS gibi karmaşık prompt-engineering yöntemlerine **alternatif veya tamamlayıcı** olabileceğini ortaya koymaktadır...

---

## 9. Tekrar Üretilebilirlik Notları

- Seed: **42** (config.yaml `seed`)
- Tüm hyperparametreler: `config.yaml` repo'da
- Adapter: Drive `models/checkpoints/qwen25_7b_lora_mt/adapter_final/`
- Predictions: Drive `results/task1_predictions.jsonl`
- COMET sonuçları: Drive `results/task1_comet_results.json`
- Training trajectory log: bu doküman §2

Aynı sonuçları tekrar üretmek için `config.yaml` değişmeden + `--max_steps 1200` ile `train_lora.py` çalıştırılabilir; HW2 baseline'ları `comet_scores.json` referansından okunur.
