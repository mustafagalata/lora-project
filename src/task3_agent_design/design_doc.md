# Task 3 — Çok Dilli Agent QA Sistem Tasarımı

> Bu doküman **tasarım** odaklıdır. Ödev gereği implementasyon zorunlu değildir; içerikler doğrudan final rapora taşınacak şekilde kaleme alınmıştır.

---

## 1. Hedef davranış

Sistem, **Türkçe** ve **İngilizce** tarih soruları alır ve dile göre farklı bilgi kaynağı kullanır:

| Soru dili | Bilgi kaynağı | Retrieval |
|-----------|---------------|-----------|
| İngilizce | Wikipedia (EN) | `WikipediaQueryRun` veya REST API |
| Türkçe    | Task 2 RAG KB | FAISS + multilingual-e5-base (top_k = 5) |

Cevap dili her zaman **sorunun dili** ile aynıdır.

---

## 2. Mimari — Bileşenler ve roller

| Bileşen | Rol |
|---------|-----|
| **Orchestrator (Agent loop)** | ReAct döngüsünü sürer: Thought → Action → Observation → ... → finish. Pratikte LangChain `AgentExecutor` veya kısa, kendi yazdığımız parser olabilir. |
| **LLM (Qwen2.5-7B-Instruct, 4-bit)** | Reasoning + generation. Dil tespiti fallback'i, ReAct adımları, cevap sentezi, reflection critique. |
| **Language Detector** | Birincil: `langdetect` (Python kütüphanesi, kısa text'lerde %95+ doğruluk). Fallback: LLM'e küçük tek-token prompt (kod-switch durumları için). |
| **Tool: turkish_rag_search(query)** | Task 2'deki `load_retriever()` üzerine ince bir wrapper. e5 `query:` prefix'i ekler, top_k = 5 chunk döndürür. |
| **Tool: wikipedia_search(query)** | `wikipedia-api` veya `langchain.tools.WikipediaQueryRun`. Top 1–2 makalenin özet/intro bölümünü string olarak döndürür. |
| **Embedding model (multilingual-e5-base)** | Task 2 KB inşası ve query embedding'i. Multilingual avantajı + HW2 ile tutarlı. |
| **Vector DB (FAISS)** | `index.faiss` + `index.pkl` snapshot; in-memory similarity search. |
| **Prompt templates** | `prompts/` altında dosya başına ayrı tutuldu (versiyonlama kolay). |
| **Reflection module (ops.)** | Üretilen cevabı kaynaklara karşı doğrular; gerekirse 1 ek retrieve + regenerate döngüsü tetikler. |

---

## 3. Workflow

```
┌─────────────────────┐
│  Kullanıcı sorusu   │
└──────────┬──────────┘
           ▼
   [1] Dil tespiti  ────►  TR | EN
           │
           ▼
   [2] Tool seçimi  ──► turkish_rag_search()      (TR)
                    └─► wikipedia_search()        (EN)
           │
           ▼
   [3] ReAct döngüsü
       Thought_i → Action_i → Observation_i
       (max 3 retrieval round; aynı tool 2+ kez de kullanılabilir, query refine ile)
           │
           ▼
   [4] Cevap üretimi (RAG/Wiki prompt)
       — context retrieved chunks ile birlikte
       — temperature = 0.0–0.3 (deterministik QA)
           │
           ▼
   [5] Reflection (opsiyonel)
       — "Cevap kaynaklara tutarlı mı? Eksik bilgi var mı?"
       — OK → finish
       — REVISE → 1 daha retrieve + regenerate
           │
           ▼
      Final cevap
```

### 3.1. Dil tespiti

- Birincil: `from langdetect import detect; lang = detect(question)`. Çıktı ISO kodu (`'tr'`, `'en'`).
- Fallback (kısa cümle / kod-switch ambiguity): LLM'e `prompts/language_detection.txt` ile sor. Bir token cevap bekle (`TR` / `EN`).
- Karar threshold'u: langdetect'in `detect_langs()` confidence < 0.9 ise LLM fallback'e geç.

### 3.2. Tool seçimi (deterministik kural)

```python
if lang == "tr":
    tool = turkish_rag_search
else:
    tool = wikipedia_search
```

ReAct prompt'unda LLM'e bu kural söylenir, ama agent yine de "Action" satırında ilgili tool'u çağırır — kural-tabanlı kısıt LLM hatalarını absorbe eder.

### 3.3. Retrieval süreci

- **Türkçe → RAG KB:** `e5_query(question)` → FAISS top_k = 5 chunk. Chunklar `[1] ...`, `[2] ...` formatında prompt'a inject edilir.
- **İngilizce → Wikipedia:** En alakalı 1-2 makale; intro veya tek bölüm (~1500 char). Çok uzun olursa truncate.
- Multi-hop sorular: ReAct, ikinci Action ile refined query ile tekrar retrieve eder (örn: "İstanbul'un fethi yılı" sonra "Fatih Sultan Mehmet kim").

### 3.4. Cevap üretimi

- Retrieved context prompt'a `{context}` placeholder'ına string olarak konur.
- Generation parametreleri: `do_sample=False`, `num_beams=1`, `max_new_tokens` (Wikipedia için ~400; RAG çoktan seçmeli için 8). Cevap dili soru ile eşleşir (system prompt'ta belirtildi).

### 3.5. Reflection (opsiyonel ama önerilen)

- `prompts/reflection.txt`: kanıt-cevap tutarlılığını ve eksiklik durumunu kontrol eder.
- Çıktı `OK` ise cevap kesinleşir. `REVISE: <reason>` ise agent bir tur daha retrieve + regenerate yapar (max 1 düzeltme, çünkü maliyet 2-3× yükselir).

---

## 4. Agent etkileşim tasarımı

- **Tool interface:** Her tool `(input: str) -> str` imzalı; observation string olarak ReAct trace'e eklenir.
- **Context injection:** Retrieved chunklar `[i] ...` formatında birleştirilir; prompt template'in `{context}` placeholder'ına geçilir.
- **Reasoning + Acting koordinasyonu:** ReAct zincir çıktısı regex parser ile satır-satır ayrıştırılır. `Action: <tool_name>(<arg>)` satırı yakalanır, tool çağrılır, dönen string `Observation: ...` olarak prompt'a eklenir, döngü devam eder.
- **Sonlandırma:** Action `finish(<answer>)` ise döngü biter; veya max 3 retrieval round sınırı dolar (sonsuz döngü guard).

---

## 5. Critical prompts — özet

Detaylı taslaklar `prompts/` klasöründe; rapora ekleneceklerse buradan kopyalanmalı.

| Dosya | Amaç |
|-------|------|
| `prompts/language_detection.txt` | Tek-token TR/EN cevap |
| `prompts/react_system.txt` | ReAct ana sistem talimatı, tool listesi, kurallar |
| `prompts/rag_qa.txt` | TR cevap üretimi (RAG context inject) |
| `prompts/wikipedia_qa.txt` | EN cevap üretimi (Wikipedia context inject) |
| `prompts/reflection.txt` | Cevabı kaynaklara karşı doğrulama |

---

## 6. Avantajlar, sınırlamalar, alternatifler

### Reflection
- **Avantaj:** Hallüsinasyon ve eksik cevap riskini düşürür; cevabın kaynak-bağlılığını artırır.
- **Sınırlama:** Maliyet 2-4×; "self-correction failure" — model kendi hatasını tanıyamayabilir (özellikle 7B sınıfı modellerde).
- **Tipik kullanım:** Karmaşık reasoning, hassas QA, faktoid kontrolü.

### ReAct
- **Avantaj:** Şeffaf reasoning trace (debug edilebilir); tool gerektiren görevlerde güçlü; faktüel doğruluk artar.
- **Sınırlama:** Adım sayısı arttıkça maliyet doğrusal artar; "loop" / "indecisive" davranış riski; tool API hataları kırılgan.
- **Tipik kullanım:** Multi-hop QA, web araması, multi-step planning.

### Alternatif tasarımlar
- **Plan-and-Execute:** Önce tüm planı yap, sonra adımları yürüt. Daha hızlı ama hata adaptasyonu zayıf.
- **Self-Consistency:** Aynı soruyu N kez farklı temperature ile cevapla, majority vote. Quality+, cost ×N.
- **Toolformer-style:** Modele tool çağrısı doğrudan öğretilir (fine-tune). Bu projede out of scope.
