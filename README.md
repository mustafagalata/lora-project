# LLM Final Project — Fine-Tuning, RAG, Agent Design

Üç görevli final projesi:

1. **LoRA Fine-Tuning** (Qwen2.5-7B-Instruct, 4-bit QLoRA) — WMT16 EN↔TR Machine Translation
2. **RAG-based QA** — TurkishMMLU History (FAISS + `intfloat/multilingual-e5-base`)
3. **Agent System Design** — çok dilli QA (Wikipedia + RAG); yalnızca tasarım dokümanı


## Hızlı başlangıç (Colab)

```python
!git clone https://github.com/<USER>/<REPO>.git /content/repo
%cd /content/repo
!pip install -r requirements.txt
```

Ardından şu notebook'lardan birini açıp sırasıyla hücreleri çalıştırın:

| Notebook | Amaç | Önerilen GPU |
|----------|------|--------------|
| `notebooks/colab_task1_lora_mt.ipynb` | Task 1: data prep + QLoRA train + inference + COMET | A100 40GB |
| `notebooks/colab_task2_rag.ipynb` | Task 2: MMLU prep + FAISS ingest + zero-shot + RAG eval | T4 yeter |

## Proje yapısı

```
src/
├── common/             # config, model loader, prompts, utilities
├── task1_lora_mt/      # WMT16 prep + QLoRA training + COMET eval
├── task2_rag/          # TurkishMMLU + FAISS ingest + RAG eval
└── task3_agent_design/ # tasarım dokümanı + prompt taslakları (yalnız tasarım)
notebooks/              # Colab orchestrator notebook'ları
data/                   # WMT16, TurkishMMLU, history books (gitignored)
models/                 # LoRA adapter çıktıları (gitignored)
results/                # Skorlar ve tahminler (JSON commit'lenir, JSONL gitignored)
```

## Donanım

- **Önerilen:** A100 40GB (Colab Pro / Pro+) → Task 1 training ~1.5-2 saat (~25 CU)
- **Minimum:** T4 16GB → Task 1 training ~6-10 saat; Task 2 için zaten yeterli

## Çıktılar

- `results/task1_comet_results.json` — LoRA fine-tuned COMET skorları (HW2 baseline ile kıyas)
- `results/task2_zero_shot_accuracy.json`, `results/task2_rag_accuracy.json`
- `models/checkpoints/qwen25_7b_lora_mt/adapter_final/` — final LoRA adapter (~150MB)
