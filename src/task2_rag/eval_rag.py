"""RAG-enhanced Qwen2.5-7B üzerinde TurkishMMLU History accuracy.

Her soru için FAISS retriever top_k chunk çeker, prompt'a inject eder, harf üretir.
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path

from src.common.utils import ensure_repo_on_syspath  # noqa: E402

ensure_repo_on_syspath()

from tqdm import tqdm  # noqa: E402

from src.common.config import DATA_DIR, FAISS_INDEX_DIR, RESULTS_DIR, TOP_K  # noqa: E402
from src.common.model_loader import load_base_model, load_tokenizer  # noqa: E402
from src.common.utils import read_jsonl, setup_logging, write_json, write_jsonl  # noqa: E402
from src.task2_rag.rag_qa import (  # noqa: E402
    generate_answer, load_retriever, parse_letter, retrieve_context,
)

log = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa_file", type=Path,
                    default=DATA_DIR / "turkish_mmlu_history" / "test.jsonl")
    ap.add_argument("--index_dir", type=Path, default=FAISS_INDEX_DIR)
    ap.add_argument("--top_k", type=int, default=TOP_K)
    ap.add_argument("--out_jsonl", type=Path, default=RESULTS_DIR / "task2_rag.jsonl")
    ap.add_argument("--out_summary", type=Path,
                    default=RESULTS_DIR / "task2_rag_accuracy.json")
    args = ap.parse_args()

    setup_logging()
    tokenizer = load_tokenizer()
    model = load_base_model(for_training=False)
    model.eval()
    retriever = load_retriever(index_dir=args.index_dir, top_k=args.top_k)

    qa = list(read_jsonl(args.qa_file))
    log.info("QA örnekleri: %d (top_k=%d)", len(qa), args.top_k)

    out: list[dict] = []
    correct = 0
    for ex in tqdm(qa, desc="rag"):
        ctx = retrieve_context(retriever, ex["question"])
        raw = generate_answer(model, tokenizer, ex["question"], ex["choices"], context=ctx)
        pred = parse_letter(raw)
        ok = pred == ex["answer"]
        correct += int(ok)
        out.append({**ex, "context": ctx, "prediction": pred, "raw": raw, "correct": ok})

    write_jsonl(out, args.out_jsonl)
    acc = correct / len(qa) if qa else 0.0
    summary = {
        "mode": "rag",
        "n": len(qa),
        "correct": correct,
        "accuracy": acc,
        "top_k": args.top_k,
    }
    write_json(summary, args.out_summary)
    log.info("Accuracy: %.4f (%d/%d)", acc, correct, len(qa))


if __name__ == "__main__":
    main()
