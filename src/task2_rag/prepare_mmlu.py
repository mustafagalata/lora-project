"""TurkishMMLU dataset'inden History (Tarih) soru-cevaplarını filtrele ve JSONL'e yaz.

Dataset şeması versiyon-bağımlı olabildiği için (Question/Soru, Section/Subject,
Choice A.. veya choice_a..) `_normalize` esnek lookup yapar.
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path

from src.common.utils import ensure_repo_on_syspath  # noqa: E402

ensure_repo_on_syspath()

from datasets import load_dataset  # noqa: E402

from src.common.config import DATA_DIR  # noqa: E402
from src.common.utils import setup_logging, write_jsonl  # noqa: E402

log = logging.getLogger(__name__)

HISTORY_KEYS = {"history", "tarih"}


def _normalize(example: dict) -> dict | None:
    """TurkishMMLU şemasını projeye özgü düz JSON'a indirger.

    Beklenen çıkış: {subject, question, choices: {A..E}, answer: "A"|...}
    """
    keys = {k.lower(): k for k in example}

    def pick(*candidates: str):
        for c in candidates:
            if c in keys:
                return example[keys[c]]
        return None

    subject = pick("subject", "section", "konu")
    question = pick("question", "soru")
    answer = pick("answer", "answer_key", "answer key", "correct", "label", "cevap")
    if subject is None or question is None or answer is None:
        return None
    if str(subject).strip().lower() not in HISTORY_KEYS:
        return None

    choices: dict[str, str] = {}
    for letter in "ABCDE":
        v = pick(
            f"choice {letter.lower()}",
            f"choice_{letter.lower()}",
            f"option_{letter.lower()}",
            f"choice{letter}",
            letter,
        )
        if v is not None and str(v).strip():
            choices[letter] = str(v).strip()
    if len(choices) < 2:
        return None

    answer_letter = str(answer).strip().upper()
    # "A) Mustafa Kemal" gibi olabilir; ilk harfi al
    for ch in answer_letter:
        if ch in "ABCDE":
            answer_letter = ch
            break
    else:
        return None

    return {
        "subject": str(subject).strip(),
        "question": str(question).strip(),
        "choices": choices,
        "answer": answer_letter,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="AYueksel/TurkishMMLU")
    ap.add_argument("--split", default="test",
                    help="HF split adı; bazı sürümler 'test' diğer ad kullanır.")
    ap.add_argument("--out", type=Path,
                    default=DATA_DIR / "turkish_mmlu_history" / "test.jsonl")
    args = ap.parse_args()

    setup_logging()
    log.info("Loading %s [%s]...", args.dataset, args.split)
    ds = load_dataset(args.dataset, split=args.split)

    total = len(ds)
    out: list[dict] = []
    for ex in ds:
        norm = _normalize(ex)
        if norm is not None:
            out.append(norm)

    log.info("History örnekleri: %d / %d", len(out), total)
    if not out:
        log.warning("HİÇBİR History örneği bulunamadı! Şemayı kontrol et: %s",
                    list(ds.features.keys()))
    write_jsonl(out, args.out)
    log.info("Yazıldı: %s", args.out)


if __name__ == "__main__":
    main()
