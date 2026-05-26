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


def _normalize(example: dict, subject_filter: bool = False) -> dict | None:
    """TurkishMMLU şemasını projeye özgü düz JSON'a indirger.

    Beklenen çıkış: {subject, question, choices: {A..E}, answer: "A"|...}

    `subject_filter=False`: config zaten subject'e göre yüklendiyse (örn.
    `--config History`) subject filtresi atlanır. True ise HISTORY_KEYS kontrolü.
    Choices hem ayrı A..E field'ı hem de tek `choices` listesi olarak gelebilir.
    Answer hem harf ("A") hem index (0/1) olabilir.
    """
    keys = {k.lower(): k for k in example}

    def pick(*candidates: str):
        for c in candidates:
            if c in keys:
                return example[keys[c]]
        return None

    question = pick("question", "soru")
    answer = pick("answer", "answer_key", "answer key", "correct", "label", "cevap")
    if question is None or answer is None:
        return None

    subject = pick("subject", "section", "konu")
    if subject_filter and subject is not None:
        if str(subject).strip().lower() not in HISTORY_KEYS:
            return None

    # Choices: ya tek liste (choices/options) ya ayrı A..E field'ları
    choices: dict[str, str] = {}
    raw = pick("choices", "options", "secenekler", "şıklar")
    if isinstance(raw, (list, tuple)):
        for i, v in enumerate(raw[:5]):
            if v is not None and str(v).strip():
                choices["ABCDE"[i]] = str(v).strip()
    else:
        for letter in "ABCDE":
            v = pick(
                f"choice {letter.lower()}", f"choice_{letter.lower()}",
                f"option_{letter.lower()}", f"choice{letter}", letter,
            )
            if v is not None and str(v).strip():
                choices[letter] = str(v).strip()
    if len(choices) < 2:
        return None

    # Answer: harf ("A") veya index (0-tabanlı ya da 1-tabanlı) olabilir
    answer_str = str(answer).strip().upper()
    answer_letter = None
    if answer_str.isdigit():
        idx = int(answer_str)
        if 0 <= idx < len(choices):           # 0-tabanlı
            answer_letter = "ABCDE"[idx]
        elif 1 <= idx <= len(choices):         # 1-tabanlı
            answer_letter = "ABCDE"[idx - 1]
    else:
        for ch in answer_str:                  # "A" veya "A) Mustafa Kemal"
            if ch in "ABCDE":
                answer_letter = ch
                break
    if answer_letter is None:
        return None

    return {
        "subject": str(subject).strip() if subject else "History",
        "question": str(question).strip(),
        "choices": choices,
        "answer": answer_letter,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="AYueksel/TurkishMMLU")
    ap.add_argument("--config", default="History",
                    help="TurkishMMLU subject config (History, Geography, All, ...)")
    ap.add_argument("--split", default="test", help="train / dev / test")
    ap.add_argument("--out", type=Path,
                    default=DATA_DIR / "turkish_mmlu_history" / "test.jsonl")
    args = ap.parse_args()

    setup_logging()
    log.info("Loading %s [config=%s, split=%s]...", args.dataset, args.config, args.split)
    ds = load_dataset(args.dataset, args.config, split=args.split)
    log.info("Şema: %s", list(ds.features.keys()))

    # config 'All' ise satır bazında History filtrele; subject-config ise filtre gereksiz
    subject_filter = args.config.lower() in ("all", "")
    total = len(ds)
    out: list[dict] = []
    for ex in ds:
        norm = _normalize(ex, subject_filter=subject_filter)
        if norm is not None:
            out.append(norm)

    log.info("History örnekleri: %d / %d", len(out), total)
    if not out:
        log.warning("HİÇBİR örnek normalize edilemedi! Şema: %s, ilk satır: %s",
                    list(ds.features.keys()), ds[0] if total else "boş")
    write_jsonl(out, args.out)
    log.info("Yazıldı: %s", args.out)


if __name__ == "__main__":
    main()
