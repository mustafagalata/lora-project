"""WMT16 EN-TR veri seti hazırlığı.

- HuggingFace `wmt16` (`tr-en` config) train/validation/test'i yükle
- Boş, çok kısa, çok uzun çiftleri filtrele
- Her çifti hem EN->TR hem TR->EN örneğine açarak bidireksiyonel eğitim sağla
- Train'i `--train_samples` ile subsample et (default 50K)
- JSONL olarak yaz: {src_lang, tgt_lang, src, tgt}
"""
from __future__ import annotations
import argparse
import logging
import random
from pathlib import Path

from src.common.utils import ensure_repo_on_syspath  # noqa: E402

ensure_repo_on_syspath()

from datasets import load_dataset  # noqa: E402

from src.common.config import DATA_DIR, SEED, WMT16_CONFIG, WMT16_TRAIN_SAMPLES  # noqa: E402
from src.common.utils import set_seed, setup_logging, write_jsonl  # noqa: E402

log = logging.getLogger(__name__)

MIN_CHARS = 5
MAX_CHARS = 2000


def build_examples(split, n_max: int | None, both_directions: bool = True) -> list[dict]:
    out: list[dict] = []
    for ex in split:
        en = (ex["translation"]["en"] or "").strip()
        tr = (ex["translation"]["tr"] or "").strip()
        if not en or not tr:
            continue
        if min(len(en), len(tr)) < MIN_CHARS or max(len(en), len(tr)) > MAX_CHARS:
            continue
        out.append({"src_lang": "English", "tgt_lang": "Turkish", "src": en, "tgt": tr})
        if both_directions:
            out.append({"src_lang": "Turkish", "tgt_lang": "English", "src": tr, "tgt": en})
    if n_max is not None and len(out) > n_max:
        random.shuffle(out)
        out = out[:n_max]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_samples", type=int, default=WMT16_TRAIN_SAMPLES,
                    help="Train subsample boyutu (-1 için tüm dataset).")
    ap.add_argument("--out_dir", type=Path, default=DATA_DIR / "wmt16_en_tr")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    setup_logging()
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cap = None if args.train_samples is not None and args.train_samples < 0 else args.train_samples

    log.info("WMT16 (%s) yükleniyor...", WMT16_CONFIG)
    ds = load_dataset("wmt16", WMT16_CONFIG)

    log.info("Train hazırlanıyor (cap=%s)...", cap)
    train_ex = build_examples(ds["train"], cap, both_directions=True)
    val_ex = build_examples(ds["validation"], None, both_directions=True)
    test_ex = build_examples(ds["test"], None, both_directions=True)

    write_jsonl(train_ex, args.out_dir / "train.jsonl")
    write_jsonl(val_ex, args.out_dir / "validation.jsonl")
    write_jsonl(test_ex, args.out_dir / "test.jsonl")

    log.info("Yazıldı: train=%d, val=%d, test=%d -> %s",
             len(train_ex), len(val_ex), len(test_ex), args.out_dir)


if __name__ == "__main__":
    main()
