"""LoRA adapter yüklü Qwen2.5-7B ile WMT16 test set üzerinde batched generation.

`--no_adapter` ile base model üzerinde de aynı script kullanılabilir
(adil karşılaştırma için ileride istenirse).
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path

from src.common.utils import ensure_repo_on_syspath  # noqa: E402

ensure_repo_on_syspath()

import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.common.config import CHECKPOINT_DIR, DATA_DIR, GEN_MAX_NEW_TOKENS, RESULTS_DIR  # noqa: E402
from src.common.model_loader import load_base_model, load_tokenizer, load_with_adapter  # noqa: E402
from src.common.prompts import mt_messages  # noqa: E402
from src.common.utils import batched, read_jsonl, setup_logging, write_jsonl  # noqa: E402

log = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path,
                    default=CHECKPOINT_DIR / "qwen25_7b_lora_mt" / "adapter_final")
    ap.add_argument("--test_file", type=Path, default=DATA_DIR / "wmt16_en_tr" / "test.jsonl")
    ap.add_argument("--out_file", type=Path, default=RESULTS_DIR / "task1_predictions.jsonl")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=GEN_MAX_NEW_TOKENS)
    ap.add_argument("--no_adapter", action="store_true", help="Base model only")
    ap.add_argument("--limit", type=int, default=-1, help="Test set kapağı (-1 = full)")
    args = ap.parse_args()

    setup_logging()
    tokenizer = load_tokenizer()
    # Decoder-only generate için sol padding kritik (sağ padding yeni token üretimini bozar)
    tokenizer.padding_side = "left"

    if args.no_adapter:
        log.info("Base model yükleniyor (adapter yok)...")
        model = load_base_model(for_training=False)
    else:
        log.info("Base model + LoRA adapter yükleniyor: %s", args.adapter)
        model = load_with_adapter(str(args.adapter))

    test = list(read_jsonl(args.test_file))
    if args.limit > 0:
        test = test[:args.limit]
    log.info("Test örnek sayısı: %d", len(test))

    out: list[dict] = []
    for batch in tqdm(list(batched(test, args.batch_size)), desc="generate"):
        prompts = [
            tokenizer.apply_chat_template(
                mt_messages(ex["src"], ex["src_lang"], ex["tgt_lang"]),
                tokenize=False,
                add_generation_prompt=True,
            )
            for ex in batch
        ]
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024,
        ).to(model.device)
        with torch.inference_mode():
            outs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
            )
        in_len = inputs["input_ids"].shape[1]
        for ex, gen_ids in zip(batch, outs):
            hyp = tokenizer.decode(gen_ids[in_len:], skip_special_tokens=True).strip()
            out.append({**ex, "hypothesis": hyp})

    write_jsonl(out, args.out_file)
    log.info("Yazıldı: %s (%d satır)", args.out_file, len(out))

    # --- Örnek çeviri önizlemesi ---
    print("\n" + "=" * 64)
    print(f"ÖRNEK ÇEVİRİLER (ilk 3 / toplam {len(out)})")
    print("=" * 64)
    for i, ex in enumerate(out[:3], 1):
        print(f"[{i}] {ex['src_lang']} -> {ex['tgt_lang']}")
        print(f"  SRC: {ex['src'][:140]}")
        print(f"  REF: {ex['tgt'][:140]}")
        print(f"  HYP: {ex['hypothesis'][:140]}")
        print()
    print("=" * 64)


if __name__ == "__main__":
    main()
