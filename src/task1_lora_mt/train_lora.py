"""Qwen2.5-7B-Instruct üzerinde 4-bit QLoRA ile WMT16 EN<->TR fine-tuning.

ChatML formatına `apply_chat_template` ile çevirir, SFTTrainer ile eğitir.
Checkpoint Drive'a yazılır (DRIVE_ROOT set ise); resume desteklidir.
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path

from src.common.utils import ensure_repo_on_syspath  # noqa: E402

ensure_repo_on_syspath()

from datasets import Dataset  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

from src.common.config import (  # noqa: E402
    CHECKPOINT_DIR, DATA_DIR, GRADIENT_ACCUMULATION_STEPS, LEARNING_RATE,
    LOGGING_STEPS, LORA_ALPHA, LORA_DROPOUT, LORA_R, LORA_TARGET_MODULES,
    MAX_SEQ_LENGTH, NUM_TRAIN_EPOCHS, PER_DEVICE_TRAIN_BATCH_SIZE,
    SAVE_STEPS, SAVE_TOTAL_LIMIT, SEED, WARMUP_RATIO,
)
from src.common.model_loader import load_base_model, load_tokenizer  # noqa: E402
from src.common.prompts import mt_messages  # noqa: E402
from src.common.utils import read_jsonl, set_seed, setup_logging  # noqa: E402

log = logging.getLogger(__name__)


def build_hf_dataset(jsonl_path: Path, tokenizer) -> Dataset:
    """JSONL'i Qwen ChatML formatına çevirip text alanlı HF Dataset üretir."""
    rows = []
    for ex in read_jsonl(jsonl_path):
        msgs = mt_messages(ex["src"], ex["src_lang"], ex["tgt_lang"], ex["tgt"])
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        rows.append({"text": text})
    return Dataset.from_list(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, default=DATA_DIR / "wmt16_en_tr")
    ap.add_argument("--output_dir", type=Path, default=CHECKPOINT_DIR / "qwen25_7b_lora_mt")
    ap.add_argument("--resume", action="store_true", help="En son checkpoint'ten devam")
    ap.add_argument("--batch_size", type=int, default=PER_DEVICE_TRAIN_BATCH_SIZE)
    ap.add_argument("--grad_accum", type=int, default=GRADIENT_ACCUMULATION_STEPS)
    ap.add_argument("--epochs", type=float, default=NUM_TRAIN_EPOCHS)
    ap.add_argument("--lr", type=float, default=LEARNING_RATE)
    ap.add_argument("--max_steps", type=int, default=-1, help="-1 = epoch'lara göre")
    ap.add_argument("--save_steps", type=int, default=SAVE_STEPS)
    ap.add_argument("--max_seq_length", type=int, default=MAX_SEQ_LENGTH)
    args = ap.parse_args()

    setup_logging()
    set_seed(SEED)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Tokenizer ve base model yükleniyor (4-bit QLoRA)...")
    tokenizer = load_tokenizer()
    model = load_base_model(for_training=True)

    log.info("LoRA adapter ekleniyor (r=%d, alpha=%d)...", LORA_R, LORA_ALPHA)
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    log.info("Dataset hazırlanıyor...")
    train_ds = build_hf_dataset(args.data_dir / "train.jsonl", tokenizer)
    val_ds = build_hf_dataset(args.data_dir / "validation.jsonl", tokenizer)
    log.info("Train=%d, Val=%d", len(train_ds), len(val_ds))

    cfg = SFTConfig(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=SAVE_TOTAL_LIMIT,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        bf16=True,
        optim="paged_adamw_8bit",
        max_grad_norm=1.0,
        max_seq_length=args.max_seq_length,
        packing=False,
        gradient_checkpointing=True,
        report_to="none",
        dataset_text_field="text",
        seed=SEED,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=cfg,
        tokenizer=tokenizer,
    )

    log.info("Eğitim başlıyor (resume=%s)...", args.resume)
    trainer.train(resume_from_checkpoint=args.resume or None)

    final_path = args.output_dir / "adapter_final"
    trainer.model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    log.info("Final adapter kaydedildi: %s", final_path)


if __name__ == "__main__":
    main()
