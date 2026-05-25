"""Qwen2.5-7B-Instruct (4-bit QLoRA) + tokenizer + opsiyonel LoRA adapter yükleyici.

HW2 ile tutarlı 4-bit NF4 quantization kullanır.
A100 / L4 üzerinde varsa flash-attention 2'ye otomatik geçer; yoksa fallback.
"""
from __future__ import annotations
import logging
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import BASE_MODEL_NAME

log = logging.getLogger(__name__)


def get_bnb_config() -> BitsAndBytesConfig:
    """HW2 ile tutarlı 4-bit NF4 + bf16 compute + double quant."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_tokenizer(model_name: str = BASE_MODEL_NAME):
    """Qwen tokenizer; pad token yoksa eos_token'a fallback (training için kritik)."""
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def load_base_model(
    model_name: str = BASE_MODEL_NAME,
    for_training: bool = False,
    use_flash_attn: bool = True,
):
    """Base modeli 4-bit yükle.

    `for_training=True` ise k-bit eğitim için hazırlar ve KV cache'i kapatır.
    `use_flash_attn=True` ise flash-attention 2 dener, paket yoksa sessizce fallback.
    """
    bnb = get_bnb_config()
    kwargs = dict(
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if use_flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except (ImportError, ValueError, RuntimeError) as e:
        if "attn_implementation" in kwargs:
            log.warning("flash-attention 2 yüklenemedi (%s); SDPA ile devam.", e)
            kwargs.pop("attn_implementation")
            model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        else:
            raise

    if for_training:
        # Geç import — peft sadece training senaryosunda gerekli
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
        model.config.use_cache = False
    else:
        model.config.use_cache = True
    return model


def load_with_adapter(adapter_path: str, model_name: str = BASE_MODEL_NAME):
    """Base modeli yükle ve üzerine eğitilmiş LoRA adapter'ı bind et (inference için)."""
    from peft import PeftModel
    base = load_base_model(model_name=model_name, for_training=False)
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model
