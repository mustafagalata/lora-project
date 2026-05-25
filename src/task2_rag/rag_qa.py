"""Task 2 RAG için ortak fonksiyonlar: retriever yükleme, bağlam çekme, generation, parse.

`eval_zero_shot.py` ve `eval_rag.py` buraya bağımlıdır.
"""
from __future__ import annotations
import logging
import re
from pathlib import Path
from typing import Optional

import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from src.common.config import (
    EMBEDDING_MODEL_NAME, FAISS_INDEX_DIR, FAISS_INDEX_NAME, TOP_K,
)
from src.common.prompts import e5_query, rag_qa_messages, zero_shot_qa_messages

log = logging.getLogger(__name__)

_PASSAGE_PREFIX = "passage: "
_LETTER_RE = re.compile(r"[ABCDE]")


def load_retriever(
    index_dir: Optional[Path] = None,
    index_name: str = FAISS_INDEX_NAME,
    top_k: int = TOP_K,
):
    """FAISS index'i diskten yükle ve LangChain retriever döndür."""
    index_dir = Path(index_dir) if index_dir else FAISS_INDEX_DIR
    emb = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vs = FAISS.load_local(
        str(index_dir),
        embeddings=emb,
        index_name=index_name,
        allow_dangerous_deserialization=True,  # pickle gerektirir
    )
    return vs.as_retriever(search_kwargs={"k": top_k})


def retrieve_context(retriever, question: str) -> str:
    """e5 query prefix'i ile retrieve et, gösterimde passage prefix'ini gizle."""
    docs = retriever.invoke(e5_query(question))
    parts = []
    for i, d in enumerate(docs, 1):
        txt = d.page_content
        if txt.startswith(_PASSAGE_PREFIX):
            txt = txt[len(_PASSAGE_PREFIX):]
        parts.append(f"[{i}] {txt}")
    return "\n\n".join(parts)


def build_messages(question: str, choices: dict, context: Optional[str]) -> list[dict]:
    if context:
        return rag_qa_messages(question, choices, context)
    return zero_shot_qa_messages(question, choices)


@torch.inference_mode()
def generate_answer(
    model, tokenizer, question: str, choices: dict, context: Optional[str],
    max_new_tokens: int = 8,
) -> str:
    """Tek bir QA örneği için generation; sadece kısa cevap (harf) bekleniyor."""
    msgs = build_messages(question, choices, context)
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def parse_letter(text: str) -> Optional[str]:
    """Üretilen çıktıdan ilk A-E harfini ayıkla."""
    m = _LETTER_RE.search(text.upper())
    return m.group(0) if m else None
