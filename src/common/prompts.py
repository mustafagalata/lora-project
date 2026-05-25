"""Ortak prompt template'leri ve formatlayıcılar.

Tüm görevler Qwen ChatML formatına `tokenizer.apply_chat_template()` üzerinden
gönderilecek mesaj listeleri (dict listesi) üretir.
"""
from __future__ import annotations
from typing import Optional


# --- Task 1: Machine Translation -------------------------------------------

MT_SYSTEM = "You are a professional translator. Translate accurately without adding commentary."


def mt_messages(
    source_text: str,
    src_lang: str,
    tgt_lang: str,
    target_text: Optional[str] = None,
) -> list[dict]:
    """ChatML mesaj listesi; eğitimde target dolu, inference'da None."""
    user = f"Translate the following text from {src_lang} to {tgt_lang}:\n\n{source_text}"
    msgs: list[dict] = [
        {"role": "system", "content": MT_SYSTEM},
        {"role": "user", "content": user},
    ]
    if target_text is not None:
        msgs.append({"role": "assistant", "content": target_text})
    return msgs


# --- Task 2: TurkishMMLU History QA ----------------------------------------

HISTORY_QA_SYSTEM_ZS = (
    "Sen Türk tarihi konusunda uzman bir öğretmensin. Aşağıdaki çoktan seçmeli "
    "soruyu cevapla. Sadece doğru seçeneğin harfini ver (A, B, C, D veya E). "
    "Başka hiçbir şey yazma."
)

HISTORY_QA_SYSTEM_RAG = (
    "Sen Türk tarihi konusunda uzman bir öğretmensin. Aşağıdaki bağlam parçalarına "
    "ve genel tarih bilgine dayanarak çoktan seçmeli soruyu cevapla. Sadece doğru "
    "seçeneğin harfini ver (A, B, C, D veya E). Başka hiçbir şey yazma."
)


def _format_choices(choices: dict[str, str]) -> str:
    return "\n".join(f"{k}) {v}" for k, v in choices.items())


def zero_shot_qa_messages(question: str, choices: dict[str, str]) -> list[dict]:
    body = f"Soru: {question}\n\n{_format_choices(choices)}\n\nCevap (sadece harf):"
    return [
        {"role": "system", "content": HISTORY_QA_SYSTEM_ZS},
        {"role": "user", "content": body},
    ]


def rag_qa_messages(question: str, choices: dict[str, str], context: str) -> list[dict]:
    body = (
        f"Bağlam:\n{context}\n\n"
        f"Soru: {question}\n\n{_format_choices(choices)}\n\n"
        f"Cevap (sadece harf):"
    )
    return [
        {"role": "system", "content": HISTORY_QA_SYSTEM_RAG},
        {"role": "user", "content": body},
    ]


# --- e5 ailesi prefix sözleşmesi -------------------------------------------
# intfloat/multilingual-e5-base her zaman "query: " / "passage: " prefix ister.

def e5_passage(text: str) -> str:
    return f"passage: {text}"


def e5_query(text: str) -> str:
    return f"query: {text}"
