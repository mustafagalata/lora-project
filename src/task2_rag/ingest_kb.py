"""Türk tarihi PDF kaynakları -> metin -> chunk -> embed -> FAISS index.

`data/history_book/*.pdf` altındaki tüm PDF'leri okur, temizler, recursive
character splitter ile ~800 char chunk'lara böler, e5 'passage:' prefix'ini
ekler ve FAISS indeksini `data/faiss_index/turkish_history.{faiss,pkl}` olarak
yazar.

Tekrar çalıştırıldığında index üzerine yazar (idempotent).
"""
from __future__ import annotations
import argparse
import logging
import re
from pathlib import Path

from src.common.utils import ensure_repo_on_syspath  # noqa: E402

ensure_repo_on_syspath()

from langchain_community.vectorstores import FAISS  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

from src.common.config import (  # noqa: E402
    CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR, EMBEDDING_MODEL_NAME,
    FAISS_INDEX_DIR, FAISS_INDEX_NAME,
)
from src.common.prompts import e5_passage  # noqa: E402
from src.common.utils import setup_logging  # noqa: E402

log = logging.getLogger(__name__)


def extract_text_from_pdf(path: Path) -> str:
    """pdfplumber ile sayfa-sayfa metin çıkarımı (Türkçe karakterler için pypdf'ten daha güvenli)."""
    import pdfplumber
    parts: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts)


_RE_HARD_LINEBREAK = re.compile(r"-\n(\w)")          # "Os-\nmanlı" -> "Osmanlı"
_RE_PAGE_NUMBER = re.compile(r"^\s*\d{1,4}\s*$", re.MULTILINE)
_RE_MULTI_BLANK = re.compile(r"\n{3,}")
_RE_MULTI_WS = re.compile(r"[ \t]+")


def clean(text: str) -> str:
    text = text.replace("\r", "")
    text = _RE_HARD_LINEBREAK.sub(r"\1", text)
    text = _RE_PAGE_NUMBER.sub("", text)
    text = _RE_MULTI_BLANK.sub("\n\n", text)
    text = _RE_MULTI_WS.sub(" ", text)
    return text.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_dir", type=Path, default=DATA_DIR / "history_book")
    ap.add_argument("--out_dir", type=Path, default=FAISS_INDEX_DIR)
    ap.add_argument("--index_name", default=FAISS_INDEX_NAME)
    ap.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    ap.add_argument("--chunk_overlap", type=int, default=CHUNK_OVERLAP)
    ap.add_argument("--embedding_batch", type=int, default=64)
    args = ap.parse_args()

    setup_logging()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(args.pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"Hiç PDF bulunamadı: {args.pdf_dir}")
    log.info("Bulundu: %d PDF", len(pdfs))

    docs: list[Document] = []
    for p in pdfs:
        log.info("Okunuyor: %s", p.name)
        raw = extract_text_from_pdf(p)
        text = clean(raw)
        docs.append(Document(page_content=text, metadata={"source": p.name}))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    log.info("Üretilen chunk sayısı: %d", len(chunks))

    # e5 sözleşmesi: chunk metnine "passage: " prefix
    for c in chunks:
        c.page_content = e5_passage(c.page_content)

    log.info("Embedding modeli yükleniyor: %s", EMBEDDING_MODEL_NAME)
    import torch
    emb = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": args.embedding_batch},
    )

    log.info("FAISS index oluşturuluyor...")
    vs = FAISS.from_documents(chunks, embedding=emb)
    vs.save_local(str(args.out_dir), index_name=args.index_name)
    log.info("Yazıldı: %s/{%s.faiss, %s.pkl}", args.out_dir, args.index_name, args.index_name)


if __name__ == "__main__":
    main()
