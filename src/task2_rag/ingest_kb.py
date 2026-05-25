"""Türk tarihi PDF kaynakları -> metin -> chunk -> embed -> FAISS index.

`data/history_book/*.pdf` altındaki tüm PDF'leri okur, ders kitabı boilerplate'ini
(içindekiler, kaynakça, dizin, sözlük, ekler) heuristic olarak ayıklar,
~800 char chunk'lara böler, gürültülü chunk'ları (çok kısa, dot-leader yoğun,
tablo/menü kalıbı) eler, e5 'passage:' prefix'ini ekler ve FAISS indeksini yazar.

Tekrar çalıştırıldığında index üzerine yazar (idempotent). `--no_filter` ile
ham (filtresiz) pipeline çalıştırılabilir (ablation veya debug için).
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


# --- PDF -> ham metin ------------------------------------------------------

def extract_text_from_pdf(path: Path) -> str:
    """pdfplumber sayfa-sayfa text (Türkçe karakterler için pypdf'ten daha güvenli)."""
    import pdfplumber
    parts: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts)


# --- Düşük seviyeli metin temizliği ----------------------------------------

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


# --- Bölüm-bazlı kesim (ders kitabı boilerplate'i için) ---------------------

# Kitap sonu bölüm başlıkları: kaynakça, dizin, sözlük, ekler vb.
_RE_END_SECTIONS = re.compile(
    r"\b(?:KAYNAK(?:LAR|ÇA)|BİBLİYOGRAFYA|REFERANSLAR|DİZİN|İNDEKS|SÖZLÜK|EKLER?)\b",
    re.IGNORECASE,
)
# İçindekiler bloğu işareti
_RE_TOC = re.compile(r"\bİÇİNDEKİLER\b", re.IGNORECASE)
# İçindekiler'den sonra atlanılacak ilk gerçek ana başlık
_RE_FIRST_HEADING = re.compile(
    r"\b(?:GİRİŞ|"
    r"(?:1|BİRİNCİ|İLK)\s*\.\s*(?:ÜNİTE|BÖLÜM)|"
    r"ÜNİTE\s*1\b|BÖLÜM\s*1\b)\b",
    re.IGNORECASE,
)


def trim_boilerplate(text: str) -> tuple[str, dict]:
    """Ders kitabının başındaki TOC ve sonundaki kaynakça/dizin'i atar.

    Konum guard'ları:
      - Sondaki kaynakça/dizin yalnızca metnin SON %30'unda bulunursa kesilir
        (ünite sonu küçük kaynak listeleri korunur).
      - İçindekiler yalnızca İLK %20'de görünürse atlanır; sonraki gerçek
        başlık metnin %30'undan önce başlamalıdır.
    """
    stats = {"trimmed_front": 0, "trimmed_end": 0}
    n = len(text)
    if n == 0:
        return text, stats

    matches = list(_RE_END_SECTIONS.finditer(text))
    if matches and matches[-1].start() > 0.70 * n:
        cut = matches[-1].start()
        stats["trimmed_end"] = n - cut
        text = text[:cut]

    toc = _RE_TOC.search(text)
    if toc and toc.start() < 0.20 * n:
        heading = _RE_FIRST_HEADING.search(text, pos=toc.end())
        if heading and heading.start() < 0.30 * n:
            stats["trimmed_front"] = heading.start()
            text = text[heading.start():]
    return text, stats


# --- Chunk-level gürültü filtresi -------------------------------------------

_RE_DOT_LEADER = re.compile(r"\.{3,}\s*\d+")   # "..... 12" gibi içindekiler izi
MIN_CHUNK_CHARS = 150
MAX_NONALNUM_RATIO = 0.40


def is_noise_chunk(text: str) -> bool:
    """Chunk'ı atmaya değer mi? (çok kısa, dot-leader yoğun, tablo benzeri)"""
    s = text.strip()
    if len(s) < MIN_CHUNK_CHARS:
        return True
    if len(_RE_DOT_LEADER.findall(s)) >= 2:
        return True
    nonalnum = sum(1 for c in s if not (c.isalnum() or c.isspace()))
    if nonalnum / len(s) > MAX_NONALNUM_RATIO:
        return True
    return False


# --- Ana pipeline -----------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_dir", type=Path, default=DATA_DIR / "history_book")
    ap.add_argument("--out_dir", type=Path, default=FAISS_INDEX_DIR)
    ap.add_argument("--index_name", default=FAISS_INDEX_NAME)
    ap.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    ap.add_argument("--chunk_overlap", type=int, default=CHUNK_OVERLAP)
    ap.add_argument("--embedding_batch", type=int, default=64)
    ap.add_argument("--no_filter", action="store_true",
                    help="Bölüm trim ve chunk filtresini kapat (ham pipeline)")
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
        if not args.no_filter:
            text, stats = trim_boilerplate(text)
            log.info("  trim: front=%d char, end=%d char",
                     stats["trimmed_front"], stats["trimmed_end"])
        docs.append(Document(page_content=text, metadata={"source": p.name}))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    log.info("Üretilen ham chunk sayısı: %d", len(chunks))

    if not args.no_filter:
        before = len(chunks)
        chunks = [c for c in chunks if not is_noise_chunk(c.page_content)]
        log.info("Noise filtre: %d -> %d (-%d)", before, len(chunks), before - len(chunks))

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
