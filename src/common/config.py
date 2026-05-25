"""Merkezi konfigürasyon.

Lokalde repo köküne, Colab'da `DRIVE_ROOT` env var ile Drive'a yönlenir.
Tüm sabitler (path, model adı, hyperparameter) tek noktada tutulur ki
script'ler argparse default'larını burdan beslesin.
"""
from __future__ import annotations
import os
from pathlib import Path

# --- Repo kökü (kod konumu) -------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]

# Veri / model / sonuç klasörleri için Drive override (Colab senaryosu)
_drive = os.environ.get("DRIVE_ROOT")


def _resolve(name: str, default: Path) -> Path:
    """DRIVE_ROOT set ise oradaki `name/`, değilse repo içindeki `default`."""
    if _drive:
        return Path(_drive) / name
    return default


DATA_DIR = Path(os.environ.get("DATA_DIR", _resolve("data", ROOT / "data")))
MODELS_DIR = Path(os.environ.get("MODELS_DIR", _resolve("models", ROOT / "models")))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", _resolve("results", ROOT / "results")))
CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", MODELS_DIR / "checkpoints"))
FAISS_INDEX_DIR = Path(os.environ.get("FAISS_INDEX_DIR", DATA_DIR / "faiss_index"))

# --- Model adları -----------------------------------------------------------
BASE_MODEL_NAME = os.environ.get("BASE_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-base")
COMET_MODEL_NAME = os.environ.get("COMET_MODEL_NAME", "Unbabel/wmt22-comet-da")

# --- Task 1 (MT / LoRA) -----------------------------------------------------
WMT16_CONFIG = "tr-en"
WMT16_TRAIN_SAMPLES = 50_000           # train subsample (None = full)
MAX_SEQ_LENGTH = 512

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

PER_DEVICE_TRAIN_BATCH_SIZE = 4        # A100 default; T4 için 1 + grad_accum 16
GRADIENT_ACCUMULATION_STEPS = 4        # effective batch = 16
LEARNING_RATE = 2e-4
NUM_TRAIN_EPOCHS = 1
WARMUP_RATIO = 0.03
SAVE_STEPS = 200
SAVE_TOTAL_LIMIT = 2
LOGGING_STEPS = 25

# --- Task 2 (RAG) -----------------------------------------------------------
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 5
FAISS_INDEX_NAME = "turkish_history"

# --- Generation (her iki task) ----------------------------------------------
GEN_MAX_NEW_TOKENS = 256
GEN_TEMPERATURE = 0.0
GEN_DO_SAMPLE = False

# --- Reproducibility --------------------------------------------------------
SEED = 42


def ensure_dirs() -> None:
    """Output klasörlerini idempotent şekilde oluşturur."""
    for d in (DATA_DIR, MODELS_DIR, RESULTS_DIR, CHECKPOINT_DIR, FAISS_INDEX_DIR):
        d.mkdir(parents=True, exist_ok=True)
