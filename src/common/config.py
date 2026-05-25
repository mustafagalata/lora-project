"""Merkezi konfigürasyon.

Tüm tunable parametreler repo kökündeki ``config.yaml`` içinde tutulur. Bu
modül YAML'ı bir kez okur ve eski Python sabit isimlerine (BASE_MODEL_NAME,
LORA_R, vb.) map'ler — böylece script'ler ``from src.common.config import X``
ile import etmeye devam edebilir.

Path'ler için Drive override destekli:
- ``DRIVE_ROOT`` env var set ise data/models/results path'leri oradan türer.
- ``CONFIG_PATH`` env var ile alternatif bir YAML dosyası yüklenebilir.
"""
from __future__ import annotations
import os
from pathlib import Path

import yaml


# --- Repo kökü -------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]

# --- YAML yükle ------------------------------------------------------------
CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", ROOT / "config.yaml"))
with open(CONFIG_PATH, "r", encoding="utf-8") as _f:
    _cfg = yaml.safe_load(_f)


# --- Path'ler (Drive override destekli) ------------------------------------
_drive = os.environ.get("DRIVE_ROOT")


def _resolve(name: str, default: Path) -> Path:
    return Path(_drive) / name if _drive else default


DATA_DIR        = Path(os.environ.get("DATA_DIR",        _resolve("data",    ROOT / "data")))
MODELS_DIR      = Path(os.environ.get("MODELS_DIR",      _resolve("models",  ROOT / "models")))
RESULTS_DIR     = Path(os.environ.get("RESULTS_DIR",     _resolve("results", ROOT / "results")))
CHECKPOINT_DIR  = Path(os.environ.get("CHECKPOINT_DIR",  MODELS_DIR / "checkpoints"))
FAISS_INDEX_DIR = Path(os.environ.get("FAISS_INDEX_DIR", DATA_DIR / "faiss_index"))


# --- Modeller --------------------------------------------------------------
BASE_MODEL_NAME      = _cfg["models"]["base"]
EMBEDDING_MODEL_NAME = _cfg["models"]["embedding"]
COMET_MODEL_NAME     = _cfg["models"]["comet"]


# --- Task 1 — MT / LoRA ----------------------------------------------------
_t1 = _cfg["task1"]
WMT16_CONFIG             = _t1["wmt16_config"]
WMT16_TRAIN_SAMPLES      = _t1["train_samples"]
TEST_LIMIT_PER_DIRECTION = _t1["test_limit_per_direction"]
MAX_SEQ_LENGTH           = _t1["max_seq_length"]

LORA_R              = _t1["lora"]["r"]
LORA_ALPHA          = _t1["lora"]["alpha"]
LORA_DROPOUT        = _t1["lora"]["dropout"]
LORA_TARGET_MODULES = list(_t1["lora"]["target_modules"])

PER_DEVICE_TRAIN_BATCH_SIZE = _t1["training"]["per_device_batch_size"]
GRADIENT_ACCUMULATION_STEPS = _t1["training"]["grad_accum_steps"]
LEARNING_RATE               = float(_t1["training"]["learning_rate"])
NUM_TRAIN_EPOCHS            = _t1["training"]["num_epochs"]
WARMUP_RATIO                = _t1["training"]["warmup_ratio"]
LOGGING_STEPS               = _t1["training"]["logging_steps"]
SAVE_STEPS                  = _t1["training"]["save_steps"]
SAVE_TOTAL_LIMIT            = _t1["training"]["save_total_limit"]

INFERENCE_BATCH_SIZE = _t1["inference"]["batch_size"]
COMET_BATCH_SIZE     = _t1["comet"]["batch_size"]


# --- Task 2 — RAG ----------------------------------------------------------
_t2 = _cfg["task2"]
CHUNK_SIZE            = _t2["chunk_size"]
CHUNK_OVERLAP         = _t2["chunk_overlap"]
TOP_K                 = _t2["top_k"]
FAISS_INDEX_NAME      = _t2["faiss_index_name"]
EMBEDDING_BATCH_SIZE  = _t2["embedding_batch_size"]

FILTER_ENABLED     = _t2["filter"]["enabled"]
MIN_CHUNK_CHARS    = _t2["filter"]["min_chunk_chars"]
MAX_NONALNUM_RATIO = _t2["filter"]["max_nonalnum_ratio"]


# --- Generation ------------------------------------------------------------
_gen = _cfg["generation"]
GEN_MAX_NEW_TOKENS = _gen["max_new_tokens"]
GEN_DO_SAMPLE      = _gen["do_sample"]
GEN_NUM_BEAMS      = _gen.get("num_beams", 1)


# --- Misc ------------------------------------------------------------------
SEED = _cfg.get("seed", 42)


def ensure_dirs() -> None:
    """Output klasörlerini idempotent oluştur."""
    for d in (DATA_DIR, MODELS_DIR, RESULTS_DIR, CHECKPOINT_DIR, FAISS_INDEX_DIR):
        d.mkdir(parents=True, exist_ok=True)
