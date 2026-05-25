"""WMT22-COMET-DA ile çeviri kalitesini ölç ve HW2 baseline ile yan yana raporla.

Predictions JSONL'ünden yön bazında (en2tr, tr2en) ortalama sistem skoru hesaplar.
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

from src.common.utils import ensure_repo_on_syspath  # noqa: E402

ensure_repo_on_syspath()

from comet import download_model, load_from_checkpoint  # noqa: E402

from src.common.config import COMET_MODEL_NAME, RESULTS_DIR, ROOT  # noqa: E402
from src.common.utils import read_jsonl, setup_logging, write_json  # noqa: E402

log = logging.getLogger(__name__)

# Predictions arasındaki src_lang -> yön kısaltması
_LANG_TO_CODE = {"english": "en", "turkish": "tr"}


def _direction(ex: dict) -> str:
    src = _LANG_TO_CODE.get(ex["src_lang"].lower(), ex["src_lang"][:2].lower())
    tgt = _LANG_TO_CODE.get(ex["tgt_lang"].lower(), ex["tgt_lang"][:2].lower())
    return f"{src}2{tgt}"


def evaluate(predictions: list[dict], comet_model, batch_size: int = 8) -> dict:
    by_dir: dict[str, list[dict]] = {}
    for ex in predictions:
        by_dir.setdefault(_direction(ex), []).append({
            "src": ex["src"], "mt": ex["hypothesis"], "ref": ex["tgt"],
        })

    scores: dict[str, float | int | dict] = {}
    for direction, data in by_dir.items():
        log.info("COMET %s (n=%d) hesaplanıyor...", direction, len(data))
        result = comet_model.predict(data, batch_size=batch_size, gpus=1, progress_bar=True)
        scores[direction] = float(result.system_score)
        scores[f"{direction}_n"] = len(data)
    return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, default=RESULTS_DIR / "task1_predictions.jsonl")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "task1_comet_results.json")
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    setup_logging()

    log.info("COMET (%s) indiriliyor / yükleniyor...", COMET_MODEL_NAME)
    model_path = download_model(COMET_MODEL_NAME)
    comet_model = load_from_checkpoint(model_path)

    preds = list(read_jsonl(args.pred))
    log.info("Tahmin sayısı: %d", len(preds))
    scores = evaluate(preds, comet_model, batch_size=args.batch_size)

    # HW2 baseline'ı çıktıya iliştir (kıyas raporlamak için)
    baseline_path = ROOT / "comet_scores.json"
    if baseline_path.exists():
        try:
            scores["_baseline_hw2"] = json.loads(baseline_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            log.warning("HW2 baseline parse edilemedi: %s", e)

    write_json(scores, args.out)
    log.info("Yazıldı: %s", args.out)

    # --- HW2 ile karşılaştırma tablosu ---
    hw2 = scores.get("_baseline_hw2", {})

    def _fmt(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) else "  —   "

    print("\n" + "=" * 78)
    print("COMET KARŞILAŞTIRMA (LoRA fine-tuned vs HW2 baseline)")
    print("=" * 78)
    print(f"{'Yön':<8}{'Zero-shot':>12}{'MAPS':>12}{'RAG':>12}{'LoRA (YENİ)':>14}{'Δ vs MAPS':>14}")
    print("-" * 78)
    for d in ("en2tr", "tr2en"):
        zs = hw2.get(f"zero_shot_{d}")
        maps_v = hw2.get(f"maps_{d}")
        rag = hw2.get(f"rag_{d}")
        lora = scores.get(d)
        delta = (
            f"{lora - maps_v:+.4f}"
            if isinstance(lora, (int, float)) and isinstance(maps_v, (int, float))
            else "   —   "
        )
        print(f"{d:<8}{_fmt(zs):>12}{_fmt(maps_v):>12}{_fmt(rag):>12}{_fmt(lora):>14}{delta:>14}")
    print("=" * 78)

    print("\nRaw JSON:")
    print(json.dumps(scores, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
