"""Generate RQ2 BoW and sentiment text-feature artifacts from the raw news workbook."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

if __package__ in {None, ""}:
    _ROOT_DIR = Path(__file__).resolve().parents[1]
    if str(_ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(_ROOT_DIR))

import scripts._path_setup  # noqa: F401

import pandas as pd

from bow import fit_bow_features
from llm_sentiment.features import DEFAULT_MODEL_ID, fit_sentiment_features


DEFAULT_NEWS_XLSX = "data/raw/text_embedding/news_with_openai_embeddings_large.xlsx"
DEFAULT_OUTPUT_ROOT = "data/processed/text_features/rq2"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate both n-gram frequency BoW and sentiment features for RQ2 FiLM experiments.",
    )
    parser.add_argument("--news-xlsx", default=DEFAULT_NEWS_XLSX, help="Raw news workbook path.")
    parser.add_argument(
        "--output-dir",
        default="",
        help=f"Output artifact directory. Defaults to {DEFAULT_OUTPUT_ROOT}/<utc-timestamp>.",
    )
    parser.add_argument("--text-column", default="LP", help="Raw text column to featurize.")
    parser.add_argument("--target-dim", type=int, default=1024, help="Fixed output vector width for both feature types.")
    parser.add_argument("--max-features", type=int, default=5000, help="Deprecated compatibility option; ignored.")
    parser.add_argument("--ngram-min", type=int, default=1, help="Minimum BoW n-gram length.")
    parser.add_argument("--ngram-max", type=int, default=2, help="Maximum BoW n-gram length.")
    parser.add_argument("--random-state", type=int, default=42, help="Deprecated compatibility option; ignored.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="HuggingFace LLaMA-style instruct model id.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Transformers inference device.")
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype for model loading.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum generated tokens per article.")
    parser.add_argument("--max-input-chars", type=int, default=6000, help="Maximum article characters sent to the model.")
    parser.add_argument("--cache-path", default="", help="JSONL cache path. Defaults to <output-dir>/llama3_sentiment_cache.jsonl.")
    parser.add_argument("--limit", type=int, default=None, help="Optional first-N article limit for smoke runs.")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Optional delay between uncached generations.")
    return parser.parse_args(list(argv) if argv is not None else None)


def _write_bow_artifacts(news_df: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> tuple[Path, dict]:
    result = fit_bow_features(
        news_df,
        text_column=str(args.text_column),
        target_dim=int(args.target_dim),
        max_features=int(args.max_features),
        ngram_range=(int(args.ngram_min), int(args.ngram_max)),
        random_state=int(args.random_state),
    )
    feature_path = output_dir / "bow_features.xlsx"
    result.frame.to_excel(feature_path, sheet_name="features", index=False)

    vocabulary_path = output_dir / "bow_vocabulary.json"
    vocabulary_path.write_text(json.dumps(result.vocabulary, indent=2), encoding="utf-8")

    manifest = {
        **result.manifest,
        "feature_path": str(feature_path),
        "vocabulary_path": str(vocabulary_path),
    }
    (output_dir / "bow_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return feature_path, manifest


def _write_sentiment_artifacts(news_df: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> tuple[Path, dict]:
    cache_path = Path(args.cache_path).expanduser() if str(args.cache_path).strip() else output_dir / "llama3_sentiment_cache.jsonl"
    result = fit_sentiment_features(
        news_df,
        text_column=str(args.text_column),
        target_dim=int(args.target_dim),
        model_id=str(args.model_id),
        device=str(args.device),
        torch_dtype=str(args.torch_dtype),
        max_new_tokens=int(args.max_new_tokens),
        max_input_chars=int(args.max_input_chars),
        cache_path=cache_path,
        limit=args.limit,
        sleep_seconds=float(args.sleep_seconds),
    )
    feature_path = output_dir / "llm_sentiment_features.xlsx"
    result.frame.to_excel(feature_path, sheet_name="features", index=False)

    manifest = {
        **result.manifest,
        "feature_path": str(feature_path),
    }
    (output_dir / "llm_sentiment_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return feature_path, manifest


def main(argv: Iterable[str] | None = None) -> Path:
    args = _parse_args(argv)
    news_path = Path(args.news_xlsx).expanduser()
    if not news_path.exists():
        raise FileNotFoundError(f"News workbook does not exist: {news_path}")

    output_dir = Path(args.output_dir).expanduser() if str(args.output_dir).strip() else Path(DEFAULT_OUTPUT_ROOT) / _timestamp()
    output_dir.mkdir(parents=True, exist_ok=True)

    news_df = pd.read_excel(news_path, engine="openpyxl", dtype=object)
    if args.limit is not None and int(args.limit) >= 0:
        news_df = news_df.head(int(args.limit)).copy()
    bow_path, bow_manifest = _write_bow_artifacts(news_df, output_dir, args)
    sentiment_path, sentiment_manifest = _write_sentiment_artifacts(news_df, output_dir, args)

    combined_manifest = {
        "news_xlsx": str(news_path),
        "output_dir": str(output_dir),
        "text_column": str(args.text_column),
        "target_dim": int(args.target_dim),
        "bow_features": str(bow_path),
        "llm_sentiment_features": str(sentiment_path),
        "bow_manifest": bow_manifest,
        "llm_sentiment_manifest": sentiment_manifest,
    }
    (output_dir / "rq2_text_features_manifest.json").write_text(
        json.dumps(combined_manifest, indent=2),
        encoding="utf-8",
    )

    print(f"BoW features written to {bow_path}")
    print(f"LLM-sentiment features written to {sentiment_path}")
    print(f"Combined manifest written to {output_dir / 'rq2_text_features_manifest.json'}")
    return output_dir


if __name__ == "__main__":
    main()
