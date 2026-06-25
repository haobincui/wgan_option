"""CLI for building RQ2 Sun-style LLaMA sentiment text features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

if __package__ in {None, ""}:
    _ROOT_DIR = Path(__file__).resolve().parents[2]
    if str(_ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(_ROOT_DIR))

import scripts._path_setup  # noqa: F401

import pandas as pd

from llm_sentiment.features import DEFAULT_MODEL_ID, fit_sentiment_features


DEFAULT_NEWS_XLSX = "data/raw/text_embedding/news_with_openai_embeddings_large.xlsx"


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Sun-style LLaMA sentiment features for RQ2 text baselines.")
    parser.add_argument("--news-xlsx", default=DEFAULT_NEWS_XLSX, help="Raw news workbook path.")
    parser.add_argument("--output-dir", required=True, help="Directory for llm_sentiment_features.xlsx.")
    parser.add_argument("--text-column", default="LP", help="Raw text column to featurize.")
    parser.add_argument("--target-dim", type=int, default=1024, help="Fixed output vector width.")
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


def main(argv: Iterable[str] | None = None) -> Path:
    args = _parse_args(argv)
    news_path = Path(args.news_xlsx).expanduser()
    if not news_path.exists():
        raise FileNotFoundError(f"News workbook does not exist: {news_path}")

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache_path).expanduser() if str(args.cache_path).strip() else output_dir / "llama3_sentiment_cache.jsonl"

    news_df = pd.read_excel(news_path, engine="openpyxl", dtype=object)
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
        "news_xlsx": str(news_path),
        "feature_path": str(feature_path),
    }
    (output_dir / "llm_sentiment_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"Sentiment features written to {feature_path}")
    return feature_path


if __name__ == "__main__":
    main()
