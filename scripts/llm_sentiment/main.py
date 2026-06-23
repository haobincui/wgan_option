"""CLI for building RQ2 dictionary sentiment text features."""

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

from llm_sentiment import fit_sentiment_features


DEFAULT_NEWS_XLSX = "data/raw/text_embedding/news_with_openai_embeddings_large.xlsx"


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dictionary sentiment features for RQ2 text baselines.")
    parser.add_argument("--news-xlsx", default=DEFAULT_NEWS_XLSX, help="Raw news workbook path.")
    parser.add_argument("--output-dir", required=True, help="Directory for llm_sentiment_features.xlsx.")
    parser.add_argument("--text-column", default="LP", help="Raw text column to featurize.")
    parser.add_argument("--target-dim", type=int, default=1024, help="Fixed output vector width.")
    parser.add_argument(
        "--dictionary-path",
        default="",
        help="Optional Loughran-McDonald dictionary CSV. If set and missing, the CLI fails unless --allow-fallback is used.",
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Use the explicit builtin fallback lexicon when dictionary-path is missing.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> Path:
    args = _parse_args(argv)
    news_path = Path(args.news_xlsx).expanduser()
    if not news_path.exists():
        raise FileNotFoundError(f"News workbook does not exist: {news_path}")

    dictionary_path = str(args.dictionary_path).strip()
    allow_builtin_fallback = bool(args.allow_fallback or not dictionary_path)

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    news_df = pd.read_excel(news_path, engine="openpyxl", dtype=object)
    result = fit_sentiment_features(
        news_df,
        text_column=str(args.text_column),
        target_dim=int(args.target_dim),
        dictionary_path=dictionary_path or None,
        allow_builtin_fallback=allow_builtin_fallback,
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
