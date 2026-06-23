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
from llm_sentiment import fit_sentiment_features


DEFAULT_NEWS_XLSX = "data/raw/text_embedding/news_with_openai_embeddings_large.xlsx"
DEFAULT_OUTPUT_ROOT = "data/processed/text_features/rq2"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate both BoW/TF-IDF and sentiment features for RQ2 FiLM experiments.",
    )
    parser.add_argument("--news-xlsx", default=DEFAULT_NEWS_XLSX, help="Raw news workbook path.")
    parser.add_argument(
        "--output-dir",
        default="",
        help=f"Output artifact directory. Defaults to {DEFAULT_OUTPUT_ROOT}/<utc-timestamp>.",
    )
    parser.add_argument("--text-column", default="LP", help="Raw text column to featurize.")
    parser.add_argument("--target-dim", type=int, default=1024, help="Fixed output vector width for both feature types.")
    parser.add_argument("--max-features", type=int, default=5000, help="Maximum TF-IDF vocabulary size.")
    parser.add_argument("--ngram-min", type=int, default=1, help="Minimum BoW n-gram length.")
    parser.add_argument("--ngram-max", type=int, default=2, help="Maximum BoW n-gram length.")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for BoW SVD when sklearn is available.")
    parser.add_argument(
        "--dictionary-path",
        default="",
        help="Optional Loughran-McDonald dictionary CSV. If set and missing, the script fails unless --allow-fallback is used.",
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Use the explicit builtin fallback lexicon when dictionary-path is missing.",
    )
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

    vectorizer_path = output_dir / "tfidf_vectorizer.joblib"
    svd_path = output_dir / "svd_model.joblib"
    manifest = {
        **result.manifest,
        "feature_path": str(feature_path),
        "vectorizer_path": str(vectorizer_path) if result.vectorizer is not None else "",
        "svd_path": str(svd_path) if result.svd is not None else "",
    }
    if result.vectorizer is not None:
        from joblib import dump

        dump(result.vectorizer, vectorizer_path)
    if result.svd is not None:
        from joblib import dump

        dump(result.svd, svd_path)
    (output_dir / "bow_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return feature_path, manifest


def _write_sentiment_artifacts(news_df: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> tuple[Path, dict]:
    dictionary_path = str(args.dictionary_path).strip()
    allow_builtin_fallback = bool(args.allow_fallback or not dictionary_path)
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
