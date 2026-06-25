"""CLI for building RQ2 n-gram frequency BoW text features."""

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

from bow import fit_bow_features


DEFAULT_NEWS_XLSX = "data/raw/text_embedding/news_with_openai_embeddings_large.xlsx"


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build n-gram frequency BoW features for RQ2 text baselines.")
    parser.add_argument("--news-xlsx", default=DEFAULT_NEWS_XLSX, help="Raw news workbook path.")
    parser.add_argument("--output-dir", required=True, help="Directory for bow_features.xlsx and bow_vocabulary.json.")
    parser.add_argument("--text-column", default="LP", help="Raw text column to featurize.")
    parser.add_argument("--target-dim", type=int, default=1024, help="Fixed output vector width.")
    parser.add_argument("--max-features", type=int, default=5000, help="Deprecated compatibility option; ignored.")
    parser.add_argument("--ngram-min", type=int, default=1, help="Minimum n-gram length.")
    parser.add_argument("--ngram-max", type=int, default=2, help="Maximum n-gram length.")
    parser.add_argument("--random-state", type=int, default=42, help="Deprecated compatibility option; ignored.")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> Path:
    args = _parse_args(argv)
    news_path = Path(args.news_xlsx).expanduser()
    if not news_path.exists():
        raise FileNotFoundError(f"News workbook does not exist: {news_path}")

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    news_df = pd.read_excel(news_path, engine="openpyxl", dtype=object)
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
        "news_xlsx": str(news_path),
        "feature_path": str(feature_path),
        "vocabulary_path": str(vocabulary_path),
    }
    (output_dir / "bow_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"BoW features written to {feature_path}")
    return feature_path


if __name__ == "__main__":
    main()
