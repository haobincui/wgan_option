"""Generate article- and pair-level LM dictionary sentiment artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

if __package__ in {None, ""}:
    _ROOT_DIR = Path(__file__).resolve().parents[2]
    if str(_ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(_ROOT_DIR))

import scripts._path_setup  # noqa: F401

import pandas as pd

from lm_sentiment.features import build_lm_feature_result, load_lm_dictionary


DEFAULT_NEWS_XLSX = "data/raw/text_embedding/news_with_openai_embeddings_large.xlsx"
DEFAULT_DICTIONARY = "data/reference/Loughran-McDonald_MasterDictionary_1993-2025.csv"
DEFAULT_PAIR_WORKBOOK = (
    "data/processed/rq3/"
    "news_first_vol_surfaces_q097_103_ttm01_38_exact_ttm_v1/"
    "tolerance_05m/merged_vol.xlsx"
)
DEFAULT_OUTPUT_ROOT = "data/processed/text_features/rq2"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Loughran--McDonald dictionary sentiment. This is a "
            "lexicon score, not the corpus-frequency BoW representation."
        )
    )
    parser.add_argument("--news-xlsx", default=DEFAULT_NEWS_XLSX)
    parser.add_argument("--news-sheet", default="Sheet1")
    parser.add_argument("--text-column", default="LP")
    parser.add_argument("--dictionary-path", default=DEFAULT_DICTIONARY)
    parser.add_argument(
        "--pair-workbook",
        default=DEFAULT_PAIR_WORKBOOK,
        help="Optional merged workbook used to create pair-level scores; pass an empty value to skip.",
    )
    parser.add_argument("--pair-sheet", default="gan_input_ready")
    parser.add_argument("--pair-text-column", default="lp_text")
    parser.add_argument("--negation-window", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        default="",
        help=(
            "Output directory. Defaults to "
            f"{DEFAULT_OUTPUT_ROOT}/lm_dictionary_<UTC timestamp>."
        ),
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _quality_summary(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "row_count": len(frame),
        "valid_word_count_sum": int(frame["lm_valid_word_count"].sum()),
        "raw_word_count_sum": int(frame["lm_raw_word_count"].sum()),
        "zero_valid_word_rows": int((frame["lm_valid_word_count"] == 0).sum()),
        "zero_sentiment_word_rows": int((~frame["lm_has_sentiment_words"]).sum()),
        "dictionary_coverage_mean": float(frame["lm_dictionary_coverage"].mean()),
        "dictionary_coverage_median": float(frame["lm_dictionary_coverage"].median()),
        "sentiment_score_mean": float(frame["lm_sentiment_score"].mean()),
        "sentiment_score_median": float(frame["lm_sentiment_score"].median()),
        "sentiment_score_min": float(frame["lm_sentiment_score"].min()),
        "sentiment_score_max": float(frame["lm_sentiment_score"].max()),
    }


def _artifact_entry(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def main(argv: Iterable[str] | None = None) -> Path:
    args = _parse_args(argv)
    news_path = Path(args.news_xlsx).expanduser().resolve()
    dictionary_path = Path(args.dictionary_path).expanduser().resolve()
    pair_path = (
        Path(args.pair_workbook).expanduser().resolve()
        if str(args.pair_workbook).strip()
        else None
    )
    for label, path in (
        ("news workbook", news_path),
        ("LM dictionary", dictionary_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if pair_path is not None and not pair_path.is_file():
        raise FileNotFoundError(f"pair workbook does not exist: {pair_path}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if str(args.output_dir).strip()
        else (Path(DEFAULT_OUTPUT_ROOT) / f"lm_dictionary_{_timestamp()}").resolve()
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    news_frame = pd.read_excel(
        news_path,
        sheet_name=str(args.news_sheet),
        usecols=lambda column: column
        in {
            "news_row_id",
            "ArticleID",
            "article_id",
            "SourceFile",
            "source_file",
            str(args.text_column),
        },
        engine="openpyxl",
    )
    pair_frame = None
    if pair_path is not None:
        pair_frame = pd.read_excel(
            pair_path,
            sheet_name=str(args.pair_sheet),
            usecols=lambda column: column
            in {
                "pair_id",
                "news_row_id",
                "article_id",
                "sample_id",
                "session_id",
                "effective_origin_utc",
                "target_anchor_utc",
                "tolerance_minutes",
                str(args.pair_text_column),
            },
            engine="openpyxl",
        )

    dictionary = load_lm_dictionary(dictionary_path)
    result = build_lm_feature_result(
        news_frame,
        dictionary,
        text_column=str(args.text_column),
        negation_window=int(args.negation_window),
        pair_frame=pair_frame,
        pair_text_column=str(args.pair_text_column),
    )

    article_path = output_dir / "lm_article_scores.csv.gz"
    result.article_scores.to_csv(article_path, index=False, compression="gzip")
    pair_output_path = None
    if result.pair_scores is not None:
        pair_output_path = output_dir / "lm_pair_scores_5m.csv"
        result.pair_scores.to_csv(pair_output_path, index=False)

    workbook_path = output_dir / "lm_sentiment_scores.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        result.article_scores.to_excel(writer, sheet_name="article_scores", index=False)
        if result.pair_scores is not None:
            result.pair_scores.to_excel(
                writer, sheet_name="pair_scores_5m", index=False
            )

    manifest = {
        **result.manifest,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "news_workbook": str(news_path),
            "news_workbook_sha256": _sha256_file(news_path),
            "pair_workbook": "" if pair_path is None else str(pair_path),
            "pair_workbook_sha256": ""
            if pair_path is None
            else _sha256_file(pair_path),
        },
        "outputs": {
            "article_scores": _artifact_entry(article_path),
            "pair_scores": _artifact_entry(pair_output_path),
            "workbook": _artifact_entry(workbook_path),
        },
        "quality": {
            "article": _quality_summary(result.article_scores),
            "pair": (
                None
                if result.pair_scores is None
                else _quality_summary(result.pair_scores)
            ),
        },
    }
    manifest_path = output_dir / "lm_sentiment_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"LM article scores written to {article_path}")
    if pair_output_path is not None:
        print(f"LM pair scores written to {pair_output_path}")
    print(f"LM workbook written to {workbook_path}")
    print(f"Manifest written to {manifest_path}")
    return output_dir


if __name__ == "__main__":
    main()
