"""Enrich merged_vol.xlsx with RQ2 BoW and sentiment feature columns."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

if __package__ in {None, ""}:
    _ROOT_DIR = Path(__file__).resolve().parents[2]
    if str(_ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(_ROOT_DIR))

import scripts._path_setup  # noqa: F401

import pandas as pd


GAN_SHEET = "gan_input_ready"
BOW_COLUMNS = ["bow_embedding", "bow_dim"]
SENTIMENT_COLUMNS = ["sentiment_embedding", "sentiment_dim", "sentiment_dictionary_source"]
_SAMPLE_ID_PATTERN = re.compile(r"^news_(\d+)(?:_|$)")


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add RQ2 text feature columns to merged_vol.xlsx.")
    parser.add_argument("--merged-vol", required=True, help="Input merged_vol.xlsx path.")
    parser.add_argument("--bow-features", required=True, help="bow_features.xlsx path.")
    parser.add_argument("--sentiment-features", required=True, help="llm_sentiment_features.xlsx path.")
    parser.add_argument("--output", required=True, help="Output enriched workbook path.")
    return parser.parse_args(list(argv) if argv is not None else None)


def _news_row_id_from_sample_id(value: object) -> int | None:
    match = _SAMPLE_ID_PATTERN.match(str(value).strip())
    if not match:
        return None
    return int(match.group(1))


def _join_key(frame: pd.DataFrame) -> pd.Series | None:
    if "news_row_id" in frame.columns:
        return pd.to_numeric(frame["news_row_id"], errors="coerce").astype("Int64")
    if "sample_id" in frame.columns:
        return frame["sample_id"].map(_news_row_id_from_sample_id).astype("Int64")
    return None


def _load_feature_frame(path: str | Path, columns: list[str]) -> pd.DataFrame:
    feature_path = Path(path).expanduser()
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature workbook does not exist: {feature_path}")
    frame = pd.read_excel(feature_path, dtype=object)
    missing = [column for column in ["news_row_id", *columns] if column not in frame.columns]
    if missing:
        raise ValueError(f"Feature workbook {feature_path} is missing columns: {missing}")
    feature_frame = frame[["news_row_id", *columns]].copy()
    feature_frame["news_row_id"] = pd.to_numeric(feature_frame["news_row_id"], errors="coerce").astype("Int64")
    return feature_frame


def _merge_one_feature_set(
    frame: pd.DataFrame,
    feature_frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    require_matches: bool,
    sheet_name: str,
) -> pd.DataFrame:
    key = _join_key(frame)
    if key is None:
        return frame
    enriched = frame.drop(columns=[column for column in feature_columns if column in frame.columns]).copy()
    enriched["_rq2_news_row_id"] = key
    feature_values = feature_frame.rename(columns={"news_row_id": "_rq2_news_row_id"})
    merged = enriched.merge(
        feature_values,
        how="left",
        on="_rq2_news_row_id",
        suffixes=("", "_feature"),
        sort=False,
    )
    missing_mask = merged[feature_columns].isna().any(axis=1)
    if require_matches and bool(missing_mask.any()):
        missing_count = int(missing_mask.sum())
        raise ValueError(f"Sheet {sheet_name} has {missing_count} rows without RQ2 feature matches.")
    return merged.drop(columns=["_rq2_news_row_id"])


def enrich_workbook(
    *,
    merged_vol_path: str | Path,
    bow_features_path: str | Path,
    sentiment_features_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Write an enriched workbook while preserving all original sheets."""

    merged_path = Path(merged_vol_path).expanduser()
    if not merged_path.exists():
        raise FileNotFoundError(f"Merged-vol workbook does not exist: {merged_path}")
    bow_frame = _load_feature_frame(bow_features_path, BOW_COLUMNS)
    sentiment_frame = _load_feature_frame(sentiment_features_path, SENTIMENT_COLUMNS)

    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelFile(merged_path) as excel, pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name in excel.sheet_names:
            frame = pd.read_excel(excel, sheet_name=sheet_name, dtype=object)
            require_matches = sheet_name == GAN_SHEET
            enriched = _merge_one_feature_set(
                frame,
                bow_frame,
                BOW_COLUMNS,
                require_matches=require_matches,
                sheet_name=sheet_name,
            )
            enriched = _merge_one_feature_set(
                enriched,
                sentiment_frame,
                SENTIMENT_COLUMNS,
                require_matches=require_matches,
                sheet_name=sheet_name,
            )
            enriched.to_excel(writer, sheet_name=sheet_name, index=False)
    return output


def main(argv: Iterable[str] | None = None) -> Path:
    args = _parse_args(argv)
    output = enrich_workbook(
        merged_vol_path=args.merged_vol,
        bow_features_path=args.bow_features,
        sentiment_features_path=args.sentiment_features,
        output_path=args.output,
    )
    print(f"RQ2 enriched workbook written to {output}")
    return output


if __name__ == "__main__":
    main()
