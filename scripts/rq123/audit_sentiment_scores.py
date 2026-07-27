#!/usr/bin/env python3
"""Prepare and evaluate frozen ChatGPT-sentiment validation artifacts."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from llm_sentiment.features import (  # noqa: E402
    SENTIMENT_DIMENSIONS,
    build_sun_prompt,
)


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _vector(value: Any) -> np.ndarray:
    if isinstance(value, (list, tuple, np.ndarray)):
        result = np.asarray(value, dtype=np.float64).reshape(-1)
    else:
        text = (
            ""
            if value is None
            or (isinstance(value, float) and np.isnan(value))
            else str(value).strip()
        )
        parsed: Any = None
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(text)
                break
            except (ValueError, SyntaxError, json.JSONDecodeError):
                continue
        if not isinstance(parsed, (list, tuple)):
            raise ValueError("Invalid sentiment_embedding vector.")
        result = np.asarray(parsed, dtype=np.float64).reshape(-1)
    if result.size < len(SENTIMENT_DIMENSIONS):
        raise ValueError(
            "Sentiment vector has fewer than three frozen score dimensions."
        )
    if not np.all(np.isfinite(result[: len(SENTIMENT_DIMENSIONS)])):
        raise ValueError("Sentiment scores contain non-finite values.")
    return result


def _stable_stratified_sample(
    frame: pd.DataFrame,
    *,
    sample_size: int,
    seed: int,
) -> pd.DataFrame:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    if frame.empty:
        raise ValueError("No non-empty LP rows are available for sentiment audit.")
    ordered = frame.sort_values(
        ["score_composite", "news_row_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    ordered["score_decile"] = np.minimum(
        (np.arange(len(ordered), dtype=np.int64) * 10) // len(ordered),
        9,
    )
    if sample_size >= len(ordered):
        return ordered
    per_decile = sample_size // 10
    remainder = sample_size % 10
    samples = []
    for decile, group in ordered.groupby("score_decile", sort=True):
        take = per_decile + (1 if int(decile) < remainder else 0)
        take = min(take, len(group))
        samples.append(
            group.sample(
                n=take,
                random_state=int(seed) + int(decile),
                replace=False,
            )
        )
    sampled = pd.concat(samples, ignore_index=True)
    if len(sampled) < sample_size:
        remaining = ordered[
            ~ordered["news_row_id"].isin(sampled["news_row_id"])
        ]
        sampled = pd.concat(
            [
                sampled,
                remaining.sample(
                    n=sample_size - len(sampled),
                    random_state=int(seed) + 100,
                    replace=False,
                ),
            ],
            ignore_index=True,
        )
    return sampled.sort_values("news_row_id").reset_index(drop=True)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def prepare(args: argparse.Namespace) -> int:
    news_path = _resolve(args.news_workbook)
    feature_path = _resolve(args.sentiment_features)
    manifest_path = _resolve(args.sentiment_manifest)
    cache_path = _resolve(args.sentiment_cache)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in (news_path, feature_path, manifest_path, cache_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    frozen_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    news = pd.read_excel(news_path).reset_index(drop=True)
    features = pd.read_excel(feature_path).reset_index(drop=True)
    news["news_row_id"] = np.arange(1, len(news) + 1, dtype=np.int64)
    features["news_row_id"] = pd.to_numeric(
        features["news_row_id"],
        errors="raise",
    ).astype(np.int64)
    if features["news_row_id"].duplicated().any():
        raise ValueError("Sentiment features contain duplicate news_row_id.")
    merged = news.merge(
        features,
        on="news_row_id",
        how="left",
        validate="one_to_one",
        suffixes=("_news", "_feature"),
    )
    if merged["sentiment_embedding"].isna().any():
        missing = int(merged["sentiment_embedding"].isna().sum())
        raise ValueError(f"Missing frozen sentiment features for {missing} rows.")

    vectors = np.vstack(
        [_vector(value)[:3] for value in merged["sentiment_embedding"]]
    )
    for index, dimension in enumerate(SENTIMENT_DIMENSIONS):
        merged[f"model_{dimension}"] = vectors[:, index]
    merged["score_composite"] = vectors.mean(axis=1)
    merged["lp_text"] = merged.get("LP", pd.Series([""] * len(merged))).fillna(
        ""
    ).astype(str)
    merged["lp_text_sha256"] = merged["lp_text"].map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
        if value
        else ""
    )
    eligible = merged[merged["lp_text"].str.strip().ne("")].copy()
    sampled = _stable_stratified_sample(
        eligible,
        sample_size=min(int(args.manual_sample_size), len(eligible)),
        seed=int(args.seed),
    )

    article_column = (
        "ArticleID"
        if "ArticleID" in sampled.columns
        else "article_id_feature"
    )
    source_column = (
        "SourceFile"
        if "SourceFile" in sampled.columns
        else "source_file_feature"
    )
    manual = pd.DataFrame(
        {
            "news_row_id": sampled["news_row_id"].astype(int),
            "article_id": sampled.get(
                article_column,
                pd.Series([""] * len(sampled), index=sampled.index),
            ),
            "source_file": sampled.get(
                source_column,
                pd.Series([""] * len(sampled), index=sampled.index),
            ),
            "lp_text_sha256": sampled["lp_text_sha256"],
            "score_decile": sampled["score_decile"].astype(int),
            "lp_text": sampled["lp_text"],
        }
    )
    for dimension in SENTIMENT_DIMENSIONS:
        manual[f"model_{dimension}"] = sampled[
            f"model_{dimension}"
        ].to_numpy()
        manual[f"human_{dimension}"] = np.nan
    manual["reviewer_id"] = ""
    manual["reviewed_at_utc"] = ""
    manual["review_notes"] = ""
    manual_path = output_dir / "sentiment_manual_annotation_sample.csv"
    manual.to_csv(manual_path, index=False)

    repeat_size = min(int(args.repeat_sample_size), len(sampled))
    repeat_sample = sampled.sample(
        n=repeat_size,
        random_state=int(args.seed) + 1000,
        replace=False,
    ).sort_values("news_row_id")
    request_path = output_dir / "sentiment_repeat_scoring_requests.jsonl"
    result_rows = []
    with request_path.open("w", encoding="utf-8") as handle:
        for row in repeat_sample.itertuples(index=False):
            original_scores = {
                dimension: float(getattr(row, f"model_{dimension}"))
                for dimension in SENTIMENT_DIMENSIONS
            }
            for repeat_index in (1, 2):
                request = {
                    "news_row_id": int(row.news_row_id),
                    "repeat_index": repeat_index,
                    "model_id": str(frozen_manifest.get("model_id", "")),
                    "prompt_version": str(
                        frozen_manifest.get("prompt_version", "")
                    ),
                    "lp_text_sha256": str(row.lp_text_sha256),
                    "prompt": build_sun_prompt(
                        str(row.lp_text),
                        max_input_chars=int(
                            frozen_manifest.get("max_input_chars", 6000)
                        ),
                    ),
                    "original_scores": original_scores,
                }
                handle.write(json.dumps(request, ensure_ascii=True) + "\n")
                result_rows.append(
                    {
                        "news_row_id": int(row.news_row_id),
                        "repeat_index": repeat_index,
                        **{
                            f"original_{dimension}": original_scores[dimension]
                            for dimension in SENTIMENT_DIMENSIONS
                        },
                        **{
                            f"repeat_{dimension}": np.nan
                            for dimension in SENTIMENT_DIMENSIONS
                        },
                        "raw_response": "",
                        "parse_status": "",
                    }
                )
    repeat_results_path = (
        output_dir / "sentiment_repeat_scoring_results.csv"
    )
    pd.DataFrame(result_rows).to_csv(repeat_results_path, index=False)

    cache_records = sum(
        1
        for line in cache_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    audit_manifest = {
        "created_at_utc": _now(),
        "status": "pending_human_annotation_and_independent_repeat_scoring",
        "seed": int(args.seed),
        "manual_sample_size": int(len(manual)),
        "repeat_article_count": repeat_size,
        "repeat_requests_per_article": 2,
        "empty_lp_rows_excluded_from_audit": int(
            merged["lp_text"].str.strip().eq("").sum()
        ),
        "model_id": frozen_manifest.get("model_id"),
        "prompt_version": frozen_manifest.get("prompt_version"),
        "sentiment_dimensions": list(SENTIMENT_DIMENSIONS),
        "source_files": {
            "news_workbook": {
                "path": str(news_path),
                "sha256": _sha256(news_path),
            },
            "sentiment_features": {
                "path": str(feature_path),
                "sha256": _sha256(feature_path),
            },
            "sentiment_manifest": {
                "path": str(manifest_path),
                "sha256": _sha256(manifest_path),
            },
            "sentiment_cache": {
                "path": str(cache_path),
                "sha256": _sha256(cache_path),
                "record_count": cache_records,
            },
        },
        "artifacts": {
            "manual_annotation_sample": {
                "path": str(manual_path),
                "sha256": _sha256(manual_path),
            },
            "repeat_scoring_requests": {
                "path": str(request_path),
                "sha256": _sha256(request_path),
            },
            "repeat_scoring_results_template": {
                "path": str(repeat_results_path),
                "sha256": _sha256(repeat_results_path),
            },
        },
        "interpretation": (
            "ChatGPT scores are a frozen compressed representation baseline, "
            "not ground-truth sentiment labels."
        ),
    }
    _write_json(output_dir / "sentiment_audit_manifest.json", audit_manifest)
    (output_dir / "README.md").write_text(
        """# ChatGPT Sentiment Audit

This directory freezes a deterministic manual-validation sample and independent
repeat-scoring requests for the RQ2 compressed ChatGPT-score baseline.

1. Complete the `human_*` columns in
   `sentiment_manual_annotation_sample.csv` without changing model columns.
2. Score every JSONL request independently with the frozen model and prompt,
   then fill the `repeat_*`, `raw_response`, and `parse_status` columns in
   `sentiment_repeat_scoring_results.csv`.
3. Run:

```bash
conda run -n py312 python scripts/rq123/audit_sentiment_scores.py evaluate \
  --audit-dir <this-directory>
```

Until both templates are complete, RQ2 must describe these scores as a frozen
low-dimensional baseline and must not treat them as verified ground truth.
""",
        encoding="utf-8",
    )
    print(output_dir)
    return 0


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def evaluate(args: argparse.Namespace) -> int:
    audit_dir = _resolve(args.audit_dir)
    manual_path = audit_dir / "sentiment_manual_annotation_sample.csv"
    repeat_path = audit_dir / "sentiment_repeat_scoring_results.csv"
    if not manual_path.is_file() or not repeat_path.is_file():
        raise FileNotFoundError("Run the sentiment audit prepare command first.")
    manual = pd.read_csv(manual_path)
    repeat = pd.read_csv(repeat_path)
    rows: list[dict[str, Any]] = []
    complete = True
    for dimension in SENTIMENT_DIMENSIONS:
        model = pd.to_numeric(
            manual[f"model_{dimension}"],
            errors="coerce",
        )
        human = pd.to_numeric(
            manual[f"human_{dimension}"],
            errors="coerce",
        )
        valid = model.notna() & human.notna()
        complete = complete and bool(valid.all())
        rows.append(
            {
                "audit": "manual_annotation",
                "dimension": dimension,
                "n": int(valid.sum()),
                "correlation": _correlation(
                    model[valid].to_numpy(dtype=np.float64),
                    human[valid].to_numpy(dtype=np.float64),
                ),
                "mean_absolute_difference": float(
                    np.abs(model[valid] - human[valid]).mean()
                )
                if valid.any()
                else float("nan"),
            }
        )
        original = pd.to_numeric(
            repeat[f"original_{dimension}"],
            errors="coerce",
        )
        rescored = pd.to_numeric(
            repeat[f"repeat_{dimension}"],
            errors="coerce",
        )
        repeat_valid = original.notna() & rescored.notna()
        complete = complete and bool(repeat_valid.all())
        rows.append(
            {
                "audit": "independent_repeat_scoring",
                "dimension": dimension,
                "n": int(repeat_valid.sum()),
                "correlation": _correlation(
                    original[repeat_valid].to_numpy(dtype=np.float64),
                    rescored[repeat_valid].to_numpy(dtype=np.float64),
                ),
                "mean_absolute_difference": float(
                    np.abs(
                        original[repeat_valid] - rescored[repeat_valid]
                    ).mean()
                )
                if repeat_valid.any()
                else float("nan"),
            }
        )
    results = pd.DataFrame(rows)
    result_path = audit_dir / "sentiment_audit_results.csv"
    results.to_csv(result_path, index=False)
    payload = {
        "evaluated_at_utc": _now(),
        "status": "complete" if complete else "incomplete",
        "all_manual_rows_annotated": bool(
            all(
                pd.to_numeric(
                    manual[f"human_{dimension}"],
                    errors="coerce",
                ).notna().all()
                for dimension in SENTIMENT_DIMENSIONS
            )
        ),
        "all_repeat_rows_scored": bool(
            all(
                pd.to_numeric(
                    repeat[f"repeat_{dimension}"],
                    errors="coerce",
                ).notna().all()
                for dimension in SENTIMENT_DIMENSIONS
            )
        ),
        "results_csv": str(result_path),
    }
    _write_json(audit_dir / "sentiment_audit_evaluation.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if complete else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument(
        "--news-workbook",
        default=(
            "data/raw/text_embedding/"
            "news_with_openai_embeddings_large.xlsx"
        ),
    )
    prepare_parser.add_argument(
        "--sentiment-features",
        default=(
            "data/processed/text_features/rq2/20260625-075653/"
            "llm_sentiment_features.xlsx"
        ),
    )
    prepare_parser.add_argument(
        "--sentiment-manifest",
        default=(
            "data/processed/text_features/rq2/20260625-075653/"
            "llm_sentiment_manifest.json"
        ),
    )
    prepare_parser.add_argument(
        "--sentiment-cache",
        default=(
            "data/processed/text_features/rq2/20260625-075653/"
            "openai_sentiment_cache.jsonl"
        ),
    )
    prepare_parser.add_argument("--output-dir", required=True)
    prepare_parser.add_argument("--manual-sample-size", type=int, default=120)
    prepare_parser.add_argument("--repeat-sample-size", type=int, default=50)
    prepare_parser.add_argument("--seed", type=int, default=20260722)
    prepare_parser.set_defaults(func=prepare)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--audit-dir", required=True)
    evaluate_parser.set_defaults(func=evaluate)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
