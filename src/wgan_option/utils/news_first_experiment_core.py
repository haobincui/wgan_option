"""Hash-bound pair-text overlays and resumable news-first training state.

The contracts in this module are deliberately opt-in.  Historical configs keep
their row-level loader and weights-only checkpoint behavior, while the RQ1--RQ3
rolling experiment can require one pair per market transition and exact
optimizer/RNG continuation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections import Counter
import ast
import hashlib
import json
import os
from pathlib import Path
import random
import re
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .merged_xlsx_types import VolSurfaceSample

NO_PAIR_TEXT_OVERLAY = "none"
PAIR_TEXT_OVERLAY_MODES = frozenset(
    {
        NO_PAIR_TEXT_OVERLAY,
        "current_only",
        "lp_mean_l2",
        "lp_shuffle",
        "bow1024",
        "sentiment_pad1024",
    }
)
PAIR_TEXT_MANIFEST_KIND = "pair_text_overlay_manifest_v1"
PAIR_TEXT_MANIFEST_SCHEMA_VERSION = 1

NO_FULL_TRAINING_STATE = "none"
SAVE_DYNAMIC_FULL_TRAINING_STATE = "save_dynamic_v1"
RESUME_DYNAMIC_FULL_TRAINING_STATE = "resume_dynamic_v1"
RESUME_FROZEN_LR_FULL_TRAINING_STATE = "resume_frozen_lr_replay_v1"
FULL_TRAINING_STATE_MODES = frozenset(
    {
        NO_FULL_TRAINING_STATE,
        SAVE_DYNAMIC_FULL_TRAINING_STATE,
        RESUME_DYNAMIC_FULL_TRAINING_STATE,
        RESUME_FROZEN_LR_FULL_TRAINING_STATE,
    }
)
FULL_TRAINING_STATE_KIND = "news_first_wgan_full_training_state_v1"
FULL_TRAINING_STATE_SCHEMA_VERSION = 1
FULL_TRAINING_STATE_CONTRACT_KIND = "full_training_state_contract_v1"
FULL_TRAINING_STATE_PHASE = "end_of_epoch_after_scheduler_step"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_REQUIRED_STATE_LINEAGE = (
    "fold_id",
    "seed",
    "arm",
    "model_contract_sha256",
    "grid_sha256",
    "training_config_payload_sha256",
    "code_sha256",
    "dataset_sha256",
    "support_sha256",
    "pair_universe_sha256",
    "text_manifest_sha256",
    "job_sha256",
)


def normalize_pair_text_overlay_mode(value: object) -> str:
    """Return one versioned pair-text representation slug."""

    mode = str(value or NO_PAIR_TEXT_OVERLAY).strip().lower().replace("-", "_")
    aliases = {"lp": "lp_mean_l2", "bow": "bow1024", "sentiment": "sentiment_pad1024"}
    mode = aliases.get(mode, mode)
    if mode not in PAIR_TEXT_OVERLAY_MODES:
        raise ValueError(
            "news_first_pair_text_overlay_mode must be one of "
            f"{sorted(PAIR_TEXT_OVERLAY_MODES)}, got {value!r}"
        )
    return mode


def normalize_full_training_state_mode(value: object) -> str:
    """Return one opt-in full-state persistence mode."""

    mode = str(value or NO_FULL_TRAINING_STATE).strip().lower()
    if mode not in FULL_TRAINING_STATE_MODES:
        raise ValueError(
            "news_first_full_training_state_mode must be one of "
            f"{sorted(FULL_TRAINING_STATE_MODES)}, got {value!r}"
        )
    return mode


def sha256_file(path: str | Path) -> str:
    """Hash a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def pair_universe_sha256(pair_ids: Sequence[object]) -> str:
    """Hash a non-empty, unique sorted pair universe."""

    normalized = [str(value).strip() for value in pair_ids]
    if not normalized or any(not value for value in normalized):
        raise ValueError("pair universe must contain non-empty pair IDs")
    if len(normalized) != len(set(normalized)):
        raise ValueError("pair universe contains duplicate pair IDs")
    return hashlib.sha256(_canonical_json_bytes(sorted(normalized))).hexdigest()


def training_config_payload_sha256(config_payload: Mapping[str, object]) -> str:
    """Hash config values without introducing a contract-SHA dependency cycle.

    The two full-state contract locator fields are normalized to empty strings
    before hashing.  The resolved config file itself may then contain the
    resulting contract path/SHA and receive its own separate file digest.
    """

    canonical = dict(config_payload)
    canonical["news_first_full_training_state_contract_path"] = ""
    canonical["news_first_full_training_state_contract_sha256"] = ""
    return hashlib.sha256(_canonical_json_bytes(canonical)).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    digest = str(value or "").strip().lower()
    if _SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"{label} must be a lowercase 64-character SHA256 digest")
    return digest


def _read_hash_bound_json(
    path: str | Path, expected_sha256: str, *, label: str
) -> Mapping[str, Any]:
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    expected = _require_sha256(expected_sha256, label=f"{label} SHA256")
    actual = sha256_file(resolved)
    if actual != expected:
        raise ValueError(f"{label} SHA256 mismatch: expected {expected}, got {actual}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be valid UTF-8 JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} root must be a JSON object")
    return payload


def _atomic_json_write(path: str | Path, payload: Mapping[str, object]) -> str:
    resolved = Path(path).expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_json_bytes(payload) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.", suffix=".tmp", dir=resolved.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, resolved)
        directory_descriptor = os.open(resolved.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return sha256_file(resolved)


@dataclass(frozen=True)
class PairTextOverlayManifest:
    """Validated in-memory form of a pair-level text overlay."""

    mode: str
    namespace: str
    embedding_dim: int
    pair_universe_sha256: str
    profile_sha256: str
    file_sha256: str
    embeddings: Mapping[str, np.ndarray]
    sessions: Mapping[str, str]
    donors: Mapping[str, str]
    transform: Mapping[str, object]


def write_pair_text_overlay_manifest(
    path: str | Path,
    *,
    mode: object,
    namespace: str,
    records: Sequence[Mapping[str, object]],
    transform: Mapping[str, object] | None = None,
) -> dict[str, str]:
    """Write a canonical, self-hashed pair-text manifest for an orchestrator."""

    normalized_mode = normalize_pair_text_overlay_mode(mode)
    if normalized_mode == NO_PAIR_TEXT_OVERLAY:
        raise ValueError("Cannot write a manifest for pair-text overlay mode 'none'")
    normalized_namespace = str(namespace).strip()
    if not normalized_namespace:
        raise ValueError("Pair-text manifest namespace must be non-empty")
    canonical_records: list[dict[str, object]] = []
    seen: set[str] = set()
    donor_by_pair: dict[str, str] = {}
    for index, raw_record in enumerate(records):
        pair_id = str(raw_record.get("pair_id", "")).strip()
        session_id = str(raw_record.get("session_id", "")).strip()
        if not pair_id or not session_id or pair_id in seen:
            raise ValueError(
                "Pair-text records require unique non-empty pair_id and session_id; "
                f"row={index}, pair_id={pair_id!r}"
            )
        seen.add(pair_id)
        vector = _normalized_embedding(
            raw_record.get("embedding"),
            mode=normalized_mode,
            pair_id=pair_id,
            embedding_dim=1024,
        )
        donor = str(raw_record.get("donor_pair_id", "")).strip()
        if normalized_mode == "lp_shuffle":
            if not donor:
                raise ValueError("lp_shuffle records require donor_pair_id")
            donor_by_pair[pair_id] = donor
        elif donor:
            raise ValueError(
                f"{normalized_mode} records must not declare donor_pair_id"
            )
        canonical_records.append(
            {
                "pair_id": pair_id,
                "session_id": session_id,
                "embedding": [float(value) for value in vector.tolist()],
                **({"donor_pair_id": donor} if donor else {}),
            }
        )
    pair_ids = sorted(seen)
    if not pair_ids:
        raise ValueError("Pair-text manifest records must be non-empty")
    if normalized_mode == "lp_shuffle":
        if set(donor_by_pair.values()) != set(pair_ids) or any(
            pair_id == donor for pair_id, donor in donor_by_pair.items()
        ):
            raise ValueError(
                "lp_shuffle donor mapping must be a fixed-point-free pair permutation"
            )
    profile = {
        "schema_version": PAIR_TEXT_MANIFEST_SCHEMA_VERSION,
        "kind": PAIR_TEXT_MANIFEST_KIND,
        "mode": normalized_mode,
        "namespace": normalized_namespace,
        "embedding_dim": 1024,
        "pair_universe_sha256": pair_universe_sha256(pair_ids),
        "records": sorted(canonical_records, key=lambda row: str(row["pair_id"])),
        **({"transform": dict(transform)} if transform is not None else {}),
    }
    _canonical_json_bytes(profile)
    profile_sha = hashlib.sha256(_canonical_json_bytes(profile)).hexdigest()
    payload = {**profile, "profile_sha256": profile_sha}
    file_sha = _atomic_json_write(path, payload)
    return {
        "manifest_path": str(Path(path).expanduser()),
        "manifest_sha256": file_sha,
        "profile_sha256": profile_sha,
        "pair_universe_sha256": str(profile["pair_universe_sha256"]),
    }


def _parsed_vector(value: object, *, dimension: int, label: str) -> np.ndarray:
    if isinstance(value, str):
        try:
            raw = json.loads(value)
        except json.JSONDecodeError:
            try:
                raw = ast.literal_eval(value)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(f"{label} is not a valid vector") from exc
    else:
        raw = value
    try:
        vector = np.asarray(raw, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not a numeric vector") from exc
    if vector.shape != (dimension,) or not np.isfinite(vector).all():
        raise ValueError(
            f"{label} must have finite shape ({dimension},), got {vector.shape}"
        )
    return vector


def _normalized_article_key(row: object) -> str:
    for field_name in ("article_id", "sample_id", "news_row_id"):
        value = getattr(row, field_name, "")
        if pd.notna(value) and str(value).strip():
            return f"{field_name}:{str(value).strip()}"
    raise ValueError("News row has no article_id, sample_id, or news_row_id")


def _l2(vector: np.ndarray) -> np.ndarray:
    output = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(output))
    return output.copy() if norm == 0.0 else (output / norm).astype(np.float32)


def _normalized_article_text(value: object) -> str:
    """Return canonical article text without turning missing values into tokens."""

    if value is None or bool(pd.isna(value)):
        return ""
    return str(value).strip()


def _ngram_counts(text: object) -> Counter[str]:
    normalized = _normalized_article_text(text)
    tokens = re.findall(r"(?u)\b\w\w+\b", normalized.lower())
    counts: Counter[str] = Counter(tokens)
    counts.update(f"{first} {second}" for first, second in zip(tokens, tokens[1:]))
    return counts


def _fixed_partition_derangement(
    pair_ids: Sequence[str],
    *,
    master_seed: int,
    namespace: str,
) -> dict[str, str]:
    receivers = sorted(str(pair_id) for pair_id in pair_ids)
    if len(receivers) < 2:
        raise ValueError(f"lp_shuffle requires at least two pairs in {namespace}")
    derived_seed = int.from_bytes(
        hashlib.sha256(
            _canonical_json_bytes(
                {
                    "method": "split_local_pair_derangement_v1",
                    "master_seed": int(master_seed),
                    "namespace": str(namespace),
                    "pair_universe_sha256": pair_universe_sha256(receivers),
                }
            )
        ).digest()[:8],
        byteorder="big",
        signed=False,
    )
    rng = np.random.default_rng(derived_seed)
    randomized = [receivers[int(index)] for index in rng.permutation(len(receivers))]
    for offset in range(len(randomized)):
        donors = randomized[offset:] + randomized[:offset]
        if all(receiver != donor for receiver, donor in zip(receivers, donors)):
            return dict(zip(receivers, donors))
    raise RuntimeError(f"Unable to construct deterministic derangement for {namespace}")


def build_pair_text_overlay_manifests(
    config: Mapping[str, Any],
    pair_universe_path: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    """Build all 40 fold/arm pair overlays for the unified RQ1--RQ3 run.

    BoW vocabulary and sentiment standardization are fitted only on each
    fold's training pairs.  Validation pairs are transformed with the frozen
    train fit.  The returned manifests cover train+validation exactly; test
    pairs remain unopened until the experiment's evaluation freeze.
    """

    if not isinstance(config, Mapping):
        raise ValueError("Unified experiment config must be a mapping")
    data = config.get("data")
    matrix = config.get("matrix")
    folds = config.get("folds")
    if (
        not isinstance(data, Mapping)
        or not isinstance(matrix, Mapping)
        or not isinstance(folds, list)
    ):
        raise ValueError("Unified config requires data, matrix, and folds sections")
    universe_path = Path(pair_universe_path).expanduser()
    universe = pd.read_csv(universe_path, dtype=str, keep_default_na=False)
    required_universe_columns = {
        "tolerance_minutes",
        "fold",
        "partition",
        "pair_id",
        "session_id",
        "pair_universe_sha256",
    }
    missing_columns = sorted(required_universe_columns - set(universe.columns))
    if missing_columns:
        raise ValueError(f"Pair-universe CSV is missing columns: {missing_columns}")
    if (
        universe.empty
        or universe.duplicated(
            ["tolerance_minutes", "fold", "partition", "pair_id"]
        ).any()
    ):
        raise ValueError("Pair-universe CSV is empty or has duplicate partition pairs")
    development_universe = universe.loc[
        universe["partition"].isin(["train", "validation"])
    ].copy()
    if development_universe.empty:
        raise ValueError("Pair-universe CSV contains no train/validation pairs")

    root = Path(str(data["root"])).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    sentiment_path = Path(str(data["sentiment_workbook_path"])).expanduser()
    if not sentiment_path.is_absolute():
        sentiment_path = Path.cwd() / sentiment_path
    sentiment_frame = pd.read_excel(
        sentiment_path,
        sheet_name="features",
        usecols=["news_row_id", "sentiment_embedding"],
    )
    # Keep the raw sentiment cells opaque until the train/validation news
    # universe is known.  In particular, malformed test-only values must not
    # be parsed during prepare.
    sentiment_frame["_news_key"] = sentiment_frame["news_row_id"].map(str)

    output_root = Path(output_dir).expanduser()
    tolerance_values = [int(value) for value in matrix.get("tolerances_minutes", [])]
    fold_ids = [str(row.get("id", "")).strip() for row in folds]
    shuffle_seed = int(matrix.get("shuffle_seed", 0))
    generated: list[Path] = []
    arm_modes = {
        "parent_current_only": "current_only",
        "continuation_no_text": "current_only",
        "lp_matched": "lp_mean_l2",
        "lp_shuffle": "lp_shuffle",
        "bow": "bow1024",
        "sentiment": "sentiment_pad1024",
    }

    for tolerance in tolerance_values:
        workbook = root / str(data["workbook_template"]).format(
            tolerance02=f"{tolerance:02d}"
        )
        frame = pd.read_excel(
            workbook,
            sheet_name=str(data["sheet_name"]),
            usecols=[
                "pair_id",
                "session_id",
                "article_id",
                "sample_id",
                "news_row_id",
                "lp_text",
                "lp_embedding",
            ],
        )
        frame["pair_id"] = frame["pair_id"].astype(str)
        frame["session_id"] = frame["session_id"].astype(str)
        required_pairs_for_tolerance = set(
            development_universe.loc[
                development_universe["tolerance_minutes"].astype(int) == tolerance,
                "pair_id",
            ]
        )
        frame = frame.loc[frame["pair_id"].isin(required_pairs_for_tolerance)].copy()
        if set(frame["pair_id"]) != required_pairs_for_tolerance:
            raise ValueError(
                f"Workbook/pair-universe coverage mismatch for {tolerance}m"
            )
        required_news_keys = set(frame["news_row_id"].map(str))
        selected_sentiment = sentiment_frame.loc[
            sentiment_frame["_news_key"].isin(required_news_keys)
        ].copy()
        if selected_sentiment["_news_key"].duplicated().any():
            raise ValueError(
                "Sentiment workbook contains duplicate train/validation "
                "news_row_id values"
            )
        sentiment_by_news = {
            str(row["_news_key"]): _parsed_vector(
                row["sentiment_embedding"],
                dimension=1024,
                label=f"sentiment news_row_id={row['news_row_id']}",
            )[:3]
            for row in selected_sentiment.to_dict(orient="records")
        }
        missing_sentiment = sorted(required_news_keys - set(sentiment_by_news))
        if missing_sentiment:
            raise ValueError(
                "Sentiment coverage is missing train/validation news rows: "
                f"{missing_sentiment[:5]}"
            )
        rows_by_pair: dict[str, list[dict[str, object]]] = {}
        dedup_audit_by_pair: dict[str, dict[str, float | int]] = {}
        for pair_id, pair_rows in frame.groupby("pair_id", sort=False):
            articles: dict[str, dict[str, object]] = {}
            dedup_audit: dict[str, float | int] = {
                "duplicate_article_row_count": 0,
                "lp_embedding_conflict_count": 0,
                "lp_embedding_conflict_max_abs": 0.0,
                "sentiment_conflict_count": 0,
                "sentiment_conflict_max_abs": 0.0,
            }
            pair_rows = pair_rows.assign(
                _news_row_sort=pd.to_numeric(pair_rows["news_row_id"], errors="raise")
            ).sort_values(["_news_row_sort", "sample_id"], kind="stable")
            for row in pair_rows.itertuples(index=False):
                article_key = _normalized_article_key(row)
                news_key = str(row.news_row_id)
                if news_key not in sentiment_by_news:
                    raise ValueError(
                        f"Sentiment coverage missing news_row_id={news_key}, pair={pair_id}"
                    )
                article = {
                    "article_key": article_key,
                    "lp_text": _normalized_article_text(row.lp_text),
                    "lp_embedding": _parsed_vector(
                        row.lp_embedding,
                        dimension=1024,
                        label=f"LP pair={pair_id}, article={article_key}",
                    ),
                    "sentiment": sentiment_by_news[news_key],
                }
                previous = articles.get(article_key)
                if previous is not None:
                    dedup_audit["duplicate_article_row_count"] += 1
                    if previous["lp_text"] != article["lp_text"]:
                        raise ValueError(
                            "Article ID maps to different text within a pair: "
                            f"{pair_id}/{article_key}"
                        )
                    lp_difference = float(
                        np.max(
                            np.abs(
                                np.asarray(previous["lp_embedding"], dtype=np.float32)
                                - np.asarray(article["lp_embedding"], dtype=np.float32)
                            )
                        )
                    )
                    if lp_difference > 0.0:
                        dedup_audit["lp_embedding_conflict_count"] += 1
                        dedup_audit["lp_embedding_conflict_max_abs"] = max(
                            float(dedup_audit["lp_embedding_conflict_max_abs"]),
                            lp_difference,
                        )
                    sentiment_difference = float(
                        np.max(
                            np.abs(
                                np.asarray(previous["sentiment"], dtype=np.float32)
                                - np.asarray(article["sentiment"], dtype=np.float32)
                            )
                        )
                    )
                    if sentiment_difference > 0.0:
                        dedup_audit["sentiment_conflict_count"] += 1
                        dedup_audit["sentiment_conflict_max_abs"] = max(
                            float(dedup_audit["sentiment_conflict_max_abs"]),
                            sentiment_difference,
                        )
                    continue
                articles[article_key] = article
            if not articles:
                raise ValueError(f"Pair has no usable articles: {pair_id}")
            rows_by_pair[str(pair_id)] = [articles[key] for key in sorted(articles)]
            dedup_audit_by_pair[str(pair_id)] = dedup_audit

        lp_by_pair = {
            pair_id: _l2(
                np.mean(
                    np.stack([article["lp_embedding"] for article in articles], axis=0),
                    axis=0,
                )
            )
            for pair_id, articles in rows_by_pair.items()
        }
        raw_sentiment_by_pair = {
            pair_id: np.mean(
                np.stack([article["sentiment"] for article in articles], axis=0),
                axis=0,
            ).astype(np.float32)
            for pair_id, articles in rows_by_pair.items()
        }

        for fold_id in fold_ids:
            fold_universe = development_universe.loc[
                (development_universe["tolerance_minutes"].astype(int) == tolerance)
                & (development_universe["fold"] == fold_id)
            ].copy()
            if set(fold_universe["partition"]) != {"train", "validation"}:
                raise ValueError(
                    f"Missing train/validation universe: {tolerance}m {fold_id}"
                )
            pair_partition: dict[str, str] = dict(
                zip(fold_universe["pair_id"], fold_universe["partition"])
            )
            session_by_pair: dict[str, str] = dict(
                zip(fold_universe["pair_id"], fold_universe["session_id"])
            )
            selected_pairs = sorted(pair_partition)
            if set(selected_pairs) - set(rows_by_pair):
                raise ValueError(
                    f"Workbook lacks frozen pair universe: {tolerance}m {fold_id}"
                )
            for partition in ("train", "validation"):
                partition_rows = fold_universe.loc[
                    fold_universe["partition"] == partition
                ]
                ids = sorted(partition_rows["pair_id"])
                actual_sha = pair_universe_sha256(ids)
                declared = set(partition_rows["pair_universe_sha256"])
                if declared != {actual_sha}:
                    raise ValueError(
                        f"Pair-universe SHA drift: {tolerance}m {fold_id} {partition}"
                    )
            train_pairs = sorted(
                pair_id
                for pair_id, partition in pair_partition.items()
                if partition == "train"
            )
            validation_pairs = sorted(
                pair_id
                for pair_id, partition in pair_partition.items()
                if partition == "validation"
            )
            train_sha = pair_universe_sha256(train_pairs)
            validation_sha = pair_universe_sha256(validation_pairs)

            vocabulary_counts: Counter[str] = Counter()
            for pair_id in train_pairs:
                for article in rows_by_pair[pair_id]:
                    vocabulary_counts.update(_ngram_counts(article["lp_text"]))
            vocabulary = [
                term
                for term, _ in sorted(
                    vocabulary_counts.items(), key=lambda item: (-item[1], item[0])
                )[:1024]
            ]
            vocabulary_index = {term: index for index, term in enumerate(vocabulary)}
            bow_by_pair: dict[str, np.ndarray] = {}
            for pair_id in selected_pairs:
                counts: Counter[str] = Counter()
                for article in rows_by_pair[pair_id]:
                    counts.update(_ngram_counts(article["lp_text"]))
                vector = np.zeros(1024, dtype=np.float32)
                for term, count in counts.items():
                    index = vocabulary_index.get(term)
                    if index is not None:
                        vector[index] = np.log1p(float(count))
                bow_by_pair[pair_id] = _l2(vector)

            sentiment_train = np.stack(
                [raw_sentiment_by_pair[pair_id] for pair_id in train_pairs], axis=0
            )
            sentiment_mean = sentiment_train.mean(axis=0)
            sentiment_std = sentiment_train.std(axis=0, ddof=0)
            sentiment_std = np.where(sentiment_std == 0.0, 1.0, sentiment_std)
            sentiment_by_pair: dict[str, np.ndarray] = {}
            for pair_id in selected_pairs:
                vector = np.zeros(1024, dtype=np.float32)
                vector[:3] = (
                    (raw_sentiment_by_pair[pair_id] - sentiment_mean) / sentiment_std
                ).astype(np.float32)
                sentiment_by_pair[pair_id] = vector

            arms = [
                "parent_current_only",
                "continuation_no_text",
                "lp_matched",
                "lp_shuffle",
                *(["bow", "sentiment"] if tolerance == 5 else []),
            ]
            for arm in arms:
                mode = arm_modes[arm]
                namespace = (
                    f"tol{tolerance:02d}/{fold_id}/{arm}/"
                    f"train_{train_sha}/validation_{validation_sha}"
                )
                donor_by_pair: dict[str, str] = {}
                if arm == "lp_shuffle":
                    for partition, pair_ids in (
                        ("train", train_pairs),
                        ("validation", validation_pairs),
                    ):
                        donor_by_pair.update(
                            _fixed_partition_derangement(
                                pair_ids,
                                master_seed=shuffle_seed,
                                namespace=f"{namespace}/{partition}",
                            )
                        )
                transform: dict[str, object] = {
                    "article_deduplication": "article_id_then_sample_id_v1",
                    "article_deduplication_selection_rule": "minimum_numeric_news_row_id_then_sample_id_v1",
                    "duplicate_article_row_count_train": sum(
                        int(dedup_audit_by_pair[pair_id]["duplicate_article_row_count"])
                        for pair_id in train_pairs
                    ),
                    "duplicate_article_row_count_train_validation": sum(
                        int(dedup_audit_by_pair[pair_id]["duplicate_article_row_count"])
                        for pair_id in selected_pairs
                    ),
                    "duplicate_lp_embedding_conflict_count_train": sum(
                        int(dedup_audit_by_pair[pair_id]["lp_embedding_conflict_count"])
                        for pair_id in train_pairs
                    ),
                    "duplicate_lp_embedding_conflict_count_train_validation": sum(
                        int(dedup_audit_by_pair[pair_id]["lp_embedding_conflict_count"])
                        for pair_id in selected_pairs
                    ),
                    "duplicate_lp_embedding_conflict_max_abs_train_validation": max(
                        (
                            float(
                                dedup_audit_by_pair[pair_id][
                                    "lp_embedding_conflict_max_abs"
                                ]
                            )
                            for pair_id in selected_pairs
                        ),
                        default=0.0,
                    ),
                    "duplicate_sentiment_conflict_count_train_validation": sum(
                        int(dedup_audit_by_pair[pair_id]["sentiment_conflict_count"])
                        for pair_id in selected_pairs
                    ),
                    "duplicate_sentiment_conflict_max_abs_train_validation": max(
                        (
                            float(
                                dedup_audit_by_pair[pair_id][
                                    "sentiment_conflict_max_abs"
                                ]
                            )
                            for pair_id in selected_pairs
                        ),
                        default=0.0,
                    ),
                    "train_pair_universe_sha256": train_sha,
                    "validation_pair_universe_sha256": validation_sha,
                }
                if arm in {"parent_current_only", "continuation_no_text"}:
                    vectors = {
                        pair_id: np.zeros(1024, dtype=np.float32)
                        for pair_id in selected_pairs
                    }
                    transform["method"] = "zero_vector_v1"
                elif arm == "lp_matched":
                    vectors = {
                        pair_id: lp_by_pair[pair_id] for pair_id in selected_pairs
                    }
                    transform["method"] = "unique_article_lp_mean_l2_v1"
                elif arm == "lp_shuffle":
                    vectors = {
                        pair_id: lp_by_pair[donor_by_pair[pair_id]]
                        for pair_id in selected_pairs
                    }
                    transform.update(
                        {
                            "method": "split_local_pair_derangement_v1",
                            "master_seed": shuffle_seed,
                            "mapping_sha256": hashlib.sha256(
                                _canonical_json_bytes(sorted(donor_by_pair.items()))
                            ).hexdigest(),
                        }
                    )
                elif arm == "bow":
                    vectors = bow_by_pair
                    missing_train_articles = sum(
                        not bool(article["lp_text"])
                        for pair_id in train_pairs
                        for article in rows_by_pair[pair_id]
                    )
                    missing_selected_articles = sum(
                        not bool(article["lp_text"])
                        for pair_id in selected_pairs
                        for article in rows_by_pair[pair_id]
                    )
                    transform.update(
                        {
                            "method": "train_pair_unigram_bigram_count_sum_log1p_l2_v1",
                            "vocabulary": vocabulary,
                            "vocabulary_size": len(vocabulary),
                            "missing_text_article_count_train": missing_train_articles,
                            "missing_text_pair_count_train": sum(
                                all(
                                    not bool(article["lp_text"])
                                    for article in rows_by_pair[pair_id]
                                )
                                for pair_id in train_pairs
                            ),
                            "missing_text_article_count_train_validation": missing_selected_articles,
                            "missing_text_pair_count_train_validation": sum(
                                all(
                                    not bool(article["lp_text"])
                                    for article in rows_by_pair[pair_id]
                                )
                                for pair_id in selected_pairs
                            ),
                        }
                    )
                else:
                    vectors = sentiment_by_pair
                    transform.update(
                        {
                            "method": "train_pair_sentiment_mean_zscore_pad1024_v1",
                            "train_mean": [float(value) for value in sentiment_mean],
                            "train_std": [float(value) for value in sentiment_std],
                        }
                    )
                records = [
                    {
                        "pair_id": pair_id,
                        "session_id": session_by_pair[pair_id],
                        "embedding": vectors[pair_id],
                        **(
                            {"donor_pair_id": donor_by_pair[pair_id]}
                            if arm == "lp_shuffle"
                            else {}
                        ),
                    }
                    for pair_id in selected_pairs
                ]
                path = (
                    output_root
                    / f"tolerance_{tolerance:02d}m"
                    / fold_id
                    / f"{arm}.json"
                )
                write_pair_text_overlay_manifest(
                    path,
                    mode=mode,
                    namespace=namespace,
                    records=records,
                    transform=transform,
                )
                generated.append(path)

    expected_count = sum(
        6 if tolerance == 5 else 4 for tolerance in tolerance_values
    ) * len(fold_ids)
    if len(generated) != expected_count or len(set(generated)) != expected_count:
        raise RuntimeError(
            f"Pair-text overlay generation count mismatch: {len(generated)} != {expected_count}"
        )
    return generated


def _normalized_embedding(
    raw: object,
    *,
    mode: str,
    pair_id: str,
    embedding_dim: int,
) -> np.ndarray:
    if mode == "current_only" and raw is None:
        return np.zeros(embedding_dim, dtype=np.float32)
    try:
        vector = np.asarray(raw, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid embedding for pair_id={pair_id!r}") from exc
    if vector.shape != (embedding_dim,) or not np.isfinite(vector).all():
        raise ValueError(
            f"Embedding for pair_id={pair_id!r} must be finite shape "
            f"({embedding_dim},), got {vector.shape}"
        )
    if mode == "current_only":
        if np.count_nonzero(vector):
            raise ValueError("current_only manifest embeddings must be exactly zero")
    elif mode in {"lp_mean_l2", "lp_shuffle", "bow1024"}:
        norm = float(np.linalg.norm(vector))
        if norm != 0.0 and not np.isclose(norm, 1.0, atol=1e-5, rtol=1e-5):
            raise ValueError(
                f"{mode} embedding must be zero or L2-normalized; "
                f"pair_id={pair_id!r}, norm={norm}"
            )
    elif mode == "sentiment_pad1024" and np.count_nonzero(vector[3:]):
        raise ValueError(
            "sentiment_pad1024 requires every coordinate after the first three "
            f"to be exactly zero; pair_id={pair_id!r}"
        )
    return vector.copy()


def load_pair_text_overlay_manifest(
    path: str | Path,
    expected_sha256: str,
    expected_profile_sha256: str,
    *,
    expected_mode: object,
) -> PairTextOverlayManifest:
    """Load a strict one-row-per-pair text overlay manifest."""

    payload = _read_hash_bound_json(path, expected_sha256, label="Pair-text manifest")
    if int(payload.get("schema_version", -1)) != PAIR_TEXT_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Pair-text manifest schema_version must be 1")
    if str(payload.get("kind", "")) != PAIR_TEXT_MANIFEST_KIND:
        raise ValueError(f"Pair-text manifest kind must be {PAIR_TEXT_MANIFEST_KIND!r}")
    mode = normalize_pair_text_overlay_mode(payload.get("mode"))
    configured_mode = normalize_pair_text_overlay_mode(expected_mode)
    if mode == NO_PAIR_TEXT_OVERLAY or mode != configured_mode:
        raise ValueError(
            f"Pair-text manifest mode mismatch: manifest={mode!r}, config={configured_mode!r}"
        )
    namespace = str(payload.get("namespace", "")).strip()
    if not namespace:
        raise ValueError("Pair-text manifest namespace must be non-empty")
    embedding_dim = int(payload.get("embedding_dim", 0))
    if embedding_dim != 1024:
        raise ValueError("Pair-text manifest embedding_dim must be exactly 1024")
    raw_records = payload.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("Pair-text manifest records must be a non-empty list")

    embeddings: dict[str, np.ndarray] = {}
    sessions: dict[str, str] = {}
    donors: dict[str, str] = {}
    canonical_records: list[dict[str, object]] = []
    for index, raw_record in enumerate(raw_records):
        if not isinstance(raw_record, Mapping):
            raise ValueError(f"Pair-text record {index} must be an object")
        pair_id = str(raw_record.get("pair_id", "")).strip()
        session_id = str(raw_record.get("session_id", "")).strip()
        if not pair_id or not session_id or pair_id in embeddings:
            raise ValueError(
                "Pair-text records require unique non-empty pair_id and session_id; "
                f"row={index}, pair_id={pair_id!r}"
            )
        vector = _normalized_embedding(
            raw_record.get("embedding"),
            mode=mode,
            pair_id=pair_id,
            embedding_dim=embedding_dim,
        )
        donor = str(raw_record.get("donor_pair_id", "")).strip()
        if mode == "lp_shuffle" and not donor:
            raise ValueError("lp_shuffle records require donor_pair_id")
        if mode != "lp_shuffle" and donor:
            raise ValueError(f"{mode} records must not declare donor_pair_id")
        embeddings[pair_id] = vector
        sessions[pair_id] = session_id
        if donor:
            donors[pair_id] = donor
        canonical_records.append(
            {
                "pair_id": pair_id,
                "session_id": session_id,
                "embedding": [float(value) for value in vector.tolist()],
                **({"donor_pair_id": donor} if donor else {}),
            }
        )

    pair_ids = sorted(embeddings)
    universe_sha = pair_universe_sha256(pair_ids)
    declared_universe_sha = _require_sha256(
        payload.get("pair_universe_sha256"), label="pair_universe_sha256"
    )
    if universe_sha != declared_universe_sha:
        raise ValueError(
            "Pair-text pair_universe_sha256 mismatch: "
            f"declared={declared_universe_sha}, actual={universe_sha}"
        )
    if mode == "lp_shuffle":
        if set(donors) != set(pair_ids) or set(donors.values()) != set(pair_ids):
            raise ValueError(
                "lp_shuffle donors must be a permutation of the pair universe"
            )
        fixed = sorted(pair for pair, donor in donors.items() if pair == donor)
        if fixed:
            raise ValueError(f"lp_shuffle donor mapping has fixed pairs: {fixed[:5]}")

    canonical_profile = {
        "schema_version": PAIR_TEXT_MANIFEST_SCHEMA_VERSION,
        "kind": PAIR_TEXT_MANIFEST_KIND,
        "mode": mode,
        "namespace": namespace,
        "embedding_dim": embedding_dim,
        "pair_universe_sha256": universe_sha,
        "records": sorted(canonical_records, key=lambda row: str(row["pair_id"])),
        **(
            {"transform": dict(payload["transform"])}
            if isinstance(payload.get("transform"), Mapping)
            else {}
        ),
    }
    if "transform" in payload and not isinstance(payload.get("transform"), Mapping):
        raise ValueError("Pair-text manifest transform must be an object")
    actual_profile_sha = hashlib.sha256(
        _canonical_json_bytes(canonical_profile)
    ).hexdigest()
    declared_profile_sha = _require_sha256(
        payload.get("profile_sha256"), label="Pair-text manifest profile_sha256"
    )
    configured_profile_sha = _require_sha256(
        expected_profile_sha256,
        label="news_first_pair_text_profile_sha256",
    )
    if (
        actual_profile_sha != declared_profile_sha
        or actual_profile_sha != configured_profile_sha
    ):
        raise ValueError(
            "Pair-text profile SHA256 mismatch: "
            f"actual={actual_profile_sha}, manifest={declared_profile_sha}, "
            f"config={configured_profile_sha}"
        )
    return PairTextOverlayManifest(
        mode=mode,
        namespace=namespace,
        embedding_dim=embedding_dim,
        pair_universe_sha256=universe_sha,
        profile_sha256=actual_profile_sha,
        file_sha256=sha256_file(path),
        embeddings=embeddings,
        sessions=sessions,
        donors=donors,
        transform=(
            dict(payload["transform"])
            if isinstance(payload.get("transform"), Mapping)
            else {}
        ),
    )


def _pair_id(sample: VolSurfaceSample) -> str:
    pair_id = str(sample.pair_id).strip()
    if not pair_id:
        raise ValueError(f"Pair-level overlay requires pair_id: {sample.sample_id}")
    return pair_id


def _equal_optional_array(first: np.ndarray | None, second: np.ndarray | None) -> bool:
    if first is None or second is None:
        return first is second
    return np.array_equal(first, second)


def _validate_pair_rows(pair_id: str, rows: Sequence[VolSurfaceSample]) -> None:
    reference = rows[0]
    scalar_fields = (
        "session_id",
        "effective_origin_utc",
        "current_snapshot_time_utc",
        "target_snapshot_time_utc",
        "surface_shape",
        "support_grid_fingerprint",
        "support_mask_fingerprint",
        "current_support_mask_fingerprint",
    )
    array_fields = (
        "current_surface",
        "target_surface",
        "strike_grid",
        "maturity_grid_days",
    )
    for row in rows[1:]:
        for field_name in scalar_fields:
            if getattr(row, field_name) != getattr(reference, field_name):
                raise ValueError(
                    f"Pair {pair_id!r} has inconsistent {field_name} across news rows"
                )
        for field_name in array_fields:
            if not np.array_equal(
                getattr(row, field_name), getattr(reference, field_name)
            ):
                raise ValueError(
                    f"Pair {pair_id!r} has inconsistent {field_name} across news rows"
                )
        if not _equal_optional_array(row.support_mask, reference.support_mask):
            raise ValueError(f"Pair {pair_id!r} has inconsistent support_mask")
        if not _equal_optional_array(
            row.current_support_mask, reference.current_support_mask
        ):
            raise ValueError(f"Pair {pair_id!r} has inconsistent current_support_mask")


def apply_pair_text_overlay(
    items: Sequence[VolSurfaceSample],
    manifest: PairTextOverlayManifest,
    *,
    require_exact_universe: bool = True,
) -> tuple[list[VolSurfaceSample], dict[str, object]]:
    """Collapse news rows to market pairs and overlay a frozen 1024-vector."""

    if not items:
        raise ValueError("Cannot apply pair-text overlay to an empty sample collection")
    grouped: dict[str, list[VolSurfaceSample]] = {}
    for sample in items:
        grouped.setdefault(_pair_id(sample), []).append(sample)
    selected_pairs = set(grouped)
    manifest_pairs = set(manifest.embeddings)
    missing = sorted(selected_pairs - manifest_pairs)
    extra = sorted(manifest_pairs - selected_pairs) if require_exact_universe else []
    if missing or extra:
        raise ValueError(
            "Pair-text manifest universe differs from selected samples: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    output: list[VolSurfaceSample] = []
    source_row_count = 0
    for pair_id in sorted(grouped):
        rows = sorted(
            grouped[pair_id],
            key=lambda sample: sample.stable_sample_key or sample.sample_id,
        )
        _validate_pair_rows(pair_id, rows)
        canonical = rows[0]
        session_id = str(canonical.session_id).strip()
        if not session_id or manifest.sessions[pair_id] != session_id:
            raise ValueError(
                "Pair-text manifest session mismatch: "
                f"pair_id={pair_id!r}, manifest={manifest.sessions[pair_id]!r}, "
                f"sample={session_id!r}"
            )
        metadata = dict(canonical.metadata)
        metadata.update(
            {
                "pair_text_overlay_mode": manifest.mode,
                "pair_text_manifest_sha256": manifest.file_sha256,
                "pair_text_profile_sha256": manifest.profile_sha256,
                "pair_text_pair_universe_sha256": manifest.pair_universe_sha256,
                "pair_text_namespace": manifest.namespace,
                "pair_text_donor_pair_id": manifest.donors.get(pair_id, ""),
                "source_news_row_count": len(rows),
                "source_sample_ids": [row.sample_id for row in rows],
                "source_news_row_ids": [
                    int(row.news_row_id) for row in rows if row.news_row_id is not None
                ],
            }
        )
        source_row_count += len(rows)
        output.append(
            replace(
                canonical,
                sample_id=f"pair::{pair_id}",
                stable_sample_key=f"pair::{pair_id}",
                global_index=min(int(row.global_index) for row in rows),
                news_row_id=None,
                text_embedding=manifest.embeddings[pair_id].copy(),
                sample_weight=1.0,
                metadata=metadata,
            )
        )
    return output, {
        "mode": manifest.mode,
        "manifest_sha256": manifest.file_sha256,
        "profile_sha256": manifest.profile_sha256,
        "pair_universe_sha256": pair_universe_sha256(sorted(selected_pairs)),
        "manifest_pair_universe_sha256": manifest.pair_universe_sha256,
        "namespace": manifest.namespace,
        "embedding_dim": manifest.embedding_dim,
        "pair_count": len(output),
        "source_row_count": source_row_count,
        "donor_mapping_sha256": (
            hashlib.sha256(
                _canonical_json_bytes(sorted(manifest.donors.items()))
            ).hexdigest()
            if manifest.donors
            else ""
        ),
    }


def _validated_state_lineage(raw: object, *, label: str) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} must be an object")
    normalized: dict[str, object] = {}
    missing = [field for field in _REQUIRED_STATE_LINEAGE if field not in raw]
    if missing:
        raise ValueError(f"{label} is missing required fields: {missing}")
    for field_name in _REQUIRED_STATE_LINEAGE:
        value = raw[field_name]
        if field_name == "seed":
            normalized[field_name] = int(value)
        elif field_name in {"fold_id", "arm"}:
            text = str(value).strip()
            if not text:
                raise ValueError(f"{label}.{field_name} must be non-empty")
            normalized[field_name] = text
        else:
            normalized[field_name] = _require_sha256(
                value, label=f"{label}.{field_name}"
            )
    extras = sorted(set(raw) - set(_REQUIRED_STATE_LINEAGE))
    for field_name in extras:
        normalized[str(field_name)] = raw[field_name]
    return normalized


@dataclass(frozen=True)
class FullTrainingStateContract:
    """Hash-validated save/resume instructions for one training job."""

    mode: str
    output_path: Path | None
    output_lineage: Mapping[str, object]
    contract_path: Path
    contract_sha256: str
    input_path: Path | None = None
    input_sha256: str = ""
    expected_input_lineage: Mapping[str, object] | None = None


def write_full_training_state_contract(
    path: str | Path,
    *,
    mode: object,
    output_path: str | Path | None,
    output_lineage: Mapping[str, object],
    input_path: str | Path | None = None,
    input_sha256: str = "",
    expected_input_lineage: Mapping[str, object] | None = None,
) -> dict[str, str]:
    """Write one canonical parent/continuation state contract."""

    normalized_mode = normalize_full_training_state_mode(mode)
    if normalized_mode == NO_FULL_TRAINING_STATE:
        raise ValueError("Cannot write a full-state contract for mode 'none'")
    output_text = str(output_path or "").strip()
    if normalized_mode != RESUME_FROZEN_LR_FULL_TRAINING_STATE and not output_text:
        raise ValueError("Dynamic full-state contracts require output_path")
    payload: dict[str, object] = {
        "schema_version": FULL_TRAINING_STATE_SCHEMA_VERSION,
        "kind": FULL_TRAINING_STATE_CONTRACT_KIND,
        "mode": normalized_mode,
        "output_path": output_text,
        "output_lineage": _validated_state_lineage(
            output_lineage, label="output_lineage"
        ),
        "input": None,
    }
    if normalized_mode in {
        RESUME_DYNAMIC_FULL_TRAINING_STATE,
        RESUME_FROZEN_LR_FULL_TRAINING_STATE,
    }:
        input_text = str(input_path or "").strip()
        if not input_text or expected_input_lineage is None:
            raise ValueError(f"{normalized_mode} requires input path and lineage")
        payload["input"] = {
            "path": input_text,
            "sha256": _require_sha256(
                input_sha256, label="input full-training-state SHA256"
            ),
            "expected_lineage": _validated_state_lineage(
                expected_input_lineage, label="expected_input_lineage"
            ),
        }
    elif input_path is not None or input_sha256 or expected_input_lineage is not None:
        raise ValueError("save_dynamic_v1 must not declare an input state")
    contract_sha = _atomic_json_write(path, payload)
    return {
        "contract_path": str(Path(path).expanduser()),
        "contract_sha256": contract_sha,
    }


def load_full_training_state_contract(
    path: str | Path,
    expected_sha256: str,
    *,
    expected_mode: object,
) -> FullTrainingStateContract:
    """Load the immutable contract that binds one parent/branch state."""

    payload = _read_hash_bound_json(
        path, expected_sha256, label="Full-training-state contract"
    )
    if int(payload.get("schema_version", -1)) != FULL_TRAINING_STATE_SCHEMA_VERSION:
        raise ValueError("Full-training-state contract schema_version must be 1")
    if str(payload.get("kind", "")) != FULL_TRAINING_STATE_CONTRACT_KIND:
        raise ValueError(
            f"Full-training-state contract kind must be {FULL_TRAINING_STATE_CONTRACT_KIND!r}"
        )
    mode = normalize_full_training_state_mode(payload.get("mode"))
    configured_mode = normalize_full_training_state_mode(expected_mode)
    if mode == NO_FULL_TRAINING_STATE or mode != configured_mode:
        raise ValueError(
            f"Full-training-state mode mismatch: contract={mode!r}, config={configured_mode!r}"
        )
    output_path_text = str(payload.get("output_path", "")).strip()
    output_path = Path(output_path_text).expanduser() if output_path_text else None
    if mode != RESUME_FROZEN_LR_FULL_TRAINING_STATE and output_path is None:
        raise ValueError(
            "Dynamic full-training-state contract output_path must be non-empty"
        )
    output_lineage = _validated_state_lineage(
        payload.get("output_lineage"), label="output_lineage"
    )
    input_path: Path | None = None
    input_sha = ""
    input_lineage: Mapping[str, object] | None = None
    raw_input = payload.get("input")
    if mode in {
        RESUME_DYNAMIC_FULL_TRAINING_STATE,
        RESUME_FROZEN_LR_FULL_TRAINING_STATE,
    }:
        if not isinstance(raw_input, Mapping):
            raise ValueError(f"{mode} requires an input state object")
        input_path = Path(str(raw_input.get("path", ""))).expanduser()
        input_sha = _require_sha256(raw_input.get("sha256"), label="input state sha256")
        input_lineage = _validated_state_lineage(
            raw_input.get("expected_lineage"), label="expected_input_lineage"
        )
    elif raw_input not in (None, {}):
        raise ValueError("save_dynamic_v1 must not declare an input state")
    return FullTrainingStateContract(
        mode=mode,
        output_path=output_path,
        output_lineage=output_lineage,
        contract_path=Path(path).expanduser(),
        contract_sha256=sha256_file(path),
        input_path=input_path,
        input_sha256=input_sha,
        expected_input_lineage=input_lineage,
    )


def _cpu_byte_rng_state(value: object, *, label: str) -> torch.Tensor:
    """Return one RNG state as the CPU ByteTensor required by PyTorch."""

    if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8:
        raise ValueError(
            f"full_training_state_v1 {label} RNG state must be a torch.uint8 tensor"
        )
    return value.detach().to(device="cpu").contiguous()


def capture_rng_state(loader_generator: torch.Generator) -> dict[str, object]:
    """Capture every RNG stream used by the news-first training job."""

    if not isinstance(loader_generator, torch.Generator):
        raise ValueError("full_training_state_v1 requires a DataLoader torch.Generator")
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": _cpu_byte_rng_state(torch.get_rng_state(), label="torch_cpu"),
        "torch_cuda": [
            _cpu_byte_rng_state(state, label=f"torch_cuda[{index}]")
            for index, state in enumerate(
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            )
        ],
        "loader_generator": _cpu_byte_rng_state(
            loader_generator.get_state(), label="loader_generator"
        ),
    }


def restore_rng_state(
    rng_state: object,
    *,
    loader_generator: torch.Generator,
) -> None:
    """Restore Python, NumPy, Torch CPU/CUDA and DataLoader RNG streams."""

    if not isinstance(rng_state, Mapping):
        raise ValueError("full_training_state_v1 rng_state must be an object")
    if not isinstance(loader_generator, torch.Generator):
        raise ValueError("full_training_state_v1 requires a DataLoader torch.Generator")
    required = {"python", "numpy", "torch_cpu", "torch_cuda", "loader_generator"}
    if set(rng_state) != required:
        raise ValueError(
            "full_training_state_v1 rng_state keys mismatch: "
            f"expected={sorted(required)}, actual={sorted(rng_state)}"
        )
    cuda_states = rng_state["torch_cuda"]
    if not isinstance(cuda_states, list):
        raise ValueError("full_training_state_v1 torch_cuda RNG state must be a list")
    current_cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if len(cuda_states) != current_cuda_count:
        raise ValueError(
            "CUDA RNG device count differs from full training state: "
            f"state={len(cuda_states)}, runtime={current_cuda_count}"
        )
    torch_cpu_state = _cpu_byte_rng_state(rng_state["torch_cpu"], label="torch_cpu")
    normalized_cuda_states = [
        _cpu_byte_rng_state(state, label=f"torch_cuda[{index}]")
        for index, state in enumerate(cuda_states)
    ]
    loader_state = _cpu_byte_rng_state(
        rng_state["loader_generator"], label="loader_generator"
    )
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(torch_cpu_state)
    if normalized_cuda_states:
        torch.cuda.set_rng_state_all(normalized_cuda_states)
    loader_generator.set_state(loader_state)


def _atomic_torch_save(payload: Mapping[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary_path)
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
        directory_descriptor = os.open(output_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def save_full_training_state(
    output_path: str | Path,
    *,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    generator_scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    discriminator_scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    loader_generator: torch.Generator,
    completed_epoch: int,
    lineage: Mapping[str, object],
    contract_sha256: str,
) -> str:
    """Atomically save a complete state after both schedulers have stepped."""

    epoch = int(completed_epoch)
    if epoch < 1:
        raise ValueError("full_training_state_v1 completed_epoch must be >= 1")
    if generator_scheduler is None or discriminator_scheduler is None:
        raise ValueError(
            "full_training_state_v1 requires Generator and Discriminator "
            "ReduceLROnPlateau schedulers"
        )
    normalized_lineage = _validated_state_lineage(lineage, label="state_lineage")
    payload: dict[str, object] = {
        "schema_version": FULL_TRAINING_STATE_SCHEMA_VERSION,
        "kind": FULL_TRAINING_STATE_KIND,
        "save_phase": FULL_TRAINING_STATE_PHASE,
        "completed_epoch": epoch,
        "lineage": normalized_lineage,
        "contract_sha256": _require_sha256(
            contract_sha256, label="full-training-state contract SHA256"
        ),
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "generator_optimizer_state_dict": generator_optimizer.state_dict(),
        "discriminator_optimizer_state_dict": discriminator_optimizer.state_dict(),
        "generator_scheduler_state_dict": generator_scheduler.state_dict(),
        "discriminator_scheduler_state_dict": discriminator_scheduler.state_dict(),
        "rng_state": capture_rng_state(loader_generator),
    }
    resolved = Path(output_path).expanduser()
    _atomic_torch_save(payload, resolved)
    return sha256_file(resolved)


def load_full_training_state(
    input_path: str | Path,
    expected_sha256: str,
    *,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    generator_scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    discriminator_scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    loader_generator: torch.Generator,
    expected_lineage: Mapping[str, object],
    restore_schedulers: bool,
    map_location: torch.device | str,
) -> dict[str, object]:
    """Validate and restore a complete state; reject weights-only checkpoints.

    The payload is staged on CPU even when the destination modules use CUDA.
    ``load_state_dict`` then applies PyTorch's canonical per-parameter placement
    for model and optimizer state.  This keeps CPU RNG tensors and Adam's CPU
    step counters on their required devices while moving parameter moments to
    the destination module device.
    """

    path = Path(input_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Full training state does not exist: {path}")
    expected_digest = _require_sha256(
        expected_sha256, label="full training state SHA256"
    )
    actual_digest = sha256_file(path)
    if actual_digest != expected_digest:
        raise ValueError(
            "Full training state SHA256 mismatch: "
            f"expected={expected_digest}, actual={actual_digest}"
        )
    destination = torch.device(map_location)
    if destination.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA full-state restore requested but CUDA is unavailable")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("Full training state root must be an object")
    if (
        int(payload.get("schema_version", -1)) != FULL_TRAINING_STATE_SCHEMA_VERSION
        or payload.get("kind") != FULL_TRAINING_STATE_KIND
    ):
        raise ValueError(
            "Continuation requires full_training_state_v1; weights-only and older "
            "checkpoints are not accepted"
        )
    if payload.get("save_phase") != FULL_TRAINING_STATE_PHASE:
        raise ValueError(
            "Full training state was not saved at end_of_epoch_after_scheduler_step"
        )
    completed_epoch = int(payload.get("completed_epoch", 0))
    if completed_epoch < 1:
        raise ValueError("Full training state completed_epoch must be >= 1")
    actual_lineage = _validated_state_lineage(
        payload.get("lineage"), label="saved state lineage"
    )
    required_lineage = _validated_state_lineage(
        expected_lineage, label="expected state lineage"
    )
    if actual_lineage != required_lineage:
        raise ValueError(
            "Full training state lineage mismatch: "
            f"expected={required_lineage}, actual={actual_lineage}"
        )
    required_keys = (
        "generator_state_dict",
        "discriminator_state_dict",
        "generator_optimizer_state_dict",
        "discriminator_optimizer_state_dict",
        "generator_scheduler_state_dict",
        "discriminator_scheduler_state_dict",
        "rng_state",
    )
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise ValueError(
            "Continuation requires complete full_training_state_v1 payload; "
            f"missing={missing}"
        )
    generator.load_state_dict(payload["generator_state_dict"], strict=True)
    discriminator.load_state_dict(payload["discriminator_state_dict"], strict=True)
    generator_optimizer.load_state_dict(payload["generator_optimizer_state_dict"])
    discriminator_optimizer.load_state_dict(
        payload["discriminator_optimizer_state_dict"]
    )
    if restore_schedulers:
        if generator_scheduler is None or discriminator_scheduler is None:
            raise ValueError("Dynamic continuation requires both plateau schedulers")
        generator_scheduler.load_state_dict(payload["generator_scheduler_state_dict"])
        discriminator_scheduler.load_state_dict(
            payload["discriminator_scheduler_state_dict"]
        )
    restore_rng_state(payload["rng_state"], loader_generator=loader_generator)
    return {
        "completed_epoch": completed_epoch,
        "lineage": actual_lineage,
        "state_sha256": actual_digest,
        "save_phase": FULL_TRAINING_STATE_PHASE,
    }


__all__ = [
    "FULL_TRAINING_STATE_CONTRACT_KIND",
    "FULL_TRAINING_STATE_KIND",
    "FULL_TRAINING_STATE_MODES",
    "FULL_TRAINING_STATE_PHASE",
    "FullTrainingStateContract",
    "NO_FULL_TRAINING_STATE",
    "NO_PAIR_TEXT_OVERLAY",
    "PAIR_TEXT_MANIFEST_KIND",
    "PAIR_TEXT_OVERLAY_MODES",
    "PairTextOverlayManifest",
    "RESUME_DYNAMIC_FULL_TRAINING_STATE",
    "RESUME_FROZEN_LR_FULL_TRAINING_STATE",
    "SAVE_DYNAMIC_FULL_TRAINING_STATE",
    "apply_pair_text_overlay",
    "build_pair_text_overlay_manifests",
    "capture_rng_state",
    "load_full_training_state",
    "load_full_training_state_contract",
    "load_pair_text_overlay_manifest",
    "normalize_full_training_state_mode",
    "normalize_pair_text_overlay_mode",
    "pair_universe_sha256",
    "restore_rng_state",
    "save_full_training_state",
    "sha256_file",
    "training_config_payload_sha256",
    "write_full_training_state_contract",
    "write_pair_text_overlay_manifest",
]
