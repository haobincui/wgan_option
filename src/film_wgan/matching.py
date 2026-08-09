"""Leakage-safe donor selection for transition--text matching losses."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .protocol import (
    MATCHING_NEGATIVE_SOURCE_PLAN_VERSION,
    TEXT_ALIGNMENT_PLAN_VERSION,
    canonical_payload_sha256,
)


_EMPTY_TOKENS = {"", "nan", "none", "null"}
_SPLIT_SEED_OFFSETS = {"train": 11, "val": 23, "test": 37}


def _readonly_int_array(values: Any, *, ndim: int, field_name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != ndim:
        raise ValueError(f"{field_name} must be a {ndim}-dimensional integer array.")
    array = np.ascontiguousarray(array.copy())
    array.setflags(write=False)
    return array


def _readonly_bool_array(values: Any, *, field_name: str) -> np.ndarray:
    array = np.asarray(values, dtype=bool)
    if array.ndim != 1:
        raise ValueError(f"{field_name} must be a one-dimensional boolean array.")
    array = np.ascontiguousarray(array.copy())
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class TextAlignmentPlan:
    """Frozen split-local native/placebo text-source mapping.

    Source indices always address the canonical native item order.  The same
    plan can therefore be shared by the matched and shuffled arms without
    letting either arm's carrier rows redefine text identity.
    """

    split: str
    base_seed: int
    effective_seed: int
    target_sample_ids: tuple[str, ...]
    target_surface_pair_ids: tuple[str, ...]
    placebo_source_indices: np.ndarray
    mapping_version: str = TEXT_ALIGNMENT_PLAN_VERSION

    def __post_init__(self) -> None:
        split = str(self.split).strip().lower()
        if not split:
            raise ValueError("Text alignment plan split must be non-empty.")
        object.__setattr__(self, "split", split)
        if str(self.mapping_version) != TEXT_ALIGNMENT_PLAN_VERSION:
            raise ValueError(
                "Unsupported text alignment plan version: "
                f"{self.mapping_version!r}."
            )
        sample_ids = tuple(str(value) for value in self.target_sample_ids)
        pair_ids = tuple(str(value) for value in self.target_surface_pair_ids)
        if not sample_ids or len(sample_ids) != len(pair_ids):
            raise ValueError(
                "Text alignment plan target sample/pair identifiers must be "
                "non-empty and have equal length."
            )
        if any(not value for value in sample_ids) or len(set(sample_ids)) != len(sample_ids):
            raise ValueError("Text alignment plan target sample IDs must be unique and non-empty.")
        if any(not value for value in pair_ids):
            raise ValueError("Text alignment plan surface-pair IDs must be non-empty.")
        indices = _readonly_int_array(
            self.placebo_source_indices,
            ndim=1,
            field_name="placebo_source_indices",
        )
        count = len(sample_ids)
        if indices.shape != (count,):
            raise ValueError(
                "placebo_source_indices length must match the target item count."
            )
        if sorted(indices.tolist()) != list(range(count)):
            raise ValueError(
                "placebo_source_indices must be a bijection over canonical native indices."
            )
        for target_index, source_index in enumerate(indices.tolist()):
            if pair_ids[target_index] == pair_ids[source_index]:
                raise ValueError(
                    "Text placebo permutation must be a surface-pair derangement; "
                    f"target_index={target_index}, source_index={source_index}, "
                    f"surface_pair_id={pair_ids[target_index]!r}."
                )
        object.__setattr__(self, "target_sample_ids", sample_ids)
        object.__setattr__(self, "target_surface_pair_ids", pair_ids)
        object.__setattr__(self, "placebo_source_indices", indices)

    @property
    def native_source_indices(self) -> np.ndarray:
        values = np.arange(len(self.target_sample_ids), dtype=np.int64)
        values.setflags(write=False)
        return values

    def positive_source_indices(self, alignment_mode: str) -> np.ndarray:
        mode = str(alignment_mode).strip().lower()
        if mode in {"matched", "native"}:
            return self.native_source_indices
        if mode in {"permuted", "shuffled", "placebo"}:
            return self.placebo_source_indices
        raise ValueError(
            "alignment_mode must be one of matched/native/permuted/shuffled/placebo, "
            f"got {alignment_mode!r}."
        )

    @property
    def sha256(self) -> str:
        return canonical_payload_sha256(
            {
                "mapping_version": self.mapping_version,
                "split": self.split,
                "base_seed": int(self.base_seed),
                "effective_seed": int(self.effective_seed),
                "target_sample_ids": self.target_sample_ids,
                "target_surface_pair_ids": self.target_surface_pair_ids,
                "placebo_source_indices": self.placebo_source_indices.tolist(),
            }
        )

    def to_frame(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for target_index, placebo_index in enumerate(self.placebo_source_indices.tolist()):
            rows.append(
                {
                    "mapping_version": self.mapping_version,
                    "split": self.split,
                    "base_seed": int(self.base_seed),
                    "effective_seed": int(self.effective_seed),
                    "target_dataset_index": target_index,
                    "target_sample_id": self.target_sample_ids[target_index],
                    "target_surface_pair_id": self.target_surface_pair_ids[target_index],
                    "native_source_index": target_index,
                    "native_source_sample_id": self.target_sample_ids[target_index],
                    "native_source_surface_pair_id": self.target_surface_pair_ids[target_index],
                    "placebo_source_index": placebo_index,
                    "placebo_source_sample_id": self.target_sample_ids[placebo_index],
                    "placebo_source_surface_pair_id": self.target_surface_pair_ids[placebo_index],
                    "text_alignment_plan_sha256": self.sha256,
                }
            )
        return pd.DataFrame(rows)

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        split: str | None = None,
    ) -> "TextAlignmentPlan":
        data = frame.copy()
        if split is not None:
            if "split" not in data.columns:
                raise ValueError("Text alignment plan frame is missing fields: ['split']")
            data = data[data["split"].astype(str).str.lower() == str(split).lower()]
        required = {
            "mapping_version",
            "split",
            "base_seed",
            "effective_seed",
            "target_dataset_index",
            "target_sample_id",
            "target_surface_pair_id",
            "native_source_index",
            "native_source_sample_id",
            "native_source_surface_pair_id",
            "placebo_source_index",
            "placebo_source_sample_id",
            "placebo_source_surface_pair_id",
            "text_alignment_plan_sha256",
        }
        missing = sorted(required - set(data.columns))
        if missing:
            raise ValueError(f"Text alignment plan frame is missing fields: {missing}")
        if data.empty:
            raise ValueError("Text alignment plan frame is empty for the requested split.")
        data = data.sort_values("target_dataset_index").reset_index(drop=True)
        expected_indices = list(range(len(data)))
        if data["target_dataset_index"].astype(int).tolist() != expected_indices:
            raise ValueError("Text alignment plan target indices are not contiguous canonical order.")
        singleton_fields = ("mapping_version", "split", "base_seed", "effective_seed")
        singleton: dict[str, Any] = {}
        for name in singleton_fields:
            values = data[name].drop_duplicates().tolist()
            if len(values) != 1:
                raise ValueError(f"Text alignment plan field {name!r} is not constant.")
            singleton[name] = values[0]
        plan = cls(
            split=str(singleton["split"]),
            mapping_version=str(singleton["mapping_version"]),
            base_seed=int(singleton["base_seed"]),
            effective_seed=int(singleton["effective_seed"]),
            target_sample_ids=tuple(data["target_sample_id"].astype(str)),
            target_surface_pair_ids=tuple(data["target_surface_pair_id"].astype(str)),
            placebo_source_indices=data["placebo_source_index"].astype(int).to_numpy(),
        )
        for row in data.to_dict(orient="records"):
            target_index = int(row["target_dataset_index"])
            native_index = int(row["native_source_index"])
            placebo_index = int(row["placebo_source_index"])
            if native_index != target_index:
                raise ValueError(
                    "Text alignment plan native source index must equal its "
                    f"target index; target={target_index}, native={native_index}."
                )
            if not 0 <= placebo_index < len(plan.target_sample_ids):
                raise ValueError(
                    "Text alignment plan placebo source index is out of canonical range."
                )
            identity_checks = (
                (
                    "native_source_sample_id",
                    row["native_source_sample_id"],
                    plan.target_sample_ids[native_index],
                ),
                (
                    "native_source_surface_pair_id",
                    row["native_source_surface_pair_id"],
                    plan.target_surface_pair_ids[native_index],
                ),
                (
                    "placebo_source_sample_id",
                    row["placebo_source_sample_id"],
                    plan.target_sample_ids[placebo_index],
                ),
                (
                    "placebo_source_surface_pair_id",
                    row["placebo_source_surface_pair_id"],
                    plan.target_surface_pair_ids[placebo_index],
                ),
            )
            for field_name, recorded_value, expected_value in identity_checks:
                if str(recorded_value) != str(expected_value):
                    raise ValueError(
                        "Text alignment plan source identity does not match its "
                        f"canonical index: field={field_name!r}, "
                        f"target_index={target_index}."
                    )
        recorded = (
            data["text_alignment_plan_sha256"]
            .drop_duplicates()
            .astype(str)
            .tolist()
        )
        if recorded != [plan.sha256]:
            raise ValueError("Text alignment plan SHA256 does not match its canonical contents.")
        return plan


@dataclass(frozen=True)
class TransitionMatchingNegativePlan:
    """Frozen symmetric negative sources for matched and shuffled arms."""

    split: str
    seed: int
    negative_count: int
    minimum_supported_cells: int
    duplicate_cosine_threshold: float
    target_sample_ids: tuple[str, ...]
    target_surface_pair_ids: tuple[str, ...]
    positive_alignment_sha256: str
    negative_source_indices: np.ndarray
    eligible_mask: np.ndarray
    audit_rows: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    summary: Mapping[str, Any] = field(default_factory=dict)
    plan_version: str = MATCHING_NEGATIVE_SOURCE_PLAN_VERSION

    def __post_init__(self) -> None:
        split = str(self.split).strip().lower()
        object.__setattr__(self, "split", split)
        if not split:
            raise ValueError("Transition-matching negative plan split must be non-empty.")
        if str(self.plan_version) != MATCHING_NEGATIVE_SOURCE_PLAN_VERSION:
            raise ValueError(
                "Unsupported transition-matching negative-source plan version: "
                f"{self.plan_version!r}."
            )
        k = int(self.negative_count)
        minimum_cells = int(self.minimum_supported_cells)
        threshold = float(self.duplicate_cosine_threshold)
        if k < 1:
            raise ValueError("negative_count must be positive.")
        if minimum_cells < 1:
            raise ValueError("minimum_supported_cells must be positive.")
        if not -1.0 < threshold < 1.0:
            raise ValueError("duplicate_cosine_threshold must be strictly between -1 and 1.")
        sample_ids = tuple(str(value) for value in self.target_sample_ids)
        pair_ids = tuple(str(value) for value in self.target_surface_pair_ids)
        if not sample_ids or len(sample_ids) != len(pair_ids):
            raise ValueError("Negative plan target sample/pair identifiers are invalid.")
        if len(set(sample_ids)) != len(sample_ids) or any(not value for value in sample_ids):
            raise ValueError("Negative plan target sample IDs must be unique and non-empty.")
        negative_indices = _readonly_int_array(
            self.negative_source_indices,
            ndim=2,
            field_name="negative_source_indices",
        )
        eligible = _readonly_bool_array(self.eligible_mask, field_name="eligible_mask")
        if negative_indices.shape != (len(sample_ids), k):
            raise ValueError(
                "negative_source_indices shape must be [target_count, negative_count]."
            )
        if eligible.shape != (len(sample_ids),):
            raise ValueError("eligible_mask length must match the target item count.")
        if bool(np.any(negative_indices[~eligible] != -1)):
            raise ValueError("Ineligible targets must contain only -1 negative-source indices.")
        if bool(np.any(negative_indices[eligible] < 0)) or bool(
            np.any(negative_indices[eligible] >= len(sample_ids))
        ):
            raise ValueError("Eligible targets contain an out-of-range negative-source index.")
        for row in negative_indices[eligible]:
            if len(set(int(value) for value in row.tolist())) != k:
                raise ValueError("A target may not reuse a negative source across ranks.")
        for rank in range(k):
            rank_values = negative_indices[eligible, rank].tolist()
            if len(set(int(value) for value in rank_values)) != len(rank_values):
                raise ValueError(
                    "Each negative rank must be a one-to-one assignment with no donor reuse."
                )
        object.__setattr__(self, "negative_count", k)
        object.__setattr__(self, "minimum_supported_cells", minimum_cells)
        object.__setattr__(self, "duplicate_cosine_threshold", threshold)
        object.__setattr__(self, "target_sample_ids", sample_ids)
        object.__setattr__(self, "target_surface_pair_ids", pair_ids)
        object.__setattr__(self, "negative_source_indices", negative_indices)
        object.__setattr__(self, "eligible_mask", eligible)
        object.__setattr__(
            self,
            "audit_rows",
            tuple(dict(row) for row in self.audit_rows),
        )
        object.__setattr__(self, "summary", dict(self.summary))

    @property
    def sha256(self) -> str:
        return canonical_payload_sha256(
            {
                "plan_version": self.plan_version,
                "split": self.split,
                "seed": int(self.seed),
                "negative_count": self.negative_count,
                "minimum_supported_cells": self.minimum_supported_cells,
                "duplicate_cosine_threshold": self.duplicate_cosine_threshold,
                "target_sample_ids": self.target_sample_ids,
                "target_surface_pair_ids": self.target_surface_pair_ids,
                "positive_alignment_sha256": self.positive_alignment_sha256,
                "eligible_mask": self.eligible_mask.astype(int).tolist(),
                "negative_source_indices": self.negative_source_indices.tolist(),
            }
        )

    def to_frame(self) -> pd.DataFrame:
        audit_by_key = {
            (int(row["target_dataset_index"]), int(row["negative_rank"])): dict(row)
            for row in self.audit_rows
        }
        rows: list[dict[str, Any]] = []
        for target_index in range(len(self.target_sample_ids)):
            for negative_rank in range(self.negative_count):
                source_index = int(self.negative_source_indices[target_index, negative_rank])
                row = {
                    "plan_version": self.plan_version,
                    "split": self.split,
                    "plan_seed": int(self.seed),
                    "negative_count": self.negative_count,
                    "minimum_supported_cells": self.minimum_supported_cells,
                    "duplicate_cosine_threshold": self.duplicate_cosine_threshold,
                    "positive_alignment_sha256": self.positive_alignment_sha256,
                    "target_dataset_index": target_index,
                    "target_sample_id": self.target_sample_ids[target_index],
                    "target_surface_pair_id": self.target_surface_pair_ids[target_index],
                    "target_eligible": int(self.eligible_mask[target_index]),
                    "negative_rank": negative_rank,
                    "negative_source_index": source_index,
                    "negative_source_sample_id": (
                        self.target_sample_ids[source_index] if source_index >= 0 else ""
                    ),
                    "negative_source_surface_pair_id": (
                        self.target_surface_pair_ids[source_index] if source_index >= 0 else ""
                    ),
                    "matching_negative_source_plan_sha256": self.sha256,
                }
                row.update(audit_by_key.get((target_index, negative_rank), {}))
                rows.append(row)
        return pd.DataFrame(rows)

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        split: str | None = None,
    ) -> "TransitionMatchingNegativePlan":
        data = frame.copy()
        if split is not None:
            if "split" not in data.columns:
                raise ValueError("Negative-source plan frame is missing fields: ['split']")
            data = data[data["split"].astype(str).str.lower() == str(split).lower()]
        required = {
            "plan_version",
            "split",
            "plan_seed",
            "negative_count",
            "minimum_supported_cells",
            "duplicate_cosine_threshold",
            "positive_alignment_sha256",
            "target_dataset_index",
            "target_sample_id",
            "target_surface_pair_id",
            "target_eligible",
            "negative_rank",
            "negative_source_index",
            "negative_source_sample_id",
            "negative_source_surface_pair_id",
            "matching_negative_source_plan_sha256",
        }
        missing = sorted(required - set(data.columns))
        if missing:
            raise ValueError(f"Negative-source plan frame is missing fields: {missing}")
        if data.empty:
            raise ValueError("Negative-source plan frame is empty for the requested split.")
        singleton_names = (
            "plan_version",
            "split",
            "plan_seed",
            "negative_count",
            "minimum_supported_cells",
            "duplicate_cosine_threshold",
            "positive_alignment_sha256",
        )
        singleton: dict[str, Any] = {}
        for name in singleton_names:
            values = data[name].drop_duplicates().tolist()
            if len(values) != 1:
                raise ValueError(f"Negative-source plan field {name!r} is not constant.")
            singleton[name] = values[0]
        k = int(singleton["negative_count"])
        target_indices = sorted(
            set(data["target_dataset_index"].astype(int).tolist())
        )
        if not target_indices or target_indices != list(range(len(target_indices))):
            raise ValueError(
                "Negative-source plan target indices are not contiguous canonical order."
            )
        target_count = len(target_indices)
        if len(data) != target_count * k:
            raise ValueError("Negative-source plan does not contain exactly K rows per target.")
        mapping = np.full((target_count, k), -1, dtype=np.int64)
        eligible = np.zeros(target_count, dtype=bool)
        sample_ids = [""] * target_count
        pair_ids = [""] * target_count
        audit_rows: list[dict[str, Any]] = []
        seen_keys: set[tuple[int, int]] = set()
        eligibility_by_target: dict[int, bool] = {}
        native_index_by_target: dict[int, int] = {}
        placebo_index_by_target: dict[int, int] = {}
        for row in data.to_dict(orient="records"):
            target_index = int(row["target_dataset_index"])
            rank = int(row["negative_rank"])
            key = (target_index, rank)
            if key in seen_keys or not 0 <= target_index < target_count or not 0 <= rank < k:
                raise ValueError("Negative-source plan contains duplicate or invalid target/rank rows.")
            seen_keys.add(key)
            source_index = int(row["negative_source_index"])
            mapping[target_index, rank] = source_index
            target_eligible = bool(int(row["target_eligible"]))
            previous_eligible = eligibility_by_target.setdefault(
                target_index, target_eligible
            )
            if previous_eligible != target_eligible:
                raise ValueError(
                    "Negative-source plan target eligibility changes across ranks."
                )
            eligible[target_index] = target_eligible
            sample_id = str(row["target_sample_id"])
            pair_id = str(row["target_surface_pair_id"])
            if sample_ids[target_index] not in {"", sample_id} or pair_ids[target_index] not in {"", pair_id}:
                raise ValueError("Negative-source plan target identity changes across ranks.")
            sample_ids[target_index] = sample_id
            pair_ids[target_index] = pair_id
            if target_eligible:
                audit_rows.append(dict(row))
        plan = cls(
            split=str(singleton["split"]),
            plan_version=str(singleton["plan_version"]),
            seed=int(singleton["plan_seed"]),
            negative_count=k,
            minimum_supported_cells=int(singleton["minimum_supported_cells"]),
            duplicate_cosine_threshold=float(singleton["duplicate_cosine_threshold"]),
            target_sample_ids=tuple(sample_ids),
            target_surface_pair_ids=tuple(pair_ids),
            positive_alignment_sha256=str(singleton["positive_alignment_sha256"]),
            negative_source_indices=mapping,
            eligible_mask=eligible,
            audit_rows=tuple(audit_rows),
            summary={},
        )
        for row in data.to_dict(orient="records"):
            target_index = int(row["target_dataset_index"])
            source_index = int(row["negative_source_index"])
            target_eligible = bool(int(row["target_eligible"]))
            if target_eligible:
                if not 0 <= source_index < len(plan.target_sample_ids):
                    raise ValueError(
                        "Eligible negative-source identity has an out-of-range index."
                    )
                expected_source_sample_id = plan.target_sample_ids[source_index]
                expected_source_pair_id = plan.target_surface_pair_ids[source_index]
            else:
                expected_source_sample_id = ""
                expected_source_pair_id = ""
            if _clean_token(row["negative_source_sample_id"]) != expected_source_sample_id:
                raise ValueError(
                    "Negative-source sample ID does not match its canonical index: "
                    f"target_index={target_index}, source_index={source_index}."
                )
            if (
                _clean_token(row["negative_source_surface_pair_id"])
                != expected_source_pair_id
            ):
                raise ValueError(
                    "Negative-source surface-pair ID does not match its canonical index: "
                    f"target_index={target_index}, source_index={source_index}."
                )

            optional_identity_fields = (
                "native_positive_source_index",
                "native_positive_sample_id",
                "native_positive_surface_pair_id",
                "placebo_positive_source_index",
                "placebo_positive_sample_id",
                "placebo_positive_surface_pair_id",
            )
            present = {
                field_name: _clean_token(row.get(field_name))
                for field_name in optional_identity_fields
                if field_name in data.columns
            }
            if target_eligible and present:
                missing_audit_values = sorted(
                    field_name
                    for field_name, value in present.items()
                    if not value
                )
                if missing_audit_values:
                    raise ValueError(
                        "Eligible negative-source audit rows contain empty identity "
                        f"fields: {missing_audit_values}."
                    )
                native_index_token = present.get("native_positive_source_index", "")
                native_sample_token = present.get("native_positive_sample_id", "")
                native_pair_token = present.get("native_positive_surface_pair_id", "")
                placebo_index_token = present.get("placebo_positive_source_index", "")
                placebo_sample_token = present.get("placebo_positive_sample_id", "")
                placebo_pair_token = present.get("placebo_positive_surface_pair_id", "")
                if native_sample_token and (
                    native_sample_token != plan.target_sample_ids[target_index]
                ):
                    raise ValueError(
                        "Negative-source audit native-positive sample ID is inconsistent."
                    )
                if native_pair_token and (
                    native_pair_token != plan.target_surface_pair_ids[target_index]
                ):
                    raise ValueError(
                        "Negative-source audit native-positive pair ID is inconsistent."
                    )
                if native_index_token:
                    native_index = int(float(native_index_token))
                    previous_native = native_index_by_target.setdefault(
                        target_index, native_index
                    )
                    if previous_native != native_index or native_index != target_index:
                        raise ValueError(
                            "Negative-source audit native-positive index is inconsistent."
                        )
                if (placebo_sample_token or placebo_pair_token) and not placebo_index_token:
                    raise ValueError(
                        "Negative-source audit placebo identity requires its source index."
                    )
                if placebo_index_token:
                    placebo_index = int(float(placebo_index_token))
                    if not 0 <= placebo_index < len(plan.target_sample_ids):
                        raise ValueError(
                            "Negative-source audit placebo-positive index is out of range."
                        )
                    previous_placebo = placebo_index_by_target.setdefault(
                        target_index, placebo_index
                    )
                    if previous_placebo != placebo_index:
                        raise ValueError(
                            "Negative-source audit placebo-positive index changes across ranks."
                        )
                    if placebo_sample_token and (
                        placebo_sample_token != plan.target_sample_ids[placebo_index]
                    ):
                        raise ValueError(
                            "Negative-source audit placebo-positive sample ID is inconsistent."
                        )
                    if placebo_pair_token and (
                        placebo_pair_token
                        != plan.target_surface_pair_ids[placebo_index]
                    ):
                        raise ValueError(
                            "Negative-source audit placebo-positive pair ID is inconsistent."
                        )
        recorded = (
            data["matching_negative_source_plan_sha256"]
            .drop_duplicates()
            .astype(str)
            .tolist()
        )
        if recorded != [plan.sha256]:
            raise ValueError("Negative-source plan SHA256 does not match its canonical contents.")
        return plan


def _clean_token(value: Any) -> str:
    rendered = str(value).strip()
    return "" if rendered.lower() in _EMPTY_TOKENS else rendered


def _token_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError:
                pass
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return {
            token
            for item in value
            if (token := _clean_token(item))
        }
    token = _clean_token(value)
    return {token} if token else set()


def _metadata_value(sample: Any, source_key: str, fallback_key: str) -> Any:
    metadata = dict(getattr(sample, "metadata", {}) or {})
    source_value = metadata.get(source_key)
    if source_value not in (None, "", [], ()):
        return source_value
    return metadata.get(fallback_key)


def _text_identity(sample: Any) -> dict[str, Any]:
    metadata = dict(getattr(sample, "metadata", {}) or {})
    return {
        "sample_id": _clean_token(
            metadata.get("text_source_sample_id") or getattr(sample, "sample_id")
        ),
        "surface_pair_id": _clean_token(
            metadata.get("text_source_surface_pair_id")
            or getattr(sample, "surface_pair_id")
        ),
        "article_ids": _token_set(
            _metadata_value(sample, "text_source_article_ids", "article_ids")
        ),
        "source_files": _token_set(
            _metadata_value(sample, "text_source_files", "source_files")
        ),
        "event_group": _clean_token(
            _metadata_value(sample, "text_source_event_group", "event_group")
        ),
        "news_cluster_id": _clean_token(
            _metadata_value(
                sample,
                "text_source_news_cluster_id",
                "news_cluster_id",
            )
        ),
        "feature_sha256": _clean_token(
            _metadata_value(
                sample,
                "text_source_feature_sha256",
                "pair_text_feature_sha256",
            )
        ),
    }


def _native_text_identity(sample: Any) -> dict[str, Any]:
    """Return identity from the canonical native row, never carrier overrides."""

    metadata = dict(getattr(sample, "metadata", {}) or {})
    return {
        "sample_id": _clean_token(getattr(sample, "sample_id", "")),
        "surface_pair_id": _clean_token(getattr(sample, "surface_pair_id", "")),
        "article_ids": _token_set(metadata.get("article_ids")),
        "source_files": _token_set(metadata.get("source_files")),
        "source_ids": _token_set(metadata.get("source_ids", metadata.get("source"))),
        "event_group": _clean_token(metadata.get("event_group")),
        "news_cluster_id": _clean_token(metadata.get("news_cluster_id")),
        "feature_sha256": _clean_token(
            metadata.get("pair_text_feature_sha256", metadata.get("feature_sha256"))
        ),
    }


def _identities_overlap(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    for name in (
        "sample_id",
        "surface_pair_id",
        "event_group",
        "news_cluster_id",
        "feature_sha256",
    ):
        if left.get(name) and left.get(name) == right.get(name):
            return True
    for name in ("article_ids", "source_files", "source_ids"):
        if set(left.get(name, set())) & set(right.get(name, set())):
            return True
    return False


def _sample_ids(samples: Sequence[Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    sample_ids = tuple(_clean_token(getattr(sample, "sample_id", "")) for sample in samples)
    pair_ids = tuple(_clean_token(getattr(sample, "surface_pair_id", "")) for sample in samples)
    if any(not value for value in sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Canonical native sample IDs must be unique and non-empty.")
    if any(not value for value in pair_ids):
        raise ValueError("Canonical native surface-pair IDs must be non-empty.")
    return sample_ids, pair_ids


def _strict_bipartite_assignment(
    candidate_rows: Mapping[int, Sequence[Mapping[str, Any]]],
    target_order: Sequence[int],
    *,
    excluded_by_target: Mapping[int, set[int]] | None = None,
) -> dict[int, Mapping[str, Any]]:
    """Return a one-to-one assignment or fail; never reuse/fallback donors."""

    exclusions = excluded_by_target or {}
    donor_to_target: dict[int, int] = {}
    target_to_row: dict[int, Mapping[str, Any]] = {}

    def assign(target_index: int, visited: set[int]) -> bool:
        for row in candidate_rows[target_index]:
            donor_index = int(row["donor_index"])
            if donor_index in exclusions.get(target_index, set()) or donor_index in visited:
                continue
            visited.add(donor_index)
            displaced = donor_to_target.get(donor_index)
            if displaced is None or assign(displaced, visited):
                donor_to_target[donor_index] = target_index
                target_to_row[target_index] = row
                return True
        return False

    for target_index in target_order:
        if not assign(int(target_index), set()):
            raise ValueError(
                "No duplicate-safe one-to-one negative assignment exists without "
                f"fallback/reuse for target index {target_index}."
            )
    if len(target_to_row) != len(target_order):
        raise ValueError("Incomplete one-to-one negative assignment.")
    return target_to_row


def _validate_text_embeddings(items: Sequence[Any], source_indices: Sequence[int]) -> dict[int, np.ndarray]:
    unit: dict[int, np.ndarray] = {}
    dimensions: set[int] = set()
    for source_index in sorted(set(int(value) for value in source_indices)):
        if not 0 <= source_index < len(items):
            raise ValueError(f"Canonical text source index is out of range: {source_index}.")
        embedding = np.asarray(items[source_index].text_embedding, dtype=np.float64).reshape(-1)
        norm = float(np.linalg.norm(embedding))
        if embedding.size <= 0 or not bool(np.all(np.isfinite(embedding))) or not np.isfinite(norm) or norm <= 1.0e-12:
            raise ValueError(
                "Every referenced transition-matching text embedding must be "
                f"finite and non-zero; source_index={source_index}."
            )
        dimensions.add(int(embedding.size))
        unit[source_index] = embedding / norm
    if len(dimensions) != 1:
        raise ValueError("All canonical transition-matching text embeddings must share one dimension.")
    return unit


def build_text_pair_derangement(pair_ids: Sequence[str], *, seed: int) -> np.ndarray:
    """Build a deterministic bijection that never stays within a pair ID."""

    normalized = tuple(_clean_token(value) for value in pair_ids)
    count = len(normalized)
    if count < 2 or len(set(normalized)) < 2 or any(not value for value in normalized):
        raise ValueError(
            "Permuted text requires at least two distinct non-empty surface pairs in each split."
        )
    candidates: dict[int, list[dict[str, Any]]] = {}
    for target_index in range(count):
        candidates[target_index] = sorted(
            (
                {
                    "donor_index": donor_index,
                    "tie_break": _stable_fraction(
                        int(seed), target_index, donor_index
                    ),
                }
                for donor_index in range(count)
                if normalized[target_index] != normalized[donor_index]
            ),
            key=lambda row: (float(row["tie_break"]), int(row["donor_index"])),
        )
    target_order = sorted(
        range(count),
        key=lambda index: (
            len(candidates[index]),
            _stable_fraction(int(seed), "target", index),
            index,
        ),
    )
    try:
        assignment = _strict_bipartite_assignment(candidates, target_order)
    except ValueError as exc:
        raise ValueError(
            "Could not build a bijective same-pair-free text permutation for this split."
        ) from exc
    donors = np.asarray(
        [int(assignment[index]["donor_index"]) for index in range(count)],
        dtype=np.int64,
    )
    if sorted(donors.tolist()) != list(range(count)):
        raise AssertionError("Internal text permutation is not bijective.")
    return donors


def build_text_alignment_plan(
    samples: Sequence[Any],
    *,
    split: str,
    seed: int,
    placebo_source_indices: Sequence[int] | None = None,
) -> TextAlignmentPlan:
    """Build or validate a deterministic split-local placebo permutation."""

    items = list(samples)
    sample_ids, pair_ids = _sample_ids(items)
    normalized_split = str(split).strip().lower()
    effective_seed = int(seed) + _SPLIT_SEED_OFFSETS.get(normalized_split, 0)
    if placebo_source_indices is None:
        indices = build_text_pair_derangement(pair_ids, seed=effective_seed)
    else:
        indices = np.asarray(placebo_source_indices, dtype=np.int64)
    plan = TextAlignmentPlan(
        split=normalized_split,
        base_seed=int(seed),
        effective_seed=effective_seed,
        target_sample_ids=sample_ids,
        target_surface_pair_ids=pair_ids,
        placebo_source_indices=indices,
    )
    referenced = [
        index
        for index, item in enumerate(items)
        if float(dict(getattr(item, "metadata", {}) or {}).get("has_text", 1.0)) > 0.0
    ]
    if referenced:
        _validate_text_embeddings(items, referenced)
    return plan


def _validate_alignment_against_items(
    plan: TextAlignmentPlan,
    items: Sequence[Any],
    *,
    split: str,
) -> None:
    sample_ids, pair_ids = _sample_ids(items)
    if plan.split != str(split).strip().lower():
        raise ValueError(
            f"Text alignment plan split mismatch: expected {split!r}, found {plan.split!r}."
        )
    if plan.target_sample_ids != sample_ids or plan.target_surface_pair_ids != pair_ids:
        raise ValueError(
            "Text alignment plan canonical item order/identities do not match the native split."
        )


def build_transition_matching_negative_source_plan(
    native_samples: Sequence[Any],
    placebo_source_indices: Sequence[int] | TextAlignmentPlan | None = None,
    *,
    text_alignment_plan: TextAlignmentPlan | None = None,
    split: str = "train",
    negative_count: int,
    minimum_supported_cells: int,
    seed: int,
    duplicate_cosine_threshold: float = 0.995,
) -> TransitionMatchingNegativePlan:
    """Build common canonical negatives protected against both arm positives.

    Eligibility uses the target transition support. Candidate identity and
    regime ranking use only native origin-time fields. The returned mapping is
    suitable for both matched and shuffled arms byte-for-byte.
    """

    items = list(native_samples)
    if isinstance(placebo_source_indices, TextAlignmentPlan):
        if text_alignment_plan is not None:
            raise ValueError("Provide the text alignment plan only once.")
        text_alignment_plan = placebo_source_indices
        placebo_source_indices = None
    if text_alignment_plan is None:
        if placebo_source_indices is None:
            raise ValueError(
                "A frozen TextAlignmentPlan or placebo_source_indices is required."
            )
        text_alignment_plan = build_text_alignment_plan(
            items,
            split=split,
            seed=seed,
            placebo_source_indices=placebo_source_indices,
        )
    elif placebo_source_indices is not None:
        supplied = np.asarray(placebo_source_indices, dtype=np.int64)
        if not np.array_equal(supplied, text_alignment_plan.placebo_source_indices):
            raise ValueError("Conflicting placebo_source_indices and text_alignment_plan.")
    _validate_alignment_against_items(text_alignment_plan, items, split=split)

    k = int(negative_count)
    minimum_cells = int(minimum_supported_cells)
    threshold = float(duplicate_cosine_threshold)
    if k < 1:
        raise ValueError("negative_count must be positive.")
    if minimum_cells < 1:
        raise ValueError("minimum_supported_cells must be positive.")
    if not -1.0 < threshold < 1.0:
        raise ValueError("duplicate_cosine_threshold must be strictly between -1 and 1.")

    sample_ids, pair_ids = _sample_ids(items)
    support_counts = np.asarray(
        [int(np.asarray(item.evaluation_support_mask, dtype=bool).sum()) for item in items],
        dtype=np.int64,
    )
    current_support_counts = np.asarray(
        [
            int(
                np.asarray(
                    item.current_support_mask
                    if item.current_support_mask is not None
                    else np.ones_like(item.current_surface, dtype=bool),
                    dtype=bool,
                ).sum()
            )
            for item in items
        ],
        dtype=np.int64,
    )
    has_text = np.asarray(
        [
            float(dict(getattr(item, "metadata", {}) or {}).get("has_text", 1.0)) > 0.0
            for item in items
        ],
        dtype=bool,
    )
    placebo_indices = text_alignment_plan.placebo_source_indices
    # Eligibility must be identical for matched and shuffled arms. A target
    # whose frozen placebo source has no text is excluded in both arms rather
    # than invalidating the whole split or contributing only to matched.
    eligible_mask = (
        has_text
        & has_text[placebo_indices]
        & (support_counts >= minimum_cells)
    )
    eligible_targets = np.flatnonzero(eligible_mask).astype(int).tolist()
    if not eligible_targets:
        raise ValueError(
            "No samples are eligible for transition matching after text/support filters."
        )
    donor_pool = [
        index
        for index in range(len(items))
        if bool(has_text[index]) and int(current_support_counts[index]) > 0
    ]
    if len(donor_pool) < len(eligible_targets):
        raise ValueError(
            "Strict no-reuse matching requires at least as many native donor texts as "
            f"eligible targets; donors={len(donor_pool)}, targets={len(eligible_targets)}."
        )
    positive_sources = set(eligible_targets)
    positive_sources.update(
        int(text_alignment_plan.placebo_source_indices[index])
        for index in eligible_targets
    )
    text_unit = _validate_text_embeddings(
        items,
        [*donor_pool, *positive_sources],
    )
    identities = {index: _native_text_identity(items[index]) for index in range(len(items))}
    regimes = {index: _current_regime(items[index]) for index in donor_pool}
    log_scale = max(float(np.std([regimes[index][0] for index in donor_pool])), 1.0e-6)
    support_scale = max(float(np.std([regimes[index][1] for index in donor_pool])), 1.0e-6)
    news_scale = max(float(np.std([regimes[index][4] for index in donor_pool])), 1.0)

    candidates: dict[int, list[dict[str, Any]]] = {}
    protected_counts: list[int] = []
    for target_index in eligible_targets:
        native_index = target_index
        placebo_index = int(text_alignment_plan.placebo_source_indices[target_index])
        target_regime = _current_regime(items[target_index])
        protected_identities = (identities[native_index], identities[placebo_index])
        protected_embeddings = (text_unit[native_index], text_unit[placebo_index])
        target_rows: list[dict[str, Any]] = []
        excluded = 0
        for donor_index in donor_pool:
            donor_identity = identities[donor_index]
            identity_overlap = any(
                _identities_overlap(donor_identity, identity)
                for identity in protected_identities
            )
            cosines = tuple(
                float(np.dot(text_unit[donor_index], positive_embedding))
                for positive_embedding in protected_embeddings
            )
            if identity_overlap or max(cosines) > threshold:
                excluded += 1
                continue
            donor_regime = regimes[donor_index]
            same_market_state = bool(
                target_regime[3]
                and donor_regime[3]
                and target_regime[3] == donor_regime[3]
            )
            same_session_bucket = target_regime[2] == donor_regime[2]
            tier = 0 if same_market_state and same_session_bucket else 1 if same_market_state else 2
            regime_distance = (
                abs(target_regime[0] - donor_regime[0]) / log_scale
                + abs(target_regime[1] - donor_regime[1]) / support_scale
                + 0.25 * abs(target_regime[2] - donor_regime[2])
                + 0.10 * abs(target_regime[4] - donor_regime[4]) / news_scale
            )
            target_rows.append(
                {
                    "donor_index": donor_index,
                    "tier": tier,
                    "regime_distance": float(regime_distance),
                    "native_positive_cosine": cosines[0],
                    "placebo_positive_cosine": cosines[1],
                    "tie_break": _stable_fraction(
                        int(seed),
                        str(split).lower(),
                        sample_ids[target_index],
                        sample_ids[donor_index],
                    ),
                }
            )
        target_rows.sort(
            key=lambda row: (
                int(row["tier"]),
                float(row["regime_distance"]),
                float(row["tie_break"]),
                int(row["donor_index"]),
            )
        )
        if len(target_rows) < k:
            raise ValueError(
                "Insufficient native/placebo-union-safe transition-matching donors for "
                f"sample={sample_ids[target_index]}: required={k}, "
                f"available={len(target_rows)}."
            )
        candidates[target_index] = target_rows
        protected_counts.append(excluded)

    mapping = np.full((len(items), k), -1, dtype=np.int64)
    selected_by_target = {index: set() for index in eligible_targets}
    audit_rows: list[dict[str, Any]] = []
    for rank in range(k):
        target_order = sorted(
            eligible_targets,
            key=lambda index: (
                _stable_fraction(int(seed), str(split).lower(), rank, sample_ids[index]),
                index,
            ),
        )
        assignment = _strict_bipartite_assignment(
            candidates,
            target_order,
            excluded_by_target=selected_by_target,
        )
        for target_index in eligible_targets:
            selected = assignment[target_index]
            donor_index = int(selected["donor_index"])
            mapping[target_index, rank] = donor_index
            selected_by_target[target_index].add(donor_index)
            native_index = target_index
            placebo_index = int(text_alignment_plan.placebo_source_indices[target_index])
            if donor_index in {native_index, placebo_index} or any(
                _identities_overlap(identities[donor_index], identities[positive_index])
                for positive_index in (native_index, placebo_index)
            ):
                raise AssertionError("Protected positive identity entered the negative plan.")
            if max(
                float(np.dot(text_unit[donor_index], text_unit[positive_index]))
                for positive_index in (native_index, placebo_index)
            ) > threshold:
                raise AssertionError("Protected near-duplicate text entered the negative plan.")
            audit_rows.append(
                {
                    "target_dataset_index": target_index,
                    "target_sample_id": sample_ids[target_index],
                    "target_surface_pair_id": pair_ids[target_index],
                    "native_positive_source_index": native_index,
                    "native_positive_sample_id": sample_ids[native_index],
                    "native_positive_surface_pair_id": pair_ids[native_index],
                    "placebo_positive_source_index": placebo_index,
                    "placebo_positive_sample_id": sample_ids[placebo_index],
                    "placebo_positive_surface_pair_id": pair_ids[placebo_index],
                    "negative_rank": rank,
                    "negative_source_index": donor_index,
                    "negative_source_sample_id": sample_ids[donor_index],
                    "negative_source_surface_pair_id": pair_ids[donor_index],
                    "selection_tier": int(selected["tier"]),
                    "regime_distance": float(selected["regime_distance"]),
                    "native_positive_cosine": float(selected["native_positive_cosine"]),
                    "placebo_positive_cosine": float(selected["placebo_positive_cosine"]),
                    "round_reuse_fallback": 0,
                    "mapping_seed": int(seed),
                    "minimum_supported_cells": minimum_cells,
                    "target_supported_cells": int(support_counts[target_index]),
                    "donor_current_supported_cells": int(current_support_counts[donor_index]),
                }
            )
    audit_rows.sort(
        key=lambda row: (int(row["target_dataset_index"]), int(row["negative_rank"]))
    )
    selected_values = mapping[eligible_mask]
    safe_candidate_counts = np.asarray(
        [len(candidates[index]) for index in eligible_targets],
        dtype=np.float64,
    )
    summary = {
        "plan_version": MATCHING_NEGATIVE_SOURCE_PLAN_VERSION,
        "split": str(split).strip().lower(),
        "seed": int(seed),
        "negative_count": k,
        "minimum_supported_cells": minimum_cells,
        "duplicate_cosine_threshold": threshold,
        "target_count": len(items),
        "eligible_target_count": len(eligible_targets),
        "eligible_target_fraction": len(eligible_targets) / len(items),
        "donor_pool_count": len(donor_pool),
        "minimum_safe_candidate_count": int(np.min(safe_candidate_counts)),
        "safe_candidate_count_p05": float(
            np.quantile(safe_candidate_counts, 0.05)
        ),
        "safe_candidate_count_median": float(
            np.median(safe_candidate_counts)
        ),
        "maximum_union_protected_count": max(protected_counts),
        "native_positive_as_negative_count": int(
            sum(target_index in mapping[target_index].tolist() for target_index in eligible_targets)
        ),
        "placebo_positive_as_negative_count": int(
            sum(
                int(text_alignment_plan.placebo_source_indices[target_index])
                in mapping[target_index].tolist()
                for target_index in eligible_targets
            )
        ),
        "rank_reuse_count": int(
            sum(
                len(selected_values[:, rank].tolist())
                - len(set(selected_values[:, rank].tolist()))
                for rank in range(k)
            )
        ),
        "fallback_count": 0,
        "positive_alignment_sha256": text_alignment_plan.sha256,
    }
    return TransitionMatchingNegativePlan(
        split=str(split).strip().lower(),
        seed=int(seed),
        negative_count=k,
        minimum_supported_cells=minimum_cells,
        duplicate_cosine_threshold=threshold,
        target_sample_ids=sample_ids,
        target_surface_pair_ids=pair_ids,
        positive_alignment_sha256=text_alignment_plan.sha256,
        negative_source_indices=mapping,
        eligible_mask=eligible_mask,
        audit_rows=tuple(audit_rows),
        summary=summary,
    )


def _stable_fraction(*parts: Any) -> float:
    digest = hashlib.sha256(
        "|".join(str(part) for part in parts).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big") / float(2**64)


def _current_regime(sample: Any) -> tuple[float, float, int, str, float]:
    current = np.asarray(sample.current_surface, dtype=np.float64).reshape(-1)
    support = (
        np.asarray(sample.current_support_mask, dtype=bool).reshape(-1)
        if sample.current_support_mask is not None
        else np.ones(current.size, dtype=bool)
    )
    if not bool(np.any(support)):
        raise ValueError(f"Sample {sample.sample_id} has empty current support.")
    mean_log_iv = float(np.mean(np.log(np.clip(current[support], 1.0e-4, None))))
    support_fraction = float(np.mean(support))
    timestamp = pd.Timestamp(sample.current_snapshot_time_utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    session_bucket = int(timestamp.tz_convert("Europe/London").hour // 4)
    metadata = dict(sample.metadata or {})
    market_state = _clean_token(metadata.get("publication_market_state", ""))
    news_count = float(metadata.get("news_count", 1) or 1)
    if not np.isfinite(news_count):
        news_count = 1.0
    return mean_log_iv, support_fraction, session_bucket, market_state, news_count


def build_transition_matching_donor_mapping(
    samples: Sequence[Any],
    *,
    negative_count: int,
    minimum_supported_cells: int,
    seed: int,
    duplicate_cosine_threshold: float = 0.995,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Build a deterministic, split-local current-regime donor mapping.

    Pair support determines whether a target transition contributes to the
    auxiliary loss.  Donor eligibility and ranking use only origin-time
    information: future surfaces, donor target support, fit errors, and model
    scores never enter donor selection. Exact lineage overlaps and
    near-duplicate text embeddings are excluded.
    """

    items = list(samples)
    k = int(negative_count)
    minimum_cells = int(minimum_supported_cells)
    threshold = float(duplicate_cosine_threshold)
    if k < 1:
        raise ValueError("negative_count must be positive.")
    if minimum_cells < 1:
        raise ValueError("minimum_supported_cells must be positive.")
    if not -1.0 < threshold < 1.0:
        raise ValueError("duplicate_cosine_threshold must be strictly between -1 and 1.")

    mapping = np.full((len(items), k), -1, dtype=np.int64)
    support_counts = np.asarray(
        [int(np.asarray(item.evaluation_support_mask, dtype=bool).sum()) for item in items],
        dtype=np.int64,
    )
    target_eligible = [
        index
        for index, item in enumerate(items)
        if float(item.metadata.get("has_text", 1.0)) > 0.0
        and int(support_counts[index]) >= minimum_cells
    ]
    current_support_counts = np.asarray(
        [
            int(
                np.asarray(
                    item.current_support_mask
                    if item.current_support_mask is not None
                    else np.ones_like(item.current_surface, dtype=bool),
                    dtype=bool,
                ).sum()
            )
            for item in items
        ],
        dtype=np.int64,
    )
    # Donor texts need not have a sufficiently observed target transition.
    # Restricting the donor pool by target availability would leak a
    # post-origin quantity into hard-negative selection.  Only text presence
    # and current-time support are required for donors.
    donor_pool = [
        index
        for index, item in enumerate(items)
        if float(item.metadata.get("has_text", 1.0)) > 0.0
        and int(current_support_counts[index]) > 0
    ]
    if not target_eligible:
        raise ValueError(
            "No training samples are eligible for transition matching after the "
            "text/support filters."
        )
    if len(donor_pool) <= k:
        raise ValueError(
            "Transition matching requires at least negative_count + 1 eligible "
            f"donor texts; got donors={len(donor_pool)} negative_count={k}."
        )

    text = np.stack(
        [
            np.asarray(items[index].text_embedding, dtype=np.float64)
            for index in donor_pool
        ],
        axis=0,
    )
    text_norms = np.linalg.norm(text, axis=1)
    if bool(np.any(~np.isfinite(text_norms))) or bool(np.any(text_norms <= 1.0e-12)):
        raise ValueError("Every transition-matching text embedding must be finite and non-zero.")
    text_unit = text / text_norms[:, None]
    position = {
        sample_index: offset for offset, sample_index in enumerate(donor_pool)
    }
    identities = {index: _text_identity(items[index]) for index in donor_pool}
    transition_regimes = {
        index: _current_regime(items[index]) for index in donor_pool
    }
    pair_to_index = {
        str(item.surface_pair_id): index for index, item in enumerate(items)
    }
    # In the shuffled-text control an item carries text from another pair.
    # Rank that text by its true origin-time regime rather than the current
    # surface of the row onto which it was permuted.
    donor_regimes = {
        index: transition_regimes[
            pair_to_index.get(identities[index]["surface_pair_id"], index)
        ]
        for index in donor_pool
    }
    log_scale = float(np.std([donor_regimes[index][0] for index in donor_pool]))
    support_scale = float(
        np.std([donor_regimes[index][1] for index in donor_pool])
    )
    news_scale = float(np.std([donor_regimes[index][4] for index in donor_pool]))
    log_scale = max(log_scale, 1.0e-6)
    support_scale = max(support_scale, 1.0e-6)
    news_scale = max(news_scale, 1.0)

    candidates: dict[int, list[dict[str, Any]]] = {}
    for target_index in target_eligible:
        target = items[target_index]
        target_identity = identities[target_index]
        target_regime = transition_regimes[target_index]
        target_rows: list[dict[str, Any]] = []
        for donor_index in donor_pool:
            if donor_index == target_index:
                continue
            donor = items[donor_index]
            donor_identity = identities[donor_index]
            if donor.surface_pair_id == target.surface_pair_id:
                continue
            if donor_identity["surface_pair_id"] == target_identity["surface_pair_id"]:
                continue
            if target_identity["article_ids"] & donor_identity["article_ids"]:
                continue
            if target_identity["source_files"] & donor_identity["source_files"]:
                continue
            if (
                target_identity["event_group"]
                and target_identity["event_group"] == donor_identity["event_group"]
            ):
                continue
            if (
                target_identity["news_cluster_id"]
                and target_identity["news_cluster_id"]
                == donor_identity["news_cluster_id"]
            ):
                continue
            if (
                target_identity["feature_sha256"]
                and target_identity["feature_sha256"]
                == donor_identity["feature_sha256"]
            ):
                continue
            cosine = float(
                np.dot(text_unit[position[target_index]], text_unit[position[donor_index]])
            )
            if cosine > threshold:
                continue

            donor_regime = donor_regimes[donor_index]
            same_market_state = bool(
                target_regime[3]
                and donor_regime[3]
                and target_regime[3] == donor_regime[3]
            )
            same_session_bucket = target_regime[2] == donor_regime[2]
            tier = 0 if same_market_state and same_session_bucket else 1 if same_market_state else 2
            regime_distance = (
                abs(target_regime[0] - donor_regime[0]) / log_scale
                + abs(target_regime[1] - donor_regime[1]) / support_scale
                + 0.25 * abs(target_regime[2] - donor_regime[2])
                + 0.10 * abs(target_regime[4] - donor_regime[4]) / news_scale
            )
            target_rows.append(
                {
                    "donor_index": donor_index,
                    "tier": tier,
                    "regime_distance": float(regime_distance),
                    "embedding_cosine": cosine,
                    "tie_break": _stable_fraction(
                        int(seed),
                        target.surface_pair_id,
                        donor.surface_pair_id,
                    ),
                }
            )
        target_rows.sort(
            key=lambda row: (
                int(row["tier"]),
                float(row["regime_distance"]),
                float(row["tie_break"]),
                int(row["donor_index"]),
            )
        )
        if len(target_rows) < k:
            raise ValueError(
                "Insufficient duplicate-safe transition-matching donors for "
                f"sample={target.sample_id}: required={k}, available={len(target_rows)}."
            )
        candidates[target_index] = target_rows

    selected_rows: list[dict[str, Any]] = []
    selected_by_target: dict[int, set[int]] = {
        index: set() for index in target_eligible
    }
    for negative_rank in range(k):
        used_this_round: set[int] = set()
        target_order = sorted(
            target_eligible,
            key=lambda index: (
                _stable_fraction(int(seed), negative_rank, items[index].surface_pair_id),
                index,
            ),
        )
        for target_index in target_order:
            available = [
                row
                for row in candidates[target_index]
                if int(row["donor_index"]) not in selected_by_target[target_index]
            ]
            preferred = [
                row
                for row in available
                if int(row["donor_index"]) not in used_this_round
            ]
            reuse_fallback = not bool(preferred)
            chosen = (preferred or available)[0]
            donor_index = int(chosen["donor_index"])
            mapping[target_index, negative_rank] = donor_index
            selected_by_target[target_index].add(donor_index)
            used_this_round.add(donor_index)
            target = items[target_index]
            donor = items[donor_index]
            selected_rows.append(
                {
                    "target_dataset_index": target_index,
                    "target_sample_id": str(target.sample_id),
                    "target_surface_pair_id": str(target.surface_pair_id),
                    "target_text_source_surface_pair_id": identities[target_index][
                        "surface_pair_id"
                    ],
                    "negative_rank": negative_rank,
                    "donor_dataset_index": donor_index,
                    "donor_sample_id": str(donor.sample_id),
                    "donor_surface_pair_id": str(donor.surface_pair_id),
                    "donor_text_source_surface_pair_id": identities[donor_index][
                        "surface_pair_id"
                    ],
                    "selection_tier": int(chosen["tier"]),
                    "regime_distance": float(chosen["regime_distance"]),
                    "embedding_cosine": float(chosen["embedding_cosine"]),
                    "round_reuse_fallback": int(reuse_fallback),
                    "mapping_seed": int(seed),
                    "minimum_supported_cells": minimum_cells,
                    "target_supported_cells": int(support_counts[target_index]),
                    "donor_current_supported_cells": int(
                        current_support_counts[donor_index]
                    ),
                }
            )

    selected_rows.sort(
        key=lambda row: (
            int(row["target_dataset_index"]),
            int(row["negative_rank"]),
        )
    )
    return mapping, selected_rows
