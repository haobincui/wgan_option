"""Audits for article identifiers, source text, and frozen LP embeddings."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .text_transform import sha256_file


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def _optional_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _vector(value: Any) -> np.ndarray:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.asarray([], dtype=np.float32)
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=np.float32).reshape(-1)
    text = str(value).strip()
    if not text:
        return np.asarray([], dtype=np.float32)
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(text)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
        if isinstance(parsed, (list, tuple)):
            return np.asarray(parsed, dtype=np.float32).reshape(-1)
    raise ValueError("LP_embedding contains an invalid serialized vector.")


def _normalized_text(value: str) -> str:
    return " ".join(_TOKEN_PATTERN.findall(value.lower()))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest() if value else ""


def _simhash64(value: str) -> str:
    tokens = _TOKEN_PATTERN.findall(value.lower())
    if not tokens:
        return ""
    accumulator = np.zeros(64, dtype=np.int64)
    for token in tokens:
        bits = int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "big")
        for index in range(64):
            accumulator[index] += 1 if (bits >> index) & 1 else -1
    fingerprint = 0
    for index, score in enumerate(accumulator):
        if score >= 0:
            fingerprint |= 1 << index
    return f"{fingerprint:016x}"


def _near_simhash_cluster_ids(
    fingerprints: list[str],
    *,
    max_hamming_distance: int = 3,
) -> list[str]:
    """Find exact/near SimHash components without an O(n^2) full scan."""

    parents = list(range(len(fingerprints)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    hashes = [
        int(value, 16) if value else None
        for value in fingerprints
    ]
    buckets: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
    for index, fingerprint in enumerate(hashes):
        if fingerprint is None:
            continue
        for band in range(4):
            buckets[(band, (fingerprint >> (band * 16)) & 0xFFFF)].append(
                index
            )
    checked: set[tuple[int, int]] = set()
    for members in buckets.values():
        for position, left in enumerate(members):
            for right in members[position + 1 :]:
                key = (min(left, right), max(left, right))
                if key in checked:
                    continue
                checked.add(key)
                if (int(hashes[left]) ^ int(hashes[right])).bit_count() <= int(
                    max_hamming_distance
                ):
                    union(left, right)

    components: defaultdict[int, list[int]] = defaultdict(list)
    for index, fingerprint in enumerate(hashes):
        if fingerprint is not None:
            components[find(index)].append(index)
    output = [""] * len(fingerprints)
    for members in components.values():
        if len(members) < 2:
            continue
        member_ids = [index + 1 for index in sorted(members)]
        cluster_id = "near_" + hashlib.sha256(
            ",".join(str(value) for value in member_ids).encode("utf-8")
        ).hexdigest()[:16]
        for index in members:
            output[index] = cluster_id
    return output


def build_text_lineage_audit(
    news_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build row and duplicate-cluster audits from the frozen news workbook."""

    required = {"LP", "LP_embedding"}
    missing = sorted(required - set(news_frame.columns))
    if missing:
        raise ValueError(f"News workbook is missing text-lineage columns: {missing}")
    rows: list[dict[str, Any]] = []
    for zero_index, source_row in news_frame.reset_index(drop=True).iterrows():
        news_row_id = int(zero_index) + 1
        article_id = _optional_text(source_row.get("ArticleID", ""))
        source_file = _optional_text(source_row.get("SourceFile", ""))
        lp_text = _optional_text(source_row.get("LP", ""))
        normalized_text = _normalized_text(lp_text)
        embedding = _vector(source_row.get("LP_embedding", ""))
        embedding_nonzero = bool(
            embedding.size and np.any(np.abs(embedding) > 0.0)
        )
        if not lp_text and embedding_nonzero:
            status = "exclude_empty_lp_nonzero_embedding"
        elif not lp_text:
            status = "exclude_empty_lp"
        elif embedding.size == 0:
            status = "exclude_missing_embedding"
        elif not np.all(np.isfinite(embedding)):
            status = "exclude_nonfinite_embedding"
        else:
            status = "usable"
        rows.append(
            {
                "news_row_id": news_row_id,
                "sample_id": f"news_{news_row_id}",
                "article_id_raw": article_id,
                "article_id_effective": article_id or f"news_row_{news_row_id}",
                "article_id_was_missing": int(not article_id),
                "source_file": source_file,
                "lp_character_count": len(lp_text),
                "lp_token_count": len(_TOKEN_PATTERN.findall(lp_text)),
                "lp_text_sha256": _sha256_text(normalized_text),
                "lp_text_simhash64": _simhash64(lp_text),
                "embedding_dim": int(embedding.size),
                "embedding_l2_norm": (
                    float(np.linalg.norm(embedding)) if embedding.size else 0.0
                ),
                "embedding_sha256": (
                    hashlib.sha256(
                        embedding.astype(np.float32).tobytes(order="C")
                    ).hexdigest()
                    if embedding.size
                    else ""
                ),
                "lineage_status": status,
            }
        )
    row_audit = pd.DataFrame(rows)
    row_audit["near_duplicate_cluster_id"] = _near_simhash_cluster_ids(
        row_audit["lp_text_simhash64"].astype(str).tolist(),
        max_hamming_distance=3,
    )

    groups: list[dict[str, Any]] = []
    definitions = (
        ("article_id", "article_id_raw"),
        ("exact_lp_text", "lp_text_sha256"),
        ("exact_embedding", "embedding_sha256"),
        ("near_text_hamming_le_3", "near_duplicate_cluster_id"),
    )
    for group_type, column in definitions:
        grouped: defaultdict[str, list[int]] = defaultdict(list)
        for row in row_audit.itertuples(index=False):
            key = str(getattr(row, column))
            if key:
                grouped[key].append(int(row.news_row_id))
        for group_key, news_row_ids in sorted(grouped.items()):
            if len(news_row_ids) < 2:
                continue
            groups.append(
                {
                    "group_type": group_type,
                    "group_key": group_key,
                    "row_count": len(news_row_ids),
                    "news_row_ids": json.dumps(news_row_ids),
                    "sample_ids": json.dumps(
                        [f"news_{news_row_id}" for news_row_id in news_row_ids]
                    ),
                }
            )
    group_audit = pd.DataFrame(
        groups,
        columns=[
            "group_type",
            "group_key",
            "row_count",
            "news_row_ids",
            "sample_ids",
        ],
    )
    dimensions = (
        row_audit.loc[row_audit["embedding_dim"] > 0, "embedding_dim"]
        .value_counts()
        .sort_index()
    )
    summary = {
        "row_count": int(len(row_audit)),
        "usable_row_count": int((row_audit["lineage_status"] == "usable").sum()),
        "excluded_row_count": int((row_audit["lineage_status"] != "usable").sum()),
        "missing_article_id_count": int(row_audit["article_id_was_missing"].sum()),
        "status_counts": {
            str(key): int(value)
            for key, value in row_audit["lineage_status"].value_counts().items()
        },
        "embedding_dimension_counts": {
            str(int(key)): int(value) for key, value in dimensions.items()
        },
        "duplicate_group_counts": {
            group_type: int((group_audit["group_type"] == group_type).sum())
            if not group_audit.empty
            else 0
            for group_type, _column in definitions
        },
    }
    return row_audit, group_audit, summary


def write_text_lineage_artifacts(
    *,
    news_workbook_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    source_path = Path(news_workbook_path)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    frame = pd.read_excel(source_path)
    row_audit, group_audit, summary = build_text_lineage_audit(frame)
    row_path = output_root / "text_lineage_rows.csv"
    group_path = output_root / "text_duplicate_groups.csv"
    manifest_path = output_root / "lp_embedding_manifest.json"
    row_audit.to_csv(row_path, index=False)
    group_audit.to_csv(group_path, index=False)
    manifest = {
        "schema_version": 1,
        "source_workbook": str(source_path),
        "source_workbook_sha256": sha256_file(source_path),
        "text_column": "LP",
        "embedding_column": "LP_embedding",
        "embedding_model": "text-embedding-3-large",
        "embedding_dimension": 1024,
        "model_source": "project documentation",
        "upstream_api_request_log_available": False,
        "upstream_preprocessing_manifest_available": False,
        "reproducibility_status": "frozen_vectors_with_incomplete_upstream_generation_metadata",
        "audit_summary": summary,
        "row_audit_sha256": sha256_file(row_path),
        "duplicate_group_audit_sha256": sha256_file(group_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "row_audit": row_path,
        "duplicate_groups": group_path,
        "manifest": manifest_path,
    }
