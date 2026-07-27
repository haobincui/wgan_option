"""Train-only text preprocessing artifacts for FiLM-WGAN experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.decomposition import PCA


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_ids_sha256(values: Sequence[str]) -> str:
    payload = "\n".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def l2_normalize_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return (array / np.clip(norms, 1e-12, None)).astype(np.float32)


@dataclass(frozen=True)
class FilmWGANTextTransform:
    """Serializable linear preprocessing fitted only on training surface pairs."""

    mode: str
    input_dim: int
    output_dim: int
    mean: np.ndarray
    scale: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray
    whiten: bool
    metadata: dict[str, Any]

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        one_row = array.ndim == 1
        if one_row:
            array = array.reshape(1, -1)
        if array.ndim != 2 or int(array.shape[1]) != int(self.input_dim):
            raise ValueError(
                f"Text transform expected shape (*, {self.input_dim}), got {tuple(array.shape)}."
            )
        normalized_mode = str(self.mode).strip().lower()
        if normalized_mode in {"raw_l2", "coordinate_zscore"}:
            transformed = l2_normalize_rows(array) if normalized_mode == "raw_l2" else array
        elif normalized_mode == "pca":
            transformed = (array - self.mean.reshape(1, -1)) @ self.components.T
            if self.whiten:
                transformed = transformed / np.sqrt(
                    np.clip(self.explained_variance.reshape(1, -1), 1e-12, None)
                )
        elif normalized_mode == "zscore_pad":
            normalized = (array - self.mean.reshape(1, -1)) / np.clip(
                self.scale.reshape(1, -1),
                1e-12,
                None,
            )
            transformed = np.zeros((array.shape[0], int(self.output_dim)), dtype=np.float32)
            transformed[:, : int(self.input_dim)] = normalized
        else:
            raise ValueError(f"Unsupported text transform mode: {self.mode}")
        transformed = np.asarray(transformed, dtype=np.float32)
        return transformed[0] if one_row else transformed

    def save(self, path: str | Path) -> tuple[Path, Path]:
        artifact_path = Path(path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            artifact_path,
            mode=np.asarray(self.mode),
            input_dim=np.asarray(self.input_dim, dtype=np.int64),
            output_dim=np.asarray(self.output_dim, dtype=np.int64),
            mean=np.asarray(self.mean, dtype=np.float32),
            scale=np.asarray(self.scale, dtype=np.float32),
            components=np.asarray(self.components, dtype=np.float32),
            explained_variance=np.asarray(self.explained_variance, dtype=np.float32),
            whiten=np.asarray(int(self.whiten), dtype=np.int8),
        )
        metadata_path = artifact_path.with_name(f"{artifact_path.stem}_metadata.json")
        payload = dict(self.metadata)
        payload.update(
            {
                "artifact_path": str(artifact_path),
                "artifact_sha256": sha256_file(artifact_path),
                "mode": self.mode,
                "input_dim": int(self.input_dim),
                "output_dim": int(self.output_dim),
                "whiten": bool(self.whiten),
            }
        )
        metadata_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        return artifact_path, metadata_path

    @classmethod
    def load(cls, path: str | Path) -> "FilmWGANTextTransform":
        artifact_path = Path(path)
        if not artifact_path.is_file():
            raise FileNotFoundError(f"text_transform_path does not exist: {artifact_path}")
        with np.load(artifact_path, allow_pickle=False) as payload:
            mode = str(payload["mode"].item())
            input_dim = int(payload["input_dim"].item())
            transform = cls(
                mode=mode,
                input_dim=input_dim,
                output_dim=int(payload["output_dim"].item()),
                mean=np.asarray(payload["mean"], dtype=np.float32),
                scale=(
                    np.asarray(payload["scale"], dtype=np.float32)
                    if "scale" in payload.files
                    else np.ones(input_dim, dtype=np.float32)
                ),
                components=np.asarray(payload["components"], dtype=np.float32),
                explained_variance=np.asarray(payload["explained_variance"], dtype=np.float32),
                whiten=bool(int(payload["whiten"].item())),
                metadata={},
            )
        metadata_path = artifact_path.with_name(f"{artifact_path.stem}_metadata.json")
        metadata: dict[str, Any] = {}
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            recorded_hash = str(metadata.get("artifact_sha256", ""))
            actual_hash = sha256_file(artifact_path)
            if recorded_hash and recorded_hash != actual_hash:
                raise ValueError(
                    f"Text transform SHA256 mismatch for {artifact_path}: "
                    f"expected {recorded_hash}, found {actual_hash}."
                )
        return cls(
            mode=transform.mode,
            input_dim=transform.input_dim,
            output_dim=transform.output_dim,
            mean=transform.mean,
            scale=transform.scale,
            components=transform.components,
            explained_variance=transform.explained_variance,
            whiten=transform.whiten,
            metadata=metadata,
        )


def fit_text_transform(
    values: np.ndarray,
    *,
    mode: str,
    components: int,
    whiten: bool,
    train_pair_ids: Sequence[str],
    input_workbook_path: str | Path,
    output_dim: int | None = None,
    input_feature_path: str | Path | None = None,
) -> FilmWGANTextTransform:
    """Fit a deterministic transform using training-pair text only."""

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError(f"Expected a non-empty 2D text matrix, got {array.shape}.")
    normalized_mode = str(mode).strip().lower()
    input_dim = int(array.shape[1])
    metadata = {
        "train_pair_count": int(len(train_pair_ids)),
        "train_pair_ids_sha256": ordered_ids_sha256(train_pair_ids),
        "input_workbook_path": str(Path(input_workbook_path)),
        "input_workbook_sha256": sha256_file(input_workbook_path),
        "fit_sample_count": int(array.shape[0]),
    }
    if input_feature_path:
        feature_path = Path(input_feature_path)
        metadata.update(
            {
                "input_feature_path": str(feature_path),
                "input_feature_sha256": sha256_file(feature_path),
            }
        )
    if normalized_mode == "pca":
        requested = int(components)
        maximum = min(int(array.shape[0]), input_dim)
        if requested > maximum:
            raise ValueError(
                f"text_pca_components={requested} exceeds min(train_pairs, input_dim)={maximum}."
            )
        estimator = PCA(
            n_components=requested,
            whiten=False,
            svd_solver="full",
        )
        estimator.fit(array)
        return FilmWGANTextTransform(
            mode=normalized_mode,
            input_dim=input_dim,
            output_dim=requested,
            mean=np.asarray(estimator.mean_, dtype=np.float32),
            scale=np.ones(input_dim, dtype=np.float32),
            components=np.asarray(estimator.components_, dtype=np.float32),
            explained_variance=np.asarray(estimator.explained_variance_, dtype=np.float32),
            whiten=bool(whiten),
            metadata={
                **metadata,
                "requested_components": requested,
                "explained_variance_ratio_sum": float(
                    np.asarray(estimator.explained_variance_ratio_, dtype=np.float64).sum()
                ),
            },
        )
    if normalized_mode == "zscore_pad":
        resolved_output_dim = int(output_dim if output_dim is not None else input_dim)
        if resolved_output_dim < input_dim:
            raise ValueError(
                f"zscore_pad output_dim={resolved_output_dim} is smaller than input_dim={input_dim}."
            )
        mean = np.mean(array, axis=0).astype(np.float32)
        scale = np.std(array, axis=0).astype(np.float32)
        scale = np.where(scale < 1e-8, 1.0, scale).astype(np.float32)
        components_matrix = np.zeros((resolved_output_dim, input_dim), dtype=np.float32)
        components_matrix[:input_dim, :] = np.eye(input_dim, dtype=np.float32)
        return FilmWGANTextTransform(
            mode=normalized_mode,
            input_dim=input_dim,
            output_dim=resolved_output_dim,
            mean=mean,
            scale=scale,
            components=components_matrix,
            explained_variance=np.var(array, axis=0).astype(np.float32),
            whiten=False,
            metadata={
                **metadata,
                "active_output_dimensions": input_dim,
                "padding_dimensions": resolved_output_dim - input_dim,
            },
        )
    if normalized_mode not in {"raw_l2", "coordinate_zscore"}:
        raise ValueError(f"Unsupported text preprocessing mode: {mode}")
    return FilmWGANTextTransform(
        mode=normalized_mode,
        input_dim=input_dim,
        output_dim=input_dim,
        mean=np.zeros(input_dim, dtype=np.float32),
        scale=np.ones(input_dim, dtype=np.float32),
        components=np.eye(input_dim, dtype=np.float32),
        explained_variance=np.ones(input_dim, dtype=np.float32),
        whiten=False,
        metadata=metadata,
    )
