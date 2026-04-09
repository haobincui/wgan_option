"""Shared inference helpers for result generation and error analysis."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from quantlib.calendar.daycount import DayCountBusN
from quantlib.calendar.holidays import embedded_calendar
from quantlib.vol_surface.algo.svi_surface import SviVolSurface
from wgan_option.config import Config
from wgan_option.models.generator import Generator
from wgan_option.models.svi_regressor import SviRegressor
from wgan_option.utils.merged_xlsx import (
    SVI_FEATURE_ORDER,
    _build_svi_matrix_from_params,
    _normalize_svi_matrix,
)

_RECONSTRUCTION_DAYCOUNT = DayCountBusN("BUS250", embedded_calendar(), 250)
_RECONSTRUCTION_VALUATION_DATE = date(2023, 1, 2)


def build_inference_device(use_cuda: bool) -> torch.device:
    """Resolve the device used for inference."""

    return torch.device("cuda:0" if (bool(use_cuda) and torch.cuda.is_available()) else "cpu")


def ensure_matching_embedding_dim(*, sample_id: str, embedding: np.ndarray, embedding_dim: int) -> None:
    """Fail fast if a sample's embedding width mismatches the checkpoint."""

    _coerce_inference_embedding(
        sample_id=sample_id,
        embedding=embedding,
        embedding_dim=embedding_dim,
    )


def _coerce_inference_embedding(*, sample_id: str, embedding: np.ndarray, embedding_dim: int) -> np.ndarray:
    """Align inference-time embeddings to the checkpoint width.

    The `none` text mode is stored at training time as an all-zero vector with a
    positive width (for example width=1), but `load_vol_surface_samples()` and
    `load_svi_paired_samples()` represent `none` rows as zero-length arrays.
    For inference we pad those zero-length arrays back to the checkpoint width
    while preserving the fail-fast behavior for real mismatches.
    """

    embedding_array = np.asarray(embedding, dtype=np.float32)
    if int(embedding_array.size) == int(embedding_dim):
        return embedding_array
    if int(embedding_array.size) == 0 and int(embedding_dim) > 0:
        return np.zeros(int(embedding_dim), dtype=np.float32)
    raise ValueError(
        f"Embedding dimension mismatch for sample {sample_id}: "
        f"expected {embedding_dim}, got {embedding_array.size}"
    )


def deterministic_noise(noise_dim: int, seed: int, global_index: int, device: torch.device) -> torch.Tensor:
    """Build deterministic generator noise for one sample."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + int(global_index))
    noise = torch.randn((1, int(noise_dim)), generator=generator, dtype=torch.float32)
    return noise.to(device)


def load_vol_generator(checkpoint_path: str | Path, sample: Any, device: torch.device) -> tuple[Generator, Config, int]:
    """Load the trained generator used by vol-surface inference."""

    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    train_config = Config(**checkpoint["config"])
    embedding_dim = int(checkpoint.get("embedding_dim", train_config.embedding_dim))

    model = Generator(
        channels=int(train_config.channels),
        embedding_dim=embedding_dim,
        noise_dim=int(train_config.noise_dim),
        surface_height=int(sample.current_surface.shape[1]),
        surface_width=int(sample.current_surface.shape[2]),
        base_channels=int(getattr(train_config, "gen_base_channels", 32)),
        res_blocks=int(getattr(train_config, "gen_res_blocks", 0)),
        text_hidden_dim=int(getattr(train_config, "gen_text_hidden_dim", 256)),
        text_out_dim=int(getattr(train_config, "gen_text_out_dim", 128)),
        hidden_dim=int(train_config.gen_hidden_dim),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, train_config, embedding_dim


def infer_vol_surface(
    model: Generator,
    sample: Any,
    *,
    noise_dim: int,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    """Run one forward pass of the vol generator and return a 2D surface."""

    current_tensor = torch.tensor(sample.current_surface, dtype=torch.float32, device=device).unsqueeze(0)
    if hasattr(model, "text_encoder") and len(model.text_encoder) > 0 and hasattr(model.text_encoder[0], "in_features"):
        embedding_dim = int(model.text_encoder[0].in_features)
    else:
        embedding_dim = int(sample.text_embedding.size)
    aligned_embedding = _coerce_inference_embedding(
        sample_id=str(sample.sample_id),
        embedding=sample.text_embedding,
        embedding_dim=embedding_dim,
    )
    text_tensor = torch.tensor(aligned_embedding, dtype=torch.float32, device=device).unsqueeze(0)
    noise = deterministic_noise(noise_dim, seed, sample.global_index, device)

    with torch.no_grad():
        generated_surface = model(current_tensor, text_tensor, noise=noise).detach().cpu().numpy()[0, 0]
    return np.asarray(generated_surface, dtype=np.float32)


def load_svi_regressor(checkpoint_path: str | Path, device: torch.device) -> tuple[SviRegressor, Dict[str, Any]]:
    """Load the trained SVI regressor used by SVI inference."""

    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    model = SviRegressor(
        current_input_dim=int(checkpoint["current_input_dim"]),
        embedding_dim=int(checkpoint["embedding_dim"]),
        regression_dim=int(checkpoint["regression_dim"]),
        count_classes=int(checkpoint["max_slices"]),
        hidden_dim=int(checkpoint.get("config", {}).get("svi_hidden_dim", 256)),
        dropout=float(checkpoint.get("config", {}).get("svi_dropout", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def build_current_svi_feature_vector(
    svi_params: Dict[str, List[float]],
    *,
    max_slices: int,
    normalization_stats: Dict[str, Any],
    sample_label: str,
) -> Tuple[np.ndarray, int]:
    """Convert SVI parameters into the normalized current-state feature vector."""

    matrix, mask, count = _build_svi_matrix_from_params(svi_params, max_slices, sample_label=sample_label)
    normalized = _normalize_svi_matrix(matrix, mask, normalization_stats)
    feature_vector = np.concatenate(
        [
            normalized.reshape(-1),
            mask.astype(np.float32),
            np.asarray([float(count) / float(max_slices)], dtype=np.float32),
        ],
        axis=0,
    )
    return feature_vector.astype(np.float32), count


def sanitize_svi_params(svi_params: Dict[str, List[float]]) -> Dict[str, List[float]]:
    """Sort SVI slices by maturity and enforce positive, strictly increasing business days."""

    rows: List[Dict[str, float]] = []
    slice_count = len(svi_params["business_days"])
    for idx in range(slice_count):
        rows.append({feature: float(svi_params[feature][idx]) for feature in SVI_FEATURE_ORDER})

    rows.sort(key=lambda row: row["business_days"])
    previous_day = 0
    sanitized: List[Dict[str, float]] = []
    for row in rows:
        business_day = max(1, int(round(row["business_days"])))
        if business_day <= previous_day:
            business_day = previous_day + 1
        previous_day = business_day
        sanitized.append(
            {
                "business_days": float(business_day),
                "a": float(row["a"]),
                "b": float(row["b"]),
                "rho": float(row["rho"]),
                "m": float(row["m"]),
                "sigma": max(float(row["sigma"]), 1e-8),
            }
        )

    return {feature: [float(row[feature]) for row in sanitized] for feature in SVI_FEATURE_ORDER}


def denormalize_svi_prediction(
    predicted_regression: np.ndarray,
    *,
    predicted_count: int,
    normalization_stats: Dict[str, Any],
    max_slices: int,
) -> Dict[str, List[float]]:
    """Map normalized SVI regression outputs back to raw SVI parameters."""

    feature_dim = len(SVI_FEATURE_ORDER)
    mean = np.asarray(normalization_stats["mean"], dtype=np.float32)
    std = np.asarray(normalization_stats["std"], dtype=np.float32)
    regression_matrix = predicted_regression.reshape(int(max_slices), feature_dim)
    denormalized = regression_matrix * std.reshape(1, feature_dim) + mean.reshape(1, feature_dim)

    raw_params = {
        feature: denormalized[:predicted_count, feature_idx].astype(float).tolist()
        for feature_idx, feature in enumerate(SVI_FEATURE_ORDER)
    }
    return sanitize_svi_params(raw_params)


def reconstruct_svi_surface(
    svi_params: Dict[str, List[float]],
    *,
    strike_grid: np.ndarray,
    maturity_days_grid: np.ndarray,
) -> np.ndarray:
    """Reconstruct a vol surface from SVI parameters on a fixed grid."""

    surface = SviVolSurface(
        valuation_date=_RECONSTRUCTION_VALUATION_DATE,
        svi_params=sanitize_svi_params(svi_params),
        vol_daycount=_RECONSTRUCTION_DAYCOUNT,
    )
    grid = surface.implied_vol_surface(
        percent_strikes=[float(value) for value in strike_grid.tolist()],
        business_days=[int(round(value)) for value in maturity_days_grid.tolist()],
        forward=1.0,
    )
    return np.asarray(grid, dtype=np.float32)


def infer_future_svi(
    model: SviRegressor,
    sample: Any,
    *,
    embedding_dim: int,
    max_slices: int,
    normalization_stats: Dict[str, Any],
    device: torch.device,
) -> tuple[Dict[str, List[float]], int, int]:
    """Run one forward pass of the SVI regressor."""

    aligned_embedding = _coerce_inference_embedding(
        sample_id=str(sample.sample_id),
        embedding=sample.text_embedding,
        embedding_dim=embedding_dim,
    )
    current_features, current_count = build_current_svi_feature_vector(
        sample.current_svi,
        max_slices=max_slices,
        normalization_stats=normalization_stats,
        sample_label=f"{sample.sample_id}:current",
    )
    current_tensor = torch.tensor(current_features, dtype=torch.float32, device=device).unsqueeze(0)
    text_tensor = torch.tensor(aligned_embedding, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        predicted_regression, predicted_count_logits = model(current_tensor, text_tensor)

    predicted_regression_np = predicted_regression.detach().cpu().numpy()[0]
    predicted_count = int(torch.argmax(predicted_count_logits, dim=1).item()) + 1
    predicted_svi = denormalize_svi_prediction(
        predicted_regression_np,
        predicted_count=predicted_count,
        normalization_stats=normalization_stats,
        max_slices=max_slices,
    )
    return predicted_svi, current_count, predicted_count
