"""Train-only PCA and Ridge baseline for fixed-grid log-IV changes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from film_wgan.data import FilmWGANSample

_VOL_FLOOR = 1e-4
_VOL_CEIL = 5.0


def _current_and_delta(samples: Sequence[FilmWGANSample]) -> tuple[np.ndarray, np.ndarray]:
    current = np.stack(
        [np.log(np.clip(sample.current_surface.reshape(-1), _VOL_FLOOR, None)) for sample in samples],
        axis=0,
    )
    target = np.stack(
        [np.log(np.clip(sample.target_surface.reshape(-1), _VOL_FLOOR, None)) for sample in samples],
        axis=0,
    )
    return current.astype(np.float64), (target - current).astype(np.float64)


@dataclass(frozen=True)
class PCARidgeSelection:
    n_components: int
    alpha: float
    val_surface_mae: float


class PCARidgeForecaster:
    """Linear quantitative benchmark with validation-only hyperparameter selection."""

    def __init__(self, *, n_components: int, alpha: float):
        self.n_components = int(n_components)
        self.alpha = float(alpha)
        self.current_scaler = StandardScaler()
        self.delta_pca = PCA(n_components=self.n_components, svd_solver="full")
        self.ridge = Ridge(alpha=self.alpha)
        self.surface_shape: tuple[int, int] | None = None

    def fit(self, samples: Sequence[FilmWGANSample]) -> "PCARidgeForecaster":
        if not samples:
            raise ValueError("PCA-Ridge requires non-empty training samples.")
        current, delta = _current_and_delta(samples)
        max_components = min(current.shape[0] - 1, delta.shape[1])
        if self.n_components > max_components:
            raise ValueError(
                f"n_components={self.n_components} exceeds the train-only limit {max_components}."
            )
        current_scaled = self.current_scaler.fit_transform(current)
        delta_scores = self.delta_pca.fit_transform(delta)
        self.ridge.fit(current_scaled, delta_scores)
        self.surface_shape = samples[0].surface_shape
        return self

    def predict(self, samples: Sequence[FilmWGANSample]) -> np.ndarray:
        if self.surface_shape is None:
            raise RuntimeError("PCA-Ridge must be fitted before prediction.")
        if not samples:
            return np.empty((0, *self.surface_shape), dtype=np.float32)
        current, _delta = _current_and_delta(samples)
        scores = self.ridge.predict(self.current_scaler.transform(current))
        predicted_delta = self.delta_pca.inverse_transform(scores)
        predicted_log = current + predicted_delta
        predicted = np.exp(np.clip(predicted_log, np.log(_VOL_FLOOR), np.log(_VOL_CEIL)))
        return predicted.reshape(len(samples), *self.surface_shape).astype(np.float32)

    @staticmethod
    def mean_surface_mae(predicted: np.ndarray, samples: Sequence[FilmWGANSample]) -> float:
        target = np.stack([sample.target_surface for sample in samples], axis=0).astype(np.float64)
        return float(np.mean(np.abs(np.asarray(predicted, dtype=np.float64) - target)))

    @classmethod
    def select(
        cls,
        *,
        train_samples: Sequence[FilmWGANSample],
        val_samples: Sequence[FilmWGANSample],
        component_grid: Sequence[int] = (4, 8, 16, 32),
        alpha_grid: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    ) -> tuple["PCARidgeForecaster", PCARidgeSelection, list[dict[str, float]]]:
        if not val_samples:
            raise ValueError("PCA-Ridge selection requires a non-empty validation split.")
        candidates: list[tuple[float, int, float, PCARidgeForecaster]] = []
        audit: list[dict[str, float]] = []
        max_components = min(len(train_samples) - 1, train_samples[0].current_surface.size)
        for raw_components in component_grid:
            components = int(raw_components)
            if components > max_components:
                continue
            for raw_alpha in alpha_grid:
                alpha = float(raw_alpha)
                model = cls(n_components=components, alpha=alpha).fit(train_samples)
                val_mae = cls.mean_surface_mae(model.predict(val_samples), val_samples)
                audit.append(
                    {
                        "n_components": float(components),
                        "alpha": alpha,
                        "val_surface_mae": val_mae,
                    }
                )
                candidates.append((val_mae, components, alpha, model))
        if not candidates:
            raise ValueError("No valid PCA-Ridge hyperparameter candidates were available.")
        val_mae, components, alpha, model = min(candidates, key=lambda item: (item[0], item[1], item[2]))
        return model, PCARidgeSelection(components, alpha, val_mae), audit
