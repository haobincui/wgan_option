"""Quantitative baselines used by the RQ1 FiLM-WGAN experiment."""

from .pca_ridge import PCARidgeForecaster, PCARidgeSelection

__all__ = ["PCARidgeForecaster", "PCARidgeSelection"]
