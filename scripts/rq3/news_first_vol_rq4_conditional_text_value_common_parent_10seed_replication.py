"""Versioned RQ4 common-parent replication wrapper.

The underlying 280-job lifecycle is intentionally reused without changing its
frozen arm or checkpoint contracts.  This wrapper only binds a fresh config,
a fresh output root, and a spawn-safe worker module for the replication.
"""

from __future__ import annotations

from typing import Sequence

from scripts.rq3 import (
    news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed as experiment,
)


DEFAULT_CONFIG = (
    "configs/rq3/"
    "news_first_vol_rq4_conditional_text_value_common_parent_10seed_replication.yaml"
)
DEFAULT_OUTPUT_DIR = (
    "outputs/experiments/"
    "rq4_conditional_text_value_common_parent_10seed_5m_replication_v1"
)
WORKER_MODULE = (
    "scripts.rq3."
    "news_first_vol_rq4_conditional_text_value_common_parent_10seed_replication"
)
WRAPPER_SOURCE = (
    "scripts/rq3/"
    "news_first_vol_rq4_conditional_text_value_common_parent_10seed_replication.py"
)
SUPERVISOR_SOURCE = (
    "scripts/rq3/"
    "news_first_vol_rq4_conditional_text_value_common_parent_10seed_replication_supervisor.py"
)
_BASE_STAGE_CODE_PATHS = experiment._stage_code_paths


def _stage_code_paths() -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (*_BASE_STAGE_CODE_PATHS(), WRAPPER_SOURCE, SUPERVISOR_SOURCE)
        )
    )


def _configure() -> None:
    experiment.DEFAULT_CONFIG = DEFAULT_CONFIG
    experiment.DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR
    experiment.WORKER_MODULE = WORKER_MODULE
    experiment._stage_code_paths = _stage_code_paths


_configure()


def main(argv: Sequence[str] | None = None) -> int:
    _configure()
    return experiment.main(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["DEFAULT_CONFIG", "DEFAULT_OUTPUT_DIR", "WORKER_MODULE", "main"]
