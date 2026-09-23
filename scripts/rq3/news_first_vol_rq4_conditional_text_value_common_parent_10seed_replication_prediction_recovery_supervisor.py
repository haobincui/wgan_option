"""Spawn-safe entrypoint for the RQ4 prediction-boundary recovery."""

from __future__ import annotations

from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    from scripts.rq3.news_first_vol_rq4_conditional_text_value_common_parent_10seed_replication_prediction_recovery import (
        main as recovery_main,
    )

    return recovery_main(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
