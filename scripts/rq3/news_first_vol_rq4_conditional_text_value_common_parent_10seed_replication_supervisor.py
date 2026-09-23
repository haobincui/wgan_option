"""Spawn-safe thin entrypoint for the RQ4 common-parent replication."""

from __future__ import annotations

from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    from scripts.rq3.news_first_vol_rq4_conditional_text_value_common_parent_10seed_replication import (
        main as experiment_main,
    )

    return experiment_main(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
