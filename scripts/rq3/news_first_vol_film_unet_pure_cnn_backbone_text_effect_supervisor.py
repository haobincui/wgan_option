"""Spawn-safe thin entrypoint for the complete text-effect pipeline.

There are deliberately no model, pandas, or torch imports at module scope.
Multiprocessing ``spawn`` may re-import this file in prediction children; the
child initializer can therefore restrict GPU visibility before loading the
experiment implementation.
"""

from __future__ import annotations

from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    from scripts.rq3.news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed import (
        main as experiment_main,
    )

    return experiment_main(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
