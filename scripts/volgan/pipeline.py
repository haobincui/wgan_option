"""End-to-end VolGAN pipeline: train → generate_result from the same trainer."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

if __package__ in {None, ""}:
    ROOT_DIR = Path(__file__).resolve().parents[2]
    SRC_DIR = ROOT_DIR / "src"
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

from volgan.config import load_train_config  # noqa: E402
from volgan.trainer import VolGANTrainer  # noqa: E402

logger = logging.getLogger("volgan.pipeline")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full VolGAN pipeline: train, then generate results from the best checkpoint.",
    )
    parser.add_argument("--config", required=True, help="Path to a VolGAN training YAML config.")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and use the latest existing run directory for generation.",
    )
    return parser


def main(argv: list[str] | None = None) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    train_config = load_train_config(args.config)
    trainer = VolGANTrainer(train_config, config_path=args.config)

    if args.skip_train:
        logger.info("Skipping training. Using the latest existing run under: %s", train_config.output_root)
        generate_dir = trainer.generate_result(config_path=args.config)
    else:
        logger.info("Starting training + generate_result with config: %s", args.config)
        generate_dir = trainer.run_pipeline(config_path=args.config)
        logger.info("Pipeline complete. Output directory: %s", generate_dir)

    print("==========================================")
    print("  VolGAN pipeline complete")
    print("==========================================")
    print(f"  Generate dir    : {generate_dir}")
    print("==========================================")

    return generate_dir


if __name__ == "__main__":
    main()
