#!/usr/bin/env python3
"""Write frozen LP text/embedding lineage and duplicate audits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from film_wgan.text_lineage import write_text_lineage_artifacts  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--news-workbook",
        default="data/raw/text_embedding/news_with_openai_embeddings_large.xlsx",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    outputs = write_text_lineage_artifacts(
        news_workbook_path=args.news_workbook,
        output_dir=args.output_dir,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
