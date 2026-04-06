from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


MetricRow = Mapping[str, Any]


def write_metrics_json(metrics_rows: Sequence[MetricRow], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(list(metrics_rows), handle, indent=2, ensure_ascii=False)
    return output


def write_metrics_csv(metrics_rows: Sequence[MetricRow], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = [dict(row) for row in metrics_rows]
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with output.open("w", encoding="utf-8", newline="") as handle:
        if fieldnames:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return output


def write_best_checkpoint(payload: Mapping[str, Any], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, ensure_ascii=False)
    return output
