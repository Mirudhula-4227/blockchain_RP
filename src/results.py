"""Shared experiment-results schema and CSV writer."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping

RESULT_COLUMNS = [
    "dataset", "model", "method", "clients", "attack", "malicious_fraction", "alpha", "seed",
    "round", "accuracy", "macro_f1", "precision", "recall", "fpr", "params",
    "model_kb", "bytes_per_round", "round_time_s",
]


def append_result(path: str | Path, row: Mapping[str, object]) -> None:
    """Append one schema-complete result row, writing the header only once."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    missing = [column for column in RESULT_COLUMNS if column not in row]
    if missing:
        raise ValueError(f"Missing result columns: {missing}")
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS, extrasaction="raise", lineterminator="\n")
        if path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow({column: row[column] for column in RESULT_COLUMNS})
