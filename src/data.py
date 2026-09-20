"""Leakage-safe tabular IDS preprocessing."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DEFAULT_DROP_PATTERNS = (
    r"^id$", r"_id$", r"flow.?id", r"timestamp", r"time", r"src.?ip", r"dst.?ip",
    r"source.?ip", r"destination.?ip", r"src.?port", r"dst.?port", r"source.?port",
    r"destination.?port", r"^port$", r"mac", r"url", r"uri",
)


def _load_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported dataset format: {path.suffix}; use CSV or Parquet")


def _counts(values: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in values.value_counts(dropna=False).items()}


def _drop_leaky_columns(frame: pd.DataFrame, label_column: str) -> tuple[pd.DataFrame, list[str]]:
    patterns = [re.compile(pattern, re.IGNORECASE) for pattern in DEFAULT_DROP_PATTERNS]
    dropped = [
        column for column in frame.columns
        if column != label_column and any(pattern.search(str(column)) for pattern in patterns)
    ]
    return frame.drop(columns=dropped), dropped


def prepare_dataset(
    raw_path: str | Path,
    output_dir: str | Path,
    label_column: str,
    drop_columns: list[str] | None = None,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> dict[str, Any]:
    """Clean, split, encode, and scale a labeled tabular dataset without leakage."""
    raw_path = Path(raw_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_frame(raw_path)
    if label_column not in frame.columns:
        raise KeyError(f"Label column {label_column!r} not found. Columns: {list(frame.columns)}")
    logs: list[dict[str, Any]] = [{"step": "loaded", "rows": len(frame), "label_counts": _counts(frame[label_column])}]

    frame, dropped_columns = _drop_leaky_columns(frame, label_column)
    explicit_drop_columns = [column for column in (drop_columns or []) if column in frame.columns and column != label_column]
    if explicit_drop_columns:
        frame = frame.drop(columns=explicit_drop_columns)
        dropped_columns.extend(explicit_drop_columns)
    logs.append({"step": "dropped_identifier_and_leaky_columns", "rows": len(frame), "dropped_columns": dropped_columns, "label_counts": _counts(frame[label_column])})
    before_dedup = len(frame)
    frame = frame.drop_duplicates().reset_index(drop=True)
    logs.append({"step": "deduplicated", "rows": len(frame), "duplicates_removed": before_dedup - len(frame), "label_counts": _counts(frame[label_column])})

    infinite_values = int(np.isinf(frame.select_dtypes(include=[np.number])).sum().sum())
    missing_values = int(frame.isna().sum().sum())
    frame = frame.replace([np.inf, -np.inf], np.nan)
    logs.append({"step": "nan_and_infinite_values_marked", "rows": len(frame), "missing_values": missing_values, "infinite_values": infinite_values, "label_counts": _counts(frame[label_column])})

    labels = frame.pop(label_column).astype(str).fillna("unknown")
    labels = labels.replace({"nan": "unknown", "None": "unknown"})
    binary_labels = (labels.str.lower() != "normal").astype(np.int64)
    class_names = sorted(labels.unique().tolist())
    label_mapping = {name: index for index, name in enumerate(class_names)}
    multiclass_labels = labels.map(label_mapping).astype(np.int64).to_numpy()
    logs.append({"step": "labels_created", "rows": len(frame), "multiclass_counts": _counts(labels), "binary_counts": _counts(binary_labels.astype(str))})

    numeric_columns = frame.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [column for column in frame.columns if column not in numeric_columns]
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    transformer = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])
    indices = np.arange(len(frame))
    train_idx, remainder_idx = train_test_split(indices, test_size=validation_size + test_size, stratify=multiclass_labels, random_state=seed)
    relative_test_size = test_size / (validation_size + test_size)
    validation_idx, test_idx = train_test_split(remainder_idx, test_size=relative_test_size, stratify=multiclass_labels[remainder_idx], random_state=seed)
    transformer.fit(frame.iloc[train_idx])
    arrays = {
        "X_train": transformer.transform(frame.iloc[train_idx]).astype(np.float32),
        "X_val": transformer.transform(frame.iloc[validation_idx]).astype(np.float32),
        "X_test": transformer.transform(frame.iloc[test_idx]).astype(np.float32),
        "y_train": multiclass_labels[train_idx], "y_val": multiclass_labels[validation_idx], "y_test": multiclass_labels[test_idx],
        "y_binary_train": binary_labels.to_numpy()[train_idx], "y_binary_val": binary_labels.to_numpy()[validation_idx], "y_binary_test": binary_labels.to_numpy()[test_idx],
    }
    logs.append({"step": "split_and_transformed", "rows": len(frame), "train_rows": len(train_idx), "validation_rows": len(validation_idx), "test_rows": len(test_idx), "feature_count": arrays["X_train"].shape[1], "train_label_counts": _counts(pd.Series(arrays["y_train"])), "validation_label_counts": _counts(pd.Series(arrays["y_val"])), "test_label_counts": _counts(pd.Series(arrays["y_test"]))})

    for name, array in arrays.items():
        np.save(output_dir / f"{name}.npy", array)
    metadata = {"source": str(raw_path), "label_column": label_column, "class_mapping": label_mapping, "binary_mapping": {"normal": 0, "attack": 1}, "feature_count": int(arrays["X_train"].shape[1]), "dropped_columns": dropped_columns, "numeric_columns": numeric_columns, "categorical_columns": categorical_columns, "seed": seed, "logs": logs}
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_path")
    parser.add_argument("output_dir")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--drop-column", action="append", default=[])
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    metadata = prepare_dataset(args.raw_path, args.output_dir, args.label_column, args.drop_column, args.validation_size, args.test_size, args.seed)
    print(json.dumps({"feature_count": metadata["feature_count"], "classes": metadata["class_mapping"], "dropped_columns": metadata["dropped_columns"]}, indent=2))


if __name__ == "__main__":
    main()
