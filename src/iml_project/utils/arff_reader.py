from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import arff


@dataclass(frozen=True)
class FoldPair:
    dataset_name: str
    fold_index: int
    train_path: Path
    test_path: Path


def _decode_arff_value(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray, np.bytes_)):
        return value.decode("utf-8")
    return value


def load_arff_table(path: str | Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load an ARFF file and return a DataFrame, feature names, and feature types."""
    data, meta = arff.loadarff(path)
    feature_names = list(meta.names())
    feature_types = [str(item).lower() for item in meta.types()]
    row_data = []
    for row in data:
        row_data.append(
            {key: _decode_arff_value(value) for key, value in zip(feature_names, row)}
        )

    frame = pd.DataFrame(row_data, columns=feature_names)
    return frame, feature_names, feature_types


def discover_fold_pairs(data_root: str | Path) -> dict[str, list[FoldPair]]:
    """Discover all complete train/test fold pairs in a data folder."""
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Data root does not exist: {root}")

    groups: dict[str, list[FoldPair]] = {}
    pattern = re.compile(
        r"^(?P<base>.+)\.fold\.(?P<fold>\d{6})\.(?P<kind>train|test)\.arff$"
    )

    for dataset_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        fold_map: dict[int, dict[str, Path]] = {}
        for arff_file in sorted(dataset_dir.glob("*.arff")):
            match = pattern.match(arff_file.name)
            if match is None:
                continue
            fold_index = int(match.group("fold"))
            kind = match.group("kind")
            fold_map.setdefault(fold_index, {})[kind] = arff_file

        pairs = [
            FoldPair(
                dataset_name=dataset_dir.name,
                fold_index=fold_index,
                train_path=paths["train"],
                test_path=paths["test"],
            )
            for fold_index, paths in sorted(fold_map.items())
            if {"train", "test"}.issubset(paths)
        ]
        groups[dataset_dir.name] = pairs

    return groups


def validate_fold_pairs(groups: dict[str, list[FoldPair]]) -> None:
    """Ensure each dataset has exactly ten complete folds and no missing indices."""
    for dataset_name, pairs in groups.items():
        if len(pairs) != 10:
            raise ValueError(
                f"Dataset '{dataset_name}' has {len(pairs)} fold pairs instead of 10."
            )

        indices = sorted(pair.fold_index for pair in pairs)
        expected = list(range(10))
        if indices != expected:
            raise ValueError(
                f"Dataset '{dataset_name}' fold indices are not contiguous: {indices}"
            )


def resolve_dataset_dir(dataset_name: str, data_root: str | Path) -> tuple[Path, str]:
    """Resolve dataset directory and canonical name, supporting common aliases."""
    root = Path(data_root)
    dataset_dir = root / dataset_name
    if dataset_dir.is_dir():
        return dataset_dir, dataset_name

    aliases = {
        "tao-grid": "grid",
        "tao_grid": "grid",
        "grid": "grid",
        "heart-statlog": "heart-statlog",
        "heart_statlog": "heart-statlog",
    }
    normalized = dataset_name.lower().replace("_", "-")
    resolved_name = aliases.get(normalized, dataset_name)
    resolved_dir = root / resolved_name
    if resolved_dir.is_dir():
        return resolved_dir, resolved_name

    for item in root.iterdir():
        if item.is_dir() and item.name.lower() == dataset_name.lower():
            return item, item.name

    raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")


def load_dataset_fold(
    dataset_name: str, fold_index: int, data_root: str | Path
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], str]:
    """Load a single train and test fold for the named dataset."""
    dataset_dir, canonical_name = resolve_dataset_dir(dataset_name, data_root)

    train_path = dataset_dir / f"{canonical_name}.fold.{fold_index:06d}.train.arff"
    test_path = dataset_dir / f"{canonical_name}.fold.{fold_index:06d}.test.arff"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Missing fold files for dataset '{dataset_name}' at index {fold_index}"
        )

    train_df, feature_names, feature_types = load_arff_table(train_path)
    test_df, _, _ = load_arff_table(test_path)

    if set(train_df.columns) != set(test_df.columns):
        raise ValueError(
            f"Train/test columns differ for dataset '{dataset_name}' fold {fold_index}."
        )

    # Strictly align test_df column ordering to match train_df column ordering
    test_df = test_df[train_df.columns].copy()

    target_name = train_df.columns[-1]
    if target_name != test_df.columns[-1]:
        raise ValueError(
            f"Target labels do not align for dataset '{dataset_name}' fold {fold_index}."
        )

    return train_df, test_df, feature_names, feature_types, target_name
