from __future__ import annotations

from pathlib import Path

from .arff_reader import FoldPair, discover_fold_pairs, validate_fold_pairs


def get_dataset_fold_map(data_root: str | Path) -> dict[str, list[FoldPair]]:
    """Return all discovered datasets and their fold metadata."""
    groups = discover_fold_pairs(data_root)
    validate_fold_pairs(groups)
    return groups


def get_dataset_names(data_root: str | Path) -> list[str]:
    """Return dataset directory names found under the data root."""
    return sorted(get_dataset_fold_map(data_root).keys())
