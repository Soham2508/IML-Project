from __future__ import annotations

from numbers import Number

import numpy as np


def _to_numeric(value):
    if isinstance(value, (np.ndarray, list, tuple)):
        return np.asarray(value, dtype=float)
    return float(value)


def euclidean_distance(
    x: np.ndarray | list[float], y: np.ndarray | list[float]
) -> float:
    """
    Compute the Euclidean distance between two vectors.
    The Euclidean distance is the square root of the sum of the squared differences
    between the corresponding elements of the two vectors.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    return float(np.linalg.norm(x_arr - y_arr))


def manhattan_distance(
    x: np.ndarray | list[float], y: np.ndarray | list[float]
) -> float:
    """
    Compute the Manhattan distance between two vectors.
    The Manhattan distance is the sum of the absolute differences between
    the corresponding elements of the two vectors.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    return float(np.abs(x_arr - y_arr).sum())


def clark_distance(x: np.ndarray | list[float], y: np.ndarray | list[float]) -> float:
    """
    Compute the Clark distance between two vectors.
    The Clark distance is a normalized version of the Euclidean distance,
    where each dimension's contribution is weighted by the sum of the
    absolute values of the two vectors in that dimension.
    """

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    numerator = np.abs(x_arr - y_arr)
    denominator = x_arr + y_arr
    with np.errstate(divide="ignore", invalid="ignore"):
        contribution = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=float),
            where=denominator != 0,
        )
    return float(np.sqrt((contribution**2).sum()))


def canberra_distance(
    x: np.ndarray | list[float], y: np.ndarray | list[float]
) -> float:
    """
    Compute the Canberra distance between two vectors.
    The Canberra distance is a weighted version of the Manhattan distance,
    where each dimension's contribution is normalized by the sum of the
    absolute values of the two vectors in that dimension.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    numerator = np.abs(x_arr - y_arr)
    denominator = np.abs(x_arr) + np.abs(y_arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        contribution = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=float),
            where=denominator != 0,
        )
    return float(contribution.sum())


def heom_distance(x: np.ndarray | list[float], y: np.ndarray | list[float]) -> float:
    """
    Heterogeneous Euclidean-Overlap Metric. For numeric data this behaves like a normalized absolute-distance measure.
    For categorical data, it behaves like an overlap measure (0 if equal, 1 if not equal)."""
    x_arr = np.asarray(x, dtype=object)
    y_arr = np.asarray(y, dtype=object)
    total = 0.0
    for left, right in zip(x_arr, y_arr):
        if isinstance(left, Number) and isinstance(right, Number):
            total += abs(float(left) - float(right))
        else:
            total += 0.0 if left == right else 1.0
    return float(total)


def compute_distance(x, y, metric: str) -> float:
    metric_name = metric.lower().replace(" ", "_")
    if metric_name == "euclidean":
        return euclidean_distance(x, y)
    if metric_name == "manhattan":
        return manhattan_distance(x, y)
    if metric_name == "canberra":
        return canberra_distance(x, y)
    if metric_name == "clark":
        return clark_distance(x, y)
    if metric_name in {"heom", "heterogeneous_euclidean_overlap_metric"}:
        return heom_distance(x, y)
    raise ValueError(f"Unsupported distance metric: {metric}")
