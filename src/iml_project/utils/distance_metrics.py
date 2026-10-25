from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


class DistanceMetric(ABC):
    """Abstract base class for distance metrics."""

    @abstractmethod
    def compute(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute the scalar distance between two individual feature vectors."""
        raise NotImplementedError

    def compute_all(self, query: np.ndarray, train_matrix: np.ndarray) -> np.ndarray:
        """Compute distances from a single query vector to all rows in train_matrix.

        Subclasses may override this for vectorized speed.
        """
        n_samples = train_matrix.shape[0]
        distances = np.empty(n_samples, dtype=np.float64)
        for i in range(n_samples):
            distances[i] = self.compute(query, train_matrix[i])
        return distances


class EuclideanDistance(DistanceMetric):
    """Euclidean distance: Minkowski distance with r = 2.

    Formula: d(x, y) = sqrt(sum_f (x_f - y_f)^2)
    """

    def compute(self, x: np.ndarray, y: np.ndarray) -> float:
        diff = x.astype(np.float64) - y.astype(np.float64)
        return float(np.sqrt(np.sum(diff * diff)))

    def compute_all(self, query: np.ndarray, train_matrix: np.ndarray) -> np.ndarray:
        diff = train_matrix.astype(np.float64) - query.astype(np.float64)
        return np.sqrt(np.sum(diff * diff, axis=1))


class ManhattanDistance(DistanceMetric):
    """Manhattan distance: Minkowski distance with r = 1.

    Formula: d(x, y) = sum_f |x_f - y_f|
    """

    def compute(self, x: np.ndarray, y: np.ndarray) -> float:
        diff = np.abs(x.astype(np.float64) - y.astype(np.float64))
        return float(np.sum(diff))

    def compute_all(self, query: np.ndarray, train_matrix: np.ndarray) -> np.ndarray:
        diff = np.abs(train_matrix.astype(np.float64) - query.astype(np.float64))
        return np.sum(diff, axis=1)


class CanberraDistance(DistanceMetric):
    """Canberra distance.

    Formula: d(x, y) = sum_f |x_f - y_f| / (|x_f| + |y_f|)
    Convention: If x_f = y_f = 0, the term is defined as 0.0.
    """

    def compute(self, x: np.ndarray, y: np.ndarray) -> float:
        x_flt = x.astype(np.float64)
        y_flt = y.astype(np.float64)
        num = np.abs(x_flt - y_flt)
        denom = np.abs(x_flt) + np.abs(y_flt)
        mask = denom > 1e-15
        terms = np.zeros_like(num)
        terms[mask] = num[mask] / denom[mask]
        return float(np.sum(terms))

    def compute_all(self, query: np.ndarray, train_matrix: np.ndarray) -> np.ndarray:
        q = query.astype(np.float64)
        mat = train_matrix.astype(np.float64)
        num = np.abs(mat - q)
        denom = np.abs(mat) + np.abs(q)
        mask = denom > 1e-15
        terms = np.zeros_like(num)
        terms[mask] = num[mask] / denom[mask]
        return np.sum(terms, axis=1)


class ClarkDistance(DistanceMetric):
    """Clark coefficient of divergence.

    Formula: d(x, y) = sqrt(sum_f ((x_f - y_f) / (x_f + y_f))^2)
    Convention: If x_f = y_f = 0, the term is defined as 0.0.
    Inputs must be non-negative (e.g. normalized [0, 1]).
    """

    def compute(self, x: np.ndarray, y: np.ndarray) -> float:
        x_flt = np.maximum(0.0, x.astype(np.float64))
        y_flt = np.maximum(0.0, y.astype(np.float64))
        diff = x_flt - y_flt
        sum_xy = x_flt + y_flt
        mask = sum_xy > 1e-15
        terms = np.zeros_like(diff)
        terms[mask] = (diff[mask] / sum_xy[mask]) ** 2
        return float(np.sqrt(np.sum(terms)))

    def compute_all(self, query: np.ndarray, train_matrix: np.ndarray) -> np.ndarray:
        q = np.maximum(0.0, query.astype(np.float64))
        mat = np.maximum(0.0, train_matrix.astype(np.float64))
        diff = mat - q
        sum_xy = mat + q
        mask = sum_xy > 1e-15
        terms = np.zeros_like(diff)
        terms[mask] = (diff[mask] / sum_xy[mask]) ** 2
        return np.sqrt(np.sum(terms, axis=1))


class HEOMDistance(DistanceMetric):
    """Heterogeneous Euclidean-Overlap Metric (Wilson & Martinez, 1997).

    Formula: d(x, y) = sqrt(sum_f d_f(x_f, y_f)^2)
    where:
      - If x_f or y_f is missing: d_f = 1.0
      - If f is nominal: d_f = 0 if x_f == y_f else 1.0 (overlap metric)
      - If f is numeric: d_f = |x_f - y_f| / range_f (if normalized [0, 1], range_f = 1.0)
    """

    def __init__(
        self,
        numeric_indices: Sequence[int],
        nominal_indices: Sequence[int],
        ranges: Sequence[float] | None = None,
    ) -> None:
        self.numeric_indices = list(numeric_indices)
        self.nominal_indices = list(nominal_indices)
        self.ranges = (
            np.array(ranges, dtype=np.float64)
            if ranges is not None
            else np.ones(len(self.numeric_indices), dtype=np.float64)
        )
        # Avoid division by zero for constant features
        self.ranges = np.where(self.ranges <= 1e-15, 1.0, self.ranges)

    def compute(self, x: np.ndarray, y: np.ndarray) -> float:
        total_sq = 0.0

        if self.numeric_indices:
            x_num = x[self.numeric_indices].astype(np.float64)
            y_num = y[self.numeric_indices].astype(np.float64)
            x_nan = np.isnan(x_num)
            y_nan = np.isnan(y_num)
            missing = x_nan | y_nan

            diff = np.abs(x_num - y_num) / self.ranges
            diff[missing] = 1.0
            total_sq += float(np.sum(diff * diff))

        if self.nominal_indices:
            x_nom = x[self.nominal_indices]
            y_nom = y[self.nominal_indices]
            for val_x, val_y in zip(x_nom, y_nom):
                if pd.isna(val_x) or pd.isna(val_y):  # noqa: SIM114
                    total_sq += 1.0
                elif str(val_x) != str(val_y):
                    total_sq += 1.0

        return float(np.sqrt(total_sq))

    def compute_all(self, query: np.ndarray, train_matrix: np.ndarray) -> np.ndarray:
        n_samples = train_matrix.shape[0]
        total_sq = np.zeros(n_samples, dtype=np.float64)

        if self.numeric_indices:
            q_num = query[self.numeric_indices].astype(np.float64)
            mat_num = train_matrix[:, self.numeric_indices].astype(np.float64)
            missing = np.isnan(q_num) | np.isnan(mat_num)
            diff = np.abs(mat_num - q_num) / self.ranges
            diff[missing] = 1.0
            total_sq += np.sum(diff * diff, axis=1)

        if self.nominal_indices:
            q_nom = query[self.nominal_indices]
            mat_nom = train_matrix[:, self.nominal_indices]
            for j, nom_idx in enumerate(self.nominal_indices):
                qv = q_nom[j]
                col = mat_nom[:, j]
                if pd.isna(qv):
                    total_sq += 1.0
                else:
                    mismatch = (col != qv) | pd.isna(col)
                    total_sq += mismatch.astype(np.float64)

        return np.sqrt(total_sq)


class IVDMDistance(DistanceMetric):
    """Interpolated Value Difference Metric (Wilson & Martinez, 1997).

    For nominal features:
      vdm_f(x_f, y_f) = sum_c |P(C=c | f=x_f) - P(C=c | f=y_f)|^2
    For numeric features:
      Continuous values are discretized into intervals based on training data quantiles,
      then the VDM formula is evaluated on the discretization bin.

    Total distance:
      d(x, y) = sqrt(sum_f vdm_f(x_f, y_f))

    Probabilities are estimated strictly on training data with Laplace smoothing.
    """

    def __init__(
        self,
        numeric_indices: Sequence[int],
        nominal_indices: Sequence[int],
        n_bins: int = 5,
    ) -> None:
        self.numeric_indices = list(numeric_indices)
        self.nominal_indices = list(nominal_indices)
        self.n_bins = n_bins
        self.classes_: list[Any] = []
        self.nominal_probs_: dict[int, dict[Any, np.ndarray]] = {}
        self.numeric_bin_edges_: dict[int, np.ndarray] = {}
        self.numeric_bin_probs_: dict[int, np.ndarray] = {}
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> IVDMDistance:
        """Estimate conditional class probabilities strictly on the training fold."""
        self.classes_ = sorted(set(y_train))
        n_classes = len(self.classes_)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}

        # Fit nominal conditional probabilities P(C | f = val)
        self.nominal_probs_ = {}
        for idx in self.nominal_indices:
            col_vals = X_train[:, idx]
            probs_for_attr: dict[Any, np.ndarray] = {}
            unique_vals = set(col_vals)
            for val in unique_vals:
                mask = col_vals == val
                count_val = np.sum(mask)
                counts_by_class = np.zeros(n_classes, dtype=np.float64)
                for c, count in zip(*np.unique(y_train[mask], return_counts=True)):
                    counts_by_class[class_to_idx[c]] = count
                # Laplace smoothing: (count(c) + 1) / (count_val + n_classes)
                probs = (counts_by_class + 1.0) / (count_val + n_classes)
                probs_for_attr[val] = probs
            self.nominal_probs_[idx] = probs_for_attr

        # Fit numeric discretization bin edges and probabilities
        self.numeric_bin_edges_ = {}
        self.numeric_bin_probs_ = {}
        for idx in self.numeric_indices:
            col_vals = X_train[:, idx].astype(np.float64)
            quantiles = np.linspace(0, 1, self.n_bins + 1)
            edges = np.unique(np.quantile(col_vals, quantiles))
            if len(edges) < 2:
                edges = np.array([col_vals.min() - 1e-5, col_vals.max() + 1e-5])
            self.numeric_bin_edges_[idx] = edges
            n_actual_bins = len(edges) - 1

            bin_indices = np.digitize(col_vals, edges[1:-1])
            probs_matrix = np.zeros((n_actual_bins, n_classes), dtype=np.float64)
            for b in range(n_actual_bins):
                b_mask = bin_indices == b
                b_count = np.sum(b_mask)
                counts = np.zeros(n_classes, dtype=np.float64)
                if b_count > 0:
                    for c, cnt in zip(*np.unique(y_train[b_mask], return_counts=True)):
                        counts[class_to_idx[c]] = cnt
                # Laplace smoothing
                probs_matrix[b] = (counts + 1.0) / (b_count + n_classes)
            self.numeric_bin_probs_[idx] = probs_matrix

        self.is_fitted = True
        return self

    def _get_nominal_prob(self, attr_idx: int, val: Any) -> np.ndarray:
        n_classes = len(self.classes_)
        attr_map = self.nominal_probs_.get(attr_idx, {})
        if val in attr_map:
            return attr_map[val]
        # Uniform Laplace fallback for unseen test category
        return np.ones(n_classes, dtype=np.float64) / n_classes

    def _get_numeric_prob(self, attr_idx: int, val: float) -> np.ndarray:
        n_classes = len(self.classes_)
        if not self.is_fitted or attr_idx not in self.numeric_bin_edges_:
            return np.ones(n_classes, dtype=np.float64) / n_classes
        edges = self.numeric_bin_edges_[attr_idx]
        bin_idx = int(np.digitize(val, edges[1:-1]))
        bin_idx = max(0, min(bin_idx, len(edges) - 2))
        return self.numeric_bin_probs_[attr_idx][bin_idx]

    def compute(self, x: np.ndarray, y: np.ndarray) -> float:
        total_sq = 0.0

        for idx in self.nominal_indices:
            p_x = self._get_nominal_prob(idx, x[idx])
            p_y = self._get_nominal_prob(idx, y[idx])
            diff = p_x - p_y
            total_sq += float(np.sum(diff * diff))

        for idx in self.numeric_indices:
            p_x = self._get_numeric_prob(idx, float(x[idx]))
            p_y = self._get_numeric_prob(idx, float(y[idx]))
            diff = p_x - p_y
            total_sq += float(np.sum(diff * diff))

        return float(np.sqrt(total_sq))

    def compute_all(self, query: np.ndarray, train_matrix: np.ndarray) -> np.ndarray:
        n_samples = train_matrix.shape[0]
        n_classes = len(self.classes_)
        total_sq = np.zeros(n_samples, dtype=np.float64)

        # Precompute query probabilities
        for idx in self.nominal_indices:
            q_prob = self._get_nominal_prob(idx, query[idx])
            attr_map = self.nominal_probs_.get(idx, {})
            col_vals = train_matrix[:, idx]
            train_probs = np.array(
                [attr_map.get(v, np.ones(n_classes) / n_classes) for v in col_vals]
            )
            diff = train_probs - q_prob
            total_sq += np.sum(diff * diff, axis=1)

        for idx in self.numeric_indices:
            q_prob = self._get_numeric_prob(idx, float(query[idx]))
            edges = self.numeric_bin_edges_[idx]
            col_vals = train_matrix[:, idx].astype(np.float64)
            bin_indices = np.clip(np.digitize(col_vals, edges[1:-1]), 0, len(edges) - 2)
            train_probs = self.numeric_bin_probs_[idx][bin_indices]
            diff = train_probs - q_prob
            total_sq += np.sum(diff * diff, axis=1)

        return np.sqrt(total_sq)


def get_distance_metric(
    name: str,
    *,
    numeric_indices: Sequence[int] | None = None,
    nominal_indices: Sequence[int] | None = None,
    ranges: Sequence[float] | None = None,
    **kwargs: Any,
) -> DistanceMetric:
    """Factory returning an instance of the requested distance metric.

    Supports:
      - 'euclidean' (or 'minkowski_2')
      - 'manhattan' (or 'minkowski_1')
      - 'canberra'
      - 'clark'
      - 'heom'
      - 'ivdm'
    """
    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")

    if normalized in {"euclidean", "minkowski_2", "l2"}:
        return EuclideanDistance()
    elif normalized in {"manhattan", "minkowski_1", "l1", "cityblock"}:
        return ManhattanDistance()
    elif normalized in {"canberra"}:
        return CanberraDistance()
    elif normalized in {"clark"}:
        return ClarkDistance()
    elif normalized in {"heom"}:
        num_idx = numeric_indices if numeric_indices is not None else []
        nom_idx = nominal_indices if nominal_indices is not None else []
        return HEOMDistance(num_idx, nom_idx, ranges=ranges)
    elif normalized in {"ivdm"}:
        num_idx = numeric_indices if numeric_indices is not None else []
        nom_idx = nominal_indices if nominal_indices is not None else []
        n_bins = kwargs.get("n_bins", 5)
        return IVDMDistance(num_idx, nom_idx, n_bins=n_bins)
    else:
        raise ValueError(
            f"Unknown distance metric '{name}'. Supported metrics: "
            "euclidean, manhattan, canberra, clark, heom, ivdm."
        )
