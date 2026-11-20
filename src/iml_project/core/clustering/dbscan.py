"""DBSCAN wrapper around ``sklearn.cluster.DBSCAN`` returning a ``ClusterResult``.

This wrapper adds input and parameter validation, fit timing, int64 labels
with noise as -1, and conversion to the ``ClusterResult`` contract.

Metric / search-algorithm restrictions (scikit-learn 1.9.1, see
``METRIC_ALGORITHMS``):

* euclidean, l2, manhattan, cityblock, l1, chebyshev: any algorithm.
* braycurtis, canberra: ``ball_tree`` or ``brute`` (KD-trees only support
  Minkowski-family metrics).
* cosine, correlation, sqeuclidean: ``brute`` only (not true metrics, so no
  tree index). ``"auto"`` always resolves to a valid choice.

An incompatible (metric, algorithm) pair is rejected in ``__init__`` with a
``ValueError``; use ``compatible_algorithms(metric)`` when building grids.

Usage::

    from iml_project.core.clustering.dbscan import DBSCANClusterer, k_distance_curve

    curve = k_distance_curve(bundle.X, k=5)          # label-free eps heuristic
    dbscan = DBSCANClusterer(eps=0.35, min_samples=5, metric="manhattan")
    labels = dbscan.fit_predict(bundle.X)            # never pass y_true
    result = dbscan.get_result()                     # ClusterResult, noise = -1
"""

from __future__ import annotations

import copy
import time
import warnings
from typing import Any

import numpy as np
import numpy.typing as npt
import sklearn
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from work3.contracts import ClusterResult

VALID_ALGORITHMS: tuple[str, ...] = ("auto", "ball_tree", "kd_tree", "brute")

# Metric -> neighbour-search algorithms that accept it (checked against
# scikit-learn 1.9.1 on float data).
_TREES = ("auto", "ball_tree", "kd_tree", "brute")
_BALL = ("auto", "ball_tree", "brute")
_BRUTE = ("auto", "brute")
METRIC_ALGORITHMS: dict[str, tuple[str, ...]] = {
    "euclidean": _TREES,
    "l2": _TREES,
    "manhattan": _TREES,
    "cityblock": _TREES,
    "l1": _TREES,
    "chebyshev": _TREES,
    "braycurtis": _BALL,
    "canberra": _BALL,
    "cosine": _BRUTE,
    "correlation": _BRUTE,
    "sqeuclidean": _BRUTE,
}

NOISE_LABEL: int = -1


def _validate_feature_matrix(X: Any) -> npt.NDArray[np.float64]:
    """Return a validated float64 copy of ``X``; the caller's array is never mutated.

    Raises ``TypeError`` if ``X`` is not numeric and ``ValueError`` if it is not
    a non-empty 2-D array of finite values.
    """
    if X is None:
        raise TypeError("X must be a 2-D numeric array, got None.")
    try:
        X_valid = np.array(X, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"X must be convertible to a float64 array: {exc}") from exc
    if X_valid.ndim != 2:
        raise ValueError(
            f"X must be 2-D (n_samples, n_features), got {X_valid.ndim}-D "
            f"with shape {X_valid.shape}."
        )
    if X_valid.shape[0] == 0 or X_valid.shape[1] == 0:
        raise ValueError(f"X must be non-empty, got shape {X_valid.shape}.")
    if not np.all(np.isfinite(X_valid)):
        raise ValueError(
            "X contains NaN or infinite values; impute or drop them first."
        )
    return X_valid


def compatible_algorithms(metric: str) -> list[str]:
    """Return the neighbour-search algorithms that support ``metric``.

    Use it to build parameter grids without invalid (metric, algorithm) pairs.
    """
    if not isinstance(metric, str):
        raise TypeError(f"metric must be a string, got {type(metric).__name__}.")
    if metric not in METRIC_ALGORITHMS:
        raise ValueError(
            f"Unsupported metric {metric!r}. Supported: {list(METRIC_ALGORITHMS)}."
        )
    return list(METRIC_ALGORITHMS[metric])


def k_distance_curve(
    X: npt.NDArray[np.float64],
    k: int,
    metric: str = "euclidean",
) -> npt.NDArray[np.float64]:
    """Return every sample's distance to its k-th nearest neighbour, sorted ascending.

    Label-free heuristic for choosing ``eps`` (Ester et al., 1996): use
    ``k = min_samples`` and pick ``eps`` near the knee of the curve.
    The sample itself counts as its first neighbour, as in scikit-learn's DBSCAN.
    """
    X_valid = _validate_feature_matrix(X)
    compatible_algorithms(metric)  # reuses the metric validation
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise TypeError(f"k must be an int, got {type(k).__name__}.")
    if not 1 <= k <= X_valid.shape[0]:
        raise ValueError(
            f"k must be between 1 and n_samples={X_valid.shape[0]}, got {k}."
        )
    neighbours = NearestNeighbors(n_neighbors=int(k), metric=metric)
    distances, _ = neighbours.fit(X_valid).kneighbors(X_valid)
    return np.sort(distances[:, -1]).astype(np.float64)


class DBSCANClusterer:
    """Wrapper of ``sklearn.cluster.DBSCAN`` that produces a ``ClusterResult``."""

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = "euclidean",
        algorithm: str = "auto",
        leaf_size: int = 30,
        metric_params: dict[str, Any] | None = None,
        n_jobs: int | None = None,
    ) -> None:
        # eps: a real number, finite and > 0
        if isinstance(eps, bool) or not isinstance(
            eps, (int, float, np.integer, np.floating)
        ):
            raise TypeError(f"eps must be a number, got {type(eps).__name__}.")
        if not np.isfinite(eps) or eps <= 0:
            raise ValueError(f"eps must be finite and > 0, got {eps}.")

        # min_samples: a real int (not bool), >= 1
        if isinstance(min_samples, bool) or not isinstance(
            min_samples, (int, np.integer)
        ):
            raise TypeError(
                f"min_samples must be an int, got {type(min_samples).__name__}."
            )
        if min_samples < 1:
            raise ValueError(f"min_samples must be >= 1, got {min_samples}.")

        # metric: one of the supported names
        if not isinstance(metric, str):
            raise TypeError(f"metric must be a string, got {type(metric).__name__}.")
        if metric not in METRIC_ALGORITHMS:
            raise ValueError(
                f"Unsupported metric {metric!r}. Supported: {list(METRIC_ALGORITHMS)}."
            )

        # algorithm: one of four, and compatible with the metric
        if not isinstance(algorithm, str):
            raise TypeError(
                f"algorithm must be a string, got {type(algorithm).__name__}."
            )
        if algorithm not in VALID_ALGORITHMS:
            raise ValueError(
                f"algorithm must be one of {VALID_ALGORITHMS}, got {algorithm!r}."
            )
        if algorithm not in METRIC_ALGORITHMS[metric]:
            raise ValueError(
                f"algorithm={algorithm!r} does not support metric={metric!r}. "
                f"Valid choices for this metric: {METRIC_ALGORITHMS[metric]}."
            )

        # leaf_size: int >= 1 (only used by the tree algorithms)
        if isinstance(leaf_size, bool) or not isinstance(leaf_size, (int, np.integer)):
            raise TypeError(
                f"leaf_size must be an int, got {type(leaf_size).__name__}."
            )
        if leaf_size < 1:
            raise ValueError(f"leaf_size must be >= 1, got {leaf_size}.")

        # metric_params: None or a dict
        if metric_params is not None and not isinstance(metric_params, dict):
            raise TypeError("metric_params must be a dict or None.")

        # n_jobs: None or a non-zero int (-1 means "all CPU cores")
        if n_jobs is not None:
            if isinstance(n_jobs, bool) or not isinstance(n_jobs, (int, np.integer)):
                raise TypeError(
                    f"n_jobs must be an int or None, got {type(n_jobs).__name__}."
                )
            if n_jobs == 0:
                raise ValueError("n_jobs must be a non-zero int or None.")

        # Store as plain Python types
        self.eps = float(eps)
        self.min_samples = int(min_samples)
        self.metric = metric
        self.algorithm = algorithm
        self.leaf_size = int(leaf_size)
        self.metric_params = dict(metric_params) if metric_params is not None else None
        self.n_jobs = None if n_jobs is None else int(n_jobs)

        # "Not fitted yet" marker: fit() fills it, get_result() checks it
        self._result: ClusterResult | None = None
        self.labels_: npt.NDArray[np.int64] | None = None
        self.core_sample_indices_: npt.NDArray[np.int64] | None = None

    def get_params(self) -> dict[str, Any]:
        """Return the constructor parameters as a plain dict."""
        return {
            "eps": self.eps,
            "min_samples": self.min_samples,
            "metric": self.metric,
            "algorithm": self.algorithm,
            "leaf_size": self.leaf_size,
            "metric_params": None
            if self.metric_params is None
            else dict(self.metric_params),
            "n_jobs": self.n_jobs,
        }

    def fit(self, X: npt.NDArray[np.float64]) -> DBSCANClusterer:
        """Cluster ``X`` (finite numeric, shape ``(n_samples, n_features)``).

        Only the scikit-learn fit is timed; validation is excluded. Warnings
        raised by scikit-learn are recorded in ``metadata["warnings"]``.
        """
        self._result = None
        self.labels_ = None
        self.core_sample_indices_ = None
        X_valid = _validate_feature_matrix(X)

        model = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric,
            metric_params=self.metric_params,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            n_jobs=self.n_jobs,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            start = time.perf_counter()
            model.fit(X_valid)
            elapsed = time.perf_counter() - start

        labels = np.asarray(model.labels_, dtype=np.int64)
        if labels.shape != (X_valid.shape[0],):
            raise RuntimeError(
                f"DBSCAN returned labels of shape {labels.shape}; "
                f"expected ({X_valid.shape[0]},)."
            )
        if np.any(labels < NOISE_LABEL):
            raise RuntimeError("DBSCAN returned labels below the noise label -1.")

        cluster_ids, cluster_counts = np.unique(
            labels[labels != NOISE_LABEL], return_counts=True
        )
        n_noise = int(np.count_nonzero(labels == NOISE_LABEL))
        core_indices = np.asarray(model.core_sample_indices_, dtype=np.int64)
        n_samples, n_features = X_valid.shape

        metadata: dict[str, Any] = {
            "parameters": self.get_params(),
            "backend": f"sklearn.cluster.DBSCAN (scikit-learn {sklearn.__version__})",
            "noise_label": NOISE_LABEL,
            "n_samples": int(n_samples),
            "n_features": int(n_features),
            "n_core_samples": int(core_indices.size),
            "noise_ratio": n_noise / n_samples,
            "cluster_sizes": {
                int(c): int(n) for c, n in zip(cluster_ids, cluster_counts)
            },
            "warnings": [str(w.message) for w in caught],
        }

        self.labels_ = labels
        self.core_sample_indices_ = core_indices
        self._result = ClusterResult(
            algorithm="DBSCAN",
            labels=labels.copy(),
            memberships=None,
            n_clusters=int(cluster_ids.size),
            n_noise=n_noise,
            execution_time=float(elapsed),
            metadata=metadata,
        )
        return self

    def fit_predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
        """Fit on ``X`` and return int64 labels of shape ``(n_samples,)``; noise is -1."""
        self.fit(X)
        assert self.labels_ is not None
        return self.labels_.copy()

    def get_result(self) -> ClusterResult:
        """Return the ``ClusterResult`` of the last fit (a copy, safe to modify).

        Raises ``RuntimeError`` if called before ``fit``/``fit_predict``.
        """
        if self._result is None:
            raise RuntimeError(
                "DBSCANClusterer is not fitted; call fit() or fit_predict() first."
            )
        return copy.deepcopy(self._result)

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"DBSCANClusterer({params})"
