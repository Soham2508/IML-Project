"""DBSCAN wrapper around ``sklearn.cluster.DBSCAN`` returning a ``ClusterResult``.

This wrapper adds input and parameter validation, fit timing, int64 labels
with noise as -1, and conversion to the ``ClusterResult`` contract. The
clustering itself is scikit-learn's implementation of DBSCAN (Ester et al.,
1996); it is not reimplemented here.

Supported metrics
-----------------
Every string metric that ``sklearn.cluster.DBSCAN`` accepts and can run on a
feature matrix in scikit-learn 1.9.1 is supported. ``METRIC_ALGORITHMS`` maps
each one to the neighbour-search algorithms that accept it (verified
empirically; ``tests/test_dbscan.py`` re-checks the table against
scikit-learn):

* Any algorithm (``auto``, ``ball_tree``, ``kd_tree``, ``brute``): euclidean,
  l2, manhattan, cityblock, l1, chebyshev, minkowski.
* ``auto``, ``ball_tree`` or ``brute`` (KD-trees only support the Minkowski
  family): braycurtis, canberra, haversine, mahalanobis, seuclidean, hamming,
  and the boolean metrics dice, jaccard, rogerstanimoto, russellrao,
  sokalsneath.
* ``auto`` or ``brute`` only (no tree index available): cosine, correlation,
  sqeuclidean, nan_euclidean, and the boolean metric yule.

``"auto"`` always resolves to a valid algorithm. An incompatible (metric,
algorithm) pair is rejected in ``__init__`` with a ``ValueError``; use
``compatible_algorithms(metric)`` to build parameter grids.

Some metrics need ``metric_params`` (checked in ``__init__``):

* minkowski: ``{"p": p}`` with ``p > 0``; ``ball_tree``/``kd_tree`` need
  ``p >= 1``. (scikit-learn 1.9.1's DBSCAN fails without ``p``.)
* mahalanobis: ``{"V": cov}`` or ``{"VI": inv_cov}``.
* seuclidean: ``{"V": per_feature_variances}``.

haversine needs exactly two features (latitude, longitude in radians). The
boolean metrics are meant for binary data (e.g. one-hot columns): scikit-learn
treats every nonzero value as True. It warns about that conversion only with
``brute``; any warning is recorded in ``metadata["warnings"]``.

Not supported (rejected with an explanation): ``precomputed`` (``X`` must be a
feature matrix, not a distance matrix), ``matching`` and ``wminkowski``
(listed by DBSCAN but rejected by scikit-learn 1.9.1's neighbour search), and
callables (the interface takes ``metric: str``).

Usage (runnable from the project root once ``work3.contracts`` exists)::

    import sys
    sys.path.insert(0, "src")

    from sklearn.datasets import make_blobs
    from iml_project.core.clustering.dbscan import DBSCANClusterer, k_distance_curve

    X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=0)
    curve = k_distance_curve(X, k=5)  # label-free eps heuristic: look for the knee
    dbscan = DBSCANClusterer(eps=0.5, min_samples=5, metric="manhattan")
    labels = dbscan.fit_predict(X)  # only X; true labels are never passed
    result = dbscan.get_result()  # ClusterResult, noise = -1
    print(result.n_clusters, result.n_noise, result.execution_time)

References
----------
Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). A density-based
algorithm for discovering clusters in large spatial databases with noise.
In Proceedings of the Second International Conference on Knowledge Discovery
and Data Mining (KDD-96), pp. 226-231. AAAI Press.

Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python.
Journal of Machine Learning Research, 12, 2825-2830.
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
TREE_ALGORITHMS: tuple[str, ...] = ("ball_tree", "kd_tree")

# Metric -> neighbour-search algorithms that accept it (scikit-learn 1.9.1).
_ANY = ("auto", "ball_tree", "kd_tree", "brute")
_BALL = ("auto", "ball_tree", "brute")
_BRUTE = ("auto", "brute")
METRIC_ALGORITHMS: dict[str, tuple[str, ...]] = {
    # Minkowski family: every search algorithm
    "euclidean": _ANY,
    "l2": _ANY,
    "manhattan": _ANY,
    "cityblock": _ANY,
    "l1": _ANY,
    "chebyshev": _ANY,
    "minkowski": _ANY,
    # Ball tree or brute force
    "braycurtis": _BALL,
    "canberra": _BALL,
    "haversine": _BALL,
    "mahalanobis": _BALL,
    "seuclidean": _BALL,
    "hamming": _BALL,
    "dice": _BALL,
    "jaccard": _BALL,
    "rogerstanimoto": _BALL,
    "russellrao": _BALL,
    "sokalsneath": _BALL,
    # Brute force only
    "cosine": _BRUTE,
    "correlation": _BRUTE,
    "sqeuclidean": _BRUTE,
    "nan_euclidean": _BRUTE,
    "yule": _BRUTE,
}

# Metric names DBSCAN lists but this wrapper rejects, with the reason.
EXCLUDED_METRICS: dict[str, str] = {
    "precomputed": "X must be a feature matrix, not a precomputed distance matrix",
    "matching": "rejected by scikit-learn 1.9.1's neighbour search",
    "wminkowski": "rejected by scikit-learn 1.9.1's neighbour search",
}

# Metric -> metric_params keys it needs; at least one key of each tuple is required.
REQUIRED_METRIC_PARAMS: dict[str, tuple[str, ...]] = {
    "minkowski": ("p",),
    "mahalanobis": ("V", "VI"),
    "seuclidean": ("V",),
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


def _validate_metric(metric: Any) -> str:
    """Raise unless ``metric`` is a supported metric name; return it."""
    if not isinstance(metric, str):
        raise TypeError(f"metric must be a string, got {type(metric).__name__}.")
    if metric in EXCLUDED_METRICS:
        raise ValueError(
            f"metric={metric!r} is not supported: {EXCLUDED_METRICS[metric]}."
        )
    if metric not in METRIC_ALGORITHMS:
        raise ValueError(
            f"Unsupported metric {metric!r}. Supported: {sorted(METRIC_ALGORITHMS)}."
        )
    return metric


def _validate_metric_params(
    metric: str,
    metric_params: Any,
    algorithm: str = "auto",
) -> dict[str, Any] | None:
    """Raise unless ``metric_params`` suits ``metric``; return a deep copy."""
    if metric_params is None:
        params: dict[str, Any] = {}
    elif isinstance(metric_params, dict):
        params = metric_params
    else:
        raise TypeError(
            f"metric_params must be a dict or None, got {type(metric_params).__name__}."
        )

    required = REQUIRED_METRIC_PARAMS.get(metric)
    if required is not None and not any(key in params for key in required):
        keys = " or ".join(repr(key) for key in required)
        raise ValueError(f"metric={metric!r} requires metric_params with key {keys}.")

    if metric == "minkowski":
        p = params["p"]
        if isinstance(p, bool) or not isinstance(
            p, (int, float, np.integer, np.floating)
        ):
            raise TypeError(f"Minkowski p must be a number, got {type(p).__name__}.")
        if np.isnan(p) or p <= 0:
            raise ValueError(f"Minkowski p must be > 0, got {p}.")
        if p < 1 and algorithm in TREE_ALGORITHMS:
            raise ValueError(
                f"algorithm={algorithm!r} needs Minkowski p >= 1, got {p}; "
                "use 'brute' or 'auto' for 0 < p < 1."
            )

    return None if metric_params is None else copy.deepcopy(params)


def compatible_algorithms(metric: str) -> list[str]:
    """Return the neighbour-search algorithms that support ``metric``.

    Use it to build parameter grids without invalid (metric, algorithm) pairs.
    """
    return list(METRIC_ALGORITHMS[_validate_metric(metric)])


def k_distance_curve(
    X: npt.NDArray[np.float64],
    k: int,
    metric: str = "euclidean",
    metric_params: dict[str, Any] | None = None,
) -> npt.NDArray[np.float64]:
    """Return every sample's distance to its k-th nearest neighbour, sorted ascending.

    Label-free heuristic for choosing ``eps``, the sorted k-dist graph of
    Ester et al. (1996): use ``k = min_samples`` and pick ``eps`` near the
    knee of the curve. The sample itself counts as its first neighbour, as in
    scikit-learn's DBSCAN. ``metric_params`` follows the same rules as in
    ``DBSCANClusterer``.
    """
    X_valid = _validate_feature_matrix(X)
    _validate_metric(metric)
    params = _validate_metric_params(metric, metric_params)
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise TypeError(f"k must be an int, got {type(k).__name__}.")
    if not 1 <= k <= X_valid.shape[0]:
        raise ValueError(
            f"k must be between 1 and n_samples={X_valid.shape[0]}, got {k}."
        )

    if metric == "minkowski":
        # NearestNeighbors takes p as its own argument, not inside metric_params.
        nn_params = dict(params or {})
        p = nn_params.pop("p")
        neighbours = NearestNeighbors(
            n_neighbors=int(k), metric=metric, p=p, metric_params=nn_params or None
        )
    else:
        neighbours = NearestNeighbors(
            n_neighbors=int(k), metric=metric, metric_params=params
        )
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
        _validate_metric(metric)

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

        # metric_params: None or a dict, with the keys the metric needs
        params = _validate_metric_params(metric, metric_params, algorithm)

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
        self.metric_params = params
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
            "metric_params": copy.deepcopy(self.metric_params),
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
            metric_params=copy.deepcopy(self.metric_params),
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
        if self.labels_ is None:
            raise RuntimeError("DBSCAN fit finished without producing labels.")
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
