"""BIRCH wrapper around ``sklearn.cluster.Birch`` returning a ``ClusterResult``.

This wrapper adds input and parameter validation, fit timing, int64 labels,
and conversion to the ``ClusterResult`` contract.

Notes on scikit-learn 1.9.1 behaviour:

* ``Birch`` no longer has a ``copy`` parameter (removed in 1.8). The ``copy``
  argument is kept here for the team interface; the wrapper always fits on its
  own float64 copy, so the caller's array is never modified either way.
* ``n_clusters`` is the *requested* number of global clusters. BIRCH may
  produce fewer when the CF-tree has fewer leaf subclusters than requested
  (scikit-learn emits a ``ConvergenceWarning``). ``ClusterResult.n_clusters``
  is always the number actually produced; the requested value and any warning
  are stored in ``metadata``.
* With ``n_clusters=None`` the leaf subclusters are returned as-is, so the
  number of clusters is controlled only by ``threshold``/``branching_factor``.
* With ``compute_labels=False`` scikit-learn does not label the training
  data, so labels are obtained with ``predict(X)``; that call is included in
  ``execution_time`` and recorded as ``metadata["labels_source"]``.

Usage::

    from iml_project.core.clustering.birch import BIRCHClusterer

    birch = BIRCHClusterer(threshold=0.3, branching_factor=50, n_clusters=3)
    labels = birch.fit_predict(bundle.X)             # never pass y_true
    result = birch.get_result()                      # ClusterResult, n_noise = 0
"""

from __future__ import annotations

import copy as copy_module
import time
import warnings
from typing import Any

import numpy as np
import numpy.typing as npt
import sklearn
from sklearn.cluster import Birch

from iml_project.core.clustering.dbscan import _validate_feature_matrix
from work3.contracts import ClusterResult


class BIRCHClusterer:
    """Wrapper of ``sklearn.cluster.Birch`` that produces a ``ClusterResult``."""

    def __init__(
        self,
        threshold: float = 0.5,
        branching_factor: int = 50,
        n_clusters: int | None = 3,
        compute_labels: bool = True,
        copy: bool = True,
    ) -> None:
        # threshold: a real number, finite and > 0
        if isinstance(threshold, bool) or not isinstance(
            threshold, (int, float, np.integer, np.floating)
        ):
            raise TypeError(
                f"threshold must be a number, got {type(threshold).__name__}."
            )
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError(f"threshold must be finite and > 0, got {threshold}.")

        # branching_factor: a real int (not bool), >= 2
        if isinstance(branching_factor, bool) or not isinstance(
            branching_factor, (int, np.integer)
        ):
            raise TypeError(
                f"branching_factor must be an int, got {type(branching_factor).__name__}."
            )
        if branching_factor < 2:
            raise ValueError(f"branching_factor must be >= 2, got {branching_factor}.")

        # n_clusters: None or a positive int (upper bound n_samples is checked in fit)
        if n_clusters is not None:
            if isinstance(n_clusters, bool) or not isinstance(
                n_clusters, (int, np.integer)
            ):
                raise TypeError(
                    f"n_clusters must be an int or None, got {type(n_clusters).__name__}."
                )
            if n_clusters < 1:
                raise ValueError(f"n_clusters must be >= 1, got {n_clusters}.")

        if not isinstance(compute_labels, (bool, np.bool_)):
            raise TypeError(
                f"compute_labels must be a bool, got {type(compute_labels).__name__}."
            )
        if not isinstance(copy, (bool, np.bool_)):
            raise TypeError(f"copy must be a bool, got {type(copy).__name__}.")

        # Store as plain Python types
        self.threshold = float(threshold)
        self.branching_factor = int(branching_factor)
        self.n_clusters = None if n_clusters is None else int(n_clusters)
        self.compute_labels = bool(compute_labels)
        self.copy = bool(copy)

        # "Not fitted yet" marker: fit() fills it, get_result() checks it
        self._result: ClusterResult | None = None
        self.labels_: npt.NDArray[np.int64] | None = None
        self.subcluster_centers_: npt.NDArray[np.float64] | None = None

    def get_params(self) -> dict[str, Any]:
        """Return the constructor parameters as a plain dict."""
        return {
            "threshold": self.threshold,
            "branching_factor": self.branching_factor,
            "n_clusters": self.n_clusters,
            "compute_labels": self.compute_labels,
            "copy": self.copy,
        }

    def fit(self, X: npt.NDArray[np.float64]) -> BIRCHClusterer:
        """Cluster ``X`` (finite numeric, shape ``(n_samples, n_features)``).

        Only the scikit-learn fit (plus ``predict`` when ``compute_labels`` is
        False) is timed; validation is excluded. Warnings raised by
        scikit-learn are recorded in ``metadata["warnings"]``.
        """
        self._result = None
        self.labels_ = None
        self.subcluster_centers_ = None
        X_valid = _validate_feature_matrix(X)
        n_samples, n_features = X_valid.shape

        if self.n_clusters is not None and self.n_clusters > n_samples:
            raise ValueError(
                f"n_clusters={self.n_clusters} is greater than n_samples={n_samples}."
            )

        model = Birch(
            threshold=self.threshold,
            branching_factor=self.branching_factor,
            n_clusters=self.n_clusters,
            compute_labels=self.compute_labels,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            start = time.perf_counter()
            model.fit(X_valid)
            if self.compute_labels:
                raw_labels = model.labels_
                labels_source = "fit"
            else:
                raw_labels = model.predict(X_valid)
                labels_source = "predict"
            elapsed = time.perf_counter() - start

        labels = np.asarray(raw_labels, dtype=np.int64)
        if labels.shape != (n_samples,):
            raise RuntimeError(
                f"BIRCH returned labels of shape {labels.shape}; expected ({n_samples},)."
            )
        if np.any(labels < 0):
            raise RuntimeError(
                "BIRCH returned negative labels; BIRCH has no noise label."
            )

        cluster_ids, cluster_counts = np.unique(labels, return_counts=True)
        subcluster_centers = np.asarray(model.subcluster_centers_, dtype=np.float64)
        warning_messages = [str(w.message) for w in caught]

        metadata: dict[str, Any] = {
            "parameters": self.get_params(),
            "backend": f"sklearn.cluster.Birch (scikit-learn {sklearn.__version__})",
            "requested_n_clusters": self.n_clusters,
            "n_subclusters": int(subcluster_centers.shape[0]),
            "fewer_clusters_than_requested": (
                self.n_clusters is not None and int(cluster_ids.size) < self.n_clusters
            ),
            "labels_source": labels_source,
            "n_samples": int(n_samples),
            "n_features": int(n_features),
            "cluster_sizes": {
                int(c): int(n) for c, n in zip(cluster_ids, cluster_counts)
            },
            "warnings": warning_messages,
        }

        self.labels_ = labels
        self.subcluster_centers_ = subcluster_centers
        self._result = ClusterResult(
            algorithm="BIRCH",
            labels=labels.copy(),
            memberships=None,
            n_clusters=int(cluster_ids.size),
            n_noise=0,
            execution_time=float(elapsed),
            metadata=metadata,
        )
        return self

    def fit_predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
        """Fit on ``X`` and return int64 labels of shape ``(n_samples,)``."""
        self.fit(X)
        assert self.labels_ is not None
        return self.labels_.copy()

    def get_result(self) -> ClusterResult:
        """Return the ``ClusterResult`` of the last fit (a copy, safe to modify).

        Raises ``RuntimeError`` if called before ``fit``/``fit_predict``.
        """
        if self._result is None:
            raise RuntimeError(
                "BIRCHClusterer is not fitted; call fit() or fit_predict() first."
            )
        return copy_module.deepcopy(self._result)

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"BIRCHClusterer({params})"
