"""K-Means clustering implemented from scratch in pure Python (Lloyd's algorithm).

The algorithm follows the basic K-Means of the course (session 2, slide 7):

1. Select K points as initial centroids.
2. Repeat: assign every point to its closest centroid, then recompute the
   centroid of each cluster.
3. Until the centroids do not change.

No library implements or accelerates the clustering. All of the algorithm
(initialisation, distances, assignment, centroid update, stopping, empty
clusters, restarts) uses plain Python lists and loops plus the standard
library (``random``, ``math``, ``statistics``, ``time``). NumPy appears only at
the module boundary, because the team contract (``ClusterResult``) receives
``X`` as a NumPy array and returns labels as an ``int64`` NumPy array: ``X`` is
validated and converted to nested lists once (``X.tolist()``), and the final
labels and centroids are converted back. No NumPy operation is used inside the
algorithm. As a consequence, its run times are those of interpreted Python and
are not comparable with scikit-learn's compiled DBSCAN/BIRCH.

Distances and centroid updates
------------------------------
* ``metric="sqeuclidean"`` (default, required by the assignment): squared
  Euclidean distance with the **mean** as centroid. The objective is the SSE
  (inertia), ``sum_i ||x_i - c_{label(i)}||^2``. The mean is the exact
  minimiser of the SSE of a cluster: the gradient ``-2 * sum_i (x_i - c)``
  is zero at ``c = mean``. Both steps therefore never increase the SSE.
* ``metric="manhattan"`` (extension): L1 distance with the per-feature
  **median** as centroid (i.e. K-Medians). The mean is not appropriate for L1:
  ``sum_i |x_ij - c_j|`` is minimised by the median of each feature, not by
  the mean (the mean minimises squared deviations). With the median, both
  steps never increase the sum of L1 distances, so the algorithm still
  converges.

Algorithm details
-----------------
* Initialisation (``init="random"``): Forgy, K *distinct* data rows chosen at
  random with ``random.Random(seed)``. Distinct rows prevent two identical
  starting centroids when the data contain duplicates (e.g. CMC). ``init`` can
  also be an explicit ``(n_clusters, n_features)`` array of starting
  centroids.
* Assignment: each point goes to the closest centroid; ties go to the lowest
  cluster index.
* Stopping: the run stops when no point changes cluster (then the recomputed
  centroids are identical, which is the slide's "centroids do not change"
  rule without a floating-point tolerance), or after ``max_iter`` iterations,
  in which case ``metadata["converged"]`` is False and a warning is recorded.
* Empty clusters: if a cluster loses all its points, its centroid is moved to
  the point farthest from its own centroid (taken from a cluster with more
  than one point). The event is logged in ``metadata["empty_cluster_events"]``
  and the run keeps K clusters. This also never increases the objective.
* Restarts: ``n_init`` runs with consecutive draws from the same seeded
  generator; the run with the lowest objective is kept. The default
  ``n_init=1`` gives one run per seed, so the runner can report the mean and
  variability over seeds, as the assignment asks.

Choosing K
----------
``elbow_curve(X, k_values, seeds)`` runs K-Means for every K and seed and
returns the mean and standard deviation of the objective per K, without using
any class labels. The objective always decreases as K grows, so look for the
elbow and combine it with internal indices such as the silhouette or the
Davies-Bouldin index (computed by the evaluator with ``sklearn.metrics``).

Usage (runnable from the project root once ``work3.contracts`` exists)::

    import sys
    sys.path.insert(0, "src")

    import numpy as np
    from iml_project.core.clustering.kmeans import KMeansClusterer, elbow_curve

    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(c, 0.3, size=(100, 2)) for c in (0.0, 5.0, 10.0)])
    curve = elbow_curve(X, k_values=[1, 2, 3, 4, 5], seeds=[0, 1, 2])
    kmeans = KMeansClusterer(n_clusters=3, random_state=0)
    labels = kmeans.fit_predict(X)  # only X; true labels are never passed
    result = kmeans.get_result()  # ClusterResult
    print(result.n_clusters, result.metadata["inertia"], result.execution_time)

References
----------
Lloyd, S. P. (1982). Least squares quantization in PCM. IEEE Transactions on
Information Theory, 28(2), 129-137.

MacQueen, J. (1967). Some methods for classification and analysis of
multivariate observations. In Proceedings of the Fifth Berkeley Symposium on
Mathematical Statistics and Probability, Vol. 1, pp. 281-297.

Forgy, E. W. (1965). Cluster analysis of multivariate data: efficiency versus
interpretability of classifications. Biometrics, 21, 768-769.

Jain, A. K. (2010). Data clustering: 50 years beyond K-means. Pattern
Recognition Letters, 31(8), 651-666.

Bradley, P. S., Mangasarian, O. L., & Street, W. N. (1997). Clustering via
concave minimization. Advances in Neural Information Processing Systems 9,
pp. 368-374. (K-Median with the L1 norm and the median update.)
"""

from __future__ import annotations

import copy
import math
import random
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from work3.contracts import ClusterResult

Point = list[float]

VALID_METRICS: tuple[str, ...] = ("sqeuclidean", "manhattan")
VALID_INITS: tuple[str, ...] = ("random",)

# Name of the objective and of the centroid update for each metric.
OBJECTIVE_NAMES: dict[str, str] = {
    "sqeuclidean": "SSE (sum of squared Euclidean distances)",
    "manhattan": "sum of Manhattan (L1) distances",
}
CENTROID_UPDATES: dict[str, str] = {"sqeuclidean": "mean", "manhattan": "median"}
ALGORITHM_NAMES: dict[str, str] = {
    "sqeuclidean": "KMeans",
    "manhattan": "KMeans-manhattan",
}


# ---------------------------------------------------------------------------
# Boundary validation (NumPy only converts and checks the input array)
# ---------------------------------------------------------------------------


def _validate_feature_matrix(X: Any) -> npt.NDArray[np.float64]:
    """Return a validated float64 copy of ``X``; the caller's array is never mutated.

    Same checks and messages as the DBSCAN/BIRCH wrappers, kept local so this
    module never imports scikit-learn. Raises ``TypeError`` if ``X`` is not
    numeric and ``ValueError`` if it is not a non-empty 2-D array of finite
    values.
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


def _check_int(name: str, value: Any, minimum: int) -> int:
    """Return ``value`` as a plain int; reject bools, non-integers and values < minimum."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}.")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}.")
    return int(value)


def _validate_init(init: Any, n_clusters: int) -> str | list[Point]:
    """Return ``"random"`` or the explicit centroids as a list of lists of floats."""
    if isinstance(init, str):
        if init not in VALID_INITS:
            raise ValueError(
                f"init must be one of {VALID_INITS} or an array of shape "
                f"(n_clusters, n_features), got {init!r}."
            )
        return init
    if init is None:
        raise TypeError("init must be 'random' or an array of centroids, got None.")
    try:
        centroids = np.array(init, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"init must be convertible to a float64 array: {exc}") from exc
    if centroids.ndim != 2 or centroids.shape[1] == 0:
        raise ValueError(
            f"init must have shape (n_clusters, n_features), got {centroids.shape}."
        )
    if centroids.shape[0] != n_clusters:
        raise ValueError(
            f"init has {centroids.shape[0]} centroids but n_clusters={n_clusters}."
        )
    if not np.all(np.isfinite(centroids)):
        raise ValueError("init contains NaN or infinite values.")
    return centroids.tolist()


# ---------------------------------------------------------------------------
# Pure-Python building blocks of the algorithm
# ---------------------------------------------------------------------------


def squared_euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    """Squared Euclidean distance ``sum_j (a_j - b_j)^2``."""
    total = 0.0
    for a_j, b_j in zip(a, b):
        diff = a_j - b_j
        total += diff * diff
    return total


def manhattan(a: Sequence[float], b: Sequence[float]) -> float:
    """Manhattan (L1) distance ``sum_j |a_j - b_j|``."""
    total = 0.0
    for a_j, b_j in zip(a, b):
        total += abs(a_j - b_j)
    return total


def mean_centroid(points: list[Point]) -> Point:
    """Per-feature mean: the minimiser of the SSE of a cluster."""
    n = len(points)
    return [math.fsum(column) / n for column in zip(*points)]


def median_centroid(points: list[Point]) -> Point:
    """Per-feature median: the minimiser of the sum of L1 distances of a cluster."""
    return [float(statistics.median(column)) for column in zip(*points)]


DISTANCES: dict[str, Callable[[Sequence[float], Sequence[float]], float]] = {
    "sqeuclidean": squared_euclidean,
    "manhattan": manhattan,
}
CENTROID_FUNCTIONS: dict[str, Callable[[list[Point]], Point]] = {
    "sqeuclidean": mean_centroid,
    "manhattan": median_centroid,
}


def _forgy_init(
    distinct_points: list[tuple[float, ...]], n_clusters: int, rng: random.Random
) -> list[Point]:
    """Forgy initialisation: K distinct data rows drawn without replacement."""
    return [list(point) for point in rng.sample(distinct_points, n_clusters)]


def _assign(
    points: list[Point],
    centroids: list[Point],
    distance: Callable[[Sequence[float], Sequence[float]], float],
) -> tuple[list[int], list[float]]:
    """Assign each point to its closest centroid (ties: lowest index).

    Returns the labels and each point's distance to its assigned centroid.
    """
    labels: list[int] = []
    distances: list[float] = []
    for point in points:
        best_k = 0
        best_d = distance(point, centroids[0])
        for k in range(1, len(centroids)):
            d = distance(point, centroids[k])
            if d < best_d:
                best_k, best_d = k, d
        labels.append(best_k)
        distances.append(best_d)
    return labels, distances


def _update_centroids(
    points: list[Point],
    labels: list[int],
    distances: list[float],
    n_clusters: int,
    centroid_of: Callable[[list[Point]], Point],
    iteration: int,
    events: list[dict[str, Any]],
) -> list[Point]:
    """Recompute every centroid from its members and refill empty clusters.

    An empty cluster takes the point farthest from its current centroid,
    among clusters that keep at least one other point. Each point is used at
    most once, so several empty clusters get different points.
    """
    members: list[list[int]] = [[] for _ in range(n_clusters)]
    for index, label in enumerate(labels):
        members[label].append(index)

    new_centroids: list[Point | None] = [
        centroid_of([points[i] for i in member_ids]) if member_ids else None
        for member_ids in members
    ]

    empty_clusters = [k for k in range(n_clusters) if not members[k]]
    if empty_clusters:
        sizes = [len(member_ids) for member_ids in members]
        # Farthest first; sorted() is stable, so ties keep the lowest index first.
        candidates = iter(sorted(range(len(points)), key=lambda i: -distances[i]))
        for k in empty_clusters:
            for i in candidates:
                if sizes[labels[i]] > 1:
                    break
            else:  # pragma: no cover - impossible when n_samples >= n_clusters
                raise RuntimeError("No point available to refill an empty cluster.")
            sizes[labels[i]] -= 1
            new_centroids[k] = list(points[i])
            events.append(
                {
                    "iteration": iteration,
                    "cluster": k,
                    "point_index": i,
                    "distance": distances[i],
                }
            )
    return [centroid for centroid in new_centroids if centroid is not None]


@dataclass
class _Run:
    """Outcome of one Lloyd run (internal)."""

    labels: list[int]
    centroids: list[Point]
    objective: float
    objective_history: list[float]
    n_iter: int
    converged: bool
    empty_cluster_events: list[dict[str, Any]]


def _lloyd(
    points: list[Point],
    initial_centroids: list[Point],
    distance: Callable[[Sequence[float], Sequence[float]], float],
    centroid_of: Callable[[list[Point]], Point],
    max_iter: int,
) -> _Run:
    """Run Lloyd's algorithm from ``initial_centroids`` until labels stop changing.

    ``objective_history[0]`` is the objective of the first assignment to the
    initial centroids; entry ``t`` is the objective after iteration ``t``
    (update + reassignment). It never increases.
    """
    n_clusters = len(initial_centroids)
    centroids = [list(c) for c in initial_centroids]
    labels, distances = _assign(points, centroids, distance)
    history = [math.fsum(distances)]
    events: list[dict[str, Any]] = []
    converged = False
    n_iter = 0

    for iteration in range(1, max_iter + 1):
        centroids = _update_centroids(
            points, labels, distances, n_clusters, centroid_of, iteration, events
        )
        new_labels, distances = _assign(points, centroids, distance)
        history.append(math.fsum(distances))
        n_iter = iteration
        if new_labels == labels:  # nothing moved: centroids would not change
            converged = True
            break
        labels = new_labels

    return _Run(
        labels=labels,
        centroids=centroids,
        objective=history[-1],
        objective_history=history,
        n_iter=n_iter,
        converged=converged,
        empty_cluster_events=events,
    )


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class KMeansClusterer:
    """K-Means (Lloyd's algorithm) in pure Python, producing a ``ClusterResult``."""

    def __init__(
        self,
        n_clusters: int = 3,
        init: str | npt.ArrayLike = "random",
        n_init: int = 1,
        max_iter: int = 300,
        metric: str = "sqeuclidean",
        random_state: int | None = None,
    ) -> None:
        self.n_clusters = _check_int("n_clusters", n_clusters, 1)
        self.init = _validate_init(init, self.n_clusters)
        self.n_init = _check_int("n_init", n_init, 1)
        self.max_iter = _check_int("max_iter", max_iter, 1)

        if not isinstance(metric, str):
            raise TypeError(f"metric must be a string, got {type(metric).__name__}.")
        if metric not in VALID_METRICS:
            raise ValueError(f"metric must be one of {VALID_METRICS}, got {metric!r}.")
        self.metric = metric

        self.random_state = (
            None if random_state is None else _check_int("random_state", random_state, 0)
        )

        # "Not fitted yet" markers: fit() fills them, get_result() checks _result.
        self._result: ClusterResult | None = None
        self.labels_: npt.NDArray[np.int64] | None = None
        self.cluster_centers_: npt.NDArray[np.float64] | None = None
        self.inertia_: float | None = None
        self.n_iter_: int | None = None

    def get_params(self) -> dict[str, Any]:
        """Return the constructor parameters as a plain dict."""
        return {
            "n_clusters": self.n_clusters,
            "init": copy.deepcopy(self.init),
            "n_init": self.n_init,
            "max_iter": self.max_iter,
            "metric": self.metric,
            "random_state": self.random_state,
        }

    def fit(self, X: npt.NDArray[np.float64]) -> KMeansClusterer:
        """Cluster ``X`` (finite numeric, shape ``(n_samples, n_features)``).

        The timing covers the whole pure-Python algorithm (conversion of ``X``
        to lists, initialisation and all ``n_init`` runs); input validation and
        the packaging of the result are excluded.
        """
        self._result = None
        self.labels_ = None
        self.cluster_centers_ = None
        self.inertia_ = None
        self.n_iter_ = None
        X_valid = _validate_feature_matrix(X)
        n_samples, n_features = X_valid.shape

        if self.n_clusters > n_samples:
            raise ValueError(
                f"n_clusters={self.n_clusters} is larger than n_samples={n_samples}."
            )
        explicit_init = not isinstance(self.init, str)
        if explicit_init and len(self.init[0]) != n_features:
            raise ValueError(
                f"init has {len(self.init[0])} features but X has {n_features}."
            )

        # The seed actually used is recorded, so a random_state=None run can be repeated.
        seed = (
            random.SystemRandom().randrange(2**32)
            if self.random_state is None
            else self.random_state
        )
        warnings_list: list[str] = []
        n_runs = self.n_init
        if explicit_init and self.n_init > 1:
            n_runs = 1
            warnings_list.append(
                f"n_init={self.n_init} ignored: explicit init centroids give "
                "identical runs, so a single run was made."
            )

        distance = DISTANCES[self.metric]
        centroid_of = CENTROID_FUNCTIONS[self.metric]

        start = time.perf_counter()
        points: list[Point] = X_valid.tolist()  # NumPy -> plain Python, once
        rng = random.Random(seed)
        distinct: list[tuple[float, ...]] = []
        if not explicit_init:
            distinct = list(dict.fromkeys(tuple(point) for point in points))
            if len(distinct) < self.n_clusters:
                raise ValueError(
                    f"n_clusters={self.n_clusters} but X has only {len(distinct)} "
                    "distinct rows; Forgy initialisation needs K distinct points."
                )
        runs: list[_Run] = []
        for _ in range(n_runs):
            initial = self.init if explicit_init else _forgy_init(
                distinct, self.n_clusters, rng
            )
            runs.append(_lloyd(points, initial, distance, centroid_of, self.max_iter))
        best_index = min(range(n_runs), key=lambda r: runs[r].objective)
        best = runs[best_index]
        elapsed = time.perf_counter() - start

        if not best.converged:
            warnings_list.append(
                f"Did not converge within max_iter={self.max_iter} iterations; "
                "labels and centroids are those of the last iteration."
            )
        n_produced = len(set(best.labels))
        if n_produced < self.n_clusters:
            warnings_list.append(
                f"Only {n_produced} of {self.n_clusters} clusters are non-empty."
            )

        labels = np.asarray(best.labels, dtype=np.int64)
        centers = np.asarray(best.centroids, dtype=np.float64)
        sizes = [0] * self.n_clusters
        for label in best.labels:
            sizes[label] += 1

        metadata: dict[str, Any] = {
            "parameters": self.get_params(),
            "backend": "own pure-Python implementation (no library in the algorithm)",
            "n_samples": int(n_samples),
            "n_features": int(n_features),
            "cluster_sizes": {k: sizes[k] for k in range(self.n_clusters) if sizes[k]},
            "warnings": warnings_list,
            "random_state_used": seed,
            "init_method": "explicit" if explicit_init else "random (Forgy, distinct rows)",
            "metric": self.metric,
            "centroid_update": CENTROID_UPDATES[self.metric],
            "objective_name": OBJECTIVE_NAMES[self.metric],
            "inertia": best.objective,
            "objective_history": list(best.objective_history),
            "n_iter": best.n_iter,
            "converged": best.converged,
            "empty_cluster_events": copy.deepcopy(best.empty_cluster_events),
            "n_runs": n_runs,
            "best_run_index": best_index,
            "run_objectives": [run.objective for run in runs],
            "cluster_centers": [list(c) for c in best.centroids],
        }

        self.labels_ = labels
        self.cluster_centers_ = centers
        self.inertia_ = float(best.objective)
        self.n_iter_ = best.n_iter
        self._result = ClusterResult(
            algorithm=ALGORITHM_NAMES[self.metric],
            labels=labels.copy(),
            memberships=None,
            n_clusters=n_produced,
            n_noise=0,
            execution_time=float(elapsed),
            metadata=metadata,
        )
        return self

    def fit_predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
        """Fit on ``X`` and return int64 labels of shape ``(n_samples,)``."""
        self.fit(X)
        if self.labels_ is None:
            raise RuntimeError("KMeans fit finished without producing labels.")
        return self.labels_.copy()

    def get_result(self) -> ClusterResult:
        """Return the ``ClusterResult`` of the last fit (a copy, safe to modify).

        Raises ``RuntimeError`` if called before ``fit``/``fit_predict``.
        """
        if self._result is None:
            raise RuntimeError(
                "KMeansClusterer is not fitted; call fit() or fit_predict() first."
            )
        return copy.deepcopy(self._result)

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"KMeansClusterer({params})"


# ---------------------------------------------------------------------------
# Label-free helper for choosing K
# ---------------------------------------------------------------------------


def elbow_curve(
    X: npt.NDArray[np.float64],
    k_values: Sequence[int],
    seeds: Sequence[int],
    metric: str = "sqeuclidean",
    max_iter: int = 300,
) -> dict[str, Any]:
    """Run K-Means for every K and seed; return the objective per K (no labels used).

    Returns a dict of plain lists, aligned with ``k_values``:

    * ``"k_values"``: the K values, in the given order.
    * ``"objective_mean"`` / ``"objective_std"``: mean and population standard
      deviation (ddof=0, as in the team's aggregation) over the seeds.
    * ``"objective_per_seed"``: one list per K with one objective per seed.
    * ``"seeds"``, ``"metric"``, ``"objective_name"``.

    The objective always decreases as K grows; look for the elbow (where the
    decrease levels off) rather than the minimum.
    """
    if isinstance(k_values, (str, bytes)) or not isinstance(k_values, Sequence):
        raise TypeError("k_values must be a sequence of ints.")
    if isinstance(seeds, (str, bytes)) or not isinstance(seeds, Sequence):
        raise TypeError("seeds must be a sequence of ints.")
    if len(k_values) == 0 or len(seeds) == 0:
        raise ValueError("k_values and seeds must be non-empty.")

    per_k: list[list[float]] = []
    for k in k_values:
        objectives = []
        for seed in seeds:
            model = KMeansClusterer(
                n_clusters=k, metric=metric, max_iter=max_iter, random_state=seed
            ).fit(X)
            objectives.append(float(model.inertia_))
        per_k.append(objectives)

    means = [math.fsum(values) / len(values) for values in per_k]
    stds = [
        math.sqrt(math.fsum((v - m) ** 2 for v in values) / len(values))
        for values, m in zip(per_k, means)
    ]
    return {
        "k_values": [int(k) for k in k_values],
        "seeds": [int(s) for s in seeds],
        "metric": metric,
        "objective_name": OBJECTIVE_NAMES[metric],
        "objective_mean": means,
        "objective_std": stds,
        "objective_per_seed": per_k,
    }
