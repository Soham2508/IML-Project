"""Fuzzy C-Means (FCM) clustering implemented from scratch (Bezdek, 1981).

This is the original FCM of Bezdek (1981), written with NumPy array maths
only; no clustering or fuzzy-logic library (e.g. scikit-fuzzy) is used. Task 7
of the assignment only asks for "your own code", unlike tasks 4-5, so NumPy is
used for the matrix operations.

Model
-----
FCM minimises the objective

    J_m(U, V) = sum_i sum_j u_ij^m * ||x_i - v_j||^2

subject to ``u_ij in [0, 1]`` and ``sum_j u_ij = 1`` for every point ``i``.
``U`` is the ``(n_samples, C)`` membership matrix, ``V`` the ``(C, n_features)``
centres and ``m > 1`` the fuzzifier. Setting the derivatives of the Lagrangian
to zero gives the two alternating-optimisation (AO) updates:

* memberships: ``u_ij = 1 / sum_k (d_ij^2 / d_ik^2)^(1/(m-1))``,
* centres: ``v_j = sum_i u_ij^m x_i / sum_i u_ij^m``.

Each update minimises J_m with the other block fixed, so J_m never increases.
As ``m -> 1`` the memberships become crisp (K-Means); as ``m`` grows they tend
to ``1/C`` (maximally fuzzy).

Algorithm (session 2, slide 27)
-------------------------------
Initialise, then repeat: ``t += 1``; compute ``U_t`` from ``V_{t-1}``; compute
``V_t`` from ``U_t``; until ``t = max_iter`` or the stopping criterion holds.

* ``init="random_memberships"`` (default, Bezdek's original): a random ``U_0``
  with rows normalised to sum to 1; ``V_0`` is computed from it.
* ``init="random_centers"`` (the slide's "guess initial cluster centers"):
  ``C`` distinct data rows chosen at random as ``V_0``.
* ``init=<array>``: explicit ``(C, n_features)`` starting centres.

Stopping criteria (``stop_criterion``), each also capped by ``max_iter``:

* ``"centers"`` (default, slide): ``max |V_t - V_{t-1}| <= tol``.
* ``"memberships"`` (Bezdek et al., 1984): ``max |U_t - U_{t-1}| <= tol``.
* ``"objective"``: ``|J_{t-1} - J_t| / J_{t-1} <= tol``.

Largest absolute changes are used so ``tol`` has the same meaning for any
number of samples or clusters.

Zero distances: if a point coincides with one or more centres, the membership
formula divides by zero. Bezdek's rule is applied: the point gets membership
``1 / q`` in each of the ``q`` coinciding centres and 0 elsewhere.

Hard labels (``ClusterResult.labels``) are the cluster of largest membership
(ties: lowest index); the full matrix is ``ClusterResult.memberships``.

Choosing C
----------
``validity_indices`` computes, without labels, the partition coefficient PC
(higher is better), the partition entropy PE (lower is better) and the
Xie-Beni index XB (lower is better). ``fuzzy_validity_curve`` runs FCM for
several ``C`` and seeds and returns the mean and standard deviation of J_m, PC,
PE and XB per ``C``. PC and PE are monotonic tendencies in ``C`` (PC tends to
fall and PE to rise as ``C`` grows), so read them together with XB and the
evaluator's silhouette/Davies-Bouldin on the hard labels.

Usage (runnable from the project root once ``work3.contracts`` exists)::

    import sys
    sys.path.insert(0, "src")

    import numpy as np
    from iml_project.core.clustering.fuzzy_cmeans import (
        FuzzyCMeansClusterer,
        fuzzy_validity_curve,
    )

    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(c, 0.3, size=(100, 2)) for c in (0.0, 5.0, 10.0)])
    curve = fuzzy_validity_curve(X, c_values=[2, 3, 4, 5], seeds=[0, 1, 2], m=2.0)
    fcm = FuzzyCMeansClusterer(n_clusters=3, m=2.0, random_state=0)
    labels = fcm.fit_predict(X)  # only X; true labels are never passed
    result = fcm.get_result()  # memberships: (n_samples, 3), rows sum to 1
    print(result.metadata["partition_coefficient"], result.execution_time)

References
----------
Bezdek, J. C. (1981). Pattern Recognition with Fuzzy Objective Function
Algorithms. Plenum Press, New York.

Bezdek, J. C., Ehrlich, R., & Full, W. (1984). FCM: The fuzzy c-means
clustering algorithm. Computers & Geosciences, 10(2-3), 191-203.

Dunn, J. C. (1973). A fuzzy relative of the ISODATA process and its use in
detecting compact well-separated clusters. Journal of Cybernetics, 3(3),
32-57.

Xie, X. L., & Beni, G. (1991). A validity measure for fuzzy clustering. IEEE
Transactions on Pattern Analysis and Machine Intelligence, 13(8), 841-847.

Pal, N. R., & Bezdek, J. C. (1995). On cluster validity for the fuzzy c-means
model. IEEE Transactions on Fuzzy Systems, 3(3), 370-379.
"""

from __future__ import annotations

import copy
import math
import secrets
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from iml_project.core.clustering.kmeans import _check_int, _validate_feature_matrix
from work3.contracts import ClusterResult

VALID_INITS: tuple[str, ...] = ("random_memberships", "random_centers")
VALID_STOP_CRITERIA: tuple[str, ...] = ("centers", "memberships", "objective")


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


def _check_float(name: str, value: Any, *, greater_than: float) -> float:
    """Return ``value`` as a float; reject bools, non-numbers, NaN/inf and values <= bound."""
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}.")
    if not math.isfinite(value) or value <= greater_than:
        raise ValueError(f"{name} must be finite and > {greater_than}, got {value}.")
    return float(value)


def _validate_init(init: Any, n_clusters: int) -> str | npt.NDArray[np.float64]:
    """Return a valid init name or a float64 copy of explicit ``(C, n_features)`` centres."""
    if isinstance(init, str):
        if init not in VALID_INITS:
            raise ValueError(
                f"init must be one of {VALID_INITS} or an array of shape "
                f"(n_clusters, n_features), got {init!r}."
            )
        return init
    if init is None:
        raise TypeError(f"init must be one of {VALID_INITS} or an array, got None.")
    try:
        centers = np.array(init, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"init must be convertible to a float64 array: {exc}") from exc
    if centers.ndim != 2 or centers.shape[1] == 0:
        raise ValueError(
            f"init must have shape (n_clusters, n_features), got {centers.shape}."
        )
    if centers.shape[0] != n_clusters:
        raise ValueError(
            f"init has {centers.shape[0]} centres but n_clusters={n_clusters}."
        )
    if not np.all(np.isfinite(centers)):
        raise ValueError("init contains NaN or infinite values.")
    return centers


# ---------------------------------------------------------------------------
# FCM building blocks
# ---------------------------------------------------------------------------


def squared_distances(
    X: npt.NDArray[np.float64], centers: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """``(n_samples, C)`` matrix of squared Euclidean distances ``||x_i - v_j||^2``.

    Computed by broadcasting (not the ``|x|^2 + |v|^2 - 2 x.v`` expansion) so a
    point equal to a centre gives an exact 0, which the zero-distance rule needs.
    """
    diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
    return np.einsum("ijk,ijk->ij", diff, diff)


def update_memberships(
    sq_dist: npt.NDArray[np.float64], m: float
) -> tuple[npt.NDArray[np.float64], int]:
    """Membership update ``u_ij = 1 / sum_k (d_ij^2 / d_ik^2)^(1/(m-1))``.

    Equivalent to ``u_ij = (d_ij^2)^(-p) / sum_k (d_ik^2)^(-p)`` with
    ``p = 1/(m-1)``; computed in log space (a softmax of ``-p * log d^2``) so it
    neither overflows nor underflows when ``m`` is close to 1. Rows with a zero
    distance follow Bezdek's rule (``1/q`` on the ``q`` coinciding centres).

    Returns the memberships and the number of points that hit the zero rule.
    """
    n_samples, n_clusters = sq_dist.shape
    memberships = np.empty((n_samples, n_clusters), dtype=np.float64)
    zero = sq_dist == 0.0
    zero_rows = zero.any(axis=1)

    regular = ~zero_rows
    if np.any(regular):
        exponent = -np.log(sq_dist[regular]) / (m - 1.0)
        exponent -= exponent.max(axis=1, keepdims=True)
        weights = np.exp(exponent)
        memberships[regular] = weights / weights.sum(axis=1, keepdims=True)
    if np.any(zero_rows):
        hits = zero[zero_rows].astype(np.float64)
        memberships[zero_rows] = hits / hits.sum(axis=1, keepdims=True)
    return memberships, int(zero_rows.sum())


def update_centers(
    X: npt.NDArray[np.float64],
    memberships: npt.NDArray[np.float64],
    m: float,
    previous: npt.NDArray[np.float64] | None = None,
) -> tuple[npt.NDArray[np.float64], list[int]]:
    """Centre update ``v_j = sum_i u_ij^m x_i / sum_i u_ij^m``.

    A centre whose weights are all zero (only possible when every point sits on
    other centres) keeps its previous position; its index is returned.
    """
    weights = memberships**m  # (n, C)
    totals = weights.sum(axis=0)  # (C,)
    centers = weights.T @ X  # (C, d)
    degenerate = [int(j) for j in np.flatnonzero(totals <= 0.0)]
    ok = totals > 0.0
    centers[ok] /= totals[ok, np.newaxis]
    if degenerate:
        if previous is None:
            raise RuntimeError("A cluster received zero total membership at initialisation.")
        centers[~ok] = previous[~ok]
    return centers, degenerate


def objective(
    sq_dist: npt.NDArray[np.float64], memberships: npt.NDArray[np.float64], m: float
) -> float:
    """FCM objective ``J_m = sum_ij u_ij^m d_ij^2``."""
    return float(np.sum((memberships**m) * sq_dist))


def validity_indices(
    X: npt.NDArray[np.float64],
    memberships: npt.NDArray[np.float64],
    centers: npt.NDArray[np.float64],
    m: float,
) -> dict[str, float]:
    """Label-free fuzzy validity indices.

    * ``partition_coefficient`` PC = (1/n) sum_ij u_ij^2, in [1/C, 1]; higher
      means crisper clusters (Bezdek, 1981).
    * ``partition_entropy`` PE = -(1/n) sum_ij u_ij ln u_ij (0 ln 0 = 0), in
      [0, ln C]; lower is better (Bezdek, 1981).
    * ``xie_beni`` XB = sum_ij u_ij^m d_ij^2 / (n * min_{j != k} ||v_j - v_k||^2);
      lower means compact, well-separated clusters (Xie & Beni, 1991, with the
      general exponent m of Pal & Bezdek, 1995). NaN when C = 1 or when two
      centres coincide (undefined).
    """
    n_samples, n_clusters = memberships.shape
    pc = float(np.sum(memberships**2) / n_samples)
    positive = memberships > 0.0
    pe = float(
        -np.sum(memberships[positive] * np.log(memberships[positive])) / n_samples
    )
    xb = float("nan")
    if n_clusters > 1:
        separation = squared_distances(centers, centers)
        np.fill_diagonal(separation, np.inf)
        min_sep = float(separation.min())
        if min_sep > 0.0:
            j_m = objective(squared_distances(X, centers), memberships, m)
            xb = j_m / (n_samples * min_sep)
    return {"partition_coefficient": pc, "partition_entropy": pe, "xie_beni": xb}


def _distinct_row_indices(X: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
    """Indices of the first occurrence of each distinct row, in data order."""
    _, first = np.unique(X, axis=0, return_index=True)
    return np.sort(first).astype(np.int64)


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class FuzzyCMeansClusterer:
    """Original Fuzzy C-Means (Bezdek, 1981), producing a fuzzy ``ClusterResult``."""

    def __init__(
        self,
        n_clusters: int = 3,
        m: float = 2.0,
        init: str | npt.ArrayLike = "random_memberships",
        max_iter: int = 300,
        tol: float = 1e-5,
        stop_criterion: str = "centers",
        random_state: int | None = None,
    ) -> None:
        self.n_clusters = _check_int("n_clusters", n_clusters, 1)
        self.m = _check_float("m", m, greater_than=1.0)
        self.init = _validate_init(init, self.n_clusters)
        self.max_iter = _check_int("max_iter", max_iter, 1)
        self.tol = _check_float("tol", tol, greater_than=0.0)

        if not isinstance(stop_criterion, str):
            raise TypeError(
                f"stop_criterion must be a string, got {type(stop_criterion).__name__}."
            )
        if stop_criterion not in VALID_STOP_CRITERIA:
            raise ValueError(
                f"stop_criterion must be one of {VALID_STOP_CRITERIA}, "
                f"got {stop_criterion!r}."
            )
        self.stop_criterion = stop_criterion

        self.random_state = (
            None if random_state is None else _check_int("random_state", random_state, 0)
        )

        # "Not fitted yet" markers: fit() fills them, get_result() checks _result.
        self._result: ClusterResult | None = None
        self.labels_: npt.NDArray[np.int64] | None = None
        self.memberships_: npt.NDArray[np.float64] | None = None
        self.cluster_centers_: npt.NDArray[np.float64] | None = None
        self.objective_: float | None = None
        self.n_iter_: int | None = None

    def get_params(self) -> dict[str, Any]:
        """Return the constructor parameters as a plain dict (explicit init as a list)."""
        return {
            "n_clusters": self.n_clusters,
            "m": self.m,
            "init": self.init if isinstance(self.init, str) else self.init.tolist(),
            "max_iter": self.max_iter,
            "tol": self.tol,
            "stop_criterion": self.stop_criterion,
            "random_state": self.random_state,
        }

    def _initialise(
        self, X: npt.NDArray[np.float64], rng: np.random.Generator
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64] | None, str]:
        """Return ``(V_0, U_0 or None, init description)``."""
        if not isinstance(self.init, str):
            return self.init.copy(), None, "explicit centres"
        if self.init == "random_centers":
            distinct = _distinct_row_indices(X)
            if distinct.size < self.n_clusters:
                raise ValueError(
                    f"n_clusters={self.n_clusters} but X has only {distinct.size} "
                    "distinct rows; random_centers needs C distinct points."
                )
            chosen = rng.choice(distinct, size=self.n_clusters, replace=False)
            return X[chosen].copy(), None, "random_centers (C distinct data rows)"
        # random_memberships: Bezdek's random U_0, rows normalised to sum 1
        u0 = rng.random((X.shape[0], self.n_clusters))
        u0 /= u0.sum(axis=1, keepdims=True)
        v0, _ = update_centers(X, u0, self.m)
        return v0, u0, "random_memberships (random U_0, rows sum to 1)"

    def fit(self, X: npt.NDArray[np.float64]) -> FuzzyCMeansClusterer:
        """Cluster ``X`` (finite numeric, shape ``(n_samples, n_features)``).

        Only the algorithm (initialisation and iterations) is timed; input
        validation and the packaging of the result are excluded.
        """
        self._result = None
        self.labels_ = None
        self.memberships_ = None
        self.cluster_centers_ = None
        self.objective_ = None
        self.n_iter_ = None
        X_valid = _validate_feature_matrix(X)
        n_samples, n_features = X_valid.shape

        if self.n_clusters > n_samples:
            raise ValueError(
                f"n_clusters={self.n_clusters} is larger than n_samples={n_samples}."
            )
        if not isinstance(self.init, str) and self.init.shape[1] != n_features:
            raise ValueError(
                f"init has {self.init.shape[1]} features but X has {n_features}."
            )

        # The seed actually used is recorded, so a random_state=None run can be repeated.
        seed = secrets.randbits(32) if self.random_state is None else self.random_state
        rng = np.random.default_rng(seed)
        warnings_list: list[str] = []
        m = self.m

        start = time.perf_counter()
        centers, previous_u, init_description = self._initialise(X_valid, rng)
        sq_dist = squared_distances(X_valid, centers)
        previous_j: float | None = None
        objective_history: list[float] = []
        change_history: list[float] = []
        zero_distance_events: list[dict[str, int]] = []
        degenerate_events: list[dict[str, Any]] = []
        converged = False
        n_iter = 0

        for t in range(1, self.max_iter + 1):
            memberships, n_zero = update_memberships(sq_dist, m)  # U_t from V_{t-1}
            if n_zero:
                zero_distance_events.append({"iteration": t, "n_points": n_zero})
            new_centers, degenerate = update_centers(  # V_t from U_t
                X_valid, memberships, m, previous=centers
            )
            if degenerate:
                degenerate_events.append({"iteration": t, "clusters": degenerate})
            sq_dist = squared_distances(X_valid, new_centers)
            current_j = objective(sq_dist, memberships, m)  # J_m(U_t, V_t)
            objective_history.append(current_j)
            n_iter = t

            if self.stop_criterion == "centers":
                change = float(np.max(np.abs(new_centers - centers)))
            elif self.stop_criterion == "memberships":
                change = (
                    float("inf")
                    if previous_u is None
                    else float(np.max(np.abs(memberships - previous_u)))
                )
            elif previous_j is None:
                change = float("inf")
            else:
                change = (
                    0.0 if previous_j == 0.0 else abs(previous_j - current_j) / previous_j
                )
            change_history.append(change)

            centers, previous_u, previous_j = new_centers, memberships, current_j
            if change <= self.tol:
                converged = True
                break
        elapsed = time.perf_counter() - start

        if not converged:
            warnings_list.append(
                f"Did not converge within max_iter={self.max_iter} iterations "
                f"(stop_criterion={self.stop_criterion!r}, tol={self.tol})."
            )
        if degenerate_events:
            warnings_list.append(
                "Some clusters received zero total membership and kept their "
                "previous centre; see metadata['degenerate_cluster_events']."
            )

        labels = np.argmax(memberships, axis=1).astype(np.int64)
        cluster_ids, cluster_counts = np.unique(labels, return_counts=True)
        if cluster_ids.size < self.n_clusters:
            warnings_list.append(
                f"Only {cluster_ids.size} of {self.n_clusters} clusters are the "
                "largest membership of at least one point (hard labels)."
            )
        indices = validity_indices(X_valid, memberships, centers, m)

        metadata: dict[str, Any] = {
            "parameters": self.get_params(),
            "backend": "own NumPy implementation (no clustering library)",
            "n_samples": int(n_samples),
            "n_features": int(n_features),
            "cluster_sizes": {
                int(c): int(n) for c, n in zip(cluster_ids, cluster_counts)
            },
            "warnings": warnings_list,
            "random_state_used": int(seed),
            "init_method": init_description,
            "stop_criterion": self.stop_criterion,
            "tol": self.tol,
            "objective": objective_history[-1],
            "objective_history": objective_history,
            "change_history": change_history,
            "n_iter": n_iter,
            "converged": converged,
            "zero_distance_events": zero_distance_events,
            "degenerate_cluster_events": degenerate_events,
            "hard_labels": "argmax of memberships (ties: lowest index)",
            "cluster_centers": centers.tolist(),
            **indices,
        }

        self.labels_ = labels
        self.memberships_ = memberships.copy()
        self.cluster_centers_ = centers.copy()
        self.objective_ = objective_history[-1]
        self.n_iter_ = n_iter
        self._result = ClusterResult(
            algorithm="FuzzyCMeans",
            labels=labels.copy(),
            memberships=memberships.copy(),
            n_clusters=int(cluster_ids.size),
            n_noise=0,
            execution_time=float(elapsed),
            metadata=metadata,
        )
        return self

    def fit_predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
        """Fit on ``X`` and return the hard labels (argmax membership), int64."""
        self.fit(X)
        if self.labels_ is None:
            raise RuntimeError("FCM fit finished without producing labels.")
        return self.labels_.copy()

    def get_result(self) -> ClusterResult:
        """Return the ``ClusterResult`` of the last fit (a copy, safe to modify).

        Raises ``RuntimeError`` if called before ``fit``/``fit_predict``.
        """
        if self._result is None:
            raise RuntimeError(
                "FuzzyCMeansClusterer is not fitted; call fit() or fit_predict() first."
            )
        return copy.deepcopy(self._result)

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"FuzzyCMeansClusterer({params})"


# ---------------------------------------------------------------------------
# Label-free helper for choosing C (and m)
# ---------------------------------------------------------------------------


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Mean and population std (ddof=0) ignoring NaN; NaN if all values are NaN."""
    finite = [v for v in values if not math.isnan(v)]
    if not finite:
        return float("nan"), float("nan")
    mean = math.fsum(finite) / len(finite)
    std = math.sqrt(math.fsum((v - mean) ** 2 for v in finite) / len(finite))
    return mean, std


def fuzzy_validity_curve(
    X: npt.NDArray[np.float64],
    c_values: Sequence[int],
    seeds: Sequence[int],
    m: float = 2.0,
    init: str = "random_memberships",
    stop_criterion: str = "centers",
    tol: float = 1e-5,
    max_iter: int = 300,
) -> dict[str, Any]:
    """Run FCM for every C and seed; return J_m, PC, PE and XB per C (no labels used).

    Returns a dict of plain lists aligned with ``c_values``: for each of
    ``objective``, ``partition_coefficient``, ``partition_entropy`` and
    ``xie_beni``, the keys ``<name>_mean``, ``<name>_std`` (ddof=0, NaN
    ignored) and ``<name>_per_seed``; plus ``c_values``, ``seeds``, ``m``,
    ``init`` and ``stop_criterion``. Call it with several ``m`` values to study
    the fuzzifier.
    """
    if isinstance(c_values, (str, bytes)) or not isinstance(c_values, Sequence):
        raise TypeError("c_values must be a sequence of ints.")
    if isinstance(seeds, (str, bytes)) or not isinstance(seeds, Sequence):
        raise TypeError("seeds must be a sequence of ints.")
    if len(c_values) == 0 or len(seeds) == 0:
        raise ValueError("c_values and seeds must be non-empty.")

    names = ("objective", "partition_coefficient", "partition_entropy", "xie_beni")
    per_seed: dict[str, list[list[float]]] = {name: [] for name in names}
    for c in c_values:
        rows: dict[str, list[float]] = {name: [] for name in names}
        for seed in seeds:
            result = (
                FuzzyCMeansClusterer(
                    n_clusters=c,
                    m=m,
                    init=init,
                    max_iter=max_iter,
                    tol=tol,
                    stop_criterion=stop_criterion,
                    random_state=seed,
                )
                .fit(X)
                .get_result()
            )
            for name in names:
                rows[name].append(float(result.metadata[name]))
        for name in names:
            per_seed[name].append(rows[name])

    curve: dict[str, Any] = {
        "c_values": [int(c) for c in c_values],
        "seeds": [int(s) for s in seeds],
        "m": float(m),
        "init": init,
        "stop_criterion": stop_criterion,
    }
    for name in names:
        stats = [_mean_std(row) for row in per_seed[name]]
        curve[f"{name}_mean"] = [s[0] for s in stats]
        curve[f"{name}_std"] = [s[1] for s in stats]
        curve[f"{name}_per_seed"] = per_seed[name]
    return curve
