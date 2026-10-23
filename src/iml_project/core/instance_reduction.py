from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from typing import Any

import numpy as np

from iml_project.utils.distance_metrics import (
    DistanceMetric,
    EuclideanDistance,
    IVDMDistance,
    get_distance_metric,
)


class InstanceReducer(ABC):
    """Base interface for instance-reduction algorithms."""

    def __init__(self, distance_metric: DistanceMetric):
        self.distance_metric = distance_metric

    @abstractmethod
    def fit_resample(self, X, y):
        """
        Reduce the training instances.

        Returns
        -------
        X_reduced, y_reduced
        """
        raise NotImplementedError


class FCNN3(InstanceReducer):
    """Fast Condensed Nearest Neighbor 3 (FCNN3) algorithm.

    FCNN3 constructs a reduced training subset S that is training-set-consistent
    for the 1-nearest-neighbor rule. At each iteration, it selects the instance
    p* in S with the largest number of Voronoi enemies (opposing-class examples
    in its Voronoi cell), and adds the enemy q* nearest to p* into S.

    Reference
    ---------
    Fabrizio Angiulli, "Fast Nearest Neighbor Condensation for Large Data Sets
    Classification," IEEE Transactions on Knowledge and Data Engineering,
    vol. 19, no. 11, pp. 1450-1464, Nov. 2007.
    """

    def __init__(self, distance_metric: DistanceMetric | str | None = None) -> None:
        if distance_metric is None:
            distance_metric = EuclideanDistance()
        elif isinstance(distance_metric, str):
            distance_metric = get_distance_metric(distance_metric)
        super().__init__(distance_metric=distance_metric)
        self.selected_indices_: list[int] = []

    def fit_resample(self, X: Any, y: Any) -> tuple[Any, Any]:
        """
        Reduce the training instances using FCNN3.

        Parameters
        ----------
        X : array-like or pd.DataFrame of shape (n_samples, n_features)
            Training feature vectors.
        y : array-like or pd.Series of shape (n_samples,)
            Categorical target labels.

        Returns
        -------
        X_reduced : array-like or pd.DataFrame
            Reduced feature vectors.
        y_reduced : array-like, pd.Series, or list
            Reduced target labels.
        """
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have the same length. Got len(X)={len(X)}, len(y)={len(y)}."
            )

        n_samples = len(X)
        if n_samples == 0:
            self.selected_indices_ = []
            if hasattr(X, "iloc"):
                return X.iloc[[]], y.iloc[[]]
            if isinstance(y, list):
                return (
                    np.empty((0, 0) if not hasattr(X, "shape") else (0, X.shape[1])),
                    [],
                )
            return (
                np.empty((0, 0) if not hasattr(X, "shape") else (0, X.shape[1])),
                np.empty(0, dtype=getattr(y, "dtype", object)),
            )

        if n_samples == 1:
            self.selected_indices_ = [0]
            return X, y

        # Convert to numpy arrays for internal calculations
        if isinstance(X, np.ndarray):
            X_arr = X
        elif hasattr(X, "to_numpy"):
            X_arr = X.to_numpy()
        else:
            X_arr = np.asarray(X)

        if isinstance(y, (np.ndarray, list)):
            y_arr = np.asarray(y)
        elif hasattr(y, "to_numpy"):
            y_arr = y.to_numpy()
        else:
            y_arr = np.asarray(y)

        # Fit IVDM metric if applicable and not already fitted
        if isinstance(self.distance_metric, IVDMDistance) and not getattr(
            self.distance_metric, "is_fitted", False
        ):
            self.distance_metric.fit(X_arr, y_arr)

        # Step 1: Find a most populated class c* in T
        classes, counts = np.unique(y_arr, return_counts=True)
        # Deterministic tie-breaking: sort by count descending, then string representation ascending
        sorted_classes = sorted(
            zip(classes, counts), key=lambda item: (-item[1], str(item[0]))
        )
        c_star = sorted_classes[0][0]

        c_star_indices = [i for i, label in enumerate(y_arr) if label == c_star]
        X_c_star = X_arr[c_star_indices]

        # Step 2: Compute feature-wise mean mu[c*] of class c*
        try:
            mu_c_star = np.mean(X_c_star.astype(np.float64), axis=0)
        except (ValueError, TypeError):
            # Fallback for mixed or non-numeric types: mean for numeric, mode for nominal
            mu_c_star = np.empty(X_c_star.shape[1], dtype=object)
            for col_idx in range(X_c_star.shape[1]):
                col_vals = X_c_star[:, col_idx]
                try:
                    mu_c_star[col_idx] = np.mean(col_vals.astype(np.float64))
                except (ValueError, TypeError):
                    vals, val_counts = np.unique(col_vals, return_counts=True)
                    mu_c_star[col_idx] = vals[np.argmax(val_counts)]

        # Step 3: Choose p0, actual training instance in class c* closest to mu[c*]
        best_dist_to_mean = float("inf")
        p_0 = c_star_indices[0]
        for idx in c_star_indices:
            dist = self.distance_metric.compute(X_arr[idx], mu_c_star)
            if dist < best_dist_to_mean:
                best_dist_to_mean = dist
                p_0 = idx

        # Step 4: Initialize S = {p0}
        S: list[int] = [p_0]
        S_set: set[int] = {p_0}
        S_order: dict[int, int] = {p_0: 0}

        # Initialize Voronoi tracking for each instance in T
        best_dist = self.distance_metric.compute_all(X_arr[p_0], X_arr).astype(
            np.float64
        )
        best_p = np.full(n_samples, p_0, dtype=int)
        best_dist[p_0] = 0.0

        # Steps 5-14: Iteratively add nearest enemy of the instance with most enemies
        while len(S) < n_samples:
            # Step 7 & 8: Voronoi enemies E(p, S, T) = {x in V(p, S, T) : label(x) != label(p)}
            enemies: dict[int, list[int]] = {p: [] for p in S}
            for i in range(n_samples):
                p = best_p[i]
                if y_arr[i] != y_arr[p]:
                    enemies[p].append(i)

            # Step 10: Stopping condition: If e(p) = 0 for every p in S, stop
            max_enemies = max(len(enemies[p]) for p in S)
            if max_enemies == 0:
                break

            # Step 12: Select p* in S with the largest enemy count e(p)
            # Deterministic tie-breaking: earlier added to S (first in S_order)
            p_star = max(S, key=lambda p: (len(enemies[p]), -S_order[p]))

            # Step 13: Select q* in E(p*, S, T) minimizing d(p*, q*)
            p_star_enemies = enemies[p_star]
            q_star = min(
                p_star_enemies,
                key=lambda idx: (
                    self.distance_metric.compute(X_arr[p_star], X_arr[idx]),
                    idx,
                ),
            )

            if q_star in S_set:
                break

            # Step 14: Update S = S union {q*}
            S.append(q_star)
            S_set.add(q_star)
            S_order[q_star] = len(S) - 1

            # Update Voronoi cells and distances
            dists_from_q = self.distance_metric.compute_all(X_arr[q_star], X_arr)
            for i in range(n_samples):
                if i == q_star:
                    best_p[i] = q_star
                    best_dist[i] = 0.0
                elif dists_from_q[i] < best_dist[i]:
                    best_dist[i] = dists_from_q[i]
                    best_p[i] = q_star

        self.selected_indices_ = S

        # Return reduced feature set and labels matching original input types
        if hasattr(X, "iloc"):
            X_reduced = X.iloc[S]
        else:
            X_reduced = X_arr[S]

        if hasattr(y, "iloc"):
            y_reduced = y.iloc[S]
        elif isinstance(y, list):
            y_reduced = [y[i] for i in S]
        else:
            y_reduced = y_arr[S]

        return X_reduced, y_reduced


class RNG2(InstanceReducer):
    """Relative Neighbourhood Graph 2nd-order editing (RNG2 / RNGE-2).

    RNG2 is an editing method based on proximity graphs. It constructs the
    Relative Neighbourhood Graph (RNG) on the training set and removes an instance
    if and only if:
      1. First-order test: Its immediate RNG neighbours misclassify it via majority vote.
      2. Second-order test: A strict majority of the expanded graph neighbourhood
         surrounding the instance and its same-class RNG neighbours belongs to other classes.

    Reference
    ---------
    J. S. Sánchez, F. Pla, and F. J. Ferri, "Prototype selection for the
    nearest neighbour rule through proximity graphs," Pattern Recognition
    Letters, vol. 18, pp. 507-513, 1997.
    """

    def __init__(self, distance_metric: DistanceMetric | str | None = None) -> None:
        if distance_metric is None:
            distance_metric = EuclideanDistance()
        elif isinstance(distance_metric, str):
            distance_metric = get_distance_metric(distance_metric)
        super().__init__(distance_metric=distance_metric)
        self.selected_indices_: list[int] = []
        self.removed_indices_: list[int] = []

    def fit_resample(self, X: Any, y: Any) -> tuple[Any, Any]:
        """
        Reduce the training instances using RNG2 editing.

        Parameters
        ----------
        X : array-like or pd.DataFrame of shape (n_samples, n_features)
            Training feature vectors.
        y : array-like or pd.Series of shape (n_samples,)
            Categorical target labels.

        Returns
        -------
        X_reduced : array-like or pd.DataFrame
            Edited feature vectors.
        y_reduced : array-like, pd.Series, or list
            Edited target labels.
        """
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have the same length. Got len(X)={len(X)}, len(y)={len(y)}."
            )

        n_samples = len(X)
        if n_samples == 0:
            self.selected_indices_ = []
            self.removed_indices_ = []
            if hasattr(X, "iloc"):
                return X.iloc[[]], y.iloc[[]]
            if isinstance(y, list):
                return (
                    np.empty((0, 0) if not hasattr(X, "shape") else (0, X.shape[1])),
                    [],
                )
            return (
                np.empty((0, 0) if not hasattr(X, "shape") else (0, X.shape[1])),
                np.empty(0, dtype=getattr(y, "dtype", object)),
            )

        if n_samples == 1:
            self.selected_indices_ = [0]
            self.removed_indices_ = []
            return X, y

        # Convert to numpy arrays for internal calculations
        if isinstance(X, np.ndarray):
            X_arr = X
        elif hasattr(X, "to_numpy"):
            X_arr = X.to_numpy()
        else:
            X_arr = np.asarray(X)

        if isinstance(y, (np.ndarray, list)):
            y_arr = np.asarray(y)
        elif hasattr(y, "to_numpy"):
            y_arr = y.to_numpy()
        else:
            y_arr = np.asarray(y)

        # Fit IVDM metric if applicable and not already fitted
        if isinstance(self.distance_metric, IVDMDistance) and not getattr(
            self.distance_metric, "is_fitted", False
        ):
            self.distance_metric.fit(X_arr, y_arr)

        # Step 1: Precompute pairwise distance matrix
        D_mat = np.zeros((n_samples, n_samples), dtype=np.float64)
        for i in range(n_samples):
            D_mat[i] = self.distance_metric.compute_all(X_arr[i], X_arr)

        # Step 2: Construct the Relative Neighbourhood Graph (RNG)
        # (x_i, x_j) in E <=> for all k != i, j: d(x_i, x_j) <= max(d(x_i, x_k), d(x_j, x_k))
        # Equivalently, no k has max(d(x_i, x_k), d(x_j, x_k)) < d(x_i, x_j)
        adj: list[list[int]] = [[] for _ in range(n_samples)]
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                d_ij = D_mat[i, j]
                lune_max = np.maximum(D_mat[i], D_mat[j])
                # Check if any point k lies strictly inside the lune
                if not np.any(lune_max < d_ij - 1e-12):
                    adj[i].append(j)
                    adj[j].append(i)

        # Step 3: Evaluate instances for removal using first- and second-order tests
        R: set[int] = set()

        for p in range(n_samples):
            N_p = adj[p]
            if not N_p:
                # Conservative retention for empty neighbourhood
                continue

            # First-order graph-neighbour vote
            neighbor_labels = [y_arr[q] for q in N_p]
            label_counts = Counter(neighbor_labels)
            max_count = max(label_counts.values())
            top_classes = [c for c, cnt in label_counts.items() if cnt == max_count]

            # Deterministic tie-breaking using closest neighbor among tied classes, then string sort
            if len(top_classes) == 1:
                y_hat = top_classes[0]
            else:
                min_dist_by_class = {
                    c: min(D_mat[p, q] for q in N_p if y_arr[q] == c)
                    for c in top_classes
                }
                y_hat = min(
                    top_classes, key=lambda c: (min_dist_by_class[c], str(c))
                )

            # First-order test: candidate only considered for removal if misclassified
            if y_hat != y_arr[p]:
                # Second-order test
                # S_p = {x_p} union {x_q in N_p : y_q == y_p}
                S_p = {p}
                for q in N_p:
                    if y_arr[q] == y_arr[p]:
                        S_p.add(q)

                # N_S = union of graph neighbours of every vertex in S_p
                N_S: set[int] = set()
                for u in S_p:
                    N_S.update(adj[u])

                M_p = len(N_S)
                B_p = sum(1 for q in N_S if y_arr[q] != y_arr[p])

                # Second-order condition: strict majority from opposing classes
                if M_p > 0 and B_p > M_p / 2.0:
                    R.add(p)

        selected_indices = [i for i in range(n_samples) if i not in R]
        self.selected_indices_ = selected_indices
        self.removed_indices_ = sorted(R)

        # Return reduced feature set and labels matching original input types
        if hasattr(X, "iloc"):
            X_reduced = X.iloc[selected_indices]
        else:
            X_reduced = X_arr[selected_indices]

        if hasattr(y, "iloc"):
            y_reduced = y.iloc[selected_indices]
        elif isinstance(y, list):
            y_reduced = [y[i] for i in selected_indices]
        else:
            y_reduced = y_arr[selected_indices]

        return X_reduced, y_reduced


RNGE = RNG2
