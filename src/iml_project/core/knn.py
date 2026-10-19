from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any
import numpy as np

from iml_project.utils.distance_metrics import DistanceMetric, IVDMDistance
from .retention import CaseBase, NeverRetain, RetentionPolicy
from .voting import MajorityVoting, VotingStrategy


class KNNClassifier:
    """K-Nearest Neighbor classifier implemented from scratch."""

    def __init__(
        self,
        k: int = 3,
        distance_metric: DistanceMetric | None = None,
        voting_strategy: VotingStrategy | None = None,
        retention_policy: RetentionPolicy | None = None,
    ) -> None:
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        self.k = k
        self.distance_metric = distance_metric
        self.voting_strategy = voting_strategy or MajorityVoting()
        self.retention_policy = retention_policy or NeverRetain()

        self.case_base: CaseBase | None = None
        self.class_priors: dict[Any, float] = {}
        self.classes_: list[Any] = []

    def fit(self, X: np.ndarray, y: Sequence[Any]) -> KNNClassifier:
        """Initialize the case base from training instances."""
        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")
        if len(X) == 0:
            raise ValueError("Training data cannot be empty.")

        self.case_base = CaseBase(X, y)
        labels = list(y)
        self.classes_ = sorted(set(labels), key=str)
        total = len(labels)
        self.class_priors = {c: labels.count(c) / total for c in self.classes_}

        # If metric is IVDM, fit conditional probabilities on training fold
        if isinstance(self.distance_metric, IVDMDistance):
            self.distance_metric.fit(self.case_base.X, np.array(labels))

        return self

    def find_k_nearest(
        self, query: np.ndarray
    ) -> tuple[list[int], list[Any], list[float]]:
        """Find the k nearest neighbors and their distances deterministically."""
        if self.case_base is None or len(self.case_base) == 0:
            raise RuntimeError("Classifier has not been fitted or case base is empty.")
        if self.distance_metric is None:
            raise RuntimeError("Distance metric has not been specified.")

        n_cases = len(self.case_base)
        actual_k = min(self.k, n_cases)

        # Compute all distances from query to current case base
        distances = self.distance_metric.compute_all(query, self.case_base.X)

        # Stable sort preserving original training indices on ties
        # np.argsort with kind='stable' guarantees deterministic order
        sorted_indices = np.argsort(distances, kind="stable")
        neighbor_indices = sorted_indices[:actual_k].tolist()
        neighbor_distances = [float(distances[i]) for i in neighbor_indices]
        neighbor_classes = [self.case_base.y[i] for i in neighbor_indices]

        return neighbor_indices, neighbor_classes, neighbor_distances

    def predict_one(self, query: np.ndarray) -> tuple[Any, list[int], list[float]]:
        """Predict the class of a single query instance without updating the case base."""
        indices, classes, distances = self.find_k_nearest(query)
        prediction = self.voting_strategy.decide(
            classes, distances, class_priors=self.class_priors
        )
        return prediction, indices, distances

    def predict_and_update(
        self, query: np.ndarray, true_label: Any
    ) -> tuple[Any, list[int], list[float]]:
        """Predict class for query, then update case base with true label."""
        prediction, indices, distances = self.predict_one(query)

        # Strictly update case base after prediction has been determined
        if self.case_base is not None and self.retention_policy is not None:
            self.retention_policy.update(
                self.case_base, query, prediction, true_label, indices
            )

        return prediction, indices, distances

    def evaluate(self, X_test: np.ndarray, y_test: Sequence[Any]) -> dict[str, Any]:
        """Sequentially evaluate a test fold, applying retention policies after each query."""
        if self.case_base is None:
            raise RuntimeError("Classifier must be fitted before evaluation.")

        initial_size = len(self.case_base)
        y_true = list(y_test)
        y_pred = []
        n_queries = len(X_test)
        prediction_times: list[float] = []

        for i in range(n_queries):
            query = X_test[i]
            label = y_true[i]

            t0 = time.perf_counter()
            pred, _, _ = self.predict_and_update(query, label)
            t1 = time.perf_counter()

            prediction_times.append(t1 - t0)
            y_pred.append(pred)

        correct = sum(1 for p, t in zip(y_pred, y_true) if str(p) == str(t))
        incorrect = n_queries - correct
        accuracy = correct / n_queries if n_queries > 0 else 0.0
        final_size = len(self.case_base)
        retention_ratio = final_size / initial_size if initial_size > 0 else 1.0

        return {
            "accuracy": accuracy,
            "correct": correct,
            "incorrect": incorrect,
            "total": n_queries,
            "y_pred": y_pred,
            "y_true": y_true,
            "initial_size": initial_size,
            "final_size": final_size,
            "retention_ratio": retention_ratio,
            "mean_prediction_time": float(np.mean(prediction_times))
            if prediction_times
            else 0.0,
            "total_prediction_time": float(np.sum(prediction_times)),
        }


def kNNAlgorithm(
    train_matrix: np.ndarray,
    train_labels: Sequence[Any],
    test_matrix: np.ndarray,
    test_labels: Sequence[Any],
    *,
    k: int = 3,
    distance_metric: DistanceMetric,
    voting_strategy: VotingStrategy | None = None,
    retention_policy: RetentionPolicy | None = None,
) -> dict[str, Any]:
    """Assignment entry point function for K-IBL classification."""
    classifier = KNNClassifier(
        k=k,
        distance_metric=distance_metric,
        voting_strategy=voting_strategy,
        retention_policy=retention_policy,
    )
    classifier.fit(train_matrix, train_labels)
    return classifier.evaluate(test_matrix, test_labels)
