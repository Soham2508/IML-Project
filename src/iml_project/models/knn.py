from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .distance_metrics import compute_distance


def _choose_closest_neighbor_class(neighbor_classes: list[Any], neighbor_distances: list[float]) -> Any:
    if not neighbor_classes:
        raise ValueError("No neighboring classes available for tie-breaking.")
    min_distance = min(neighbor_distances)
    closest = [cls for cls, distance in zip(neighbor_classes, neighbor_distances) if np.isclose(distance, min_distance)]
    if len(set(closest)) == 1:
        return closest[0]
    return sorted(set(closest), key=lambda item: str(item))[0]


def _vote_majority(neighbor_classes: list[Any]) -> Any:
    counts = Counter(neighbor_classes)
    max_count = max(counts.values())
    winners = [label for label, count in counts.items() if count == max_count]
    return sorted(winners, key=lambda item: str(item))[0]


def _vote_inverse_distance(neighbor_classes: list[Any], neighbor_distances: list[float]) -> Any:
    zero_distance_mask = np.isclose(neighbor_distances, 0.0)
    if np.any(zero_distance_mask):
        zero_labels = [label for label, is_zero in zip(neighbor_classes, zero_distance_mask) if is_zero]
        if zero_labels:
            counts = Counter(zero_labels)
            max_count = max(counts.values())
            winners = [label for label, count in counts.items() if count == max_count]
            return sorted(winners, key=lambda item: str(item))[0]

    votes: dict[Any, float] = {}
    for label, distance in zip(neighbor_classes, neighbor_distances):
        if np.isclose(distance, 0.0):
            continue
        weight = 1.0 / float(distance)
        votes[label] = votes.get(label, 0.0) + weight

    if not votes:
        return _choose_closest_neighbor_class(neighbor_classes, neighbor_distances)

    max_vote = max(votes.values())
    winners = [label for label, value in votes.items() if np.isclose(value, max_vote)]
    if len(winners) > 1:
        return _choose_closest_neighbor_class(neighbor_classes, neighbor_distances)
    return winners[0]


def kNNAlgorithm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    k: int = 3,
    metric: str = "euclidean",
    voting_scheme: str = "majority_class",
    retention_policy: str = "never_retain",
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Perform a deterministic custom k-NN evaluation, returning predictions and per-query metadata."""
    if k <= 0:
        raise ValueError("k must be positive.")

    predictions: list[Any] = []
    details: list[dict[str, Any]] = []

    case_base = [row.copy() for row in X_train]
    labels = [label for label in y_train]

    for query in X_test:
        distances = [compute_distance(case, query, metric) for case in case_base]
        indexed = sorted(enumerate(distances), key=lambda pair: (pair[1], pair[0]))[:k]
        neighbor_indices = [idx for idx, _ in indexed]
        neighbor_distances = [distance for _, distance in indexed]
        neighbor_classes = [labels[idx] for idx in neighbor_indices]

        if voting_scheme == "majority_class":
            predicted = _vote_majority(neighbor_classes)
        elif voting_scheme == "inverse_distance_weighted":
            predicted = _vote_inverse_distance(neighbor_classes, neighbor_distances)
        else:
            raise ValueError(f"Unsupported voting scheme: {voting_scheme}")

        predictions.append(predicted)
        details.append(
            {
                "query": np.asarray(query, dtype=float).copy(),
                "k": k,
                "metric": metric,
                "voting_scheme": voting_scheme,
                "retention_policy": retention_policy,
                "neighbor_indices": neighbor_indices,
                "neighbor_distances": neighbor_distances,
                "neighbor_classes": neighbor_classes,
                "predicted_label": predicted,
            }
        )

        if retention_policy == "never_retain":
            continue
        if retention_policy == "always_retain":
            case_base.append(np.asarray(query, dtype=float).copy())
            labels.append(query[-1] if isinstance(query, np.ndarray) and query.size else None)
            continue

        # The run-time retention policies here are intentionally kept as scaffolding for the full Week 2 implementation.
        # Detailed policy-specific logic is handled in the retention helper and by the runner.

    return np.asarray(predictions), details
