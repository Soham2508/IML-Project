from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np


class VotingStrategy(ABC):
    """Abstract base class for KNN voting schemes."""

    @abstractmethod
    def decide(
        self,
        neighbor_classes: Sequence[Any],
        neighbor_distances: Sequence[float],
        class_priors: dict[Any, float] | None = None,
    ) -> Any:
        """Decide the predicted class label given nearest neighbors and their distances."""
        raise NotImplementedError


def _break_ties(
    tied_classes: list[Any],
    neighbor_classes: Sequence[Any],
    neighbor_distances: Sequence[float],
    class_priors: dict[Any, float] | None = None,
) -> Any:
    """Deterministically break class ties.

    Rule:
      1. Choose the class that has the closest individual neighbor (1-NN tie-break).
      2. If still tied, choose the class with higher prior probability in the training set.
      3. If still tied, use stable lexicographical sorting of class representations.
    """
    if len(tied_classes) == 1:
        return tied_classes[0]

    # 1. Closest individual neighbor among tied classes
    min_dist_by_class: dict[Any, float] = {}
    for cls, dist in zip(neighbor_classes, neighbor_distances):
        if cls in tied_classes and (
            cls not in min_dist_by_class or dist < min_dist_by_class[cls]
        ):
            min_dist_by_class[cls] = float(dist)

    sorted_by_min_dist = sorted(
        tied_classes, key=lambda c: min_dist_by_class.get(c, float("inf"))
    )
    best_dist = min_dist_by_class.get(sorted_by_min_dist[0], float("inf"))
    cand_after_dist = [
        c
        for c in sorted_by_min_dist
        if np.isclose(min_dist_by_class.get(c, float("inf")), best_dist)
    ]

    if len(cand_after_dist) == 1:
        return cand_after_dist[0]

    # 2. Prior probability
    if class_priors:
        cand_after_prior = sorted(
            cand_after_dist, key=lambda c: class_priors.get(c, 0.0), reverse=True
        )
        best_prior = class_priors.get(cand_after_prior[0], 0.0)
        final_cand = [
            c
            for c in cand_after_prior
            if np.isclose(class_priors.get(c, 0.0), best_prior)
        ]
        if len(final_cand) == 1:
            return final_cand[0]
        cand_after_dist = final_cand

    # 3. Stable string sort
    return min(cand_after_dist, key=lambda c: str(c))


class MajorityVoting(VotingStrategy):
    """Majority voting scheme.

    Each of the k nearest neighbors casts 1 vote.
    """

    def decide(
        self,
        neighbor_classes: Sequence[Any],
        neighbor_distances: Sequence[float],
        class_priors: dict[Any, float] | None = None,
    ) -> Any:
        if not neighbor_classes:
            raise ValueError("No neighbor classes provided to voting scheme.")

        counts = Counter(neighbor_classes)
        max_votes = max(counts.values())
        tied_classes = [c for c, cnt in counts.items() if cnt == max_votes]

        return _break_ties(
            tied_classes, neighbor_classes, neighbor_distances, class_priors
        )


class InverseDistanceWeightedVoting(VotingStrategy):
    """Inverse-distance voting; zero-distance neighbours, if any, decide alone."""

    def __init__(self, p: float = 1.0) -> None:
        if p <= 0:
            raise ValueError(f"Power parameter p must be > 0, got {p}")
        self.p = p

    def decide(
        self,
        neighbor_classes: Sequence[Any],
        neighbor_distances: Sequence[float],
        class_priors: dict[Any, float] | None = None,
    ) -> Any:
        if not neighbor_classes:
            raise ValueError("No neighbor classes provided to voting scheme.")

        dist_arr = np.array(neighbor_distances, dtype=np.float64)
        zero_mask = np.isclose(dist_arr, 0.0)

        # Exact match / zero-distance case
        if np.any(zero_mask):
            zero_classes = [
                cls for cls, is_zero in zip(neighbor_classes, zero_mask) if is_zero
            ]
            zero_counts = Counter(zero_classes)
            max_c = max(zero_counts.values())
            tied = [c for c, cnt in zero_counts.items() if cnt == max_c]
            return _break_ties(tied, neighbor_classes, neighbor_distances, class_priors)

        # Normal weighted voting
        weights = 1.0 / (dist_arr**self.p)
        score_by_class: dict[Any, float] = {}
        for cls, w in zip(neighbor_classes, weights):
            score_by_class[cls] = score_by_class.get(cls, 0.0) + float(w)

        max_score = max(score_by_class.values())
        tied_classes = [
            c for c, score in score_by_class.items() if np.isclose(score, max_score)
        ]

        return _break_ties(
            tied_classes, neighbor_classes, neighbor_distances, class_priors
        )


def get_voting_strategy(name: str, **kwargs: Any) -> VotingStrategy:
    """Factory returning an instance of the requested voting strategy."""
    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")

    if normalized in {"majority", "majority_voting", "uniform"}:
        return MajorityVoting()
    elif normalized in {
        "inverse_distance",
        "inverse_distance_weighted",
        "idw",
        "distance",
    }:
        p = kwargs.get("p", 1.0)
        return InverseDistanceWeightedVoting(p=p)
    else:
        raise ValueError(
            f"Unknown voting strategy '{name}'. Supported: majority, inverse_distance."
        )
