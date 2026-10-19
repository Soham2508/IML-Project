from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any
import numpy as np


class CaseBase:
    """Manages the stored instances and goodness values for KNN generalization."""

    def __init__(self, X: np.ndarray, y: Sequence[Any]) -> None:
        self.X = np.array(X, dtype=object if X.dtype == object else np.float64)
        self.y = list(y)
        self.initial_size = len(self.y)
        self.goodness: list[float] = [0.5] * self.initial_size

    def __len__(self) -> int:
        return len(self.y)

    def add_case(
        self, x: np.ndarray, label: Any, initial_goodness: float = 0.5
    ) -> None:
        """Append a newly retained case to the case base."""
        x_row = np.array([x], dtype=self.X.dtype)
        self.X = np.vstack([self.X, x_row])
        self.y.append(label)
        self.goodness.append(initial_goodness)

    def remove_cases(self, indices_to_remove: set[int]) -> None:
        """Remove specified indices from the case base."""
        if not indices_to_remove:
            return
        keep_mask = np.ones(len(self.y), dtype=bool)
        for idx in indices_to_remove:
            if 0 <= idx < len(self.y):
                keep_mask[idx] = False

        self.X = self.X[keep_mask]
        self.y = [label for i, label in enumerate(self.y) if keep_mask[i]]
        self.goodness = [g for i, g in enumerate(self.goodness) if keep_mask[i]]


class RetentionPolicy(ABC):
    """Abstract base class for instance retention and forgetting policies."""

    @abstractmethod
    def update(
        self,
        case_base: CaseBase,
        query: np.ndarray,
        predicted_class: Any,
        true_class: Any,
        retrieved_indices: Sequence[int],
    ) -> None:
        """Update the case base after a query instance has been classified."""
        raise NotImplementedError


class NeverRetain(RetentionPolicy):
    """Never Retain (NR): Training instances remain strictly fixed."""

    def update(
        self,
        case_base: CaseBase,
        query: np.ndarray,
        predicted_class: Any,
        true_class: Any,
        retrieved_indices: Sequence[int],
    ) -> None:
        # Case base remains unchanged
        pass


class AlwaysRetain(RetentionPolicy):
    """Always Retain (AR): Retains all new solved instances in the instance base."""

    def update(
        self,
        case_base: CaseBase,
        query: np.ndarray,
        predicted_class: Any,
        true_class: Any,
        retrieved_indices: Sequence[int],
    ) -> None:
        case_base.add_case(query, true_class)


class DifferentClass(RetentionPolicy):
    """Different Class retention (DC): Retains instances solved incorrectly."""

    def update(
        self,
        case_base: CaseBase,
        query: np.ndarray,
        predicted_class: Any,
        true_class: Any,
        retrieved_indices: Sequence[int],
    ) -> None:
        if str(predicted_class) != str(true_class):
            case_base.add_case(query, true_class)


class OblivionByGoodness(RetentionPolicy):
    """Oblivion by Goodness (OG):

    - Case goodness initialized to G_0 = 0.5.
    - Retrieved k neighbors update goodness based on reward:
        r(c) = 1 if class(c) == class(q) else 0
        G_t(c) = G_{t-1}(c) + alpha * (r(c) - G_{t-1}(c)), alpha = 0.2
    - A case is removed whenever G_t(c) < 0.5.
    - If retain_solved=True, newly retained cases inherit initial goodness G_0 = 0.5.
    """

    def __init__(
        self,
        alpha: float = 0.2,
        initial_goodness: float = 0.5,
        retain_solved: bool = False,
    ) -> None:
        self.alpha = alpha
        self.initial_goodness = initial_goodness
        self.retain_solved = retain_solved

    def update(
        self,
        case_base: CaseBase,
        query: np.ndarray,
        predicted_class: Any,
        true_class: Any,
        retrieved_indices: Sequence[int],
    ) -> None:
        indices_to_remove: set[int] = set()

        # Update goodness only for retrieved neighbors
        for idx in retrieved_indices:
            if idx >= len(case_base.goodness):
                continue
            neighbor_class = case_base.y[idx]
            reward = 1.0 if str(neighbor_class) == str(true_class) else 0.0
            old_g = case_base.goodness[idx]
            new_g = old_g + self.alpha * (reward - old_g)
            case_base.goodness[idx] = new_g

            # Forgetting condition: remove if current goodness < initial goodness
            if new_g < self.initial_goodness:
                indices_to_remove.add(idx)

        # Apply forgetting
        case_base.remove_cases(indices_to_remove)

        # Optional retention of newly solved instance
        if self.retain_solved:
            case_base.add_case(
                query, true_class, initial_goodness=self.initial_goodness
            )


def get_retention_policy(name: str, **kwargs: Any) -> RetentionPolicy:
    """Factory returning an instance of the requested retention policy."""
    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")

    if normalized in {"nr", "never", "never_retain"}:
        return NeverRetain()
    elif normalized in {"ar", "always", "always_retain"}:
        return AlwaysRetain()
    elif normalized in {"dc", "different_class", "different_class_retention"}:
        return DifferentClass()
    elif normalized in {"og", "oblivion", "oblivion_by_goodness"}:
        alpha = kwargs.get("alpha", 0.2)
        initial_goodness = kwargs.get("initial_goodness", 0.5)
        retain_solved = kwargs.get("retain_solved", False)
        return OblivionByGoodness(
            alpha=alpha,
            initial_goodness=initial_goodness,
            retain_solved=retain_solved,
        )
    else:
        raise ValueError(
            f"Unknown retention policy '{name}'. Supported: nr, ar, dc, og."
        )
