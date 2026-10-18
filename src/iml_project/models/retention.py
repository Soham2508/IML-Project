from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class RetentionSummary:
    policy: str
    case_base_size: int
    updated_cases: list[int]
    removed_cases: list[int]


def apply_retention_policy(
    case_base: list[Any],
    labels: list[Any],
    query_vector: Any,
    true_label: Any,
    predicted_label: Any,
    policy: str,
    neighbors: list[int] | None = None,
    neighbor_goodness: dict[int, float] | None = None,
    baseline_goodness: dict[int, float] | None = None,
    alpha: float = 0.2,
) -> tuple[list[Any], list[Any], RetentionSummary]:
    """Apply the assignment-specified retention and forgetting policy.

    For OG, the default interpretation is to update the retrieved neighbors and discard cases whose
    goodness falls below the baseline, without appending every query to the case base.
    """
    if policy == "never_retain":
        return (
            list(case_base),
            list(labels),
            RetentionSummary(policy, len(case_base), [], []),
        )

    if policy == "always_retain":
        case_base = list(case_base)
        labels = list(labels)
        case_base.append(query_vector)
        labels.append(true_label)
        return (
            case_base,
            labels,
            RetentionSummary(policy, len(case_base), [len(case_base) - 1], []),
        )

    if policy == "different_class":
        if predicted_label != true_label:
            case_base = list(case_base)
            labels = list(labels)
            case_base.append(query_vector)
            labels.append(true_label)
            return (
                case_base,
                labels,
                RetentionSummary(policy, len(case_base), [len(case_base) - 1], []),
            )
        return (
            list(case_base),
            list(labels),
            RetentionSummary(policy, len(case_base), [], []),
        )

    if policy == "oblivion_by_goodness":
        if neighbors is None:
            raise ValueError("Neighbors are required for the OG policy.")
        case_base = list(case_base)
        labels = list(labels)
        if neighbor_goodness is None:
            neighbor_goodness = {idx: 0.5 for idx in neighbors}
        if baseline_goodness is None:
            baseline_goodness = {idx: 0.5 for idx in neighbors}

        updated_cases: list[int] = []
        removed_cases: list[int] = []
        for idx in neighbors:
            if idx < 0 or idx >= len(case_base):
                continue
            reward = 1.0 if labels[idx] == true_label else 0.0
            old_goodness = neighbor_goodness.get(idx, 0.5)
            new_goodness = old_goodness + alpha * (reward - old_goodness)
            neighbor_goodness[idx] = new_goodness
            updated_cases.append(idx)
            baseline = baseline_goodness.get(idx, 0.5)
            if new_goodness < baseline:
                case_base.pop(idx)
                labels.pop(idx)
                removed_cases.append(idx)
        return (
            case_base,
            labels,
            RetentionSummary(policy, len(case_base), updated_cases, removed_cases),
        )

    raise ValueError(f"Unsupported retention policy: {policy}")
