from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support


def rfAlgorithm(
    train_matrix: np.ndarray,
    train_labels: Sequence[Any],
    test_matrix: np.ndarray,
    test_labels: Sequence[Any],
    *,
    n_estimators: int = 100,
    criterion: str = "gini",
    max_depth: int | None = None,
    random_state: int = 42,
    n_jobs: int = -1,
) -> dict[str, Any]:
    """Random Forest classification algorithm using scikit-learn's RandomForestClassifier.

    Parameters
    ----------
    train_matrix : np.ndarray
        Training feature matrix (n_train, n_features).
    train_labels : Sequence[Any]
        Training target labels.
    test_matrix : np.ndarray
        Test feature matrix (n_test, n_features).
    test_labels : Sequence[Any]
        Test target labels.
    n_estimators : int, default=100
        The number of trees in the forest (strictly positive integer).
    criterion : str, default="gini"
        The function to measure the quality of a split ('gini', 'entropy', or 'log_loss').
    max_depth : int | None, default=None
        The maximum depth of the tree. If None, nodes are expanded until all leaves are pure.
    random_state : int, default=42
        Controls the randomness of the bootstrapping and feature splitting.
    n_jobs : int, default=-1
        The number of jobs to run in parallel. -1 means using all processors.

    Returns
    -------
    dict[str, Any]
        Dictionary containing evaluation metrics, timings, and predictions.
    """
    if n_estimators <= 0:
        raise ValueError(f"n_estimators must be strictly positive, got {n_estimators}")
    if criterion not in {"gini", "entropy", "log_loss"}:
        raise ValueError(
            f"criterion must be 'gini', 'entropy', or 'log_loss', got '{criterion}'"
        )
    if max_depth is not None and max_depth <= 0:
        raise ValueError(f"max_depth must be strictly positive when specified, got {max_depth}")

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    t_fit_start = time.perf_counter()
    clf.fit(train_matrix, train_labels)
    fit_time = time.perf_counter() - t_fit_start

    t_pred_start = time.perf_counter()
    y_pred = clf.predict(test_matrix)
    total_prediction_time = time.perf_counter() - t_pred_start

    n_test = len(test_labels)
    mean_prediction_time = total_prediction_time / n_test if n_test > 0 else 0.0

    y_true_list = list(test_labels)
    y_pred_list = list(y_pred)

    correct = sum(1 for p, t in zip(y_pred_list, y_true_list) if str(p) == str(t))
    incorrect = n_test - correct
    accuracy = correct / n_test if n_test > 0 else 0.0

    # Extended classification metrics
    if n_test > 0:
        balanced_acc = float(balanced_accuracy_score(y_true_list, y_pred_list))
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true_list, y_pred_list, average="macro", zero_division=0
        )
    else:
        balanced_acc, prec, rec, f1 = 0.0, 0.0, 0.0, 0.0

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "correct": correct,
        "incorrect": incorrect,
        "total": n_test,
        "y_pred": y_pred_list,
        "y_true": y_true_list,
        "fit_time": fit_time,
        "mean_prediction_time": mean_prediction_time,
        "total_prediction_time": total_prediction_time,
        "initial_size": len(train_matrix),
        "final_size": len(train_matrix),
        "retention_ratio": 1.0,
    }
