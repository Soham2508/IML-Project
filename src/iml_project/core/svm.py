from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support
from sklearn.svm import SVC


def svmAlgorithm(
    train_matrix: np.ndarray,
    train_labels: Sequence[Any],
    test_matrix: np.ndarray,
    test_labels: Sequence[Any],
    *,
    kernel: str = "linear",
    C: float = 1.0,
    gamma: str | float = "scale",
    random_state: int = 42,
) -> dict[str, Any]:
    """Support Vector Machine classification algorithm using scikit-learn's SVC.

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
    kernel : str, default="linear"
        Specifies the kernel type to be used in the algorithm ('linear', 'rbf', etc.).
    C : float, default=1.0
        Regularization parameter (strictly positive).
    gamma : str | float, default="scale"
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. Can be 'scale', 'auto', or a float.
        Note: For 'linear' kernel, gamma is ignored by the underlying libsvm implementation.
    random_state : int, default=42
        Random state for pseudo-random number generator.

    Returns
    -------
    dict[str, Any]
        Dictionary containing evaluation metrics, timings, and predictions.
    """
    if C <= 0:
        raise ValueError(f"C must be strictly positive, got {C}")

    # Parse numeric gamma if passed as string numeric
    parsed_gamma: str | float = gamma
    if isinstance(gamma, str):
        try:
            parsed_gamma = float(gamma)
        except ValueError:
            parsed_gamma = gamma

    clf = SVC(
        C=C,
        kernel=kernel,
        gamma=parsed_gamma,
        random_state=random_state,
    )

    t_fit_start = time.perf_counter()
    clf.fit(train_matrix, train_labels)
    fit_time = time.perf_counter() - t_fit_start

    t_pred_start = time.perf_counter()
    y_pred = clf.predict(test_matrix)
    total_prediction_time = time.perf_counter() - t_pred_start

    n_test = len(test_labels)
    correct = int(np.sum([str(p) == str(t) for p, t in zip(y_pred, test_labels)]))
    incorrect = n_test - correct
    accuracy = correct / n_test if n_test > 0 else 0.0

    if n_test > 0:
        balanced_acc = float(balanced_accuracy_score(test_labels, y_pred))
        prec, rec, f1, _ = precision_recall_fscore_support(
            test_labels, y_pred, average="macro", zero_division=0
        )
    else:
        balanced_acc, prec, rec, f1 = 0.0, 0.0, 0.0, 0.0

    mean_prediction_time = total_prediction_time / n_test if n_test > 0 else 0.0
    initial_size = len(train_labels)

    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "correct": correct,
        "incorrect": incorrect,
        "total": n_test,
        "fit_time": float(fit_time),
        "mean_prediction_time": float(mean_prediction_time),
        "total_prediction_time": float(total_prediction_time),
        "initial_size": initial_size,
        "final_size": initial_size,
        "retention_ratio": 1.0,
        "y_pred": list(y_pred),
        "y_true": list(test_labels),
    }
