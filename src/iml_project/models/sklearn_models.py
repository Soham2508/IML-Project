from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def svmAlgorithm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    kernel: str = "linear",
    C: float = 1.0,
    gamma: str | float = "scale",
) -> tuple[np.ndarray, dict[str, float]]:
    """Train and evaluate a scikit-learn SVC configured for the assignment grid."""
    start = time.perf_counter()
    model = SVC(kernel=kernel, C=C, gamma=gamma)
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - start

    predict_start = time.perf_counter()
    predictions = model.predict(X_test)
    predict_time = time.perf_counter() - predict_start
    return predictions, {
        "fit_time": fit_time,
        "prediction_time": predict_time,
        "model": model,
    }


def rfAlgorithm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_estimators: int = 100,
    criterion: str = "gini",
    random_state: int = 42,
) -> tuple[np.ndarray, dict[str, float]]:
    """Train and evaluate a scikit-learn RandomForestClassifier configured for the assignment grid."""
    start = time.perf_counter()
    model = RandomForestClassifier(
        n_estimators=n_estimators, criterion=criterion, random_state=random_state
    )
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - start

    predict_start = time.perf_counter()
    predictions = model.predict(X_test)
    predict_time = time.perf_counter() - predict_start
    return predictions, {
        "fit_time": fit_time,
        "prediction_time": predict_time,
        "model": model,
    }
