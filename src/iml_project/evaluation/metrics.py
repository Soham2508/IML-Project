from __future__ import annotations

import numpy as np


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise ValueError("Prediction and truth arrays differ in length.")
    correct = np.sum(y_true_arr == y_pred_arr)
    return float(correct / len(y_true_arr)) if len(y_true_arr) else 0.0
