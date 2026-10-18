from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..data.arff_reader import load_dataset_fold
from ..data.folds import get_dataset_fold_map
from ..data.preprocessing import fit_and_transform
from ..evaluation.metrics import accuracy_score
from ..models.sklearn_models import rfAlgorithm, svmAlgorithm


def _prepare_dataframe(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_names: list[str],
    feature_types: list[str],
    target_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_scaled, test_scaled, _ = fit_and_transform(
        train_df, test_df, feature_names, feature_types, target_name
    )
    return (
        train_scaled.drop(columns=[target_name]),
        test_scaled.drop(columns=[target_name]),
    )


def _evaluate_model_config(
    dataset_name: str,
    fold_index: int,
    config_name: str,
    params: dict[str, object],
    data_root: str | Path,
) -> dict[str, object]:
    train_df, test_df, feature_names, feature_types, target_name = load_dataset_fold(
        dataset_name, fold_index, data_root
    )
    train_scaled, test_scaled, _ = fit_and_transform(
        train_df, test_df, feature_names, feature_types, target_name
    )

    X_train = train_scaled.drop(columns=[target_name]).to_numpy(dtype=float)
    X_test = test_scaled.drop(columns=[target_name]).to_numpy(dtype=float)
    y_train = train_scaled[target_name].to_numpy()
    y_test = test_scaled[target_name].to_numpy()

    if config_name == "svm":
        predictions, timings = svmAlgorithm(
            X_train,
            y_train,
            X_test,
            kernel=params["kernel"],
            C=float(params["C"]),
            gamma=params.get("gamma", "scale"),
        )
    elif config_name == "random_forest":
        predictions, timings = rfAlgorithm(
            X_train,
            y_train,
            X_test,
            n_estimators=int(params["n_estimators"]),
            criterion=params["criterion"],
            random_state=int(params.get("random_state", 42)),
        )
    else:
        raise ValueError(f"Unsupported config name: {config_name}")

    correct = int((predictions == y_test).sum())
    accuracy = accuracy_score(y_test, predictions)
    return {
        "dataset": dataset_name,
        "fold_index": fold_index,
        "model_name": config_name,
        "model_params": json.dumps(params, sort_keys=True),
        "num_test_cases": len(y_test),
        "correct_count": correct,
        "incorrect_count": int(len(y_test) - correct),
        "accuracy": accuracy,
        "fit_time": float(timings["fit_time"]),
        "prediction_time": float(timings["prediction_time"]),
        "average_prediction_time": float(timings["prediction_time"] / len(y_test))
        if len(y_test)
        else 0.0,
    }


def run_week1_experiments(
    data_root: str | Path = "data", output_dir: str | Path = "results/week1"
) -> dict[str, pd.DataFrame]:
    """Run the Week 1 SVM and Random Forest experiments and save CSV results."""
    root = Path(data_root)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset_map = get_dataset_fold_map(root)
    svm_rows: list[dict[str, object]] = []
    rf_rows: list[dict[str, object]] = []

    svm_configs = [
        {"kernel": "linear", "C": 0.1, "gamma": "scale"},
        {"kernel": "linear", "C": 1.0, "gamma": "scale"},
        {"kernel": "linear", "C": 10.0, "gamma": "scale"},
        {"kernel": "rbf", "C": 0.1, "gamma": "scale"},
        {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
        {"kernel": "rbf", "C": 10.0, "gamma": "scale"},
    ]
    rf_configs = [
        {"n_estimators": 50, "criterion": "gini", "random_state": 42},
        {"n_estimators": 100, "criterion": "gini", "random_state": 42},
        {"n_estimators": 200, "criterion": "gini", "random_state": 42},
        {"n_estimators": 50, "criterion": "entropy", "random_state": 42},
        {"n_estimators": 100, "criterion": "entropy", "random_state": 42},
        {"n_estimators": 200, "criterion": "entropy", "random_state": 42},
    ]

    for dataset_name, folds in dataset_map.items():
        for fold in folds:
            for config in svm_configs:
                row = _evaluate_model_config(
                    dataset_name, fold.fold_index, "svm", config, root
                )
                svm_rows.append(row)
            for config in rf_configs:
                row = _evaluate_model_config(
                    dataset_name, fold.fold_index, "random_forest", config, root
                )
                rf_rows.append(row)

    svm_df = pd.DataFrame(svm_rows)
    rf_df = pd.DataFrame(rf_rows)

    svm_summary = (
        svm_df.groupby(["dataset", "model_name", "model_params"], as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            total_correct=("correct_count", "sum"),
            total_cases=("num_test_cases", "sum"),
            mean_prediction_time=("average_prediction_time", "mean"),
            mean_fit_time=("fit_time", "mean"),
        )
        .sort_values(
            ["dataset", "model_name", "mean_accuracy"], ascending=[True, True, False]
        )
    )

    rf_summary = (
        rf_df.groupby(["dataset", "model_name", "model_params"], as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            total_correct=("correct_count", "sum"),
            total_cases=("num_test_cases", "sum"),
            mean_prediction_time=("average_prediction_time", "mean"),
            mean_fit_time=("fit_time", "mean"),
        )
        .sort_values(
            ["dataset", "model_name", "mean_accuracy"], ascending=[True, True, False]
        )
    )

    svm_path = output_path / "svm_fold_results.csv"
    svm_summary_path = output_path / "svm_summary.csv"
    rf_path = output_path / "random_forest_fold_results.csv"
    rf_summary_path = output_path / "random_forest_summary.csv"

    svm_df.to_csv(svm_path, index=False)
    svm_summary.to_csv(svm_summary_path, index=False)
    rf_df.to_csv(rf_path, index=False)
    rf_summary.to_csv(rf_summary_path, index=False)

    return {
        "svm_fold": svm_df,
        "svm_summary": svm_summary,
        "random_forest_fold": rf_df,
        "random_forest_summary": rf_summary,
    }


def validate_data_files(data_root: str | Path = "data") -> dict[str, list[str]]:
    dataset_map = get_dataset_fold_map(data_root)
    report: dict[str, list[str]] = {}
    for dataset_name, folds in dataset_map.items():
        report[dataset_name] = [f"fold_{fold.fold_index:06d}" for fold in folds]
    return report
