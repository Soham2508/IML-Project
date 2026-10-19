from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from iml_project.core.knn import KNNClassifier
from iml_project.core.retention import get_retention_policy
from iml_project.core.voting import get_voting_strategy
from iml_project.utils.arff_reader import load_dataset_fold, resolve_dataset_dir
from iml_project.utils.distance_metrics import (
    DistanceMetric,
    get_distance_metric,
)
from iml_project.utils.preprocessing import Preprocessor


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str
    fold: int
    distance_metric: str
    k: int
    voting_strategy: str
    retention_policy: str


@dataclass
class ExperimentResult:
    dataset: str
    fold: int
    distance_metric: str
    k: int
    voting_strategy: str
    retention_policy: str
    accuracy: float
    correct: int
    incorrect: int
    total: int
    initial_size: int
    final_size: int
    retention_ratio: float
    mean_prediction_time: float
    total_prediction_time: float


def prepare_fold_data(
    dataset: str,
    fold: int,
    distance_metric_name: str,
    data_root: str | Path = "data",
) -> tuple[np.ndarray, list[Any], np.ndarray, list[Any], DistanceMetric]:
    """Load and preprocess fold data with zero test leakage, tailored to the chosen distance metric."""
    train_df, test_df, names, types, target_name = load_dataset_fold(
        dataset, fold, data_root
    )

    # HEOM and IVDM need intact nominal columns and normalized numeric columns
    use_raw_categoricals = distance_metric_name.lower() in {"heom", "ivdm"}

    preprocessor = Preprocessor.fit(
        train_df,
        feature_names=names,
        feature_types=types,
        target_name=target_name,
        encode_categorical=not use_raw_categoricals,
    )

    trans_train = preprocessor.transform(train_df)
    trans_test = preprocessor.transform(test_df)

    pred_cols = [c for c in trans_train.columns if c != target_name]
    X_train = trans_train[pred_cols].to_numpy()
    y_train = trans_train[target_name].tolist()
    X_test = trans_test[pred_cols].to_numpy()
    y_test = trans_test[target_name].tolist()

    # Determine metric indices and build metric instance
    if use_raw_categoricals:
        num_indices = [i for i, c in enumerate(pred_cols) if c in preprocessor.numeric_features_]
        nom_indices = [i for i, c in enumerate(pred_cols) if c in preprocessor.categorical_features_]
        metric = get_distance_metric(
            distance_metric_name,
            numeric_indices=num_indices,
            nominal_indices=nom_indices,
        )
    else:
        metric = get_distance_metric(distance_metric_name)

    return X_train, y_train, X_test, y_test, metric


def run_single_experiment(
    config: ExperimentConfig,
    data_root: str | Path = "data",
) -> ExperimentResult:
    """Run a single fold experiment configuration with strict state isolation."""
    X_train, y_train, X_test, y_test, metric = prepare_fold_data(
        config.dataset,
        config.fold,
        config.distance_metric,
        data_root=data_root,
    )

    voting = get_voting_strategy(config.voting_strategy)
    retention = get_retention_policy(config.retention_policy)

    classifier = KNNClassifier(
        k=config.k,
        distance_metric=metric,
        voting_strategy=voting,
        retention_policy=retention,
    )

    classifier.fit(X_train, y_train)
    eval_metrics = classifier.evaluate(X_test, y_test)

    return ExperimentResult(
        dataset=config.dataset,
        fold=config.fold,
        distance_metric=config.distance_metric,
        k=config.k,
        voting_strategy=config.voting_strategy,
        retention_policy=config.retention_policy,
        accuracy=eval_metrics["accuracy"],
        correct=eval_metrics["correct"],
        incorrect=eval_metrics["incorrect"],
        total=eval_metrics["total"],
        initial_size=eval_metrics["initial_size"],
        final_size=eval_metrics["final_size"],
        retention_ratio=eval_metrics["retention_ratio"],
        mean_prediction_time=eval_metrics["mean_prediction_time"],
        total_prediction_time=eval_metrics["total_prediction_time"],
    )


def run_experiment_matrix(
    datasets: Sequence[str],
    folds: Sequence[int] = range(10),
    distances: Sequence[str] = (
        "euclidean",
        "manhattan",
        "canberra",
        "clark",
        "heom",
        "ivdm",
    ),
    k_values: Sequence[int] = (1, 3, 5, 7, 9),
    voting_strategies: Sequence[str] = ("majority", "inverse_distance"),
    retention_policies: Sequence[str] = ("nr", "ar", "dc", "og"),
    data_root: str | Path = "data",
    output_dir: str | Path = "results",
    progress_callback: Any = None,
) -> pd.DataFrame:
    """Run full or partial experiment grid across specified dimensions."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    total_runs = (
        len(datasets)
        * len(folds)
        * len(distances)
        * len(k_values)
        * len(voting_strategies)
        * len(retention_policies)
    )

    count = 0

    for ds in datasets:
        ds_results: list[dict[str, Any]] = []
        for fold in folds:
            for dist in distances:
                # Prepare fold data once per (ds, fold, dist) to optimize speed
                X_tr, y_tr, X_te, y_te, metric = prepare_fold_data(
                    ds, fold, dist, data_root=data_root
                )

                for k, vote, ret in itertools.product(
                    k_values, voting_strategies, retention_policies
                ):
                    voting_obj = get_voting_strategy(vote)
                    retention_obj = get_retention_policy(ret)

                    # Fresh classifier instance to guarantee state reset per run
                    clf = KNNClassifier(
                        k=k,
                        distance_metric=metric,
                        voting_strategy=voting_obj,
                        retention_policy=retention_obj,
                    )
                    clf.fit(X_tr, y_tr)
                    res = clf.evaluate(X_te, y_te)

                    row = {
                        "dataset": ds,
                        "fold": fold,
                        "distance_metric": dist,
                        "k": k,
                        "voting_strategy": vote,
                        "retention_policy": ret,
                        "accuracy": res["accuracy"],
                        "correct": res["correct"],
                        "incorrect": res["incorrect"],
                        "total": res["total"],
                        "initial_size": res["initial_size"],
                        "final_size": res["final_size"],
                        "retention_ratio": res["retention_ratio"],
                        "mean_prediction_time": res["mean_prediction_time"],
                        "total_prediction_time": res["total_prediction_time"],
                    }
                    results.append(row)
                    ds_results.append(row)
                    count += 1

                    if progress_callback and count % 100 == 0:
                        progress_callback(count, total_runs)

        # Save per-dataset CSV incrementally
        ds_df = pd.DataFrame(ds_results)
        canonical_ds_name = resolve_dataset_dir(ds, data_root)[1]
        ds_df.to_csv(out_path / f"results_{canonical_ds_name}.csv", index=False)

    df_all = pd.DataFrame(results)
    df_all.to_csv(out_path / "all_experiments_results.csv", index=False)
    return df_all

