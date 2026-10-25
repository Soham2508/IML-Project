from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from iml_project.core.random_forest import rfAlgorithm
from iml_project.core.svm import svmAlgorithm
from iml_project.experiments.evaluation import (
    aggregate_eager_results,
    compare_with_knn_baseline,
    find_best_eager_configurations,
    generate_eager_statistical_comparisons,
)
from iml_project.experiments.runner import (
    prepare_eager_fold_data,
    run_rf_experiments,
    run_svm_experiments,
)


class TestEagerAlgorithms(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(42)
        self.X_train = np.array([
            [1.0, 2.0],
            [1.5, 1.8],
            [5.0, 8.0],
            [6.0, 9.0],
        ])
        self.y_train = [0, 0, 1, 1]
        self.X_test = np.array([
            [1.2, 1.9],
            [5.5, 8.5],
        ])
        self.y_test = [0, 1]

    def test_svm_algorithm_linear_and_rbf(self) -> None:
        for kernel in ["linear", "rbf"]:
            res = svmAlgorithm(
                self.X_train,
                self.y_train,
                self.X_test,
                self.y_test,
                kernel=kernel,
                C=1.0,
                random_state=42,
            )
            self.assertEqual(res["total"], 2)
            self.assertEqual(res["accuracy"], 1.0)
            self.assertEqual(res["balanced_accuracy"], 1.0)
            self.assertEqual(res["precision_macro"], 1.0)
            self.assertEqual(res["recall_macro"], 1.0)
            self.assertEqual(res["f1_macro"], 1.0)
            self.assertGreaterEqual(res["fit_time"], 0.0)
            self.assertGreaterEqual(res["mean_prediction_time"], 0.0)
            self.assertEqual(res["initial_size"], 4)
            self.assertEqual(res["final_size"], 4)
            self.assertEqual(res["retention_ratio"], 1.0)

    def test_rf_algorithm_gini_and_entropy(self) -> None:
        for crit in ["gini", "entropy"]:
            res = rfAlgorithm(
                self.X_train,
                self.y_train,
                self.X_test,
                self.y_test,
                n_estimators=10,
                criterion=crit,
                random_state=42,
                n_jobs=1,
            )
            self.assertEqual(res["total"], 2)
            self.assertEqual(res["accuracy"], 1.0)
            self.assertEqual(res["balanced_accuracy"], 1.0)
            self.assertEqual(res["precision_macro"], 1.0)
            self.assertEqual(res["recall_macro"], 1.0)
            self.assertEqual(res["f1_macro"], 1.0)
            self.assertGreaterEqual(res["fit_time"], 0.0)
            self.assertGreaterEqual(res["mean_prediction_time"], 0.0)
            self.assertEqual(res["initial_size"], 4)
            self.assertEqual(res["final_size"], 4)
            self.assertEqual(res["retention_ratio"], 1.0)


class TestEagerPipelineAndRunner(unittest.TestCase):
    def test_prepare_eager_fold_data(self) -> None:
        X_train, y_train, X_test, y_test = prepare_eager_fold_data(
            "heart-statlog", fold=0, data_root="data"
        )
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))
        self.assertEqual(X_train.shape[1], X_test.shape[1])
        self.assertGreater(len(X_train), 0)
        self.assertGreater(len(X_test), 0)
        # Check numeric type
        self.assertEqual(X_train.dtype, np.float64)
        self.assertEqual(X_test.dtype, np.float64)

    def test_run_eager_experiments_single_fold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Test SVM single fold
            svm_df = run_svm_experiments(
                datasets=["heart-statlog"],
                folds=[0],
                kernels=["linear"],
                data_root="data",
                output_dir=tmp_dir,
            )
            self.assertEqual(len(svm_df), 1)
            self.assertIn("accuracy", svm_df.columns)
            self.assertIn("balanced_accuracy", svm_df.columns)
            self.assertIn("f1_macro", svm_df.columns)
            self.assertIn("fit_time", svm_df.columns)

            # Test RF single fold
            rf_df = run_rf_experiments(
                datasets=["heart-statlog"],
                folds=[0],
                n_estimators_list=[10],
                criteria_list=["gini"],
                n_jobs=1,
                data_root="data",
                output_dir=tmp_dir,
            )
            self.assertEqual(len(rf_df), 1)
            self.assertIn("accuracy", rf_df.columns)
            self.assertIn("balanced_accuracy", rf_df.columns)
            self.assertIn("f1_macro", rf_df.columns)
            self.assertIn("fit_time", rf_df.columns)


class TestEagerEvaluation(unittest.TestCase):
    def setUp(self) -> None:
        # Synthetic 10-fold SVM results
        rows_svm = []
        for f in range(10):
            rows_svm.append({
                "dataset": "heart-statlog",
                "fold": f,
                "model": "svm",
                "kernel": "linear",
                "C": 1.0,
                "gamma": "scale",
                "n_estimators": None,
                "criterion": None,
                "max_depth": None,
                "accuracy": 0.8 + 0.01 * f,
                "balanced_accuracy": 0.8 + 0.01 * f,
                "precision_macro": 0.79,
                "recall_macro": 0.81,
                "f1_macro": 0.80,
                "fit_time": 0.005,
                "mean_prediction_time": 0.001,
                "total_prediction_time": 0.01,
                "retention_ratio": 1.0,
            })
            rows_svm.append({
                "dataset": "heart-statlog",
                "fold": f,
                "model": "svm",
                "kernel": "rbf",
                "C": 1.0,
                "gamma": "scale",
                "n_estimators": None,
                "criterion": None,
                "max_depth": None,
                "accuracy": 0.75 + 0.01 * f,
                "balanced_accuracy": 0.75 + 0.01 * f,
                "precision_macro": 0.74,
                "recall_macro": 0.76,
                "f1_macro": 0.75,
                "fit_time": 0.006,
                "mean_prediction_time": 0.001,
                "total_prediction_time": 0.01,
                "retention_ratio": 1.0,
            })
        self.svm_df = pd.DataFrame(rows_svm)

    def test_aggregate_and_best_eager_results(self) -> None:
        agg = aggregate_eager_results(self.svm_df, "svm")
        self.assertEqual(len(agg), 2)
        self.assertIn("pooled_accuracy", agg.columns)
        self.assertIn("mean_accuracy", agg.columns)
        self.assertIn("mean_balanced_accuracy", agg.columns)
        self.assertIn("mean_f1_macro", agg.columns)
        self.assertIn("mean_fit_time_s", agg.columns)
        self.assertIn("rank_accuracy", agg.columns)

        best = find_best_eager_configurations(agg, top_n=1)
        self.assertEqual(len(best), 1)
        # Higher accuracy kernel gets rank 1
        self.assertEqual(best.iloc[0]["kernel"], "linear")

    def test_generate_eager_statistical_comparisons(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            stat_df = generate_eager_statistical_comparisons(
                self.svm_df, "svm", tmp_dir
            )
            self.assertFalse(stat_df.empty)
            # In single-dataset mode, summary report is produced
            self.assertIn("mode", stat_df.columns)
            self.assertEqual(stat_df.iloc[0]["mode"], "single_dataset")

    def test_compare_with_knn_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            # Create synthetic KNN baseline
            knn_rows = []
            for f in range(10):
                knn_rows.append({
                    "dataset": "heart-statlog",
                    "fold": f,
                    "distance_metric": "euclidean",
                    "k": 3,
                    "voting_strategy": "majority",
                    "retention_policy": "nr",
                    "accuracy": 0.7 + 0.01 * f,
                    "correct": int((0.7 + 0.01 * f) * 100),
                    "incorrect": int((0.3 - 0.01 * f) * 100),
                    "total": 100,
                    "balanced_accuracy": 0.7 + 0.01 * f,
                    "precision_macro": 0.7,
                    "recall_macro": 0.7,
                    "f1_macro": 0.7,
                    "mean_prediction_time": 0.002,
                    "total_prediction_time": 0.2,
                    "training_size_before_reduction": 900,
                    "retained_size": 900,
                    "retention_ratio": 1.0,
                })
            pd.DataFrame(knn_rows).to_csv(
                tmp_path / "all_experiments_results.csv", index=False
            )

            out_dir = tmp_path / "out"
            comp_df = compare_with_knn_baseline(
                self.svm_df, tmp_path, "svm", out_dir
            )
            self.assertIsNotNone(comp_df)
            self.assertEqual(len(comp_df), 1)
            self.assertGreater(comp_df.iloc[0]["accuracy_difference"], 0)
            self.assertEqual(comp_df.iloc[0]["winner_accuracy"], "svm")
            self.assertIn("speedup_vs_knn", comp_df.columns)


class TestCLIValidation(unittest.TestCase):
    def test_cli_rejects_knn_args_for_svm(self) -> None:
        cmd = [
            sys.executable,
            "main.py",
            "--model",
            "svm",
            "--k",
            "5",
        ]
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("--k cannot be used when --model svm is selected", proc.stderr)

    def test_cli_rejects_svm_args_for_knn(self) -> None:
        cmd = [
            sys.executable,
            "main.py",
            "--model",
            "knn",
            "--svm-kernels",
            "linear",
        ]
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("--svm-kernels cannot be used when --model knn is selected", proc.stderr)

    def test_cli_rejects_rf_args_for_svm(self) -> None:
        cmd = [
            sys.executable,
            "main.py",
            "--model",
            "svm",
            "--rf-n-estimators",
            "100",
        ]
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("--rf-n-estimators cannot be used when --model svm is selected", proc.stderr)


if __name__ == "__main__":
    unittest.main()
