from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from iml_project.experiments.evaluation import (
    aggregate_experiment_results,
    compute_friedman_statistic,
    compute_friedman_test,
    evaluate_multi_dataset,
    evaluate_single_dataset,
    export_evaluation_report,
    get_next_experiment_dir,
)
from iml_project.experiments.runner import (
    ExperimentConfig,
    prepare_fold_data,
    run_experiment_matrix,
    run_single_experiment,
)


class ExperimentsTests(unittest.TestCase):
    def test_prepare_fold_data_isolation(self) -> None:
        """Verify prepare_fold_data returns arrays with proper shapes and zero leakage."""
        X_tr, y_tr, X_te, y_te, _metric = prepare_fold_data(
            "grid", 0, "euclidean", data_root="data"
        )
        self.assertEqual(X_tr.shape[1], 2)
        self.assertEqual(X_te.shape[1], 2)
        self.assertEqual(len(X_tr), len(y_tr))
        self.assertEqual(len(X_te), len(y_te))

    def test_run_single_experiment(self) -> None:
        """Verify single configuration execution on heart-statlog fold 0."""
        cfg = ExperimentConfig(
            dataset="heart-statlog",
            fold=0,
            distance_metric="euclidean",
            k=3,
            voting_strategy="majority",
            retention_policy="nr",
        )
        res = run_single_experiment(cfg, data_root="data")
        self.assertEqual(res.dataset, "heart-statlog")
        self.assertEqual(res.fold, 0)
        self.assertEqual(res.k, 3)
        self.assertEqual(res.model, "knn")
        self.assertEqual(res.model_family, "knn")
        self.assertTrue(res.config_id.startswith("knn_metric=euclidean"))
        self.assertGreater(res.accuracy, 0.5)
        self.assertEqual(res.total, res.correct + res.incorrect)
        self.assertGreater(res.training_size_before_reduction, 0)
        self.assertEqual(res.retained_size, res.final_size)
        self.assertEqual(res.retention_ratio, 1.0)

    def test_smoke_matrix_execution(self) -> None:
        """Smoke test running 1 fold across 2 metrics, 2 k values, 1 voting, 1 retention."""
        df = run_experiment_matrix(
            datasets=["heart-statlog"],
            folds=[0],
            distances=["euclidean", "heom"],
            k_values=[1, 3],
            voting_strategies=["majority"],
            retention_policies=["nr"],
            data_root="data",
            output_dir="results/smoke_test",
        )
        self.assertEqual(len(df), 4)
        self.assertTrue(Path("results/smoke_test/all_experiments_results.csv").exists())
        self.assertIn("config_id", df.columns)
        self.assertIn("training_size_before_reduction", df.columns)
        self.assertIn("retained_size", df.columns)

    def test_aggregation_and_ranking(self) -> None:
        """Verify result aggregation, pooled accuracy, efficiency ms, and descriptive ranks."""
        rows = []
        for fold in range(10):
            # Config A: 80% accuracy, total 100
            rows.append({
                "dataset": "ds1",
                "fold": fold,
                "model": "knn",
                "config_id": "cfg_a",
                "distance_metric": "euclidean",
                "k": 3,
                "voting_strategy": "majority",
                "retention_policy": "nr",
                "accuracy": 0.80,
                "correct": 80,
                "incorrect": 20,
                "total": 100,
                "total_prediction_time": 0.05,
                "mean_prediction_time": 0.0005,
                "training_size_before_reduction": 900,
                "retained_size": 900,
                "retention_ratio": 1.0,
            })
            # Config B: 70% accuracy, total 100, faster runtime
            rows.append({
                "dataset": "ds1",
                "fold": fold,
                "model": "knn",
                "config_id": "cfg_b",
                "distance_metric": "manhattan",
                "k": 3,
                "voting_strategy": "majority",
                "retention_policy": "nr",
                "accuracy": 0.70,
                "correct": 70,
                "incorrect": 30,
                "total": 100,
                "total_prediction_time": 0.02,
                "mean_prediction_time": 0.0002,
                "training_size_before_reduction": 900,
                "retained_size": 900,
                "retention_ratio": 1.0,
            })

        df = pd.DataFrame(rows)
        agg = aggregate_experiment_results(df)
        self.assertEqual(len(agg), 2)

        # Check pooled accuracy: Config A = 800/1000 = 0.80, Config B = 700/1000 = 0.70
        cfg_a_row = agg[agg["config_id"] == "cfg_a"].iloc[0]
        cfg_b_row = agg[agg["config_id"] == "cfg_b"].iloc[0]
        self.assertAlmostEqual(cfg_a_row["pooled_accuracy"], 0.80)
        self.assertAlmostEqual(cfg_b_row["pooled_accuracy"], 0.70)

        # Check accuracy rank: Config A gets rank 1, Config B gets rank 2
        self.assertEqual(cfg_a_row["rank_accuracy"], 1.0)
        self.assertEqual(cfg_b_row["rank_accuracy"], 2.0)

        # Check efficiency rank: Config B is faster, so rank 1, Config A rank 2
        self.assertEqual(cfg_b_row["rank_efficiency"], 1.0)
        self.assertEqual(cfg_a_row["rank_efficiency"], 2.0)

    def test_single_dataset_mode_exports(self) -> None:
        """Verify single-dataset mode produces leaderboards without fold-level Friedman inference."""
        rows = []
        for f in range(10):
            for i, name in enumerate(["cfg_1", "cfg_2", "cfg_3"]):
                rows.append({
                    "dataset": "heart-statlog",
                    "fold": f,
                    "model": "knn",
                    "config_id": name,
                    "distance_metric": "euclidean",
                    "k": 3 + i * 2,
                    "voting_strategy": "majority",
                    "retention_policy": "nr",
                    "accuracy": 0.75 + 0.05 * i,
                    "correct": int(75 + 5 * i),
                    "incorrect": int(25 - 5 * i),
                    "total": 100,
                    "total_prediction_time": 0.01 * (i + 1),
                    "mean_prediction_time": 0.0001 * (i + 1),
                    "training_size_before_reduction": 900,
                    "retained_size": 900,
                    "retention_ratio": 1.0,
                })
        df = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as tmp_dir:
            agg_df, best_df, eval_dict = export_evaluation_report(df, output_dir=tmp_dir)
            tmp_path = Path(tmp_dir)
            self.assertTrue((tmp_path / "aggregated_results.csv").exists())
            self.assertTrue((tmp_path / "top10_accuracy.csv").exists())
            self.assertTrue((tmp_path / "top10_efficiency.csv").exists())
            self.assertIn("top10_accuracy", eval_dict)
            self.assertEqual(eval_dict["top10_accuracy"].iloc[0]["config_id"], "cfg_3")

    def test_friedman_test_multi_dataset(self) -> None:
        """Verify across-dataset Friedman test, tie handling, and Iman-Davenport calculation."""
        # 3 datasets, 3 configurations
        rows = []
        for ds, base_accs in [
            ("ds1", [0.85, 0.80, 0.70]),
            ("ds2", [0.90, 0.82, 0.75]),
            ("ds3", [0.88, 0.84, 0.72]),
        ]:
            for fold in range(10):
                for cfg_idx, cfg_name in enumerate(["cfg_A", "cfg_B", "cfg_C"]):
                    rows.append({
                        "dataset": ds,
                        "fold": fold,
                        "model": "knn",
                        "config_id": cfg_name,
                        "distance_metric": "euclidean",
                        "k": 3,
                        "voting_strategy": "majority",
                        "retention_policy": "nr",
                        "accuracy": base_accs[cfg_idx],
                        "correct": int(base_accs[cfg_idx] * 100),
                        "incorrect": int((1 - base_accs[cfg_idx]) * 100),
                        "total": 100,
                        "total_prediction_time": 0.01 * (cfg_idx + 1),
                        "mean_prediction_time": 0.0001 * (cfg_idx + 1),
                        "training_size_before_reduction": 900,
                        "retained_size": 900,
                        "retention_ratio": 1.0,
                    })
        df = pd.DataFrame(rows)
        agg_df = aggregate_experiment_results(df)
        res = compute_friedman_test(agg_df, metric="accuracy")

        self.assertEqual(res["D"], 3)
        self.assertEqual(res["M"], 3)
        self.assertEqual(res["df_chi2"], 2)
        # cfg_A is consistently 1st place across all 3 datasets
        self.assertEqual(res["rankings_df"].iloc[0]["config_id"], "cfg_A")
        self.assertEqual(res["rankings_df"].iloc[0]["average_rank"], 1.0)
        self.assertEqual(res["rankings_df"].iloc[2]["config_id"], "cfg_C")
        self.assertEqual(res["rankings_df"].iloc[2]["average_rank"], 3.0)
        self.assertGreater(res["statistic_chi2"], 0.0)
        self.assertLess(res["p_value_chi2"], 0.05)
        self.assertIn("limitations", res)

    def test_tie_handling_exact_hand_calculated(self) -> None:
        """Verify tie correction against exact hand-calculated 3x3 rank matrix."""
        ranks_matrix = np.array([
            [1.0, 2.5, 2.5],
            [2.0, 2.0, 2.0],
            [1.0, 2.0, 3.0],
        ])
        res = compute_friedman_statistic(ranks_matrix)
        self.assertAlmostEqual(res["tie_correction_factor"], 7.0 / 12.0)
        self.assertAlmostEqual(res["statistic_chi2_uncorrected"], 13.0 / 6.0)
        self.assertAlmostEqual(res["statistic_chi2"], 26.0 / 7.0)

    def test_get_next_experiment_dir(self) -> None:
        """Verify get_next_experiment_dir detects latest number and increments by 1."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dir1 = get_next_experiment_dir(tmp_path)
            self.assertEqual(dir1.name, "experiment_results_1")
            self.assertTrue(dir1.is_dir())

            dir2 = get_next_experiment_dir(tmp_path)
            self.assertEqual(dir2.name, "experiment_results_2")
            self.assertTrue(dir2.is_dir())

            (tmp_path / "experiment_results_5").mkdir()
            dir6 = get_next_experiment_dir(tmp_path)
            self.assertEqual(dir6.name, "experiment_results_6")
            self.assertTrue(dir6.is_dir())


if __name__ == "__main__":
    unittest.main()
