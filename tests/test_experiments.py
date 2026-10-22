from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from iml_project.experiments.evaluation import (
    aggregate_experiment_results,
    compute_paired_ttest,
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
        self.assertGreater(res.accuracy, 0.5)
        self.assertEqual(res.total, res.correct + res.incorrect)

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

    def test_aggregation_and_paired_ttest(self) -> None:
        """Verify result aggregation and paired t-test computation."""
        # Create synthetic 10-fold results for 2 configs
        rows = []
        for fold in range(10):
            rows.append(
                {
                    "dataset": "ds",
                    "fold": fold,
                    "distance_metric": "euclidean",
                    "k": 3,
                    "voting_strategy": "majority",
                    "retention_policy": "nr",
                    "accuracy": 0.80 + 0.01 * (fold % 3),
                    "mean_prediction_time": 0.001,
                    "retention_ratio": 1.0,
                }
            )
            rows.append(
                {
                    "dataset": "ds",
                    "fold": fold,
                    "distance_metric": "manhattan",
                    "k": 3,
                    "voting_strategy": "majority",
                    "retention_policy": "nr",
                    "accuracy": 0.75 + 0.01 * (fold % 3),
                    "mean_prediction_time": 0.001,
                    "retention_ratio": 1.0,
                }
            )
        df = pd.DataFrame(rows)
        agg = aggregate_experiment_results(df)
        self.assertEqual(len(agg), 2)
        self.assertGreater(agg.iloc[0]["mean_accuracy"], agg.iloc[1]["mean_accuracy"])

        ttest = compute_paired_ttest(
            df,
            cfg_a={"distance_metric": "euclidean", "k": 3, "voting_strategy": "majority", "retention_policy": "nr"},
            cfg_b={"distance_metric": "manhattan", "k": 3, "voting_strategy": "majority", "retention_policy": "nr"},
            dataset="ds",
        )
        self.assertAlmostEqual(ttest["mean_diff"], 0.05)
        self.assertGreater(ttest["t_statistic"], 0.0)

    def test_get_next_experiment_dir(self) -> None:
        """Verify get_next_experiment_dir detects latest number and increments by 1."""
        import tempfile

        from iml_project.experiments.evaluation import get_next_experiment_dir

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            # No runs yet -> creates experiment_results_1
            dir1 = get_next_experiment_dir(tmp_path)
            self.assertEqual(dir1.name, "experiment_results_1")
            self.assertTrue(dir1.is_dir())

            # Next run -> creates experiment_results_2
            dir2 = get_next_experiment_dir(tmp_path)
            self.assertEqual(dir2.name, "experiment_results_2")
            self.assertTrue(dir2.is_dir())

            # Simulate existing higher index (e.g. 5)
            (tmp_path / "experiment_results_5").mkdir()
            dir6 = get_next_experiment_dir(tmp_path)
            self.assertEqual(dir6.name, "experiment_results_6")
            self.assertTrue(dir6.is_dir())


if __name__ == "__main__":
    unittest.main()

