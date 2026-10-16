import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from iml_project.data.arff_reader import load_dataset_fold, discover_fold_pairs, validate_fold_pairs
from iml_project.data.preprocessing import fit_and_transform
from iml_project.evaluation.runner import run_week1_experiments
from iml_project.models.distance_metrics import canberra_distance, euclidean_distance, heom_distance, manhattan_distance
from iml_project.models.knn import kNNAlgorithm


class ProjectScaffoldTests(unittest.TestCase):
    def setUp(self):
        self.data_root = ROOT / "data"

    def test_arff_reader_preserves_dataset_shape(self):
        dataset_map = discover_fold_pairs(self.data_root)
        validate_fold_pairs(dataset_map)
        self.assertEqual(sorted(dataset_map.keys()), ["grid", "heart-statlog"])
        for dataset_name, folds in dataset_map.items():
            self.assertEqual(len(folds), 10)
            first_train_df, first_test_df, feature_names, feature_types, target_name = load_dataset_fold(dataset_name, 0, self.data_root)
            self.assertGreater(len(first_train_df), 0)
            self.assertGreater(len(first_test_df), 0)
            self.assertIn(target_name, first_train_df.columns)
            self.assertEqual(len(feature_names), len(feature_types))

    def test_preprocessing_keeps_values_in_range(self):
        train_df = pd.DataFrame(
            {
                "x": [0.0, 1.0, np.nan, 2.0],
                "y": [5.0, 5.0, 5.0, 5.0],
                "class": ["A", "B", "A", "B"],
            }
        )
        test_df = pd.DataFrame(
            {
                "x": [np.nan, 1.5],
                "y": [5.0, 5.0],
                "class": ["A", "B"],
            }
        )
        train_scaled, test_scaled, _ = fit_and_transform(train_df, test_df, ["x", "y"], ["real", "real"], "class")
        self.assertTrue((train_scaled["x"].between(0.0, 1.0, inclusive="both")).all())
        self.assertTrue((test_scaled["x"].between(0.0, 1.0, inclusive="both")).all())
        self.assertTrue(np.allclose(train_scaled["y"].to_numpy(), 0.0))

    def test_distance_metrics_examples(self):
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        self.assertAlmostEqual(euclidean_distance(a, b), 5.0)
        self.assertAlmostEqual(manhattan_distance(a, b), 7.0)
        self.assertAlmostEqual(canberra_distance(np.array([0.0, 0.0]), np.array([0.0, 0.0])), 0.0)
        self.assertAlmostEqual(heom_distance([0.0, "A"], [1.0, "B"]), 2.0)

    def test_knn_predicts_deterministically(self):
        X_train = np.array([[0.0], [2.0], [4.0]], dtype=float)
        y_train = np.array(["A", "A", "B"])
        X_test = np.array([[0.0], [2.0], [3.0]], dtype=float)
        predictions, _ = kNNAlgorithm(X_train, y_train, X_test, k=3, metric="euclidean", voting_scheme="majority_class")
        self.assertEqual(list(predictions), ["A", "A", "A"])

    def test_week1_outputs_are_generated(self):
        results = run_week1_experiments(self.data_root, ROOT / "results" / "week1")
        self.assertIn("svm_fold", results)
        self.assertIn("random_forest_fold", results)
        self.assertTrue((ROOT / "results" / "week1" / "svm_fold_results.csv").exists())
        self.assertTrue((ROOT / "results" / "week1" / "random_forest_fold_results.csv").exists())
        self.assertTrue((ROOT / "results" / "week1" / "svm_summary.csv").exists())
        self.assertTrue((ROOT / "results" / "week1" / "random_forest_summary.csv").exists())
        self.assertGreater(len(results["svm_fold"]), 0)


if __name__ == "__main__":
    unittest.main()
