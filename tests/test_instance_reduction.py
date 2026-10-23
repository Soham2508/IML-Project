from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from iml_project.core.instance_reduction import FCNN3, RNG2, RNGE2, InstanceReducer
from iml_project.core.knn import KNNClassifier
from iml_project.utils.distance_metrics import EuclideanDistance, ManhattanDistance


class TestFCNN3(unittest.TestCase):
    """Unit tests specific to the FCNN3 instance reduction algorithm."""

    def test_subclass_of_instance_reducer(self) -> None:
        """Verify FCNN3 inherits from InstanceReducer."""
        fcnn = FCNN3()
        self.assertIsInstance(fcnn, InstanceReducer)
        self.assertIsInstance(fcnn.distance_metric, EuclideanDistance)

    def test_empty_and_single_sample_inputs(self) -> None:
        """Verify behavior on boundary cases: empty and single-sample datasets."""
        fcnn = FCNN3()

        # Empty dataset
        X_empty = np.empty((0, 2))
        y_empty = np.array([])
        X_red, y_red = fcnn.fit_resample(X_empty, y_empty)
        self.assertEqual(len(X_red), 0)
        self.assertEqual(len(y_red), 0)
        self.assertEqual(fcnn.selected_indices_, [])

        # Single sample dataset
        X_single = np.array([[1.0, 2.0]])
        y_single = np.array(["A"])
        X_red, y_red = fcnn.fit_resample(X_single, y_single)
        self.assertEqual(len(X_red), 1)
        self.assertEqual(len(y_red), 1)
        self.assertEqual(fcnn.selected_indices_, [0])

    def test_single_class_reduction(self) -> None:
        """In a single-class dataset, FCNN3 should reduce to the single centroid instance."""
        np.random.seed(42)
        X = np.random.randn(25, 3)
        y = ["class_0"] * 25

        fcnn = FCNN3()
        X_red, y_red = fcnn.fit_resample(X, y)
        self.assertEqual(len(X_red), 1)
        self.assertEqual(len(y_red), 1)
        self.assertEqual(y_red[0], "class_0")
        self.assertEqual(len(fcnn.selected_indices_), 1)

    def test_initial_point_is_closest_to_mean(self) -> None:
        """Verify that p0 is the actual training instance closest to the mean of most populated class."""
        X = np.array([
            [10.0, 10.0],  # idx 0: Class A, dist 14.14
            [0.1, 0.1],    # idx 1: Class A, dist 0.1414 (closest to mean)
            [-0.2, 0.2],   # idx 2: Class A, dist 0.2828
            [5.0, 5.0],    # idx 3: Class B
            [6.0, 6.0],    # idx 4: Class B
        ])
        y = ["A", "A", "A", "B", "B"]

        fcnn = FCNN3(EuclideanDistance())
        fcnn.fit_resample(X, y)
        # p0 should be index 1 (closest to mean of class A)
        self.assertEqual(fcnn.selected_indices_[0], 1)

    def test_separable_two_class_consistency(self) -> None:
        """Verify reduction preserves 1-NN classification consistency on separable data."""
        np.random.seed(42)
        X1 = np.random.randn(20, 2) + np.array([0, 0])
        y1 = ["class_A"] * 20
        X2 = np.random.randn(30, 2) + np.array([5, 5])
        y2 = ["class_B"] * 30

        X = np.vstack([X1, X2])
        y = y1 + y2

        fcnn = FCNN3(EuclideanDistance())
        X_red, y_red = fcnn.fit_resample(X, y)

        self.assertLess(len(X_red), len(X))
        self.assertGreater(len(X_red), 0)

        # Verify 1-NN training consistency
        knn = KNNClassifier(k=1, distance_metric=EuclideanDistance())
        knn.fit(X_red, y_red)
        for i in range(len(X)):
            pred, _, _ = knn.predict_one(X[i])
            self.assertEqual(pred, y[i])

    def test_multi_class_and_pandas_support(self) -> None:
        """Verify DataFrame and Series preservation and multi-class consistency."""
        np.random.seed(123)
        X1 = np.random.randn(15, 3) + np.array([0, 0, 0])
        y1 = ["C1"] * 15
        X2 = np.random.randn(20, 3) + np.array([6, 6, 0])
        y2 = ["C2"] * 20
        X3 = np.random.randn(25, 3) + np.array([0, 6, 6])
        y3 = ["C3"] * 25

        X = np.vstack([X1, X2, X3])
        y = y1 + y2 + y3

        df_X = pd.DataFrame(X, columns=["feat_1", "feat_2", "feat_3"])
        s_y = pd.Series(y, name="label")

        fcnn = FCNN3(ManhattanDistance())
        df_red, s_red = fcnn.fit_resample(df_X, s_y)

        self.assertIsInstance(df_red, pd.DataFrame)
        self.assertIsInstance(s_red, pd.Series)
        self.assertEqual(len(df_red), len(s_red))
        self.assertLess(len(df_red), len(df_X))

        # Verify 1-NN training consistency
        knn = KNNClassifier(k=1, distance_metric=ManhattanDistance())
        knn.fit(df_red.to_numpy(), s_red.to_numpy())
        for i in range(len(df_X)):
            pred, _, _ = knn.predict_one(df_X.iloc[i].to_numpy())
            self.assertEqual(pred, s_y.iloc[i])


class TestRNG2(unittest.TestCase):
    """Unit tests specific to the RNG2 / RNGE-2 instance editing algorithm."""

    def test_subclass_and_alias(self) -> None:
        """Verify RNG2 inherits from InstanceReducer and RNGE2 is an alias."""
        rng = RNG2()
        self.assertIsInstance(rng, InstanceReducer)
        self.assertIs(RNGE2, RNG2)
        self.assertIsInstance(rng.distance_metric, EuclideanDistance)

    def test_empty_and_single_sample_inputs(self) -> None:
        """Verify behavior on empty and single-sample datasets."""
        rng = RNG2()

        # Empty dataset
        X_empty = np.empty((0, 2))
        y_empty = np.array([])
        X_red, y_red = rng.fit_resample(X_empty, y_empty)
        self.assertEqual(len(X_red), 0)
        self.assertEqual(len(y_red), 0)

        # Single sample dataset
        X_single = np.array([[1.0, 2.0]])
        y_single = np.array(["A"])
        X_red, y_red = rng.fit_resample(X_single, y_single)
        self.assertEqual(len(X_red), 1)
        self.assertEqual(len(y_red), 1)
        self.assertEqual(rng.selected_indices_, [0])
        self.assertEqual(rng.removed_indices_, [])

    def test_clean_separable_data_no_removal(self) -> None:
        """On clean, well-separated clusters, no instance should be edited out."""
        np.random.seed(42)
        X1 = np.random.randn(20, 2) + np.array([0, 0])
        y1 = ["A"] * 20
        X2 = np.random.randn(20, 2) + np.array([20, 20])
        y2 = ["B"] * 20

        X = np.vstack([X1, X2])
        y = y1 + y2

        rng = RNG2()
        X_red, y_red = rng.fit_resample(X, y)
        self.assertEqual(len(X_red), len(X))
        self.assertEqual(len(rng.removed_indices_), 0)

    def test_noisy_outlier_removal(self) -> None:
        """Verify that an opposing-class outlier surrounded by a homogeneous cluster is removed."""
        # 10 points of class A clustered near origin
        np.random.seed(42)
        X_A = np.array([
            [0.0, 1.0], [0.0, -1.0], [1.0, 0.0], [-1.0, 0.0],
            [0.7, 0.7], [-0.7, 0.7], [0.7, -0.7], [-0.7, -0.7],
        ])
        y_A = ["A"] * len(X_A)

        # 1 noise point of class B right at the center of class A points
        X_noise = np.array([[0.0, 0.0]])
        y_noise = ["B"]

        # Main cluster of class B further away
        X_B = np.array([
            [10.0, 10.0], [10.5, 10.0], [10.0, 10.5], [10.5, 10.5],
        ])
        y_B = ["B"] * len(X_B)

        X = np.vstack([X_A, X_noise, X_B])
        y = y_A + y_noise + y_B
        noise_idx = len(X_A)  # index 8

        rng = RNG2()
        X_red, y_red = rng.fit_resample(X, y)

        self.assertIn(noise_idx, rng.removed_indices_)
        self.assertNotIn(noise_idx, rng.selected_indices_)
        self.assertEqual(len(X_red), len(X) - 1)

    def test_pandas_support_and_metrics(self) -> None:
        """Verify DataFrame and Series type preservation and metric support."""
        X = np.array([
            [0.0, 0.0], [0.1, 0.0], [0.0, 0.1],
            [5.0, 5.0], [5.1, 5.0], [5.0, 5.1],
        ])
        y = ["A", "A", "A", "B", "B", "B"]

        df_X = pd.DataFrame(X, columns=["x1", "x2"])
        s_y = pd.Series(y, name="target")

        rng = RNG2("manhattan")
        df_red, s_red = rng.fit_resample(df_X, s_y)

        self.assertIsInstance(df_red, pd.DataFrame)
        self.assertIsInstance(s_red, pd.Series)
        self.assertEqual(len(df_red), len(df_X))


if __name__ == "__main__":
    unittest.main()
