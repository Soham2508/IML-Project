from __future__ import annotations

import unittest

import numpy as np
from sklearn.datasets import make_blobs

from iml_project.core.clustering.dbscan import (
    METRIC_ALGORITHMS,
    DBSCANClusterer,
    compatible_algorithms,
    k_distance_curve,
)
from work3.contracts import ClusterResult


def _blobs_with_outliers() -> tuple[np.ndarray, int]:
    """Three tight, well-separated blobs plus 3 far-away isolated outliers."""
    X, _ = make_blobs(
        n_samples=150,
        centers=[[0.0, 0.0], [10.0, 10.0], [-10.0, 10.0]],
        cluster_std=0.4,
        random_state=0,
    )
    outliers = np.array([[50.0, 50.0], [-50.0, -50.0], [50.0, -50.0]])
    return np.vstack([X, outliers]), outliers.shape[0]


class DBSCANClustererTests(unittest.TestCase):
    def setUp(self) -> None:
        self.X, self.n_outliers = _blobs_with_outliers()

    def _check_result_invariants(self, result: ClusterResult, n_samples: int) -> None:
        self.assertIsInstance(result, ClusterResult)
        self.assertEqual(result.algorithm, "DBSCAN")
        self.assertEqual(result.labels.dtype, np.int64)
        self.assertEqual(result.labels.shape, (n_samples,))
        self.assertIsNone(result.memberships)
        self.assertTrue(np.all(result.labels >= -1))
        non_noise = set(np.unique(result.labels).tolist()) - {-1}
        self.assertEqual(result.n_clusters, len(non_noise))
        self.assertEqual(result.n_noise, int(np.sum(result.labels == -1)))
        self.assertIsInstance(result.execution_time, float)
        self.assertGreaterEqual(result.execution_time, 0.0)

    def test_recovers_blobs_and_marks_outliers_as_noise(self) -> None:
        clusterer = DBSCANClusterer(eps=1.0, min_samples=5)
        labels = clusterer.fit_predict(self.X)
        result = clusterer.get_result()

        self._check_result_invariants(result, self.X.shape[0])
        np.testing.assert_array_equal(labels, result.labels)
        self.assertEqual(result.n_clusters, 3)
        self.assertEqual(result.n_noise, self.n_outliers)
        np.testing.assert_array_equal(labels[-self.n_outliers :], [-1, -1, -1])

    def test_fit_returns_self(self) -> None:
        clusterer = DBSCANClusterer()
        self.assertIs(clusterer.fit(self.X), clusterer)

    def test_multiple_eps_and_min_samples(self) -> None:
        for eps in (0.3, 0.5, 1.0, 2.0, 5.0):
            for min_samples in (1, 3, 5, 10):
                with self.subTest(eps=eps, min_samples=min_samples):
                    clusterer = DBSCANClusterer(eps=eps, min_samples=min_samples)
                    clusterer.fit(self.X)
                    self._check_result_invariants(
                        clusterer.get_result(), self.X.shape[0]
                    )

    def test_min_samples_one_produces_no_noise(self) -> None:
        clusterer = DBSCANClusterer(eps=0.5, min_samples=1).fit(self.X)
        self.assertEqual(clusterer.get_result().n_noise, 0)

    def test_tiny_eps_gives_all_noise(self) -> None:
        result = DBSCANClusterer(eps=1e-6, min_samples=2).fit(self.X).get_result()
        self.assertEqual(result.n_clusters, 0)
        self.assertEqual(result.n_noise, self.X.shape[0])

    def test_several_metrics(self) -> None:
        eps_by_metric = {
            "euclidean": 1.0,
            "manhattan": 1.5,
            "chebyshev": 1.0,
            "cosine": 0.01,
        }
        for metric, eps in eps_by_metric.items():
            with self.subTest(metric=metric):
                result = (
                    DBSCANClusterer(eps=eps, min_samples=5, metric=metric)
                    .fit(self.X)
                    .get_result()
                )
                self._check_result_invariants(result, self.X.shape[0])
                self.assertEqual(result.metadata["parameters"]["metric"], metric)
                self.assertGreaterEqual(result.n_clusters, 1)

    def test_search_algorithms_give_identical_labels(self) -> None:
        for metric in ("euclidean", "manhattan", "canberra", "cosine"):
            labels_per_algorithm = {
                algorithm: DBSCANClusterer(
                    eps=0.5, min_samples=5, metric=metric, algorithm=algorithm
                ).fit_predict(self.X)
                for algorithm in compatible_algorithms(metric)
            }
            reference = labels_per_algorithm["brute"]
            for algorithm, labels in labels_per_algorithm.items():
                with self.subTest(metric=metric, algorithm=algorithm):
                    np.testing.assert_array_equal(labels, reference)

    def test_incompatible_metric_algorithm_rejected(self) -> None:
        for metric, algorithm in (
            ("cosine", "kd_tree"),
            ("cosine", "ball_tree"),
            ("canberra", "kd_tree"),
        ):
            with (
                self.subTest(metric=metric, algorithm=algorithm),
                self.assertRaises(ValueError),
            ):
                DBSCANClusterer(metric=metric, algorithm=algorithm)

    def test_invalid_parameters(self) -> None:
        cases = [
            ({"eps": 0.0}, ValueError),
            ({"eps": -1.0}, ValueError),
            ({"eps": float("nan")}, ValueError),
            ({"eps": float("inf")}, ValueError),
            ({"eps": "0.5"}, TypeError),
            ({"eps": True}, TypeError),
            ({"min_samples": 0}, ValueError),
            ({"min_samples": 2.5}, TypeError),
            ({"min_samples": True}, TypeError),
            ({"metric": "not-a-metric"}, ValueError),
            ({"metric": 3}, TypeError),
            ({"algorithm": "octree"}, ValueError),
            ({"algorithm": None}, TypeError),
            ({"leaf_size": 0}, ValueError),
            ({"leaf_size": 1.5}, TypeError),
            ({"metric_params": [1, 2]}, TypeError),
            ({"n_jobs": 0}, ValueError),
            ({"n_jobs": 1.0}, TypeError),
        ]
        for kwargs, error in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(error):
                DBSCANClusterer(**kwargs)

    def test_invalid_input(self) -> None:
        cases = [
            (np.array([1.0, 2.0, 3.0]), ValueError),
            (np.zeros((2, 2, 2)), ValueError),
            (np.empty((0, 2)), ValueError),
            (np.empty((3, 0)), ValueError),
            (np.array([[0.0, np.nan], [1.0, 1.0]]), ValueError),
            (np.array([[0.0, np.inf], [1.0, 1.0]]), ValueError),
            (np.array([["a", "b"], ["c", "d"]]), TypeError),
            (None, TypeError),
        ]
        for X, error in cases:
            with self.subTest(X=X), self.assertRaises(error):
                DBSCANClusterer().fit(X)

    def test_get_result_before_fit_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            DBSCANClusterer().get_result()

    def test_failed_refit_clears_previous_result(self) -> None:
        clusterer = DBSCANClusterer().fit(self.X)
        with self.assertRaises(ValueError):
            clusterer.fit(np.array([1.0, 2.0]))
        with self.assertRaises(RuntimeError):
            clusterer.get_result()

    def test_accepts_lists_and_integer_arrays(self) -> None:
        X_int = np.array([[0, 0], [0, 1], [1, 0], [10, 10], [10, 11], [11, 10]])
        labels = DBSCANClusterer(eps=1.5, min_samples=2).fit_predict(X_int.tolist())
        self.assertEqual(labels.dtype, np.int64)
        self.assertEqual(len(set(labels.tolist())), 2)

    def test_input_is_not_mutated(self) -> None:
        X_before = self.X.copy()
        DBSCANClusterer(eps=1.0).fit(self.X)
        np.testing.assert_array_equal(self.X, X_before)

    def test_result_is_a_copy(self) -> None:
        clusterer = DBSCANClusterer(eps=1.0).fit(self.X)
        first = clusterer.get_result()
        first.labels[:] = 99
        first.metadata["parameters"]["eps"] = -1
        second = clusterer.get_result()
        self.assertTrue(np.all(second.labels != 99))
        self.assertEqual(second.metadata["parameters"]["eps"], 1.0)

    def test_metadata_contents(self) -> None:
        params = {
            "eps": 1.0,
            "min_samples": 4,
            "metric": "manhattan",
            "algorithm": "ball_tree",
            "leaf_size": 20,
            "metric_params": None,
            "n_jobs": None,
        }
        result = DBSCANClusterer(**params).fit(self.X).get_result()
        self.assertEqual(result.metadata["parameters"], params)
        self.assertEqual(result.metadata["noise_label"], -1)
        self.assertEqual(result.metadata["n_samples"], self.X.shape[0])
        self.assertEqual(result.metadata["n_features"], self.X.shape[1])
        self.assertEqual(
            sum(result.metadata["cluster_sizes"].values()) + result.n_noise,
            self.X.shape[0],
        )
        self.assertAlmostEqual(
            result.metadata["noise_ratio"], result.n_noise / self.X.shape[0]
        )


class DBSCANHelperTests(unittest.TestCase):
    def test_compatible_algorithms(self) -> None:
        self.assertEqual(compatible_algorithms("cosine"), ["auto", "brute"])
        self.assertIn("kd_tree", compatible_algorithms("euclidean"))
        self.assertNotIn("kd_tree", compatible_algorithms("canberra"))
        with self.assertRaises(ValueError):
            compatible_algorithms("unknown")
        with self.assertRaises(TypeError):
            compatible_algorithms(None)  # type: ignore[arg-type]

    def test_metric_table_is_consistent_with_constructor(self) -> None:
        for metric, algorithms in METRIC_ALGORITHMS.items():
            for algorithm in algorithms:
                with self.subTest(metric=metric, algorithm=algorithm):
                    DBSCANClusterer(metric=metric, algorithm=algorithm)

    def test_k_distance_curve(self) -> None:
        X, _ = _blobs_with_outliers()
        curve = k_distance_curve(X, k=5)
        self.assertEqual(curve.shape, (X.shape[0],))
        self.assertEqual(curve.dtype, np.float64)
        self.assertTrue(np.all(np.diff(curve) >= 0))
        # k=1 is the sample itself, so every distance is zero.
        np.testing.assert_allclose(k_distance_curve(X, k=1), 0.0)
        for bad_k, error in (
            (0, ValueError),
            (X.shape[0] + 1, ValueError),
            (2.0, TypeError),
        ):
            with self.subTest(k=bad_k), self.assertRaises(error):
                k_distance_curve(X, k=bad_k)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
