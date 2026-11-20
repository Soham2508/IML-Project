from __future__ import annotations

import inspect
import unittest

import numpy as np
from sklearn.datasets import make_blobs

from iml_project.core.clustering.birch import BIRCHClusterer
from work3.contracts import ClusterResult


def _three_blobs() -> np.ndarray:
    X, _ = make_blobs(
        n_samples=300,
        centers=[[0.0, 0.0], [10.0, 10.0], [-10.0, 10.0]],
        cluster_std=0.5,
        random_state=0,
    )
    return X


class BIRCHClustererTests(unittest.TestCase):
    def setUp(self) -> None:
        self.X = _three_blobs()

    def _check_result_invariants(self, result: ClusterResult, n_samples: int) -> None:
        self.assertIsInstance(result, ClusterResult)
        self.assertEqual(result.algorithm, "BIRCH")
        self.assertEqual(result.labels.dtype, np.int64)
        self.assertEqual(result.labels.shape, (n_samples,))
        self.assertIsNone(result.memberships)
        self.assertEqual(result.n_noise, 0)
        self.assertTrue(np.all(result.labels >= 0))
        self.assertEqual(result.n_clusters, len(np.unique(result.labels)))
        self.assertIsInstance(result.execution_time, float)
        self.assertGreaterEqual(result.execution_time, 0.0)

    def test_recovers_three_blobs(self) -> None:
        clusterer = BIRCHClusterer(threshold=0.5, n_clusters=3)
        labels = clusterer.fit_predict(self.X)
        result = clusterer.get_result()

        self._check_result_invariants(result, self.X.shape[0])
        np.testing.assert_array_equal(labels, result.labels)
        self.assertEqual(result.n_clusters, 3)
        # Each blob of 100 consecutive samples from make_blobs is not ordered,
        # so check the cluster sizes instead of label identity.
        self.assertEqual(
            sorted(result.metadata["cluster_sizes"].values()), [100, 100, 100]
        )

    def test_fit_returns_self(self) -> None:
        clusterer = BIRCHClusterer()
        self.assertIs(clusterer.fit(self.X), clusterer)

    def test_parameter_grid(self) -> None:
        for threshold in (0.1, 0.5, 1.0, 3.0):
            for branching_factor in (2, 10, 50):
                for n_clusters in (None, 2, 3, 5):
                    with self.subTest(
                        threshold=threshold,
                        branching_factor=branching_factor,
                        n_clusters=n_clusters,
                    ):
                        clusterer = BIRCHClusterer(
                            threshold=threshold,
                            branching_factor=branching_factor,
                            n_clusters=n_clusters,
                        ).fit(self.X)
                        result = clusterer.get_result()
                        self._check_result_invariants(result, self.X.shape[0])
                        self.assertEqual(
                            result.metadata["requested_n_clusters"], n_clusters
                        )
                        if n_clusters is not None:
                            self.assertLessEqual(result.n_clusters, n_clusters)

    def test_n_clusters_none_returns_leaf_subclusters(self) -> None:
        result = BIRCHClusterer(threshold=0.5, n_clusters=None).fit(self.X).get_result()
        self.assertEqual(result.n_clusters, result.metadata["n_subclusters"])
        self.assertGreater(result.n_clusters, 3)

    def test_records_fewer_clusters_than_requested(self) -> None:
        # A huge threshold collapses everything into a single subcluster.
        result = BIRCHClusterer(threshold=100.0, n_clusters=5).fit(self.X).get_result()
        self.assertEqual(result.n_clusters, 1)
        self.assertEqual(result.metadata["requested_n_clusters"], 5)
        self.assertTrue(result.metadata["fewer_clusters_than_requested"])
        self.assertTrue(any("less than" in w for w in result.metadata["warnings"]))

    def test_compute_labels_false_matches_true(self) -> None:
        labels_fit = BIRCHClusterer(threshold=0.5, compute_labels=True).fit_predict(
            self.X
        )
        clusterer = BIRCHClusterer(threshold=0.5, compute_labels=False)
        labels_predict = clusterer.fit_predict(self.X)
        np.testing.assert_array_equal(labels_fit, labels_predict)
        self.assertEqual(clusterer.get_result().metadata["labels_source"], "predict")

    def test_n_clusters_greater_than_n_samples(self) -> None:
        with self.assertRaises(ValueError):
            BIRCHClusterer(n_clusters=11).fit(self.X[:10])

    def test_n_clusters_equal_to_n_samples(self) -> None:
        X = np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 0.0]])
        result = BIRCHClusterer(threshold=0.1, n_clusters=3).fit(X).get_result()
        self.assertEqual(result.n_clusters, 3)

    def test_invalid_parameters(self) -> None:
        cases = [
            ({"threshold": 0.0}, ValueError),
            ({"threshold": -0.5}, ValueError),
            ({"threshold": float("nan")}, ValueError),
            ({"threshold": "0.5"}, TypeError),
            ({"threshold": True}, TypeError),
            ({"branching_factor": 1}, ValueError),
            ({"branching_factor": 2.0}, TypeError),
            ({"branching_factor": True}, TypeError),
            ({"n_clusters": 0}, ValueError),
            ({"n_clusters": -3}, ValueError),
            ({"n_clusters": 2.5}, TypeError),
            ({"n_clusters": True}, TypeError),
            ({"compute_labels": 1}, TypeError),
            ({"copy": "yes"}, TypeError),
        ]
        for kwargs, error in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(error):
                BIRCHClusterer(**kwargs)

    def test_invalid_input(self) -> None:
        cases = [
            (np.array([1.0, 2.0, 3.0]), ValueError),
            (np.zeros((2, 2, 2)), ValueError),
            (np.empty((0, 2)), ValueError),
            (np.array([[0.0, np.nan], [1.0, 1.0]]), ValueError),
            (np.array([["a", "b"], ["c", "d"]]), TypeError),
            (None, TypeError),
        ]
        for X, error in cases:
            with self.subTest(X=X), self.assertRaises(error):
                BIRCHClusterer(n_clusters=None).fit(X)

    def test_get_result_before_fit_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            BIRCHClusterer().get_result()

    def test_input_is_not_mutated_even_with_copy_false(self) -> None:
        X_before = self.X.copy()
        BIRCHClusterer(copy=False).fit(self.X)
        np.testing.assert_array_equal(self.X, X_before)

    def test_result_is_a_copy(self) -> None:
        clusterer = BIRCHClusterer().fit(self.X)
        first = clusterer.get_result()
        first.labels[:] = 99
        second = clusterer.get_result()
        self.assertTrue(np.all(second.labels != 99))

    def test_metadata_contents(self) -> None:
        params = {
            "threshold": 0.7,
            "branching_factor": 20,
            "n_clusters": 3,
            "compute_labels": True,
            "copy": True,
        }
        result = BIRCHClusterer(**params).fit(self.X).get_result()
        self.assertEqual(result.metadata["parameters"], params)
        self.assertEqual(result.metadata["labels_source"], "fit")
        self.assertFalse(result.metadata["fewer_clusters_than_requested"])
        self.assertEqual(result.metadata["n_samples"], self.X.shape[0])
        self.assertEqual(
            sum(result.metadata["cluster_sizes"].values()), self.X.shape[0]
        )

    def test_public_signature_matches_work_plan(self) -> None:
        expected_init = [
            ("threshold", "float", 0.5),
            ("branching_factor", "int", 50),
            ("n_clusters", "int | None", 3),
            ("compute_labels", "bool", True),
            ("copy", "bool", True),
        ]
        init = inspect.signature(BIRCHClusterer.__init__)
        params = list(init.parameters.values())[1:]
        self.assertEqual(
            [(p.name, p.annotation, p.default) for p in params], expected_init
        )
        self.assertEqual(init.return_annotation, "None")

        fit = inspect.signature(BIRCHClusterer.fit)
        self.assertEqual(list(fit.parameters), ["self", "X"])
        self.assertEqual(fit.parameters["X"].annotation, "npt.NDArray[np.float64]")
        self.assertEqual(fit.return_annotation, "BIRCHClusterer")

        fit_predict = inspect.signature(BIRCHClusterer.fit_predict)
        self.assertEqual(list(fit_predict.parameters), ["self", "X"])
        self.assertEqual(
            fit_predict.parameters["X"].annotation, "npt.NDArray[np.float64]"
        )
        self.assertEqual(fit_predict.return_annotation, "npt.NDArray[np.int64]")

        get_result = inspect.signature(BIRCHClusterer.get_result)
        self.assertEqual(list(get_result.parameters), ["self"])
        self.assertEqual(get_result.return_annotation, "ClusterResult")

    def test_execution_time_is_fit_only_by_default(self) -> None:
        result = BIRCHClusterer().fit(self.X).get_result()
        self.assertEqual(result.metadata["labels_source"], "fit")
        self.assertEqual(result.metadata["parameters"]["copy"], True)


if __name__ == "__main__":
    unittest.main()
