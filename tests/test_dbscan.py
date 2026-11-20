from __future__ import annotations

import inspect
import unittest
from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs

from iml_project.core.clustering.dbscan import (
    EXCLUDED_METRICS,
    METRIC_ALGORITHMS,
    REQUIRED_METRIC_PARAMS,
    VALID_ALGORITHMS,
    DBSCANClusterer,
    compatible_algorithms,
    k_distance_curve,
)
from work3.contracts import ClusterResult


def _metric_case(metric: str) -> tuple[np.ndarray, dict[str, Any] | None]:
    """Data and the metric_params ``metric`` needs, for table checks."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    if metric == "haversine":
        return rng.uniform(-1.0, 1.0, size=(40, 2)), None
    if metric == "minkowski":
        return X, {"p": 3}
    if metric == "mahalanobis":
        return X, {"VI": np.linalg.inv(np.cov(X.T))}
    if metric == "seuclidean":
        return X, {"V": X.var(axis=0)}
    return X, None


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

    def test_metric_table_matches_sklearn_exactly(self) -> None:
        """Every listed pair fits; every unlisted pair is rejected by sklearn itself."""
        for metric, algorithms in METRIC_ALGORITHMS.items():
            X, params = _metric_case(metric)
            for algorithm in VALID_ALGORITHMS:
                with self.subTest(metric=metric, algorithm=algorithm):
                    if algorithm in algorithms:
                        result = (
                            DBSCANClusterer(
                                metric=metric, algorithm=algorithm, metric_params=params
                            )
                            .fit(X)
                            .get_result()
                        )
                        self.assertEqual(result.labels.shape, (X.shape[0],))
                    else:
                        with self.assertRaises(ValueError):
                            DBSCAN(
                                metric=metric, algorithm=algorithm, metric_params=params
                            ).fit(X)

    def test_table_covers_every_sklearn_dbscan_metric(self) -> None:
        sklearn_metrics: set[str] = set()
        for constraint in DBSCAN._parameter_constraints["metric"]:
            sklearn_metrics |= set(getattr(constraint, "options", ()))
        self.assertTrue(sklearn_metrics)
        self.assertEqual(
            sklearn_metrics, set(METRIC_ALGORITHMS) | set(EXCLUDED_METRICS)
        )
        self.assertFalse(set(METRIC_ALGORITHMS) & set(EXCLUDED_METRICS))

    def test_excluded_metrics_rejected(self) -> None:
        for metric in EXCLUDED_METRICS:
            with self.subTest(metric=metric), self.assertRaises(ValueError) as ctx:
                DBSCANClusterer(metric=metric)
            self.assertIn("not supported", str(ctx.exception))
        X, _ = _metric_case("euclidean")
        for metric in ("matching", "wminkowski"):
            with self.subTest(metric=metric), self.assertRaises(ValueError):
                DBSCAN(metric=metric).fit(X)

    def test_required_metric_params(self) -> None:
        for metric in REQUIRED_METRIC_PARAMS:
            for bad_params in (None, {}, {"unrelated": 1}):
                with (
                    self.subTest(metric=metric, metric_params=bad_params),
                    self.assertRaises(ValueError),
                ):
                    DBSCANClusterer(metric=metric, metric_params=bad_params)
        X, _ = _metric_case("euclidean")
        DBSCANClusterer(metric="mahalanobis", metric_params={"V": np.cov(X.T)}).fit(X)

    def test_minkowski_p_validation(self) -> None:
        cases = [
            ({"p": 0}, "auto", ValueError),
            ({"p": -1.0}, "brute", ValueError),
            ({"p": float("nan")}, "auto", ValueError),
            ({"p": "2"}, "auto", TypeError),
            ({"p": True}, "auto", TypeError),
            ({"p": 0.5}, "kd_tree", ValueError),
            ({"p": 0.5}, "ball_tree", ValueError),
        ]
        for params, algorithm, error in cases:
            with (
                self.subTest(params=params, algorithm=algorithm),
                self.assertRaises(error),
            ):
                DBSCANClusterer(
                    metric="minkowski", algorithm=algorithm, metric_params=params
                )
        X, _ = _metric_case("euclidean")
        for p in (0.5, 1, 2, float("inf")):
            with self.subTest(p=p):
                DBSCANClusterer(metric="minkowski", metric_params={"p": p}).fit(X)

    def test_minkowski_p2_matches_euclidean(self) -> None:
        X, _ = _blobs_with_outliers()
        labels_minkowski = DBSCANClusterer(
            eps=1.0, metric="minkowski", metric_params={"p": 2}
        ).fit_predict(X)
        labels_euclidean = DBSCANClusterer(eps=1.0).fit_predict(X)
        np.testing.assert_array_equal(labels_minkowski, labels_euclidean)

    def test_metric_params_are_copied(self) -> None:
        X, params = _metric_case("seuclidean")
        assert params is not None
        clusterer = DBSCANClusterer(metric="seuclidean", metric_params=params)
        params["V"][:] = -1.0
        self.assertTrue(np.all(clusterer.get_params()["metric_params"]["V"] > 0))
        clusterer.fit(X)

    def test_public_signature_matches_work_plan(self) -> None:
        expected_init = [
            ("eps", "float", 0.5),
            ("min_samples", "int", 5),
            ("metric", "str", "euclidean"),
            ("algorithm", "str", "auto"),
            ("leaf_size", "int", 30),
            ("metric_params", "dict[str, Any] | None", None),
            ("n_jobs", "int | None", None),
        ]
        init = inspect.signature(DBSCANClusterer.__init__)
        params = list(init.parameters.values())[1:]
        self.assertEqual(
            [(p.name, p.annotation, p.default) for p in params], expected_init
        )
        self.assertEqual(init.return_annotation, "None")

        fit = inspect.signature(DBSCANClusterer.fit)
        self.assertEqual(list(fit.parameters), ["self", "X"])
        self.assertEqual(fit.parameters["X"].annotation, "npt.NDArray[np.float64]")
        self.assertEqual(fit.return_annotation, "DBSCANClusterer")

        fit_predict = inspect.signature(DBSCANClusterer.fit_predict)
        self.assertEqual(list(fit_predict.parameters), ["self", "X"])
        self.assertEqual(
            fit_predict.parameters["X"].annotation, "npt.NDArray[np.float64]"
        )
        self.assertEqual(fit_predict.return_annotation, "npt.NDArray[np.int64]")

        get_result = inspect.signature(DBSCANClusterer.get_result)
        self.assertEqual(list(get_result.parameters), ["self"])
        self.assertEqual(get_result.return_annotation, "ClusterResult")

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

    def test_k_distance_curve_with_metric_params(self) -> None:
        X, _ = _blobs_with_outliers()
        np.testing.assert_allclose(
            k_distance_curve(X, k=5, metric="minkowski", metric_params={"p": 2}),
            k_distance_curve(X, k=5),
        )
        np.testing.assert_allclose(
            k_distance_curve(X, k=5, metric="minkowski", metric_params={"p": 1}),
            k_distance_curve(X, k=5, metric="manhattan"),
        )
        curve = k_distance_curve(
            X, k=5, metric="mahalanobis", metric_params={"V": np.cov(X.T)}
        )
        self.assertEqual(curve.shape, (X.shape[0],))
        with self.assertRaises(ValueError):
            k_distance_curve(X, k=5, metric="minkowski")
        with self.assertRaises(ValueError):
            k_distance_curve(X, k=5, metric="precomputed")


if __name__ == "__main__":
    unittest.main()
