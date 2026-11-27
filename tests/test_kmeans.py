from __future__ import annotations

import ast
import inspect
import unittest
from itertools import pairwise
from pathlib import Path

import numpy as np

from iml_project.core.clustering import kmeans as kmeans_module
from iml_project.core.clustering.kmeans import (
    KMeansClusterer,
    elbow_curve,
    manhattan,
    squared_euclidean,
)
from work3.contracts import ClusterResult

CENTERS = np.array([[0.0, 0.0], [10.0, 10.0], [-10.0, 10.0]])


def _three_blobs(n_per_blob: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Three well-separated Gaussian blobs and their generating group (for tests only)."""
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(c, 0.5, size=(n_per_blob, 2)) for c in CENTERS])
    groups = np.repeat(np.arange(len(CENTERS)), n_per_blob)
    return X, groups


def _uniform(n: int = 200, d: int = 3, seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).random((n, d))


def _same_partition(a: np.ndarray, b: np.ndarray) -> bool:
    """True if two labelings define the same partition (IDs may be permuted)."""
    pairs = set(zip(a.tolist(), b.tolist()))
    return len(pairs) == len(set(a.tolist())) == len(set(b.tolist()))


class KMeansClustererTests(unittest.TestCase):
    def setUp(self) -> None:
        self.X, self.groups = _three_blobs()

    def _check_result_invariants(
        self, result: ClusterResult, n_samples: int, n_clusters: int
    ) -> None:
        self.assertIsInstance(result, ClusterResult)
        self.assertEqual(result.labels.dtype, np.int64)
        self.assertEqual(result.labels.shape, (n_samples,))
        self.assertIsNone(result.memberships)
        self.assertEqual(result.n_noise, 0)
        self.assertTrue(np.all(result.labels >= 0))
        self.assertTrue(np.all(result.labels < n_clusters))
        self.assertEqual(result.n_clusters, len(np.unique(result.labels)))
        self.assertIsInstance(result.execution_time, float)
        self.assertGreaterEqual(result.execution_time, 0.0)

    # --- correctness ---------------------------------------------------------

    def test_recovers_three_blobs(self) -> None:
        model = KMeansClusterer(n_clusters=3, n_init=5, random_state=0)
        labels = model.fit_predict(self.X)
        result = model.get_result()
        self._check_result_invariants(result, len(self.X), 3)
        self.assertEqual(result.algorithm, "KMeans")
        self.assertEqual(result.n_clusters, 3)
        self.assertTrue(_same_partition(labels, self.groups))
        self.assertTrue(result.metadata["converged"])

    def test_every_point_is_assigned_to_its_nearest_centroid(self) -> None:
        X = _uniform()
        for metric, dist in (("sqeuclidean", squared_euclidean), ("manhattan", manhattan)):
            with self.subTest(metric=metric):
                model = KMeansClusterer(n_clusters=4, metric=metric, random_state=3).fit(X)
                centers = model.cluster_centers_.tolist()
                for x, label in zip(X.tolist(), model.labels_.tolist()):
                    distances = [dist(x, c) for c in centers]
                    self.assertEqual(label, distances.index(min(distances)))

    def test_converged_centroids_are_means(self) -> None:
        X = _uniform()
        model = KMeansClusterer(n_clusters=4, random_state=2).fit(X)
        self.assertTrue(model.get_result().metadata["converged"])
        for k in range(4):
            np.testing.assert_allclose(
                model.cluster_centers_[k], X[model.labels_ == k].mean(axis=0)
            )

    def test_manhattan_centroids_are_medians(self) -> None:
        X = _uniform()
        model = KMeansClusterer(n_clusters=4, metric="manhattan", random_state=2).fit(X)
        result = model.get_result()
        self.assertEqual(result.algorithm, "KMeans-manhattan")
        self.assertEqual(result.metadata["centroid_update"], "median")
        self.assertTrue(result.metadata["converged"])
        for k in range(4):
            np.testing.assert_allclose(
                model.cluster_centers_[k], np.median(X[model.labels_ == k], axis=0)
            )

    def test_objective_history_never_increases(self) -> None:
        X = _uniform(n=300, d=4)
        for metric in ("sqeuclidean", "manhattan"):
            for seed in range(5):
                with self.subTest(metric=metric, seed=seed):
                    result = (
                        KMeansClusterer(n_clusters=6, metric=metric, random_state=seed)
                        .fit(X)
                        .get_result()
                    )
                    history = result.metadata["objective_history"]
                    self.assertEqual(len(history), result.metadata["n_iter"] + 1)
                    for before, after in pairwise(history):
                        self.assertLessEqual(after, before + 1e-9)

    def test_inertia_matches_recomputed_sse(self) -> None:
        model = KMeansClusterer(n_clusters=3, random_state=0).fit(self.X)
        centers = model.cluster_centers_[model.labels_]
        sse = float(np.sum((self.X - centers) ** 2))
        self.assertAlmostEqual(model.inertia_, sse, places=8)
        self.assertAlmostEqual(model.get_result().metadata["inertia"], sse, places=8)

    def test_k_equals_one_gives_global_mean(self) -> None:
        model = KMeansClusterer(n_clusters=1, random_state=0).fit(self.X)
        self.assertTrue(np.all(model.labels_ == 0))
        np.testing.assert_allclose(model.cluster_centers_[0], self.X.mean(axis=0))

    def test_k_equals_n_gives_zero_sse(self) -> None:
        X = _uniform(n=12, d=2)
        model = KMeansClusterer(n_clusters=12, random_state=0).fit(X)
        self.assertEqual(model.get_result().n_clusters, 12)
        self.assertAlmostEqual(model.inertia_, 0.0)

    # --- initialisation and randomness ---------------------------------------

    def test_same_seed_gives_identical_results(self) -> None:
        X = _uniform()
        a = KMeansClusterer(n_clusters=5, random_state=7).fit(X).get_result()
        b = KMeansClusterer(n_clusters=5, random_state=7).fit(X).get_result()
        np.testing.assert_array_equal(a.labels, b.labels)
        self.assertEqual(a.metadata["cluster_centers"], b.metadata["cluster_centers"])

    def test_different_seeds_can_differ(self) -> None:
        X = _uniform(n=300, d=4)
        objectives = {
            KMeansClusterer(n_clusters=8, random_state=s).fit(X).inertia_
            for s in range(10)
        }
        self.assertGreater(len(objectives), 1)

    def test_random_state_none_records_a_reproducible_seed(self) -> None:
        X = _uniform()
        first = KMeansClusterer(n_clusters=5).fit(X).get_result()
        seed = first.metadata["random_state_used"]
        again = KMeansClusterer(n_clusters=5, random_state=seed).fit(X).get_result()
        np.testing.assert_array_equal(first.labels, again.labels)

    def test_forgy_uses_distinct_rows(self) -> None:
        # 3 distinct rows, each repeated: K=3 must pick all three, giving SSE 0.
        X = np.repeat(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), 5, axis=0)
        for seed in range(10):
            model = KMeansClusterer(n_clusters=3, random_state=seed).fit(X)
            self.assertAlmostEqual(model.inertia_, 0.0)
            self.assertEqual(model.get_result().n_clusters, 3)
        with self.assertRaises(ValueError):
            KMeansClusterer(n_clusters=4, random_state=0).fit(X)

    def test_explicit_init_is_used(self) -> None:
        init = CENTERS + 0.1
        model = KMeansClusterer(n_clusters=3, init=init).fit(self.X)
        result = model.get_result()
        self.assertEqual(result.metadata["init_method"], "explicit")
        np.testing.assert_array_equal(model.labels_, self.groups)

    def test_explicit_init_with_n_init_runs_once_and_warns(self) -> None:
        model = KMeansClusterer(n_clusters=3, init=CENTERS, n_init=4).fit(self.X)
        result = model.get_result()
        self.assertEqual(result.metadata["n_runs"], 1)
        self.assertTrue(any("n_init" in w for w in result.metadata["warnings"]))

    def test_n_init_keeps_lowest_objective(self) -> None:
        X = _uniform(n=300, d=4)
        result = KMeansClusterer(n_clusters=8, n_init=6, random_state=0).fit(X).get_result()
        objectives = result.metadata["run_objectives"]
        self.assertEqual(len(objectives), 6)
        self.assertEqual(result.metadata["inertia"], min(objectives))
        self.assertEqual(
            objectives[result.metadata["best_run_index"]], min(objectives)
        )

    # --- empty clusters and stopping ------------------------------------------

    def test_empty_cluster_is_refilled_with_farthest_point(self) -> None:
        # The third start centroid is far from every point, so its cluster is
        # empty after the first assignment.
        X = np.array([[0.0], [0.1], [0.2], [5.0], [10.0], [10.1], [10.2]])
        init = np.array([[0.1], [10.1], [100.0]])
        model = KMeansClusterer(n_clusters=3, init=init).fit(X)
        result = model.get_result()
        events = result.metadata["empty_cluster_events"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["cluster"], 2)
        self.assertEqual(events[0]["iteration"], 1)
        self.assertEqual(events[0]["point_index"], 3)  # 5.0 is farthest from its centroid
        self.assertEqual(result.n_clusters, 3)
        self.assertEqual(len(set(model.labels_.tolist())), 3)
        self.assertTrue(result.metadata["converged"])

    def test_two_empty_clusters_get_different_points(self) -> None:
        X = np.array([[0.0], [1.0], [2.0], [3.0], [20.0]])
        init = np.array([[1.0], [100.0], [200.0]])
        result = KMeansClusterer(n_clusters=3, init=init).fit(X).get_result()
        first_events = [e for e in result.metadata["empty_cluster_events"] if e["iteration"] == 1]
        self.assertEqual(len(first_events), 2)
        self.assertNotEqual(first_events[0]["point_index"], first_events[1]["point_index"])
        self.assertEqual(result.n_clusters, 3)

    def test_max_iter_reached_records_warning(self) -> None:
        X = _uniform(n=300, d=4)
        result = KMeansClusterer(n_clusters=8, max_iter=1, random_state=0).fit(X).get_result()
        self.assertFalse(result.metadata["converged"])
        self.assertEqual(result.metadata["n_iter"], 1)
        self.assertTrue(any("max_iter" in w for w in result.metadata["warnings"]))

    # --- validation ----------------------------------------------------------

    def test_invalid_parameters_are_rejected(self) -> None:
        cases = [
            ({"n_clusters": 0}, ValueError),
            ({"n_clusters": True}, TypeError),
            ({"n_clusters": 2.0}, TypeError),
            ({"n_init": 0}, ValueError),
            ({"max_iter": 0}, ValueError),
            ({"max_iter": 1.5}, TypeError),
            ({"metric": "cosine"}, ValueError),
            ({"metric": 1}, TypeError),
            ({"init": "k-means++"}, ValueError),
            ({"init": None}, TypeError),
            ({"init": [[0.0, 0.0]]}, ValueError),  # 1 centroid for n_clusters=3
            ({"init": [[np.nan, 0.0]] * 3}, ValueError),
            ({"init": [0.0, 1.0, 2.0]}, ValueError),  # 1-D
            ({"random_state": -1}, ValueError),
            ({"random_state": False}, TypeError),
            ({"random_state": 1.0}, TypeError),
        ]
        for kwargs, error in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(error):
                KMeansClusterer(**kwargs)

    def test_numpy_integer_parameters_are_accepted(self) -> None:
        model = KMeansClusterer(n_clusters=np.int64(3), random_state=np.int32(0))
        self.assertIs(type(model.n_clusters), int)
        self.assertIs(type(model.random_state), int)

    def test_invalid_input_is_rejected(self) -> None:
        model = KMeansClusterer(n_clusters=2, random_state=0)
        cases = [
            (None, TypeError),
            ([["a", "b"]], TypeError),
            (np.zeros(5), ValueError),
            (np.zeros((0, 2)), ValueError),
            (np.array([[0.0, np.nan], [1.0, 1.0]]), ValueError),
            (np.array([[0.0, np.inf], [1.0, 1.0]]), ValueError),
            (np.zeros((1, 2)), ValueError),  # n_clusters > n_samples
        ]
        for X, error in cases:
            with self.subTest(X=X), self.assertRaises(error):
                model.fit(X)

    def test_init_feature_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            KMeansClusterer(n_clusters=3, init=np.zeros((3, 5))).fit(self.X)

    # --- contract behaviour ----------------------------------------------------

    def test_get_result_before_fit_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            KMeansClusterer().get_result()

    def test_failed_refit_clears_previous_result(self) -> None:
        model = KMeansClusterer(n_clusters=3, random_state=0).fit(self.X)
        with self.assertRaises(ValueError):
            model.fit(np.zeros(4))
        self.assertIsNone(model.labels_)
        self.assertIsNone(model.cluster_centers_)
        with self.assertRaises(RuntimeError):
            model.get_result()

    def test_input_is_not_modified(self) -> None:
        X = self.X.copy()
        KMeansClusterer(n_clusters=3, random_state=0).fit(X)
        np.testing.assert_array_equal(X, self.X)

    def test_get_result_returns_a_copy(self) -> None:
        model = KMeansClusterer(n_clusters=3, random_state=0).fit(self.X)
        result = model.get_result()
        result.labels[:] = 99
        result.metadata["cluster_centers"].clear()
        fresh = model.get_result()
        self.assertTrue(np.all(fresh.labels < 3))
        self.assertEqual(len(fresh.metadata["cluster_centers"]), 3)

    def test_fit_predict_matches_result_labels(self) -> None:
        model = KMeansClusterer(n_clusters=3, random_state=0)
        labels = model.fit_predict(self.X)
        np.testing.assert_array_equal(labels, model.get_result().labels)

    def test_metadata_contents(self) -> None:
        result = KMeansClusterer(n_clusters=3, random_state=0).fit(self.X).get_result()
        meta = result.metadata
        for key in (
            "parameters", "backend", "n_samples", "n_features", "cluster_sizes",
            "warnings", "random_state_used", "init_method", "metric",
            "centroid_update", "objective_name", "inertia", "objective_history",
            "n_iter", "converged", "empty_cluster_events", "n_runs",
            "best_run_index", "run_objectives", "cluster_centers",
        ):
            self.assertIn(key, meta)
        self.assertEqual(meta["parameters"]["n_clusters"], 3)
        self.assertEqual(sum(meta["cluster_sizes"].values()), len(self.X))
        self.assertEqual(meta["random_state_used"], 0)

    def test_signature_and_repr(self) -> None:
        params = list(inspect.signature(KMeansClusterer.__init__).parameters)
        self.assertEqual(
            params,
            ["self", "n_clusters", "init", "n_init", "max_iter", "metric", "random_state"],
        )
        self.assertTrue(repr(KMeansClusterer()).startswith("KMeansClusterer(n_clusters=3"))

    def test_module_imports_no_third_party_library_except_numpy(self) -> None:
        tree = ast.parse(Path(inspect.getfile(kmeans_module)).read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        allowed = {
            "__future__", "copy", "math", "random", "statistics", "time",
            "collections", "dataclasses", "typing", "numpy", "work3",
        }
        self.assertLessEqual(imported, allowed)


class ElbowCurveTests(unittest.TestCase):
    def test_shapes_and_values(self) -> None:
        X, _ = _three_blobs(n_per_blob=40)
        curve = elbow_curve(X, k_values=[1, 2, 3, 4], seeds=[0, 1, 2])
        self.assertEqual(curve["k_values"], [1, 2, 3, 4])
        self.assertEqual(len(curve["objective_mean"]), 4)
        self.assertEqual(len(curve["objective_std"]), 4)
        self.assertTrue(all(len(row) == 3 for row in curve["objective_per_seed"]))
        for row, mean, std in zip(
            curve["objective_per_seed"], curve["objective_mean"], curve["objective_std"]
        ):
            self.assertAlmostEqual(mean, float(np.mean(row)))
            self.assertAlmostEqual(std, float(np.std(row)))  # ddof=0
        # K=1 has a single possible solution, so no spread across seeds.
        self.assertAlmostEqual(curve["objective_std"][0], 0.0)
        # Best run per K: the large drop happens up to the true K=3, then levels off.
        # (The mean can be pulled up at K=3 by a seed stuck in a local optimum.)
        best = [min(row) for row in curve["objective_per_seed"]]
        self.assertGreater(best[1] - best[2], 10 * (best[2] - best[3]))

    def test_invalid_arguments(self) -> None:
        X, _ = _three_blobs(n_per_blob=10)
        with self.assertRaises(ValueError):
            elbow_curve(X, k_values=[], seeds=[0])
        with self.assertRaises(TypeError):
            elbow_curve(X, k_values=3, seeds=[0])  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            elbow_curve(X, k_values=[2], seeds=[0], metric="cosine")


if __name__ == "__main__":
    unittest.main()
