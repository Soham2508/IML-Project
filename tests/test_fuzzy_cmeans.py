from __future__ import annotations

import ast
import inspect
import math
import unittest
from itertools import pairwise
from pathlib import Path

import numpy as np

from iml_project.core.clustering import fuzzy_cmeans as fcm_module
from iml_project.core.clustering.fuzzy_cmeans import (
    FuzzyCMeansClusterer,
    fuzzy_validity_curve,
    squared_distances,
    update_centers,
    update_memberships,
    validity_indices,
)
from work3.contracts import ClusterResult

CENTERS = np.array([[0.0, 0.0], [10.0, 10.0], [-10.0, 10.0]])


def _three_blobs(n_per_blob: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Three well-separated Gaussian blobs and their generating group (for tests only)."""
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(c, 0.5, size=(n_per_blob, 2)) for c in CENTERS])
    return X, np.repeat(np.arange(len(CENTERS)), n_per_blob)


def _uniform(n: int = 200, d: int = 3, seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).random((n, d))


def _same_partition(a: np.ndarray, b: np.ndarray) -> bool:
    pairs = set(zip(a.tolist(), b.tolist()))
    return len(pairs) == len(set(a.tolist())) == len(set(b.tolist()))


class FCMBuildingBlockTests(unittest.TestCase):
    def test_membership_formula_by_hand(self) -> None:
        # One point at distance^2 1 and 4 from two centres, m = 2 (p = 1):
        # u_1 = 1 / (1/1 + 1/4) = 0.8 and u_2 = 0.2.
        u, n_zero = update_memberships(np.array([[1.0, 4.0]]), m=2.0)
        np.testing.assert_allclose(u, [[0.8, 0.2]])
        self.assertEqual(n_zero, 0)

    def test_membership_formula_matches_definition(self) -> None:
        rng = np.random.default_rng(3)
        d2 = rng.random((20, 4)) + 0.1
        for m in (1.3, 2.0, 4.0):
            u, _ = update_memberships(d2, m)
            expected = 1.0 / np.sum(
                (d2[:, :, None] / d2[:, None, :]) ** (1.0 / (m - 1.0)), axis=2
            )
            np.testing.assert_allclose(u, expected, rtol=1e-12)

    def test_zero_distance_rule(self) -> None:
        d2 = np.array([[0.0, 4.0, 9.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
        u, n_zero = update_memberships(d2, m=2.0)
        np.testing.assert_allclose(u[0], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(u[1], [0.5, 0.5, 0.0])
        np.testing.assert_allclose(u[2], [1 / 3, 1 / 3, 1 / 3])
        self.assertEqual(n_zero, 2)

    def test_memberships_do_not_overflow_for_m_near_one(self) -> None:
        d2 = np.array([[1e-8, 1.0], [5.0, 1e3]])
        u, _ = update_memberships(d2, m=1.001)
        self.assertTrue(np.all(np.isfinite(u)))
        np.testing.assert_allclose(u.sum(axis=1), 1.0)
        np.testing.assert_allclose(u, [[1.0, 0.0], [1.0, 0.0]], atol=1e-12)

    def test_center_update_is_weighted_mean(self) -> None:
        X = np.array([[0.0], [2.0], [10.0]])
        u = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
        centers, degenerate = update_centers(X, u, m=2.0)
        # cluster 0: (1*0 + 0.25*2) / 1.25 = 0.4; cluster 1: (0.25*2 + 1*10) / 1.25 = 8.4
        np.testing.assert_allclose(centers, [[0.4], [8.4]])
        self.assertEqual(degenerate, [])

    def test_center_with_zero_weight_keeps_previous(self) -> None:
        X = np.array([[0.0], [1.0]])
        u = np.array([[1.0, 0.0], [1.0, 0.0]])
        centers, degenerate = update_centers(X, u, 2.0, previous=np.array([[0.0], [7.0]]))
        self.assertEqual(degenerate, [1])
        np.testing.assert_allclose(centers, [[0.5], [7.0]])

    def test_squared_distances_exact_zero(self) -> None:
        X = np.array([[0.1, 0.7], [3.0, 4.0]])
        d2 = squared_distances(X, X[[0]])
        self.assertEqual(d2[0, 0], 0.0)
        self.assertAlmostEqual(d2[1, 0], 2.9**2 + 3.3**2)

    def test_validity_indices_by_hand(self) -> None:
        X = np.array([[0.0], [1.0], [4.0], [5.0]])
        u = np.array([[1.0, 0.0], [0.5, 0.5], [0.5, 0.5], [0.0, 1.0]])
        v = np.array([[0.0], [5.0]])
        idx = validity_indices(X, u, v, m=2.0)
        self.assertAlmostEqual(idx["partition_coefficient"], (1 + 0.5 + 0.5 + 1) / 4)
        self.assertAlmostEqual(idx["partition_entropy"], 2 * math.log(2) / 4)
        # J = 0.25*(1 + 16) + 0.25*(16 + 1) = 8.5; min separation^2 = 25.
        self.assertAlmostEqual(idx["xie_beni"], 8.5 / (4 * 25))

    def test_validity_indices_single_cluster(self) -> None:
        X = _uniform(n=10)
        idx = validity_indices(X, np.ones((10, 1)), X.mean(axis=0, keepdims=True), 2.0)
        self.assertAlmostEqual(idx["partition_coefficient"], 1.0)
        self.assertAlmostEqual(idx["partition_entropy"], 0.0)
        self.assertTrue(math.isnan(idx["xie_beni"]))


class FuzzyCMeansClustererTests(unittest.TestCase):
    def setUp(self) -> None:
        self.X, self.groups = _three_blobs()

    def _check_result_invariants(self, result: ClusterResult, n: int, c: int) -> None:
        self.assertIsInstance(result, ClusterResult)
        self.assertEqual(result.algorithm, "FuzzyCMeans")
        self.assertEqual(result.labels.dtype, np.int64)
        self.assertEqual(result.labels.shape, (n,))
        self.assertEqual(result.memberships.dtype, np.float64)
        self.assertEqual(result.memberships.shape, (n, c))
        self.assertTrue(np.all(result.memberships >= 0.0))
        self.assertTrue(np.all(result.memberships <= 1.0))
        np.testing.assert_allclose(result.memberships.sum(axis=1), 1.0)
        np.testing.assert_array_equal(result.labels, np.argmax(result.memberships, axis=1))
        self.assertEqual(result.n_noise, 0)
        self.assertEqual(result.n_clusters, len(np.unique(result.labels)))
        self.assertIsInstance(result.execution_time, float)
        self.assertGreaterEqual(result.execution_time, 0.0)

    # --- correctness ---------------------------------------------------------

    def test_recovers_three_blobs_with_every_init(self) -> None:
        for init in ("random_memberships", "random_centers", CENTERS + 0.5):
            with self.subTest(init=init if isinstance(init, str) else "explicit"):
                model = FuzzyCMeansClusterer(n_clusters=3, init=init, random_state=0)
                labels = model.fit_predict(self.X)
                result = model.get_result()
                self._check_result_invariants(result, len(self.X), 3)
                self.assertEqual(result.n_clusters, 3)
                self.assertTrue(_same_partition(labels, self.groups))
                self.assertTrue(result.metadata["converged"])
                self.assertGreater(result.metadata["partition_coefficient"], 0.9)

    def test_objective_never_increases(self) -> None:
        X = _uniform(n=300, d=4)
        for init in ("random_memberships", "random_centers"):
            for criterion in ("centers", "memberships", "objective"):
                for seed in range(3):
                    with self.subTest(init=init, criterion=criterion, seed=seed):
                        result = (
                            FuzzyCMeansClusterer(
                                n_clusters=5,
                                init=init,
                                stop_criterion=criterion,
                                tol=1e-8,
                                random_state=seed,
                            )
                            .fit(X)
                            .get_result()
                        )
                        history = result.metadata["objective_history"]
                        self.assertEqual(len(history), result.metadata["n_iter"])
                        for before, after in pairwise(history):
                            self.assertLessEqual(after, before * (1 + 1e-12) + 1e-12)

    def test_converged_solution_is_a_fixed_point(self) -> None:
        model = FuzzyCMeansClusterer(n_clusters=3, tol=1e-10, random_state=0).fit(self.X)
        u_again, _ = update_memberships(
            squared_distances(self.X, model.cluster_centers_), model.m
        )
        np.testing.assert_allclose(u_again, model.memberships_, atol=1e-7)
        v_again, _ = update_centers(self.X, model.memberships_, model.m)
        np.testing.assert_allclose(v_again, model.cluster_centers_, atol=1e-9)

    def test_objective_matches_recomputed_value(self) -> None:
        model = FuzzyCMeansClusterer(n_clusters=3, random_state=0).fit(self.X)
        d2 = squared_distances(self.X, model.cluster_centers_)
        expected = float(np.sum(model.memberships_**model.m * d2))
        self.assertAlmostEqual(model.objective_, expected, places=8)

    def test_single_cluster_gives_global_mean(self) -> None:
        model = FuzzyCMeansClusterer(n_clusters=1, random_state=0).fit(self.X)
        np.testing.assert_allclose(model.memberships_, 1.0)
        np.testing.assert_allclose(model.cluster_centers_[0], self.X.mean(axis=0))
        self.assertTrue(math.isnan(model.get_result().metadata["xie_beni"]))

    def test_fuzzifier_controls_softness(self) -> None:
        pcs = [
            FuzzyCMeansClusterer(n_clusters=3, m=m, random_state=0)
            .fit(self.X)
            .get_result()
            .metadata["partition_coefficient"]
            for m in (1.1, 2.0, 4.0, 10.0)
        ]
        self.assertGreater(pcs[0], 0.999)  # nearly crisp
        for before, after in pairwise(pcs):
            self.assertGreater(before, after)  # fuzzier as m grows
        self.assertLess(pcs[-1], 0.6)  # towards 1/C

    def test_duplicate_rows_and_point_on_centre_are_handled(self) -> None:
        X = np.repeat(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), 4, axis=0)
        result = (
            FuzzyCMeansClusterer(n_clusters=3, init="random_centers", random_state=0)
            .fit(X)
            .get_result()
        )
        self._check_result_invariants(result, len(X), 3)
        self.assertGreater(len(result.metadata["zero_distance_events"]), 0)
        self.assertAlmostEqual(result.metadata["objective"], 0.0)
        with self.assertRaises(ValueError):
            FuzzyCMeansClusterer(n_clusters=4, init="random_centers", random_state=0).fit(X)

    # --- stopping ------------------------------------------------------------

    def test_each_stop_criterion_is_recorded_and_met(self) -> None:
        for criterion in ("centers", "memberships", "objective"):
            with self.subTest(criterion=criterion):
                meta = (
                    FuzzyCMeansClusterer(
                        n_clusters=3, stop_criterion=criterion, tol=1e-6, random_state=0
                    )
                    .fit(self.X)
                    .get_result()
                    .metadata
                )
                self.assertEqual(meta["stop_criterion"], criterion)
                self.assertTrue(meta["converged"])
                self.assertLessEqual(meta["change_history"][-1], 1e-6)
                self.assertTrue(all(c > 1e-6 for c in meta["change_history"][:-1]))

    def test_max_iter_reached_records_warning(self) -> None:
        result = (
            FuzzyCMeansClusterer(n_clusters=5, max_iter=2, tol=1e-12, random_state=0)
            .fit(_uniform())
            .get_result()
        )
        self.assertFalse(result.metadata["converged"])
        self.assertEqual(result.metadata["n_iter"], 2)
        self.assertTrue(any("max_iter" in w for w in result.metadata["warnings"]))

    # --- initialisation and randomness -----------------------------------------

    def test_same_seed_gives_identical_results(self) -> None:
        X = _uniform()
        for init in ("random_memberships", "random_centers"):
            with self.subTest(init=init):
                a = FuzzyCMeansClusterer(4, init=init, random_state=7).fit(X).get_result()
                b = FuzzyCMeansClusterer(4, init=init, random_state=7).fit(X).get_result()
                np.testing.assert_array_equal(a.memberships, b.memberships)

    def test_different_seeds_can_differ(self) -> None:
        X = _uniform(n=300, d=4)
        centres = {
            tuple(
                np.round(
                    FuzzyCMeansClusterer(6, init="random_centers", random_state=s)
                    .fit(X)
                    .cluster_centers_[0],
                    6,
                )
            )
            for s in range(6)
        }
        self.assertGreater(len(centres), 1)

    def test_random_state_none_records_a_reproducible_seed(self) -> None:
        X = _uniform()
        first = FuzzyCMeansClusterer(4).fit(X).get_result()
        seed = first.metadata["random_state_used"]
        again = FuzzyCMeansClusterer(4, random_state=seed).fit(X).get_result()
        np.testing.assert_array_equal(first.memberships, again.memberships)

    def test_init_method_is_recorded(self) -> None:
        expected = {
            "random_memberships": "random_memberships",
            "random_centers": "random_centers",
        }
        for init, prefix in expected.items():
            meta = FuzzyCMeansClusterer(3, init=init, random_state=0).fit(self.X).get_result().metadata
            self.assertTrue(meta["init_method"].startswith(prefix))
        meta = FuzzyCMeansClusterer(3, init=CENTERS).fit(self.X).get_result().metadata
        self.assertEqual(meta["init_method"], "explicit centres")

    def test_default_init_is_random_memberships(self) -> None:
        self.assertEqual(FuzzyCMeansClusterer().init, "random_memberships")

    # --- validation ----------------------------------------------------------

    def test_invalid_parameters_are_rejected(self) -> None:
        cases = [
            ({"n_clusters": 0}, ValueError),
            ({"n_clusters": True}, TypeError),
            ({"m": 1.0}, ValueError),
            ({"m": 0.5}, ValueError),
            ({"m": float("nan")}, ValueError),
            ({"m": float("inf")}, ValueError),
            ({"m": True}, TypeError),
            ({"m": "2"}, TypeError),
            ({"tol": 0.0}, ValueError),
            ({"tol": -1e-3}, ValueError),
            ({"max_iter": 0}, ValueError),
            ({"stop_criterion": "labels"}, ValueError),
            ({"stop_criterion": None}, TypeError),
            ({"init": "k-means++"}, ValueError),
            ({"init": None}, TypeError),
            ({"init": [[0.0, 0.0]]}, ValueError),
            ({"init": [[np.inf, 0.0]] * 3}, ValueError),
            ({"random_state": -1}, ValueError),
            ({"random_state": 1.5}, TypeError),
        ]
        for kwargs, error in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(error):
                FuzzyCMeansClusterer(**kwargs)

    def test_numpy_scalars_are_accepted(self) -> None:
        model = FuzzyCMeansClusterer(n_clusters=np.int64(2), m=np.float32(1.5), tol=np.float64(1e-4))
        self.assertIs(type(model.n_clusters), int)
        self.assertIs(type(model.m), float)

    def test_invalid_input_is_rejected(self) -> None:
        model = FuzzyCMeansClusterer(n_clusters=2, random_state=0)
        cases = [
            (None, TypeError),
            ([["a", "b"]], TypeError),
            (np.zeros(5), ValueError),
            (np.zeros((0, 2)), ValueError),
            (np.array([[0.0, np.nan], [1.0, 1.0]]), ValueError),
            (np.zeros((1, 2)), ValueError),  # n_clusters > n_samples
        ]
        for X, error in cases:
            with self.subTest(X=X), self.assertRaises(error):
                model.fit(X)
        with self.assertRaises(ValueError):
            FuzzyCMeansClusterer(n_clusters=3, init=np.zeros((3, 5))).fit(self.X)

    # --- contract behaviour ----------------------------------------------------

    def test_get_result_before_fit_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            FuzzyCMeansClusterer().get_result()

    def test_failed_refit_clears_previous_result(self) -> None:
        model = FuzzyCMeansClusterer(n_clusters=3, random_state=0).fit(self.X)
        with self.assertRaises(ValueError):
            model.fit(np.zeros(4))
        self.assertIsNone(model.memberships_)
        with self.assertRaises(RuntimeError):
            model.get_result()

    def test_input_is_not_modified(self) -> None:
        X = self.X.copy()
        FuzzyCMeansClusterer(n_clusters=3, random_state=0).fit(X)
        np.testing.assert_array_equal(X, self.X)

    def test_get_result_returns_a_copy(self) -> None:
        model = FuzzyCMeansClusterer(n_clusters=3, random_state=0).fit(self.X)
        result = model.get_result()
        result.memberships[:] = 0.0
        result.labels[:] = 99
        fresh = model.get_result()
        np.testing.assert_allclose(fresh.memberships.sum(axis=1), 1.0)
        self.assertTrue(np.all(fresh.labels < 3))

    def test_fit_predict_matches_result_labels(self) -> None:
        model = FuzzyCMeansClusterer(n_clusters=3, random_state=0)
        np.testing.assert_array_equal(model.fit_predict(self.X), model.get_result().labels)

    def test_metadata_contents(self) -> None:
        meta = FuzzyCMeansClusterer(n_clusters=3, random_state=0).fit(self.X).get_result().metadata
        for key in (
            "parameters", "backend", "n_samples", "n_features", "cluster_sizes",
            "warnings", "random_state_used", "init_method", "stop_criterion", "tol",
            "objective", "objective_history", "change_history", "n_iter", "converged",
            "zero_distance_events", "degenerate_cluster_events", "hard_labels",
            "cluster_centers", "partition_coefficient", "partition_entropy", "xie_beni",
        ):
            self.assertIn(key, meta)
        self.assertEqual(meta["parameters"]["m"], 2.0)
        self.assertEqual(sum(meta["cluster_sizes"].values()), len(self.X))

    def test_signature_and_repr(self) -> None:
        params = list(inspect.signature(FuzzyCMeansClusterer.__init__).parameters)
        self.assertEqual(
            params,
            ["self", "n_clusters", "m", "init", "max_iter", "tol", "stop_criterion", "random_state"],
        )
        self.assertTrue(repr(FuzzyCMeansClusterer()).startswith("FuzzyCMeansClusterer(n_clusters=3"))

    def test_module_uses_no_clustering_library(self) -> None:
        tree = ast.parse(Path(inspect.getfile(fcm_module)).read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        allowed = {
            "__future__", "copy", "math", "secrets", "time", "collections",
            "typing", "numpy", "work3", "iml_project",
        }
        self.assertLessEqual(imported, allowed)


class FuzzyValidityCurveTests(unittest.TestCase):
    def test_indices_pick_true_c_on_blobs(self) -> None:
        X, _ = _three_blobs(n_per_blob=40)
        curve = fuzzy_validity_curve(X, c_values=[2, 3, 4, 5], seeds=[0, 1, 2])
        self.assertEqual(curve["c_values"], [2, 3, 4, 5])
        best_pc = curve["c_values"][int(np.argmax(curve["partition_coefficient_mean"]))]
        best_pe = curve["c_values"][int(np.argmin(curve["partition_entropy_mean"]))]
        best_xb = curve["c_values"][int(np.argmin(curve["xie_beni_mean"]))]
        self.assertEqual((best_pc, best_pe, best_xb), (3, 3, 3))
        for name in ("objective", "partition_coefficient", "partition_entropy", "xie_beni"):
            per_seed = curve[f"{name}_per_seed"]
            self.assertEqual(len(per_seed), 4)
            for row, mean, std in zip(per_seed, curve[f"{name}_mean"], curve[f"{name}_std"]):
                self.assertAlmostEqual(mean, float(np.mean(row)))
                self.assertAlmostEqual(std, float(np.std(row)))  # ddof=0

    def test_single_cluster_xie_beni_is_nan(self) -> None:
        X, _ = _three_blobs(n_per_blob=10)
        curve = fuzzy_validity_curve(X, c_values=[1], seeds=[0, 1])
        self.assertTrue(math.isnan(curve["xie_beni_mean"][0]))
        self.assertTrue(math.isnan(curve["xie_beni_std"][0]))

    def test_invalid_arguments(self) -> None:
        X, _ = _three_blobs(n_per_blob=10)
        with self.assertRaises(ValueError):
            fuzzy_validity_curve(X, c_values=[], seeds=[0])
        with self.assertRaises(TypeError):
            fuzzy_validity_curve(X, c_values=3, seeds=[0])  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            fuzzy_validity_curve(X, c_values=[2], seeds=[0], m=1.0)


if __name__ == "__main__":
    unittest.main()
