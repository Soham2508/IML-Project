from __future__ import annotations

import unittest
import numpy as np

from iml_project.core.knn import KNNClassifier, kNNAlgorithm
from iml_project.core.retention import (
    AlwaysRetain,
    CaseBase,
    DifferentClass,
    NeverRetain,
    OblivionByGoodness,
    get_retention_policy,
)
from iml_project.core.voting import (
    InverseDistanceWeightedVoting,
    MajorityVoting,
    get_voting_strategy,
)
from iml_project.utils.distance_metrics import (
    CanberraDistance,
    ClarkDistance,
    EuclideanDistance,
    HEOMDistance,
    IVDMDistance,
    ManhattanDistance,
    get_distance_metric,
)


class CoreAlgorithmsTests(unittest.TestCase):
    def test_euclidean_and_manhattan_distances(self) -> None:
        """Hand-calculated verification of Euclidean and Manhattan distances."""
        x = np.array([0.0, 3.0])
        y = np.array([4.0, 0.0])

        euclid = EuclideanDistance()
        # sqrt((0-4)^2 + (3-0)^2) = sqrt(16 + 9) = 5.0
        self.assertAlmostEqual(euclid.compute(x, y), 5.0)

        manhattan = ManhattanDistance()
        # |0-4| + |3-0| = 4 + 3 = 7.0
        self.assertAlmostEqual(manhattan.compute(x, y), 7.0)

    def test_canberra_and_clark_distances_with_zero_handling(self) -> None:
        """Hand-calculated verification of Canberra and Clark, including 0/0 -> 0.0."""
        x = np.array([0.0, 2.0, 0.0])
        y = np.array([0.0, 6.0, 0.0])

        # Canberra:
        # term 0: 0/0 = 0.0
        # term 1: |2-6| / (2+6) = 4 / 8 = 0.5
        # term 2: 0/0 = 0.0
        # Total = 0.5
        canberra = CanberraDistance()
        self.assertAlmostEqual(canberra.compute(x, y), 0.5)

        # Clark:
        # term 0: 0/0 = 0.0
        # term 1: ((2-6) / (2+6))^2 = (-4/8)^2 = (-0.5)^2 = 0.25
        # term 2: 0/0 = 0.0
        # sqrt(0.25) = 0.5
        clark = ClarkDistance()
        self.assertAlmostEqual(clark.compute(x, y), 0.5)

    def test_heom_distance_heterogeneous_features(self) -> None:
        """Verify HEOM on mixed numerical and nominal attributes."""
        # Feature 0: numeric (range = 1.0)
        # Feature 1: nominal
        heom = HEOMDistance(numeric_indices=[0], nominal_indices=[1], ranges=[1.0])

        x = np.array([0.2, "red"], dtype=object)
        y_same = np.array([0.5, "red"], dtype=object)
        y_diff = np.array([0.5, "blue"], dtype=object)

        # y_same: numeric diff = |0.2 - 0.5| = 0.3. nominal = 0. sqrt(0.3^2) = 0.3
        self.assertAlmostEqual(heom.compute(x, y_same), 0.3)

        # y_diff: numeric diff = 0.3. nominal = 1.0. sqrt(0.3^2 + 1.0^2) = sqrt(0.09 + 1) = sqrt(1.09)
        self.assertAlmostEqual(heom.compute(x, y_diff), float(np.sqrt(1.09)))

    def test_ivdm_distance_probability_estimation(self) -> None:
        """Verify IVDM fits conditional class probabilities strictly from training data."""
        # 4 train samples, 1 nominal feature, binary class
        # x=0 -> class 0 (twice)
        # x=1 -> class 1 (twice)
        X_train = np.array([[0], [0], [1], [1]], dtype=object)
        y_train = np.array(["neg", "neg", "pos", "pos"])

        ivdm = IVDMDistance(numeric_indices=[], nominal_indices=[0])
        ivdm.fit(X_train, y_train)

        # Distance between 0 and 1:
        # For x=0: P(neg|0) = (2+1)/(2+2) = 3/4 = 0.75, P(pos|0) = (0+1)/(2+2) = 0.25
        # For x=1: P(neg|1) = (0+1)/(2+2) = 0.25, P(pos|1) = (2+1)/(2+2) = 0.75
        # diff_sq = (0.75 - 0.25)^2 + (0.25 - 0.75)^2 = 0.25 + 0.25 = 0.50
        # dist = sqrt(0.50)
        d_01 = ivdm.compute(np.array([0], dtype=object), np.array([1], dtype=object))
        self.assertAlmostEqual(d_01, float(np.sqrt(0.50)))

        # Distance between identical values is 0.0
        d_00 = ivdm.compute(np.array([0], dtype=object), np.array([0], dtype=object))
        self.assertAlmostEqual(d_00, 0.0)

    def test_distance_metric_factory(self) -> None:
        """Verify string factory for all 6 metrics and error on unknown name."""
        for name in ["euclidean", "manhattan", "canberra", "clark", "heom", "ivdm"]:
            metric = get_distance_metric(name, numeric_indices=[0], nominal_indices=[])
            self.assertIsNotNone(metric)

        with self.assertRaises(ValueError):
            get_distance_metric("non_existent_metric")

    def test_majority_and_inverse_distance_voting(self) -> None:
        """Verify majority voting and IDW voting, including zero-distance dominance."""
        classes = ["pos", "pos", "neg"]
        distances = [1.0, 2.0, 0.5]

        # Majority: pos (2 votes) vs neg (1 vote) -> pos
        maj = MajorityVoting()
        self.assertEqual(maj.decide(classes, distances), "pos")

        # IDW with p=1:
        # pos weight = 1/1.0 + 1/2.0 = 1.0 + 0.5 = 1.5
        # neg weight = 1/0.5 = 2.0
        # Winner -> neg!
        idw = InverseDistanceWeightedVoting(p=1.0)
        self.assertEqual(idw.decide(classes, distances), "neg")

        # Zero-distance neighbor test:
        # One neighbor has d=0 -> should strictly dominate despite count
        classes_zero = ["neg", "pos", "pos"]
        distances_zero = [0.0, 1.0, 1.0]
        self.assertEqual(idw.decide(classes_zero, distances_zero), "neg")

    def test_voting_tie_breaking_by_1nn_and_priors(self) -> None:
        """Verify deterministic tie-breaking: nearest single neighbor (1-NN), then prior."""
        maj = MajorityVoting()
        classes = ["pos", "neg"]
        # Tied votes (1 each), but pos is nearer (d=0.2 < d=0.8)
        self.assertEqual(maj.decide(classes, [0.2, 0.8]), "pos")
        # Tied votes and equidistant (d=0.5, d=0.5), prior breaks tie
        priors = {"pos": 0.3, "neg": 0.7}
        self.assertEqual(maj.decide(classes, [0.5, 0.5], class_priors=priors), "neg")

    def test_knn_classifier_fit_and_predict(self) -> None:
        """Verify end-to-end KNN neighbor retrieval and classification."""
        X_train = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]])
        y_train = ["A", "A", "B"]

        knn = KNNClassifier(
            k=2,
            distance_metric=EuclideanDistance(),
            voting_strategy=MajorityVoting(),
        )
        knn.fit(X_train, y_train)

        pred, idxs, dists = knn.predict_one(np.array([0.1, 0.1]))
        self.assertEqual(pred, "A")
        self.assertEqual(idxs, [0, 1])

    def test_retention_policies_sequencing_and_effects(self) -> None:
        """Verify NR, AR, DC, and OG retention behaviors and strict update sequencing."""
        X_train = np.array([[0.0], [10.0]])
        y_train = ["classA", "classB"]

        # 1. Never Retain: case base stays length 2
        knn_nr = KNNClassifier(
            k=1,
            distance_metric=EuclideanDistance(),
            retention_policy=NeverRetain(),
        )
        knn_nr.fit(X_train, y_train)
        res_nr = knn_nr.evaluate(np.array([[1.0], [9.0]]), ["classA", "classB"])
        self.assertEqual(res_nr["final_size"], 2)

        # 2. Always Retain: case base grows after each query
        knn_ar = KNNClassifier(
            k=1,
            distance_metric=EuclideanDistance(),
            retention_policy=AlwaysRetain(),
        )
        knn_ar.fit(X_train, y_train)
        res_ar = knn_ar.evaluate(np.array([[1.0], [2.0]]), ["classA", "classA"])
        self.assertEqual(res_ar["final_size"], 4)

        # 3. Different Class: grows only on incorrect predictions
        knn_dc = KNNClassifier(
            k=1,
            distance_metric=EuclideanDistance(),
            retention_policy=DifferentClass(),
        )
        knn_dc.fit(X_train, y_train)
        # Query 1.0 is near 0.0 -> predicts classA. True label is classB (incorrect -> retained)
        # Query 9.0 is near 10.0 -> predicts classB. True label is classB (correct -> NOT retained)
        res_dc = knn_dc.evaluate(np.array([[1.0], [9.0]]), ["classB", "classB"])
        self.assertEqual(res_dc["final_size"], 3)

        # 4. Oblivion by Goodness:
        # Initial G_0 = 0.5.
        # Query nearest to neighbor 0.
        # If neighbor 0 predicts incorrectly (r=0):
        # G_1 = 0.5 + 0.2*(0 - 0.5) = 0.4 < 0.5 -> Neighbor 0 must be removed!
        knn_og = KNNClassifier(
            k=1,
            distance_metric=EuclideanDistance(),
            retention_policy=OblivionByGoodness(alpha=0.2, initial_goodness=0.5),
        )
        knn_og.fit(X_train, y_train)
        self.assertEqual(len(knn_og.case_base), 2)
        # Query at 0.1: retrieved neighbor is index 0 ("classA").
        # True label is "classB" (incorrect). Reward r=0. Neighbor 0 dropped!
        knn_og.predict_and_update(np.array([0.1]), "classB")
        self.assertEqual(len(knn_og.case_base), 1)
        # The remaining neighbor is index 1 ("classB" at 10.0)
        self.assertEqual(knn_og.case_base.y[0], "classB")


if __name__ == "__main__":
    unittest.main()
