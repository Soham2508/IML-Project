from __future__ import annotations

import unittest

import pandas as pd

from iml_project.utils.arff_reader import (
    discover_fold_pairs,
    load_dataset_fold,
    resolve_dataset_dir,
    validate_fold_pairs,
)
from iml_project.utils.preprocessing import (
    Preprocessor,
    fit_and_transform,
    infer_feature_type,
)


class DataPipelineTests(unittest.TestCase):
    def test_discover_and_validate_predefined_folds(self) -> None:
        """Verify that predefined folds for grid and heart-statlog are discovered and validated."""
        groups = discover_fold_pairs("data")
        validate_fold_pairs(groups)
        self.assertIn("grid", groups)
        self.assertIn("heart-statlog", groups)
        self.assertEqual(len(groups["grid"]), 10)
        self.assertEqual(len(groups["heart-statlog"]), 10)

    def test_dataset_alias_resolution(self) -> None:
        """Verify dataset directory alias resolution (tao-grid -> grid)."""
        resolved_dir, canonical = resolve_dataset_dir("tao-grid", "data")
        self.assertEqual(canonical, "grid")
        self.assertTrue(resolved_dir.is_dir())

        resolved_dir2, canonical2 = resolve_dataset_dir("heart_statlog", "data")
        self.assertEqual(canonical2, "heart-statlog")
        self.assertTrue(resolved_dir2.is_dir())

    def test_column_alignment_and_target_isolation_across_all_folds(self) -> None:
        """Verify schema alignment and target name consistency across all 10 folds."""
        for ds in ["grid", "heart-statlog"]:
            for fold in range(10):
                train_df, test_df, _names, _types, target = load_dataset_fold(
                    ds, fold, "data"
                )
                self.assertEqual(target, "class")
                self.assertEqual(list(train_df.columns), list(test_df.columns))
                self.assertEqual(train_df.columns[-1], "class")
                self.assertEqual(test_df.columns[-1], "class")
                self.assertFalse(train_df.empty)
                self.assertFalse(test_df.empty)

    def test_feature_type_inference_rules(self) -> None:
        """Test deterministic inference across numeric, binary, categorical, and overrides."""
        # Continuous fractional -> numeric
        s_cont = pd.Series([1.2, 3.4, 5.6])
        t_cont, r_cont = infer_feature_type(s_cont)
        self.assertEqual(t_cont, "numeric")
        self.assertIn("fractional", r_cont)

        # Binary indicator -> nominal
        s_bin = pd.Series([0.0, 1.0, 0.0, 1.0])
        t_bin, r_bin = infer_feature_type(s_bin)
        self.assertEqual(t_bin, "nominal")
        self.assertIn("binary", r_bin)

        # String nominal -> nominal
        s_str = pd.Series(["cat", "dog", "cat"])
        t_str, _r_str = infer_feature_type(s_str)
        self.assertEqual(t_str, "nominal")

        # Explicit override
        s_ovr = pd.Series([1.2, 3.4, 5.6])
        t_ovr, r_ovr = infer_feature_type(s_ovr, override="nominal")
        self.assertEqual(t_ovr, "nominal")
        self.assertEqual(r_ovr, "explicit override")

    def test_no_data_leakage_between_train_and_test_inference(self) -> None:
        """Verify that test values cannot alter training inference statistics."""
        train_df = pd.DataFrame(
            {
                "feat1": [10.0, 20.0, 30.0],
                "feat2": [0.0, 1.0, 0.0],
                "target": ["A", "B", "A"],
            }
        )
        # Test fold has extreme values and unseen category
        test_df = pd.DataFrame(
            {
                "feat1": [1000.0, -500.0],
                "feat2": [1.0, 0.0],
                "target": ["B", "A"],
            }
        )

        prep = Preprocessor.fit(
            train_df,
            feature_names=["feat1", "feat2", "target"],
            feature_types=["real", "real", "nominal"],
            target_name="target",
            encode_categorical=False,
        )

        # Inferred min/max must be from train fold only
        self.assertEqual(prep.mins_["feat1"], 10.0)
        self.assertEqual(prep.maxs_["feat1"], 30.0)

        # Transformation on test applies train scaling
        transformed_test = prep.transform(test_df)
        # 1000.0 scaled with train min=10, max=30 is (1000-10)/(30-10) = 990/20 = 49.5
        self.assertAlmostEqual(transformed_test.loc[0, "feat1"], 49.5)
        # Target column is preserved without transformation
        self.assertEqual(transformed_test["target"].tolist(), ["B", "A"])

    def test_constant_feature_handling(self) -> None:
        """Verify that constant numeric features (min == max) avoid division by zero."""
        train_df = pd.DataFrame(
            {
                "constant_feat": [5.0, 5.0, 5.0],
                "label": [0, 1, 0],
            }
        )
        test_df = pd.DataFrame(
            {
                "constant_feat": [5.0, 10.0],
                "label": [1, 0],
            }
        )

        transformed_train, transformed_test, _prep = fit_and_transform(
            train_df,
            test_df,
            ["constant_feat", "label"],
            ["numeric", "nominal"],
            "label",
            encode_categorical=False,
        )

        self.assertEqual(transformed_train["constant_feat"].tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(transformed_test["constant_feat"].tolist(), [0.0, 0.0])

    def test_encode_categorical_flag_preserves_raw_categories_for_heom(self) -> None:
        """Verify that encode_categorical=False preserves intact nominal columns."""
        train_df = pd.DataFrame(
            {
                "num": [10.0, 20.0, 30.0],
                "cat": ["low", "medium", "high"],
                "class": ["yes", "no", "yes"],
            }
        )
        test_df = pd.DataFrame(
            {
                "num": [15.0, 25.0],
                "cat": ["low", "unseen_val"],
                "class": ["yes", "no"],
            }
        )

        tr_norm, te_norm, _prep = fit_and_transform(
            train_df,
            test_df,
            ["num", "cat", "class"],
            ["real", "nominal", "nominal"],
            "class",
            encode_categorical=False,
        )

        # Numeric normalized to [0, 1]
        self.assertEqual(tr_norm["num"].tolist(), [0.0, 0.5, 1.0])
        self.assertEqual(te_norm["num"].tolist(), [0.25, 0.75])

        # Nominal preserved as-is without one-hot expansion
        self.assertEqual(tr_norm["cat"].tolist(), ["low", "medium", "high"])
        self.assertEqual(te_norm["cat"].tolist(), ["low", "unseen_val"])
        self.assertIn("cat", tr_norm.columns)
        self.assertNotIn("cat_low", tr_norm.columns)


if __name__ == "__main__":
    unittest.main()
