from __future__ import annotations

import unittest

import pandas as pd

from iml_project.utils.preprocessing import fit_and_transform


class PreprocessingTests(unittest.TestCase):
    def test_infers_feature_kinds_and_transforms_unseen_categories(self) -> None:
        train = pd.DataFrame(
            {
                "measurement": [10.0, 20.0, 30.0],
                "color": ["red", "blue", "red"],
                "label": [0, 1, 0],
            }
        )
        test = pd.DataFrame(
            {
                "measurement": [15.0, 40.0],
                "color": ["green", "blue"],
                "label": [1, 0],
            }
        )

        transformed_train, transformed_test, preprocessor = fit_and_transform(
            train,
            test,
            ["measurement", "color", "label"],
            ["nominal", "real", "numeric"],
            "label",
        )

        self.assertEqual(preprocessor.numeric_features_, ["measurement"])
        self.assertEqual(preprocessor.categorical_features_, ["color"])
        self.assertEqual(
            set(transformed_train.columns),
            {"measurement", "label", "color_blue", "color_red"},
        )
        self.assertEqual(transformed_train["measurement"].tolist(), [0.0, 0.5, 1.0])
        self.assertEqual(transformed_test["measurement"].tolist(), [0.25, 1.5])
        self.assertEqual(transformed_test.loc[0, "color_blue"], 0.0)
        self.assertEqual(transformed_test.loc[0, "color_red"], 0.0)
        self.assertEqual(transformed_test.loc[1, "color_blue"], 1.0)
        self.assertEqual(transformed_test["label"].tolist(), [1, 0])

    def test_imputes_numeric_and_categorical_missing_values(self) -> None:
        train = pd.DataFrame(
            {
                "measurement": [2.0, None, 6.0],
                "group": ["a", None, "a"],
                "label": ["yes", "no", "yes"],
            }
        )

        transformed_train, _, _ = fit_and_transform(
            train,
            train,
            ["measurement", "group", "label"],
            ["numeric", "string", "nominal"],
            "label",
        )

        self.assertEqual(transformed_train["measurement"].tolist(), [0.0, 0.5, 1.0])
        self.assertEqual(transformed_train["group_a"].tolist(), [1.0, 1.0, 1.0])
        self.assertEqual(transformed_train["label"].tolist(), ["yes", "no", "yes"])


if __name__ == "__main__":
    unittest.main()
