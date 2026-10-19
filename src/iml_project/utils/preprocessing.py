from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

import numpy as np
import pandas as pd


@dataclass
class Preprocessor:
    """Train-only normalization and imputation for ARFF features."""

    feature_names: list[str]
    feature_types: list[str]
    target_name: str
    mins_: dict[str, float]
    maxs_: dict[str, float]
    medians_: dict[str, float]
    modes_: dict[str, object]

    @classmethod
    def fit(
        cls,
        train_df: pd.DataFrame,
        feature_names: Iterable[str],
        feature_types: Iterable[str],
        target_name: str,
    ) -> "Preprocessor":
        names = list(feature_names)
        types = [str(item).lower() for item in feature_types]
        if len(names) != len(types):
            raise ValueError("Feature names and feature types lengths do not match.")

        numeric_cols = [
            name
            for name, feature_type in zip(names, types)
            if name != target_name and feature_type in {"real", "numeric", "integer"}
        ]
        nominal_cols = [
            name
            for name, feature_type in zip(names, types)
            if name != target_name and feature_type in {"nominal", "string"}
        ]

        medians: dict[str, float] = {}
        for col in numeric_cols:
            series = train_df[col].replace({None: np.nan})
            medians[col] = float(series.median()) if not series.empty else 0.0

        modes: dict[str, object] = {}
        for col in nominal_cols:
            series = train_df[col].replace({None: np.nan})
            non_missing = series.dropna()
            modes[col] = non_missing.mode().iloc[0] if not non_missing.empty else None

        mins: dict[str, float] = {}
        maxs: dict[str, float] = {}
        for col in numeric_cols:
            series = train_df[col].replace({None: np.nan})
            min_value = float(series.min()) if not series.empty else 0.0
            max_value = float(series.max()) if not series.empty else 0.0
            mins[col] = min_value
            maxs[col] = max_value

        return cls(
            feature_names=names,
            feature_types=types,
            target_name=target_name,
            mins_=mins,
            maxs_=maxs,
            medians_=medians,
            modes_=modes,
        )

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        transformed = frame.copy()
        names = list(self.feature_names)
        types = list(self.feature_types)

        for name, feature_type in zip(names, types):
            if name == self.target_name:
                continue

            if feature_type in {"real", "numeric", "integer"}:
                if name not in transformed.columns:
                    continue
                series = transformed[name].replace({None: np.nan})
                if name in self.medians_:
                    series = series.fillna(self.medians_[name])
                if name in self.mins_ and name in self.maxs_:
                    minimum = self.mins_[name]
                    maximum = self.maxs_[name]
                    if np.isclose(minimum, maximum):
                        transformed[name] = 0.0
                    else:
                        transformed[name] = (series - minimum) / (maximum - minimum)
                else:
                    transformed[name] = series
                transformed[name] = pd.to_numeric(
                    transformed[name], errors="coerce"
                ).fillna(0.0)
            elif feature_type in {"nominal", "string"}:
                if name not in transformed.columns:
                    continue
                series = transformed[name].replace({None: np.nan})
                if name in self.modes_ and self.modes_[name] is not None:
                    series = series.fillna(self.modes_[name])
                transformed[name] = series

        return transformed


def fit_and_transform(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_names: list[str],
    feature_types: list[str],
    target_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, Preprocessor]:
    preprocessor = Preprocessor.fit(train_df, feature_names, feature_types, target_name)
    transformed_train = preprocessor.transform(train_df)
    transformed_test = preprocessor.transform(test_df)
    return transformed_train, transformed_test, preprocessor
