from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


@dataclass
class Preprocessor:
    """Train-only normalization, imputation, and categorical encoding."""

    feature_names: list[str]
    feature_types: list[str]
    target_name: str
    numeric_features_: list[str]
    categorical_features_: list[str]
    mins_: dict[str, float]
    maxs_: dict[str, float]
    medians_: dict[str, float]
    modes_: dict[str, object]
    encoder_: OneHotEncoder | None

    @classmethod
    def fit(
        cls,
        train_df: pd.DataFrame,
        feature_names: Iterable[str],
        feature_types: Iterable[str],
        target_name: str,
    ) -> Preprocessor:
        names = list(feature_names)
        types = [str(item).lower() for item in feature_types]
        if len(names) != len(types):
            raise ValueError("Feature names and feature types lengths do not match.")

        input_features = [name for name in names if name != target_name]
        numeric_features = [
            name
            for name in input_features
            if pd.api.types.is_numeric_dtype(train_df[name].dtype)
            and not pd.api.types.is_bool_dtype(train_df[name].dtype)
        ]
        categorical_features = [
            name for name in input_features if name not in numeric_features
        ]

        medians: dict[str, float] = {}
        mins: dict[str, float] = {}
        maxs: dict[str, float] = {}
        for col in numeric_features:
            series = pd.to_numeric(
                train_df[col].replace({None: np.nan}), errors="coerce"
            )
            median = series.median()
            observed = series.dropna()
            medians[col] = float(median) if pd.notna(median) else 0.0
            mins[col] = float(observed.min()) if not observed.empty else 0.0
            maxs[col] = float(observed.max()) if not observed.empty else 0.0

        modes: dict[str, object] = {}
        for col in categorical_features:
            non_missing = train_df[col].replace({None: np.nan}).dropna()
            modes[col] = non_missing.mode().iloc[0] if not non_missing.empty else None

        encoder: OneHotEncoder | None = None
        if categorical_features:
            categorical_data = train_df[categorical_features].copy()
            for col in categorical_features:
                if modes[col] is not None:
                    categorical_data[col] = categorical_data[col].fillna(modes[col])
            encoder = OneHotEncoder(handle_unknown="ignore")
            encoder.fit(categorical_data)

        return cls(
            feature_names=names,
            feature_types=types,
            target_name=target_name,
            numeric_features_=numeric_features,
            categorical_features_=categorical_features,
            mins_=mins,
            maxs_=maxs,
            medians_=medians,
            modes_=modes,
            encoder_=encoder,
        )

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        transformed = frame.copy()

        for name in self.numeric_features_:
            if name not in transformed.columns:
                continue
            series = pd.to_numeric(
                transformed[name].replace({None: np.nan}), errors="coerce"
            ).fillna(self.medians_[name])
            minimum = self.mins_[name]
            maximum = self.maxs_[name]
            if np.isclose(minimum, maximum):
                transformed[name] = 0.0
            else:
                transformed[name] = (series - minimum) / (maximum - minimum)
            transformed[name] = pd.to_numeric(
                transformed[name], errors="coerce"
            ).fillna(0.0)

        present_categorical = [
            name
            for name in self.categorical_features_
            if name in transformed.columns
        ]
        if present_categorical:
            categorical_data = transformed[present_categorical].copy()
            for col in present_categorical:
                if self.modes_[col] is not None:
                    categorical_data[col] = categorical_data[col].fillna(
                        self.modes_[col]
                    )

            if self.encoder_ is None:
                raise RuntimeError("Categorical features have no fitted encoder.")
            encoded = self.encoder_.transform(
                categorical_data.reindex(columns=self.categorical_features_)
            ).toarray()
            encoded_names = self.encoder_.get_feature_names_out(
                self.categorical_features_
            )
            encoded_frame = pd.DataFrame(
                encoded,
                index=transformed.index,
                columns=encoded_names,
            )
            transformed = transformed.drop(columns=present_categorical)
            transformed = pd.concat([transformed, encoded_frame], axis=1)

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
