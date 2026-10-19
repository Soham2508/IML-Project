from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def infer_feature_type(
    series: pd.Series,
    declared_type: str | None = None,
    col_name: str | None = None,
    override: str | None = None,
) -> tuple[str, str]:
    """Deterministically infer whether a predictor is 'numeric' or 'nominal' strictly from training data.

    Returns (inferred_type, rationale).
    """
    if override is not None:
        ov = override.lower()
        if ov in {"numeric", "numerical", "continuous", "real"}:
            return "numeric", "explicit override"
        if ov in {"nominal", "categorical", "discrete"}:
            return "nominal", "explicit override"
        return ov, "explicit override"

    non_null = series.replace({None: np.nan}).dropna()
    if len(non_null) == 0:
        return "numeric", "all missing values, defaulting to numeric"

    # Boolean columns are nominal
    if pd.api.types.is_bool_dtype(series.dtype):
        return "nominal", "boolean dtype"

    # Check for categorical or string dtypes
    is_cat = isinstance(series.dtype, pd.CategoricalDtype)
    if (
        pd.api.types.is_object_dtype(series.dtype)
        or is_cat
        or pd.api.types.is_string_dtype(series.dtype)
    ):
        numeric_series = pd.to_numeric(non_null, errors="coerce")
        if numeric_series.isna().any():
            return "nominal", "non-numeric string/object values"
        non_null = numeric_series

    numeric_series = pd.to_numeric(non_null, errors="coerce")
    if numeric_series.isna().any():
        return "nominal", "contains unparseable non-numeric values"

    # Continuous fractional values (e.g. 0.1, 1.25, -3.75) are numeric
    is_all_integers = np.all(np.isclose(numeric_series, np.round(numeric_series)))
    if not is_all_integers:
        return "numeric", "contains continuous fractional values"

    # All observed values are integer-valued
    unique_vals = np.unique(numeric_series)
    num_unique = len(unique_vals)
    val_set = set(unique_vals)
    val_range = float(numeric_series.max() - numeric_series.min())
    n_samples = len(numeric_series)

    # Strictly binary flags with standard {0, 1} encoding
    if val_set.issubset({0.0, 1.0}):
        return "nominal", f"binary 0/1 indicator (cardinality={num_unique})"

    # Low-cardinality discrete integer codes in non-trivial datasets
    if (
        n_samples >= 20
        and num_unique <= 6
        and val_range <= 10.0
        and (num_unique / n_samples < 0.05)
    ):
        return (
            "nominal",
            f"discrete low-cardinality integer code (cardinality={num_unique}, range={val_range:.1f})",
        )

    # High-cardinality or small-sample integer measurements
    return "numeric", f"continuous integer measurement/count (cardinality={num_unique})"


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
    inferred_types_: dict[str, str] = field(default_factory=dict)
    inference_provenance_: dict[str, str] = field(default_factory=dict)
    encode_categorical_: bool = True

    @classmethod
    def fit(
        cls,
        train_df: pd.DataFrame,
        feature_names: Iterable[str],
        feature_types: Iterable[str],
        target_name: str,
        encode_categorical: bool = True,
        feature_type_overrides: dict[str, str] | None = None,
    ) -> Preprocessor:
        names = list(feature_names)
        types = [str(item).lower() for item in feature_types]
        if len(names) != len(types):
            raise ValueError("Feature names and feature types lengths do not match.")

        overrides = feature_type_overrides or {}
        # input_features = [name for name in names if name != target_name]

        numeric_features: list[str] = []
        categorical_features: list[str] = []
        inferred_types: dict[str, str] = {}
        inference_provenance: dict[str, str] = {}

        for name, decl_type in zip(names, types):
            if name == target_name:
                continue

            override = overrides.get(name)
            inferred, reason = infer_feature_type(
                train_df[name],
                declared_type=decl_type,
                col_name=name,
                override=override,
            )
            inferred_types[name] = inferred
            inference_provenance[name] = reason

            if inferred == "numeric":
                numeric_features.append(name)
            else:
                categorical_features.append(name)

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
        if encode_categorical and categorical_features:
            categorical_data = train_df[categorical_features].copy()
            for col in categorical_features:
                if modes[col] is not None:
                    categorical_data[col] = categorical_data[col].fillna(modes[col])
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
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
            inferred_types_=inferred_types,
            inference_provenance_=inference_provenance,
            encode_categorical_=encode_categorical,
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
            name for name in self.categorical_features_ if name in transformed.columns
        ]

        if present_categorical:
            for col in present_categorical:
                if self.modes_[col] is not None:
                    transformed[col] = transformed[col].fillna(self.modes_[col])

            if self.encode_categorical_:
                if self.encoder_ is None:
                    raise RuntimeError("Categorical features have no fitted encoder.")
                cat_subset = transformed[self.categorical_features_].copy()
                encoded = self.encoder_.transform(cat_subset)
                encoded_names = list(
                    self.encoder_.get_feature_names_out(self.categorical_features_)
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
    encode_categorical: bool = True,
    feature_type_overrides: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Preprocessor]:
    preprocessor = Preprocessor.fit(
        train_df,
        feature_names,
        feature_types,
        target_name,
        encode_categorical=encode_categorical,
        feature_type_overrides=feature_type_overrides,
    )
    transformed_train = preprocessor.transform(train_df)
    transformed_test = preprocessor.transform(test_df)
    return transformed_train, transformed_test, preprocessor
