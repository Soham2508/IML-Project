from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def aggregate_experiment_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-fold results across the 10 predefined folds."""
    group_cols = [
        "dataset",
        "distance_metric",
        "k",
        "voting_strategy",
        "retention_policy",
    ]

    agg_df = (
        results_df.groupby(group_cols)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_efficiency_ms=("mean_prediction_time", lambda x: np.mean(x) * 1000.0),
            std_efficiency_ms=("mean_prediction_time", lambda x: np.std(x) * 1000.0),
            mean_retention_ratio=("retention_ratio", "mean"),
            std_retention_ratio=("retention_ratio", "std"),
            total_folds=("fold", "count"),
        )
        .reset_index()
    )

    agg_df = agg_df.sort_values(
        by=["dataset", "mean_accuracy", "mean_efficiency_ms"],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    return agg_df


def find_best_configurations(agg_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Return top N performing configurations for each dataset."""
    best_list = []
    for ds in agg_df["dataset"].unique():
        ds_subset = agg_df[agg_df["dataset"] == ds].copy()
        best_list.append(ds_subset.head(top_n))
    return pd.concat(best_list, ignore_index=True)


def compute_paired_ttest(
    results_df: pd.DataFrame,
    cfg_a: dict[str, Any],
    cfg_b: dict[str, Any],
    dataset: str,
) -> dict[str, Any]:
    """Perform a paired t-test between two configurations across the 10 predefined folds."""
    def _filter(cfg: dict[str, Any]) -> pd.Series:
        mask = results_df["dataset"] == dataset
        for k, v in cfg.items():
            mask = mask & (results_df[k] == v)
        sub = results_df[mask].sort_values("fold")
        return sub["accuracy"].to_numpy()

    acc_a = _filter(cfg_a)
    acc_b = _filter(cfg_b)

    if len(acc_a) != 10 or len(acc_b) != 10:
        raise ValueError(
            f"Expected exactly 10 folds for paired comparison, got {len(acc_a)} and {len(acc_b)}"
        )

    diff = acc_a - acc_b
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))

    # Standard paired t-test
    if np.isclose(std_diff, 0.0):
        if np.isclose(mean_diff, 0.0):
            t_stat = 0.0
            p_val = 1.0
        else:
            t_stat = float("inf") if mean_diff > 0 else float("-inf")
            p_val = 0.0
    else:
        res = stats.ttest_rel(acc_a, acc_b)
        t_stat = float(res.statistic)
        p_val = float(res.pvalue)

    return {
        "dataset": dataset,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "significant_005": bool(p_val < 0.05),
        "cfg_a": cfg_a,
        "cfg_b": cfg_b,
    }


def export_evaluation_report(
    results_df: pd.DataFrame,
    output_dir: str | Path = "results",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Export summary tables and top configurations to CSV and Markdown."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    agg_df = aggregate_experiment_results(results_df)
    agg_df.to_csv(out_path / "aggregated_results.csv", index=False)

    best_df = find_best_configurations(agg_df, top_n=5)
    best_df.to_csv(out_path / "best_configurations.csv", index=False)

    return agg_df, best_df
