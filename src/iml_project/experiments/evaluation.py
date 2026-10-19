from __future__ import annotations

import re
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


def get_next_experiment_dir(base_results_dir: str | Path = "results") -> Path:
    """Find the next experiment directory: {base_results_dir}/experiment_results_{N}."""
    base = Path(base_results_dir)
    base.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(r"^experiment_results_(\d+)$")
    highest_num = 0
    for item in base.iterdir():
        if item.is_dir():
            match = pattern.match(item.name)
            if match:
                num = int(match.group(1))
                highest_num = max(highest_num, num)

    next_num = highest_num + 1
    next_dir = base / f"experiment_results_{next_num}"
    next_dir.mkdir(parents=True, exist_ok=True)
    return next_dir


def generate_statistical_comparisons(
    results_df: pd.DataFrame,
    output_dir: str | Path,
) -> pd.DataFrame:
    """Compute pairwise paired t-tests comparing metrics, voting, retention, and best vs baseline."""
    out_path = Path(output_dir)
    comparisons: list[dict[str, Any]] = []

    for ds in results_df["dataset"].unique():
        base_cfg = {
            "distance_metric": "euclidean",
            "k": 3,
            "voting_strategy": "majority",
            "retention_policy": "nr",
        }

        ds_subset = results_df[results_df["dataset"] == ds]
        agg_ds = aggregate_experiment_results(ds_subset)
        if not agg_ds.empty:
            best_row = agg_ds.iloc[0]
            best_cfg = {
                "distance_metric": best_row["distance_metric"],
                "k": int(best_row["k"]),
                "voting_strategy": best_row["voting_strategy"],
                "retention_policy": best_row["retention_policy"],
            }
            try:
                t_res = compute_paired_ttest(results_df, best_cfg, base_cfg, ds)
                comparisons.append(
                    {
                        "comparison": "Best_vs_Baseline",
                        "dataset": ds,
                        "mean_diff": t_res["mean_diff"],
                        "std_diff": t_res["std_diff"],
                        "t_stat": t_res["t_statistic"],
                        "p_value": t_res["p_value"],
                        "sig_005": t_res["significant_005"],
                        "cfg_a": str(best_cfg),
                        "cfg_b": str(base_cfg),
                    }
                )
            except (ValueError, KeyError):
                continue

        for m in ["manhattan", "canberra", "clark", "heom", "ivdm"]:
            comp_cfg = {
                "distance_metric": m,
                "k": 3,
                "voting_strategy": "majority",
                "retention_policy": "nr",
            }
            try:
                t_m = compute_paired_ttest(results_df, comp_cfg, base_cfg, ds)
                comparisons.append(
                    {
                        "comparison": f"{m.capitalize()}_vs_Euclidean",
                        "dataset": ds,
                        "mean_diff": t_m["mean_diff"],
                        "std_diff": t_m["std_diff"],
                        "t_stat": t_m["t_statistic"],
                        "p_value": t_m["p_value"],
                        "sig_005": t_m["significant_005"],
                        "cfg_a": str(comp_cfg),
                        "cfg_b": str(base_cfg),
                    }
                )
            except (ValueError, KeyError):
                continue

        cfg_idw = {
            "distance_metric": "euclidean",
            "k": 3,
            "voting_strategy": "inverse_distance",
            "retention_policy": "nr",
        }
        try:
            t_v = compute_paired_ttest(results_df, cfg_idw, base_cfg, ds)
            comparisons.append(
                {
                    "comparison": "IDW_vs_Majority",
                    "dataset": ds,
                    "mean_diff": t_v["mean_diff"],
                    "std_diff": t_v["std_diff"],
                    "t_stat": t_v["t_statistic"],
                    "p_value": t_v["p_value"],
                    "sig_005": t_v["significant_005"],
                    "cfg_a": str(cfg_idw),
                    "cfg_b": str(base_cfg),
                }
            )
        except (ValueError, KeyError):
            continue

        for r in ["ar", "dc", "og"]:
            cfg_r = {
                "distance_metric": "euclidean",
                "k": 3,
                "voting_strategy": "majority",
                "retention_policy": r,
            }
            try:
                t_r = compute_paired_ttest(results_df, cfg_r, base_cfg, ds)
                comparisons.append(
                    {
                        "comparison": f"{r.upper()}_vs_NR",
                        "dataset": ds,
                        "mean_diff": t_r["mean_diff"],
                        "std_diff": t_r["std_diff"],
                        "t_stat": t_r["t_statistic"],
                        "p_value": t_r["p_value"],
                        "sig_005": t_r["significant_005"],
                        "cfg_a": str(cfg_r),
                        "cfg_b": str(base_cfg),
                    }
                )
            except (ValueError, KeyError):
                continue

    comp_df = pd.DataFrame(comparisons)
    comp_df.to_csv(out_path / "statistical_comparisons.csv", index=False)
    return comp_df


def export_evaluation_report(
    results_df: pd.DataFrame,
    output_dir: str | Path = "results",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Export summary tables, top configurations, and statistical comparisons."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    agg_df = aggregate_experiment_results(results_df)
    agg_df.to_csv(out_path / "aggregated_results.csv", index=False)

    best_df = find_best_configurations(agg_df, top_n=5)
    best_df.to_csv(out_path / "best_configurations.csv", index=False)

    stat_df = generate_statistical_comparisons(results_df, out_path)

    return agg_df, best_df, stat_df
