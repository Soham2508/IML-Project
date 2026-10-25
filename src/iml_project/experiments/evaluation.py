from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from iml_project.utils.arff_reader import resolve_dataset_dir


# ==============================================================================
# Fold Aggregation & Descriptive Ranking (Section 3)
# ==============================================================================

def aggregate_experiment_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-fold results across the predefined folds for each dataset and configuration.

    Follows Section 3 of the evaluation plan:
    - Section 3.1: Pooled out-of-fold accuracy:
        A[c, d] = sum(correct[c, d, i]) / sum(test_size[d, i])
    - Section 3.2: Efficiency / Execution time:
        mean_efficiency_ms = sum(total_prediction_time) / sum(test_size) * 1000.0
    - Section 3.3: Storage / retention ratio:
        Q[c, d] = sum(retained_size) / sum(training_size_before_reduction)
      (100% / 1.0 for models operating on the full training set without reduction).
    - Descriptive within-dataset rankings:
      Accuracy: descending (rank 1 is best).
      Efficiency: ascending runtime (rank 1 is best).
      Retention: ascending retention ratio (rank 1 is best).
      Tied scores receive average ranks.
    """
    if results_df.empty:
        return pd.DataFrame()

    df = results_df.copy()

    # Determine grouping columns: dataset + config_id or all hyperparameters
    potential_meta = [
        "config_id",
        "model",
        "model_family",
        "distance_metric",
        "k",
        "voting_strategy",
        "retention_policy",
        "kernel",
        "C",
        "gamma",
        "n_estimators",
        "criterion",
        "max_depth",
        "reduction_method",
        "reduction_distance",
    ]
    group_cols = ["dataset"] + [c for c in potential_meta if c in df.columns]

    def _agg_group(group: pd.DataFrame) -> pd.Series:
        mean_acc = float(group["accuracy"].mean()) if "accuracy" in group.columns else 0.0
        std_acc = float(group["accuracy"].std(ddof=1)) if "accuracy" in group.columns and len(group) > 1 else 0.0

        if "total" in group.columns:
            total_test = int(group["total"].sum())
        elif "correct" in group.columns and "incorrect" in group.columns:
            total_test = int(group["incorrect"].sum() + group["correct"].sum())
        else:
            total_test = len(group)

        if "correct" in group.columns:
            correct = int(group["correct"].sum())
            pooled_acc = float(correct / total_test) if total_test > 0 else 0.0
        elif "total" in group.columns and "accuracy" in group.columns:
            correct = int(np.round((group["accuracy"] * group["total"]).sum()))
            pooled_acc = float(correct / total_test) if total_test > 0 else 0.0
        else:
            correct = int(np.round(mean_acc * total_test))
            pooled_acc = mean_acc

        # Efficiency: problem solving time per test instance in milliseconds
        if "total_prediction_time" in group.columns and total_test > 0:
            mean_eff_ms = float((group["total_prediction_time"].sum() / total_test) * 1000.0)
        elif "mean_prediction_time" in group.columns:
            mean_eff_ms = float(group["mean_prediction_time"].mean() * 1000.0)
        else:
            mean_eff_ms = 0.0

        std_eff_ms = (
            float((group["mean_prediction_time"] * 1000.0).std(ddof=1))
            if "mean_prediction_time" in group.columns and len(group) > 1
            else 0.0
        )

        mean_fit_s = (
            float(group["fit_time"].mean())
            if "fit_time" in group.columns
            else 0.0
        )

        # Storage / retention ratio: counts aggregation (Section 3.3)
        train_before_col = (
            "training_size_before_reduction"
            if "training_size_before_reduction" in group.columns
            else ("initial_size" if "initial_size" in group.columns else None)
        )
        retained_col = (
            "retained_size"
            if "retained_size" in group.columns
            else ("final_size" if "final_size" in group.columns else None)
        )

        if train_before_col and retained_col:
            tot_before = float(group[train_before_col].sum())
            tot_retained = float(group[retained_col].sum())
            ret_ratio = float(tot_retained / tot_before) if tot_before > 0 else 1.0
        elif "retention_ratio" in group.columns:
            ret_ratio = float(group["retention_ratio"].mean())
        else:
            ret_ratio = 1.0

        mean_ret_ratio = (
            float(group["retention_ratio"].mean())
            if "retention_ratio" in group.columns
            else ret_ratio
        )
        std_ret_ratio = (
            float(group["retention_ratio"].std(ddof=1))
            if "retention_ratio" in group.columns and len(group) > 1
            else 0.0
        )

        row_dict: dict[str, Any] = {
            "pooled_accuracy": pooled_acc,
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "mean_efficiency_ms": mean_eff_ms,
            "std_efficiency_ms": std_eff_ms,
            "mean_fit_time_s": mean_fit_s,
            "retention_ratio": ret_ratio,
            "mean_retention_ratio": mean_ret_ratio,
            "std_retention_ratio": std_ret_ratio,
            "total_test_samples": total_test,
            "total_correct": correct,
            "total_folds": len(group),
        }

        for metric_col in ["balanced_accuracy", "precision_macro", "recall_macro", "f1_macro"]:
            if metric_col in group.columns:
                row_dict[f"mean_{metric_col}"] = float(group[metric_col].mean())
                row_dict[f"std_{metric_col}"] = (
                    float(group[metric_col].std(ddof=1)) if len(group) > 1 else 0.0
                )

        return pd.Series(row_dict)

    # Group by dataset and configuration
    agg_df = (
        df.groupby(group_cols, dropna=False)
        .apply(_agg_group, include_groups=False)
        .reset_index()
    )

    # Compute descriptive within-dataset ranks with average tie handling
    ranked_chunks: list[pd.DataFrame] = []
    for ds in agg_df["dataset"].unique():
        sub = agg_df[agg_df["dataset"] == ds].copy()
        # Accuracy: descending (highest score is rank 1)
        sub["rank_accuracy"] = sub["pooled_accuracy"].rank(
            ascending=False, method="average"
        )
        # Efficiency: ascending (lowest runtime is rank 1)
        sub["rank_efficiency"] = sub["mean_efficiency_ms"].rank(
            ascending=True, method="average"
        )
        # Retention: ascending (lowest retention ratio is rank 1)
        sub["rank_retention"] = sub["retention_ratio"].rank(
            ascending=True, method="average"
        )
        ranked_chunks.append(sub)

    agg_df = pd.concat(ranked_chunks, ignore_index=True)

    agg_df = agg_df.sort_values(
        by=["dataset", "pooled_accuracy", "mean_efficiency_ms"],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    return agg_df


def aggregate_eager_results(results_df: pd.DataFrame, model_type: str) -> pd.DataFrame:
    """Aggregate per-fold results across folds for SVM or Random Forest."""
    return aggregate_experiment_results(results_df)


def find_best_configurations(
    agg_df: pd.DataFrame,
    top_n: int = 5,
    metric: str = "accuracy",
) -> pd.DataFrame:
    """Return top N performing configurations for each dataset based on specified metric rank."""
    if agg_df.empty:
        return pd.DataFrame()

    rank_col = (
        "rank_efficiency"
        if metric == "efficiency"
        else ("rank_retention" if metric == "retention" else "rank_accuracy")
    )

    best_list: list[pd.DataFrame] = []
    for ds in agg_df["dataset"].unique():
        ds_subset = agg_df[agg_df["dataset"] == ds].copy()
        sorted_sub = ds_subset.sort_values(by=rank_col, ascending=True)
        best_list.append(sorted_sub.head(top_n))

    return pd.concat(best_list, ignore_index=True) if best_list else pd.DataFrame()


def find_best_eager_configurations(
    agg_df: pd.DataFrame,
    top_n: int = 5,
    metric: str = "accuracy",
) -> pd.DataFrame:
    """Return top N performing configurations for eager models."""
    return find_best_configurations(agg_df, top_n=top_n, metric=metric)


# ==============================================================================
# Across-Dataset Friedman Analysis & Critical Difference (Section 4)
# ==============================================================================

def compute_friedman_statistic(ranks_matrix: np.ndarray) -> dict[str, Any]:
    """Calculate the Friedman test statistic, tie correction, and Iman-Davenport F_F.

    Parameters
    ----------
    ranks_matrix : np.ndarray
        Array of shape (D, M) containing ranks for M configurations across D datasets.
        rows = datasets (blocks), columns = configurations (treatments).

    Returns
    -------
    dict[str, Any]
        Dictionary with D, M, uncorrected chi2, tie-corrected chi2, p-values,
        Iman-Davenport statistic, and Nemenyi Critical Differences.
    """
    D, M = ranks_matrix.shape
    if D < 1 or M < 2:
        raise ValueError(
            f"Friedman test requires at least 1 dataset and 2 configurations, got D={D}, M={M}"
        )

    # Average ranks across datasets for each configuration: R[j] = (1/D) * sum(r[j, d])
    mean_ranks = np.mean(ranks_matrix, axis=0)

    # Uncorrected Friedman statistic:
    # chi_F^2 = (12*D / (M*(M+1))) * (sum(R[j]^2) - M*(M+1)^2/4)
    ss = float(np.sum(mean_ranks ** 2))
    chi2_uncorrected = float((12.0 * D / (M * (M + 1))) * (ss - M * (M + 1) ** 2 / 4.0))

    # Tie correction: C = 1 - sum(t_k^3 - t_k) / (D * (M^3 - M))
    tie_sum = 0.0
    for d in range(D):
        _, counts = np.unique(ranks_matrix[d], return_counts=True)
        tie_sum += float(np.sum(counts ** 3 - counts))

    denom_tie = float(D * (M ** 3 - M))
    tie_correction_factor = (
        float(1.0 - tie_sum / denom_tie) if denom_tie > 0 else 1.0
    )

    if tie_correction_factor <= 1e-12:
        chi2_corrected = 0.0
    else:
        chi2_corrected = float(chi2_uncorrected / tie_correction_factor)

    df_chi2 = int(M - 1)
    p_value_chi2 = float(stats.chi2.sf(chi2_corrected, df_chi2))

    # Iman-Davenport FF statistic:
    # FF = ((D - 1) * chi_F^2) / (D * (M - 1) - chi_F^2)
    df_f1 = int(M - 1)
    df_f2 = int((M - 1) * (D - 1))
    f_denom = float(D * (M - 1) - chi2_corrected)

    if df_f2 <= 0:
        f_stat = np.nan
        p_value_f = np.nan
    elif f_denom <= 1e-12:
        f_stat = float("inf") if chi2_corrected > 0 else 0.0
        p_value_f = 0.0 if chi2_corrected > 0 else 1.0
    else:
        f_stat = float(((D - 1) * chi2_corrected) / f_denom)
        p_value_f = float(stats.f.sf(f_stat, df_f1, df_f2))

    # Nemenyi Critical Difference at alpha = 0.05 and alpha = 0.10:
    # CD = q_alpha * sqrt(M*(M+1) / (6*D))
    try:
        q_005 = float(stats.studentized_range.ppf(0.95, M, np.inf) / np.sqrt(2))
        q_010 = float(stats.studentized_range.ppf(0.90, M, np.inf) / np.sqrt(2))
        cd_nemenyi_005 = float(q_005 * np.sqrt(M * (M + 1) / (6.0 * D)))
        cd_nemenyi_010 = float(q_010 * np.sqrt(M * (M + 1) / (6.0 * D)))
    except Exception:
        cd_nemenyi_005 = np.nan
        cd_nemenyi_010 = np.nan

    # Bonferroni-Dunn Critical Difference against a control at alpha = 0.05
    try:
        q_dunn = float(stats.norm.ppf(1.0 - 0.05 / (2.0 * max(1, M - 1))))
        cd_dunn_005 = float(q_dunn * np.sqrt(M * (M + 1) / (6.0 * D)))
    except Exception:
        cd_dunn_005 = np.nan

    return {
        "D": D,
        "M": M,
        "statistic_chi2_uncorrected": chi2_uncorrected,
        "statistic_chi2": chi2_corrected,
        "df_chi2": df_chi2,
        "p_value_chi2": p_value_chi2,
        "significant_chi2_005": bool(p_value_chi2 < 0.05),
        "statistic_f": f_stat,
        "df_f1": df_f1,
        "df_f2": df_f2,
        "p_value_f": p_value_f,
        "significant_f_005": bool(p_value_f < 0.05) if not np.isnan(p_value_f) else False,
        "tie_correction_factor": tie_correction_factor,
        "cd_nemenyi_005": cd_nemenyi_005,
        "cd_nemenyi_010": cd_nemenyi_010,
        "cd_bonferroni_dunn_005": cd_dunn_005,
        "mean_ranks": mean_ranks.tolist(),
    }


def compute_friedman_test(
    agg_df: pd.DataFrame,
    metric: str = "accuracy",
) -> dict[str, Any]:
    """Run Friedman test across datasets on aggregated configuration results.

    Parameters
    ----------
    agg_df : pd.DataFrame
        DataFrame of aggregated results containing 'dataset', configuration identifiers,
        and scores.
    metric : str, default='accuracy'
        One of 'accuracy', 'efficiency', or 'retention'.

    Returns
    -------
    dict[str, Any]
        Statistical test results and ranked leaderboard table.
    """
    if agg_df.empty:
        raise ValueError("Cannot perform Friedman test on empty aggregated DataFrame.")

    datasets = sorted(agg_df["dataset"].unique())
    D = len(datasets)
    if D < 2:
        raise ValueError(
            f"Friedman test requires at least 2 datasets, got {D} ('{datasets}'). "
            f"Use single-dataset mode for descriptive leaderboards."
        )

    # Select target score and ranking direction
    if metric == "accuracy":
        score_col = "pooled_accuracy" if "pooled_accuracy" in agg_df.columns else "mean_accuracy"
        higher_is_better = True
    elif metric == "efficiency":
        score_col = "mean_efficiency_ms"
        higher_is_better = False
    elif metric == "retention":
        score_col = "retention_ratio"
        higher_is_better = False
    else:
        raise ValueError(f"Unknown metric '{metric}'. Expected 'accuracy', 'efficiency', or 'retention'.")

    # Group configurations that exist in all datasets (complete blocks)
    cfg_id_col = "config_id" if "config_id" in agg_df.columns else "distance_metric"
    cfgs_per_ds = [set(agg_df[agg_df["dataset"] == ds][cfg_id_col]) for ds in datasets]
    common_cfgs = sorted(set.intersection(*cfgs_per_ds))

    if len(common_cfgs) < 2:
        raise ValueError(
            f"Need at least 2 common configurations across all {D} datasets, found {len(common_cfgs)}"
        )

    filtered = agg_df[agg_df[cfg_id_col].isin(common_cfgs)].copy()
    M = len(common_cfgs)

    # Compute ranks within each dataset
    ranks_matrix = np.zeros((D, M), dtype=np.float64)
    scores_matrix = np.zeros((D, M), dtype=np.float64)

    for d_idx, ds in enumerate(datasets):
        ds_sub = filtered[filtered["dataset"] == ds].set_index(cfg_id_col).loc[common_cfgs]
        scores = ds_sub[score_col].to_numpy()
        scores_matrix[d_idx] = scores

        # Ranks: ascending=False for accuracy (higher score is rank 1), ascending=True for time/storage
        if higher_is_better:
            r = stats.rankdata(-scores, method="average")
        else:
            r = stats.rankdata(scores, method="average")
        ranks_matrix[d_idx] = r

    friedman_res = compute_friedman_statistic(ranks_matrix)

    # Two-classifier Wilcoxon signed-ranks test if M == 2 (Section 4 & Demšar 2006)
    wilcoxon_stat = np.nan
    wilcoxon_p = np.nan
    if M == 2:
        diffs = scores_matrix[:, 0] - scores_matrix[:, 1]
        if not np.all(np.isclose(diffs, 0.0)):
            try:
                w_res = stats.wilcoxon(scores_matrix[:, 0], scores_matrix[:, 1])
                wilcoxon_stat = float(w_res.statistic)
                wilcoxon_p = float(w_res.pvalue)
            except Exception:
                pass

    # Build across-dataset leaderboard table
    avg_ranks = friedman_res["mean_ranks"]
    order = np.argsort(avg_ranks)

    leaderboard_rows: list[dict[str, Any]] = []
    meta_cols = [
        "model",
        "model_family",
        "distance_metric",
        "k",
        "voting_strategy",
        "retention_policy",
        "kernel",
        "C",
        "gamma",
        "n_estimators",
        "criterion",
        "max_depth",
        "reduction_method",
        "reduction_distance",
    ]

    for rank_pos, idx in enumerate(order):
        cid = common_cfgs[idx]
        row_cfg = filtered[filtered[cfg_id_col] == cid].iloc[0]

        row_entry: dict[str, Any] = {
            "rank": rank_pos + 1,
            "config_id": cid,
            "average_rank": float(avg_ranks[idx]),
        }

        for mc in meta_cols:
            if mc in row_cfg.index and pd.notna(row_cfg[mc]):
                row_entry[mc] = row_cfg[mc]

        # Add per-dataset scores and ranks
        for d_idx, ds in enumerate(datasets):
            row_entry[f"{ds}_{score_col}"] = float(scores_matrix[d_idx, idx])
            row_entry[f"{ds}_rank"] = float(ranks_matrix[d_idx, idx])

        leaderboard_rows.append(row_entry)

    rankings_df = pd.DataFrame(leaderboard_rows)

    limitations = (
        f"Evaluated across D={D} datasets and M={M} configurations. "
        f"Demsar (2006) recommends a rule of thumb of D > 10 datasets and M > 5 classifiers "
        f"for asymptotic chi-square validity. With D={D}, this omnibus Friedman test and nominal "
        f"p-value are reported for exploratory purposes and must not be presented as strong confirmatory "
        f"evidence. Cross-validation folds are not independent datasets and must not be substituted for D."
    )

    friedman_res.update({
        "metric": metric,
        "score_column": score_col,
        "datasets": datasets,
        "configurations_tested": M,
        "rankings_df": rankings_df,
        "wilcoxon_statistic": wilcoxon_stat,
        "wilcoxon_p_value": wilcoxon_p,
        "limitations": limitations,
    })

    return friedman_res


# ==============================================================================
# Single-Dataset & Multi-Dataset Evaluation Modes
# ==============================================================================

def evaluate_single_dataset(
    agg_df: pd.DataFrame,
    dataset_name: str,
    output_dir: str | Path,
    is_reduction: bool = False,
) -> dict[str, pd.DataFrame]:
    """Single-dataset mode: produce descriptive rankings and leaderboards without inferential fold Friedman."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    ds_sub = agg_df[agg_df["dataset"] == dataset_name].copy()
    if ds_sub.empty:
        return {}

    # Sort and export Top 10 by Accuracy
    top10_acc = ds_sub.sort_values(by="rank_accuracy", ascending=True).head(10).reset_index(drop=True)
    top10_acc.to_csv(out_path / f"top10_accuracy_{dataset_name}.csv", index=False)

    # Sort and export Top 10 by Efficiency
    top10_eff = ds_sub.sort_values(by="rank_efficiency", ascending=True).head(10).reset_index(drop=True)
    top10_eff.to_csv(out_path / f"top10_efficiency_{dataset_name}.csv", index=False)

    tables: dict[str, pd.DataFrame] = {
        "top10_accuracy": top10_acc,
        "top10_efficiency": top10_eff,
    }

    # If instance reduction is performed, export Top 10 by Retention Ratio
    if is_reduction or ds_sub["retention_ratio"].nunique() > 1:
        top10_ret = ds_sub.sort_values(by="rank_retention", ascending=True).head(10).reset_index(drop=True)
        top10_ret.to_csv(out_path / f"top10_retention_{dataset_name}.csv", index=False)
        tables["top10_retention"] = top10_ret

    # Descriptive summary metadata
    summary_df = pd.DataFrame([
        {
            "mode": "single_dataset",
            "dataset": dataset_name,
            "configurations_evaluated": len(ds_sub),
            "top_accuracy_config": top10_acc.iloc[0]["config_id"] if "config_id" in top10_acc.columns else "N/A",
            "top_pooled_accuracy": float(top10_acc.iloc[0]["pooled_accuracy"]),
            "top_efficiency_config": top10_eff.iloc[0]["config_id"] if "config_id" in top10_eff.columns else "N/A",
            "top_mean_efficiency_ms": float(top10_eff.iloc[0]["mean_efficiency_ms"]),
            "notes": (
                "Single-dataset descriptive mode active. As detailed in the statistical plan, "
                "cross-validation folds within a single dataset are not independent datasets and "
                "must not be used to compute inferential Friedman p-values."
            ),
        }
    ])
    summary_df.to_csv(out_path / f"single_dataset_summary_{dataset_name}.csv", index=False)
    tables["summary"] = summary_df

    return tables


def evaluate_multi_dataset(
    agg_df: pd.DataFrame,
    output_dir: str | Path,
    is_reduction: bool = False,
) -> dict[str, Any]:
    """All-datasets mode: run Friedman test across datasets for accuracy, efficiency, and retention."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}
    friedman_summary_rows: list[dict[str, Any]] = []

    # 1. Friedman Test for Accuracy
    f_acc = compute_friedman_test(agg_df, metric="accuracy")
    results["accuracy_test"] = f_acc
    top10_acc = f_acc["rankings_df"].head(10)
    top10_acc.to_csv(out_path / "top10_accuracy_across_datasets.csv", index=False)
    results["top10_accuracy"] = top10_acc

    friedman_summary_rows.append({
        "metric": "accuracy",
        "D": f_acc["D"],
        "M": f_acc["M"],
        "statistic_chi2_uncorrected": f_acc["statistic_chi2_uncorrected"],
        "statistic_chi2_corrected": f_acc["statistic_chi2"],
        "df_chi2": f_acc["df_chi2"],
        "p_value_chi2": f_acc["p_value_chi2"],
        "significant_chi2_005": f_acc["significant_chi2_005"],
        "statistic_f": f_acc["statistic_f"],
        "df_f1": f_acc["df_f1"],
        "df_f2": f_acc["df_f2"],
        "p_value_f": f_acc["p_value_f"],
        "significant_f_005": f_acc["significant_f_005"],
        "tie_correction_factor": f_acc["tie_correction_factor"],
        "cd_nemenyi_005": f_acc["cd_nemenyi_005"],
        "cd_nemenyi_010": f_acc["cd_nemenyi_010"],
        "cd_bonferroni_dunn_005": f_acc["cd_bonferroni_dunn_005"],
        "limitations": f_acc["limitations"],
    })

    # 2. Friedman Test for Efficiency / Solving Time
    f_eff = compute_friedman_test(agg_df, metric="efficiency")
    results["efficiency_test"] = f_eff
    top10_eff = f_eff["rankings_df"].head(10)
    top10_eff.to_csv(out_path / "top10_efficiency_across_datasets.csv", index=False)
    results["top10_efficiency"] = top10_eff

    friedman_summary_rows.append({
        "metric": "efficiency",
        "D": f_eff["D"],
        "M": f_eff["M"],
        "statistic_chi2_uncorrected": f_eff["statistic_chi2_uncorrected"],
        "statistic_chi2_corrected": f_eff["statistic_chi2"],
        "df_chi2": f_eff["df_chi2"],
        "p_value_chi2": f_eff["p_value_chi2"],
        "significant_chi2_005": f_eff["significant_chi2_005"],
        "statistic_f": f_eff["statistic_f"],
        "df_f1": f_eff["df_f1"],
        "df_f2": f_eff["df_f2"],
        "p_value_f": f_eff["p_value_f"],
        "significant_f_005": f_eff["significant_f_005"],
        "tie_correction_factor": f_eff["tie_correction_factor"],
        "cd_nemenyi_005": f_eff["cd_nemenyi_005"],
        "cd_nemenyi_010": f_eff["cd_nemenyi_010"],
        "cd_bonferroni_dunn_005": f_eff["cd_bonferroni_dunn_005"],
        "limitations": f_eff["limitations"],
    })

    # 3. Friedman Test for Retention Ratio (if instance reduction is active)
    if is_reduction or agg_df["retention_ratio"].nunique() > 1:
        try:
            f_ret = compute_friedman_test(agg_df, metric="retention")
            results["retention_test"] = f_ret
            top10_ret = f_ret["rankings_df"].head(10)
            top10_ret.to_csv(out_path / "top10_retention_across_datasets.csv", index=False)
            results["top10_retention"] = top10_ret

            friedman_summary_rows.append({
                "metric": "retention",
                "D": f_ret["D"],
                "M": f_ret["M"],
                "statistic_chi2_uncorrected": f_ret["statistic_chi2_uncorrected"],
                "statistic_chi2_corrected": f_ret["statistic_chi2"],
                "df_chi2": f_ret["df_chi2"],
                "p_value_chi2": f_ret["p_value_chi2"],
                "significant_chi2_005": f_ret["significant_chi2_005"],
                "statistic_f": f_ret["statistic_f"],
                "df_f1": f_ret["df_f1"],
                "df_f2": f_ret["df_f2"],
                "p_value_f": f_ret["p_value_f"],
                "significant_f_005": f_ret["significant_f_005"],
                "tie_correction_factor": f_ret["tie_correction_factor"],
                "cd_nemenyi_005": f_ret["cd_nemenyi_005"],
                "cd_nemenyi_010": f_ret["cd_nemenyi_010"],
                "cd_bonferroni_dunn_005": f_ret["cd_bonferroni_dunn_005"],
                "limitations": f_ret["limitations"],
            })
        except Exception:
            pass

    # Save Friedman summary table
    friedman_df = pd.DataFrame(friedman_summary_rows)
    friedman_df.to_csv(out_path / "friedman_test_results.csv", index=False)
    results["friedman_summary_df"] = friedman_df

    # Also save dataset-specific top 10 tables for visibility into disagreements
    for ds in agg_df["dataset"].unique():
        ds_sub = agg_df[agg_df["dataset"] == ds]
        ds_top_acc = ds_sub.sort_values(by="rank_accuracy", ascending=True).head(10)
        ds_top_acc.to_csv(out_path / f"top10_accuracy_{ds}.csv", index=False)
        ds_top_eff = ds_sub.sort_values(by="rank_efficiency", ascending=True).head(10)
        ds_top_eff.to_csv(out_path / f"top10_efficiency_{ds}.csv", index=False)

    return results


# ==============================================================================
# Master Export Routines
# ==============================================================================

def export_evaluation_report(
    results_df: pd.DataFrame,
    output_dir: str | Path = "results",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Export aggregated results, best configurations, and statistical tests (single or multi-dataset)."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    agg_df = aggregate_experiment_results(results_df)
    agg_df.to_csv(out_path / "aggregated_results.csv", index=False)

    best_df = find_best_configurations(agg_df, top_n=5)
    best_df.to_csv(out_path / "best_configurations.csv", index=False)

    # Check whether instance reduction is evaluated
    is_reduction = False
    if "reduction_method" in results_df.columns:
        valid_reds = results_df["reduction_method"].dropna().astype(str).str.lower()
        is_reduction = bool((valid_reds != "none").any())

    unique_datasets = agg_df["dataset"].unique()
    eval_dict: dict[str, Any] = {}

    if len(unique_datasets) == 1:
        # Single-dataset mode
        ds_name = unique_datasets[0]
        eval_dict = evaluate_single_dataset(
            agg_df, dataset_name=ds_name, output_dir=out_path, is_reduction=is_reduction
        )
    else:
        # Multi-dataset scenario: Friedman test
        eval_dict = evaluate_multi_dataset(
            agg_df, output_dir=out_path, is_reduction=is_reduction
        )

    # Legacy compatibility helper: produce statistical_comparisons.csv
    stat_summary_file = out_path / "statistical_comparisons.csv"
    if "friedman_summary_df" in eval_dict:
        eval_dict["friedman_summary_df"].to_csv(stat_summary_file, index=False)
    elif "summary" in eval_dict:
        eval_dict["summary"].to_csv(stat_summary_file, index=False)

    return agg_df, best_df, eval_dict


def export_eager_evaluation_report(
    results_df: pd.DataFrame,
    model_type: str,
    output_dir: str | Path = "results",
    knn_baseline_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame | None]:
    """Export summary tables, top configurations, and statistical tests for eager models."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    agg_df = aggregate_experiment_results(results_df)
    agg_df.to_csv(out_path / f"{model_type}_aggregated_results.csv", index=False)

    best_df = find_best_eager_configurations(agg_df, top_n=5)
    best_df.to_csv(out_path / f"{model_type}_best_configurations.csv", index=False)

    is_reduction = False
    if "reduction_method" in results_df.columns:
        valid_reds = results_df["reduction_method"].dropna().astype(str).str.lower()
        is_reduction = bool((valid_reds != "none").any())

    unique_datasets = agg_df["dataset"].unique()
    eval_dict: dict[str, Any] = {}

    if len(unique_datasets) == 1:
        eval_dict = evaluate_single_dataset(
            agg_df, dataset_name=unique_datasets[0], output_dir=out_path, is_reduction=is_reduction
        )
    else:
        eval_dict = evaluate_multi_dataset(
            agg_df, output_dir=out_path, is_reduction=is_reduction
        )

    stat_summary_file = out_path / f"{model_type}_statistical_comparisons.csv"
    if "friedman_summary_df" in eval_dict:
        eval_dict["friedman_summary_df"].to_csv(stat_summary_file, index=False)
    elif "summary" in eval_dict:
        eval_dict["summary"].to_csv(stat_summary_file, index=False)

    knn_comp_df: pd.DataFrame | None = None
    if knn_baseline_dir:
        knn_comp_df = compare_with_knn_baseline(
            results_df, knn_baseline_dir, model_type, out_path
        )

    return agg_df, best_df, eval_dict, knn_comp_df


def generate_statistical_comparisons(
    results_df: pd.DataFrame,
    output_dir: str | Path,
) -> pd.DataFrame:
    """Non-parametric evaluation dispatch replacing old t-test comparison."""
    agg_df = aggregate_experiment_results(results_df)
    unique_datasets = agg_df["dataset"].unique()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if len(unique_datasets) >= 2:
        res = evaluate_multi_dataset(agg_df, out_path)
        return res.get("friedman_summary_df", pd.DataFrame())
    elif len(unique_datasets) == 1:
        res = evaluate_single_dataset(agg_df, unique_datasets[0], out_path)
        return res.get("summary", pd.DataFrame())
    return pd.DataFrame()


def generate_eager_statistical_comparisons(
    results_df: pd.DataFrame,
    model_type: str,
    output_dir: str | Path,
) -> pd.DataFrame:
    """Non-parametric evaluation dispatch for eager models replacing old t-test comparison."""
    return generate_statistical_comparisons(results_df, output_dir)


# ==============================================================================
# Model Baseline & Assignment Stage Comparisons (Stages D, E, F)
# ==============================================================================

def compare_with_knn_baseline(
    eager_results_df: pd.DataFrame,
    knn_baseline_dir: str | Path,
    model_type: str,
    output_dir: str | Path,
) -> pd.DataFrame | None:
    """Compare best eager model configuration against best KNN baseline on each dataset and across datasets.

    Eliminates paired t-tests over CV folds. Replaces with pooled accuracy,
    mean efficiency, retention comparison, and cross-dataset ranks/Wilcoxon test.
    """
    baseline_path = Path(knn_baseline_dir)
    if not baseline_path.exists():
        raise FileNotFoundError(f"KNN baseline directory not found: {baseline_path}")

    knn_file = baseline_path / "all_experiments_results.csv"
    if not knn_file.exists():
        candidates = list(baseline_path.glob("results_*.csv")) + list(baseline_path.glob("*.csv"))
        if candidates:
            knn_file = candidates[0]
        else:
            raise FileNotFoundError(f"No results CSV found inside KNN baseline directory: {baseline_path}")

    knn_df = pd.read_csv(knn_file)
    agg_knn = aggregate_experiment_results(knn_df)
    agg_eager = aggregate_experiment_results(eager_results_df)

    if agg_knn.empty or agg_eager.empty:
        return None

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    comparisons: list[dict[str, Any]] = []

    for ds in eager_results_df["dataset"].unique():
        canonical_ds = resolve_dataset_dir(ds, "data")[1]
        ds_matches = [
            d for d in agg_knn["dataset"].unique()
            if resolve_dataset_dir(d, "data")[1] == canonical_ds
        ]
        if not ds_matches:
            continue

        knn_ds = agg_knn[agg_knn["dataset"] == ds_matches[0]].sort_values(
            by="rank_accuracy", ascending=True
        )
        eager_ds = agg_eager[agg_eager["dataset"] == ds].sort_values(
            by="rank_accuracy", ascending=True
        )

        if knn_ds.empty or eager_ds.empty:
            continue

        best_knn = knn_ds.iloc[0]
        best_eager = eager_ds.iloc[0]

        acc_eager = float(best_eager["pooled_accuracy"])
        acc_knn = float(best_knn["pooled_accuracy"])
        acc_diff = acc_eager - acc_knn

        eff_eager = float(best_eager["mean_efficiency_ms"])
        eff_knn = float(best_knn["mean_efficiency_ms"])
        eff_diff = eff_eager - eff_knn
        speedup = (eff_knn / eff_eager) if eff_eager > 0 else np.nan

        ret_eager = float(best_eager["retention_ratio"])
        ret_knn = float(best_knn["retention_ratio"])

        comparisons.append({
            "dataset": ds,
            "eager_model": model_type,
            "eager_best_config": best_eager.get("config_id", f"eager_{ds}"),
            "eager_pooled_accuracy": acc_eager,
            "eager_mean_efficiency_ms": eff_eager,
            "eager_retention_ratio": ret_eager,
            "knn_best_config": best_knn.get("config_id", f"knn_{ds}"),
            "knn_pooled_accuracy": acc_knn,
            "knn_mean_efficiency_ms": eff_knn,
            "knn_retention_ratio": ret_knn,
            "accuracy_difference": acc_diff,
            "efficiency_difference_ms": eff_diff,
            "speedup_vs_knn": speedup,
            "winner_accuracy": model_type if acc_diff > 0 else ("knn" if acc_diff < 0 else "tie"),
            "winner_efficiency": model_type if eff_diff < 0 else ("knn" if eff_diff > 0 else "tie"),
        })

    if not comparisons:
        return None

    comp_df = pd.DataFrame(comparisons)
    comp_df.to_csv(out_path / "model_comparison.csv", index=False)
    return comp_df


def evaluate_stage_d_comparison(
    knn_results_df: pd.DataFrame,
    svm_results_df: pd.DataFrame,
    rf_results_df: pd.DataFrame,
    output_dir: str | Path = "results",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Stage D: Compare the best KNN, SVM, and RF configurations on accuracy and efficiency."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    agg_knn = aggregate_experiment_results(knn_results_df)
    agg_svm = aggregate_experiment_results(svm_results_df)
    agg_rf = aggregate_experiment_results(rf_results_df)

    best_rows: list[dict[str, Any]] = []

    # Collect best configuration per family and dataset
    datasets = sorted(
        set(agg_knn["dataset"]).intersection(set(agg_svm["dataset"])).intersection(set(agg_rf["dataset"]))
    )

    for ds in datasets:
        b_knn = agg_knn[agg_knn["dataset"] == ds].sort_values("rank_accuracy").iloc[0]
        b_svm = agg_svm[agg_svm["dataset"] == ds].sort_values("rank_accuracy").iloc[0]
        b_rf = agg_rf[agg_rf["dataset"] == ds].sort_values("rank_accuracy").iloc[0]

        for b, m in [(b_knn, "knn"), (b_svm, "svm"), (b_rf, "rf")]:
            best_rows.append({
                "dataset": ds,
                "model_family": m,
                "config_id": b.get("config_id", f"{m}_{ds}"),
                "pooled_accuracy": float(b["pooled_accuracy"]),
                "mean_efficiency_ms": float(b["mean_efficiency_ms"]),
                "retention_ratio": float(b["retention_ratio"]),
            })

    stage_d_df = pd.DataFrame(best_rows)
    stage_d_df.to_csv(out_path / "stage_d_best_family_comparison.csv", index=False)

    friedman_results: dict[str, Any] = {}
    if len(datasets) >= 2:
        # Run Friedman across the 3 best-of-family configurations (M=3)
        ranks_acc = np.zeros((len(datasets), 3))
        ranks_eff = np.zeros((len(datasets), 3))
        for d_idx, ds in enumerate(datasets):
            ds_sub = stage_d_df[stage_d_df["dataset"] == ds].set_index("model_family").loc[["knn", "svm", "rf"]]
            ranks_acc[d_idx] = stats.rankdata(-ds_sub["pooled_accuracy"].values, method="average")
            ranks_eff[d_idx] = stats.rankdata(ds_sub["mean_efficiency_ms"].values, method="average")

        f_acc = compute_friedman_statistic(ranks_acc)
        f_eff = compute_friedman_statistic(ranks_eff)
        friedman_results = {
            "accuracy_friedman": f_acc,
            "efficiency_friedman": f_eff,
            "model_families": ["knn", "svm", "rf"],
            "accuracy_average_ranks": f_acc["mean_ranks"],
            "efficiency_average_ranks": f_eff["mean_ranks"],
        }

    return stage_d_df, friedman_results


def evaluate_stage_e_knn_reduction(
    unreduced_df: pd.DataFrame,
    reduced_df: pd.DataFrame,
    output_dir: str | Path = "results",
) -> pd.DataFrame:
    """Stage E: Evaluate KNN instance reduction methods against unreduced baseline."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    agg_unred = aggregate_experiment_results(unreduced_df)
    agg_red = aggregate_experiment_results(reduced_df)

    rows: list[dict[str, Any]] = []
    for ds in agg_red["dataset"].unique():
        unred_sub = agg_unred[agg_unred["dataset"] == ds].sort_values("rank_accuracy")
        if unred_sub.empty:
            continue
        base = unred_sub.iloc[0]

        red_sub = agg_red[agg_red["dataset"] == ds]
        for _, red_row in red_sub.iterrows():
            acc_base = float(base["pooled_accuracy"])
            acc_red = float(red_row["pooled_accuracy"])
            eff_base = float(base["mean_efficiency_ms"])
            eff_red = float(red_row["mean_efficiency_ms"])
            ret_ratio = float(red_row["retention_ratio"])

            rows.append({
                "dataset": ds,
                "reduction_method": red_row.get("reduction_method", "none"),
                "reduction_distance": red_row.get("reduction_distance", "none"),
                "config_id": red_row.get("config_id", ""),
                "unreduced_accuracy": acc_base,
                "reduced_accuracy": acc_red,
                "accuracy_change": acc_red - acc_base,
                "unreduced_efficiency_ms": eff_base,
                "reduced_efficiency_ms": eff_red,
                "efficiency_speedup": (eff_base / eff_red) if eff_red > 0 else np.nan,
                "retention_ratio": ret_ratio,
                "storage_reduction_pct": (1.0 - ret_ratio) * 100.0,
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path / "stage_e_knn_reduction_comparison.csv", index=False)
    return out_df


def evaluate_stage_f_eager_reduction(
    unreduced_df: pd.DataFrame,
    reduced_df: pd.DataFrame,
    model_type: str,
    output_dir: str | Path = "results",
) -> pd.DataFrame:
    """Stage F: Evaluate reduction impact on SVM or Random Forest."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    agg_unred = aggregate_experiment_results(unreduced_df)
    agg_red = aggregate_experiment_results(reduced_df)

    rows: list[dict[str, Any]] = []
    for ds in agg_red["dataset"].unique():
        unred_sub = agg_unred[agg_unred["dataset"] == ds]
        red_sub = agg_red[agg_red["dataset"] == ds]

        for _, red_row in red_sub.iterrows():
            cid = red_row.get("config_id", "")
            base_matches = unred_sub[unred_sub["config_id"] == cid.replace(f"_red={red_row.get('reduction_method')}", "")]
            if base_matches.empty:
                base_matches = unred_sub.sort_values("rank_accuracy")
            base = base_matches.iloc[0]

            acc_base = float(base["pooled_accuracy"])
            acc_red = float(red_row["pooled_accuracy"])
            eff_base = float(base["mean_efficiency_ms"])
            eff_red = float(red_row["mean_efficiency_ms"])
            ret_ratio = float(red_row["retention_ratio"])

            rows.append({
                "dataset": ds,
                "model_type": model_type,
                "reduction_method": red_row.get("reduction_method", "none"),
                "reduced_config_id": cid,
                "unreduced_accuracy": acc_base,
                "reduced_accuracy": acc_red,
                "accuracy_change": acc_red - acc_base,
                "unreduced_efficiency_ms": eff_base,
                "reduced_efficiency_ms": eff_red,
                "efficiency_speedup": (eff_base / eff_red) if eff_red > 0 else np.nan,
                "retention_ratio": ret_ratio,
                "storage_reduction_pct": (1.0 - ret_ratio) * 100.0,
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path / f"stage_f_{model_type}_reduction_comparison.csv", index=False)
    return out_df


# ==============================================================================
# Helper: Experiment Directory Auto-Versioning
# ==============================================================================

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
