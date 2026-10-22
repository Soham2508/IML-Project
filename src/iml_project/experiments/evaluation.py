from __future__ import annotations

import itertools
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from iml_project.utils.arff_reader import resolve_dataset_dir


def aggregate_experiment_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-fold results across the 10 predefined folds for KNN."""
    group_cols = [
        "dataset",
        "distance_metric",
        "k",
        "voting_strategy",
        "retention_policy",
    ]

    agg_dict: dict[str, Any] = {
        "mean_accuracy": ("accuracy", "mean"),
        "std_accuracy": ("accuracy", "std"),
    }
    if "balanced_accuracy" in results_df.columns:
        agg_dict["mean_balanced_accuracy"] = ("balanced_accuracy", "mean")
        agg_dict["std_balanced_accuracy"] = ("balanced_accuracy", "std")
    if "precision_macro" in results_df.columns:
        agg_dict["mean_precision_macro"] = ("precision_macro", "mean")
        agg_dict["std_precision_macro"] = ("precision_macro", "std")
    if "recall_macro" in results_df.columns:
        agg_dict["mean_recall_macro"] = ("recall_macro", "mean")
        agg_dict["std_recall_macro"] = ("recall_macro", "std")
    if "f1_macro" in results_df.columns:
        agg_dict["mean_f1_macro"] = ("f1_macro", "mean")
        agg_dict["std_f1_macro"] = ("f1_macro", "std")

    agg_dict.update(
        {
            "mean_efficiency_ms": (
                "mean_prediction_time",
                lambda x: float(np.mean(x) * 1000.0),
            ),
            "std_efficiency_ms": (
                "mean_prediction_time",
                lambda x: float(np.std(x) * 1000.0),
            ),
            "mean_retention_ratio": ("retention_ratio", "mean"),
            "std_retention_ratio": ("retention_ratio", "std"),
            "total_folds": ("fold", "count"),
        }
    )

    agg_df = (
        results_df.groupby(group_cols)
        .agg(**agg_dict)
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


def compute_paired_arrays_ttest(
    acc_a: Sequence[float],
    acc_b: Sequence[float],
    dataset: str,
    cfg_a: Any = "A",
    cfg_b: Any = "B",
) -> dict[str, Any]:
    """Perform a two-sided paired Student's t-test between two arrays across matching folds."""
    arr_a = np.array(acc_a, dtype=np.float64)
    arr_b = np.array(acc_b, dtype=np.float64)

    if len(arr_a) != len(arr_b) or len(arr_a) < 2:
        raise ValueError(
            f"Expected equal array lengths (at least 2), got {len(arr_a)} and {len(arr_b)}"
        )

    diff = arr_a - arr_b
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))

    if np.isclose(std_diff, 0.0):
        if np.isclose(mean_diff, 0.0):
            t_stat = 0.0
            p_val = 1.0
        else:
            t_stat = float("inf") if mean_diff > 0 else float("-inf")
            p_val = 0.0
    else:
        res = stats.ttest_rel(arr_a, arr_b)
        t_stat = float(res.statistic)
        p_val = float(res.pvalue)

    return {
        "dataset": dataset,
        "n_pairs": len(arr_a),
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "significant_005": bool(p_val < 0.05),
        "cfg_a": cfg_a,
        "cfg_b": cfg_b,
    }


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

    return compute_paired_arrays_ttest(acc_a, acc_b, dataset, cfg_a=cfg_a, cfg_b=cfg_b)


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
    """Compute pairwise paired t-tests comparing metrics, voting, retention, and best vs baseline for KNN."""
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
                pass

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
                pass

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
            pass

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
                pass

    comp_df = pd.DataFrame(comparisons)
    comp_df.to_csv(out_path / "statistical_comparisons.csv", index=False)
    return comp_df


def export_evaluation_report(
    results_df: pd.DataFrame,
    output_dir: str | Path = "results",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Export summary tables, top configurations, and statistical comparisons for KNN."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    agg_df = aggregate_experiment_results(results_df)
    agg_df.to_csv(out_path / "aggregated_results.csv", index=False)

    best_df = find_best_configurations(agg_df, top_n=5)
    best_df.to_csv(out_path / "best_configurations.csv", index=False)

    stat_df = generate_statistical_comparisons(results_df, out_path)

    return agg_df, best_df, stat_df


# ==============================================================================
# Eager Models (SVM & Random Forest) Evaluation & Statistical Comparison
# ==============================================================================

def aggregate_eager_results(results_df: pd.DataFrame, model_type: str) -> pd.DataFrame:
    """Aggregate per-fold results across the 10 predefined folds for SVM or Random Forest."""
    if model_type == "svm":
        group_cols = ["dataset", "model", "kernel", "C", "gamma"]
    elif model_type == "rf":
        group_cols = ["dataset", "model", "n_estimators", "criterion", "max_depth"]
    else:
        raise ValueError(f"Unknown model_type '{model_type}', expected 'svm' or 'rf'")

    agg_dict: dict[str, Any] = {
        "mean_accuracy": ("accuracy", "mean"),
        "std_accuracy": ("accuracy", "std"),
    }
    if "balanced_accuracy" in results_df.columns:
        agg_dict["mean_balanced_accuracy"] = ("balanced_accuracy", "mean")
        agg_dict["std_balanced_accuracy"] = ("balanced_accuracy", "std")
    if "precision_macro" in results_df.columns:
        agg_dict["mean_precision_macro"] = ("precision_macro", "mean")
        agg_dict["std_precision_macro"] = ("precision_macro", "std")
    if "recall_macro" in results_df.columns:
        agg_dict["mean_recall_macro"] = ("recall_macro", "mean")
        agg_dict["std_recall_macro"] = ("recall_macro", "std")
    if "f1_macro" in results_df.columns:
        agg_dict["mean_f1_macro"] = ("f1_macro", "mean")
        agg_dict["std_f1_macro"] = ("f1_macro", "std")

    agg_dict.update(
        {
            "mean_fit_time_s": ("fit_time", "mean"),
            "std_fit_time_s": ("fit_time", "std"),
            "mean_efficiency_ms": (
                "mean_prediction_time",
                lambda x: float(np.mean(x) * 1000.0),
            ),
            "std_efficiency_ms": (
                "mean_prediction_time",
                lambda x: float(np.std(x) * 1000.0),
            ),
            "mean_retention_ratio": ("retention_ratio", "mean"),
            "total_folds": ("fold", "count"),
        }
    )

    agg_df = (
        results_df.groupby(group_cols, dropna=False)
        .agg(**agg_dict)
        .reset_index()
    )

    agg_df = agg_df.sort_values(
        by=["dataset", "mean_accuracy", "mean_efficiency_ms"],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    return agg_df


def find_best_eager_configurations(agg_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Return top N performing configurations for each dataset for an eager model."""
    best_list = []
    for ds in agg_df["dataset"].unique():
        ds_subset = agg_df[agg_df["dataset"] == ds].copy()
        best_list.append(ds_subset.head(top_n))
    return pd.concat(best_list, ignore_index=True) if best_list else pd.DataFrame()


def generate_eager_statistical_comparisons(
    results_df: pd.DataFrame,
    model_type: str,
    output_dir: str | Path,
) -> pd.DataFrame:
    """Generate paired t-tests across configurations for SVM or Random Forest."""
    out_path = Path(output_dir)
    comparisons: list[dict[str, Any]] = []

    for ds in results_df["dataset"].unique():
        ds_sub = results_df[results_df["dataset"] == ds]

        if model_type == "svm":
            # Compare other kernels against linear (or first kernel)
            kernels = list(ds_sub["kernel"].unique())
            baseline_kernel = "linear" if "linear" in kernels else kernels[0]

            acc_base = ds_sub[ds_sub["kernel"] == baseline_kernel].sort_values("fold")["accuracy"].to_numpy()

            for k in kernels:
                if k == baseline_kernel:
                    continue
                acc_k = ds_sub[ds_sub["kernel"] == k].sort_values("fold")["accuracy"].to_numpy()
                if len(acc_base) == 10 and len(acc_k) == 10:
                    t_res = compute_paired_arrays_ttest(
                        acc_k,
                        acc_base,
                        dataset=ds,
                        cfg_a=f"kernel={k}",
                        cfg_b=f"kernel={baseline_kernel}",
                    )
                    comparisons.append(
                        {
                            "comparison": f"{k.capitalize()}_vs_{baseline_kernel.capitalize()}",
                            "dataset": ds,
                            "model": "svm",
                            "n_pairs": t_res["n_pairs"],
                            "mean_diff": t_res["mean_diff"],
                            "std_diff": t_res["std_diff"],
                            "t_stat": t_res["t_statistic"],
                            "p_value": t_res["p_value"],
                            "sig_005": t_res["significant_005"],
                            "cfg_a": str(t_res["cfg_a"]),
                            "cfg_b": str(t_res["cfg_b"]),
                        }
                    )

        elif model_type == "rf":
            # Baseline: n_estimators=100, criterion='gini' (or lowest available)
            n_ests = sorted(ds_sub["n_estimators"].unique())
            crits = list(ds_sub["criterion"].unique())

            base_n = 100 if 100 in n_ests else n_ests[0]
            base_crit = "gini" if "gini" in crits else crits[0]

            mask_base = (ds_sub["n_estimators"] == base_n) & (ds_sub["criterion"] == base_crit)
            acc_base = ds_sub[mask_base].sort_values("fold")["accuracy"].to_numpy()

            # Compare all other combinations against baseline
            for n_est, crit in itertools.product(n_ests, crits):
                if n_est == base_n and crit == base_crit:
                    continue
                mask_comp = (ds_sub["n_estimators"] == n_est) & (ds_sub["criterion"] == crit)
                acc_comp = ds_sub[mask_comp].sort_values("fold")["accuracy"].to_numpy()
                if len(acc_base) == 10 and len(acc_comp) == 10:
                    t_res = compute_paired_arrays_ttest(
                        acc_comp,
                        acc_base,
                        dataset=ds,
                        cfg_a=f"n_est={n_est},crit={crit}",
                        cfg_b=f"n_est={base_n},crit={base_crit}",
                    )
                    comparisons.append(
                        {
                            "comparison": f"RF(n={n_est},{crit})_vs_Baseline(n={base_n},{base_crit})",
                            "dataset": ds,
                            "model": "rf",
                            "n_pairs": t_res["n_pairs"],
                            "mean_diff": t_res["mean_diff"],
                            "std_diff": t_res["std_diff"],
                            "t_stat": t_res["t_statistic"],
                            "p_value": t_res["p_value"],
                            "sig_005": t_res["significant_005"],
                            "cfg_a": str(t_res["cfg_a"]),
                            "cfg_b": str(t_res["cfg_b"]),
                        }
                    )

            # Compare criteria directly at matching n_estimators (e.g. entropy vs gini)
            if "entropy" in crits and "gini" in crits:
                for n_est in n_ests:
                    acc_ent = ds_sub[(ds_sub["n_estimators"] == n_est) & (ds_sub["criterion"] == "entropy")].sort_values("fold")["accuracy"].to_numpy()
                    acc_gin = ds_sub[(ds_sub["n_estimators"] == n_est) & (ds_sub["criterion"] == "gini")].sort_values("fold")["accuracy"].to_numpy()
                    if len(acc_ent) == 10 and len(acc_gin) == 10:
                        t_res = compute_paired_arrays_ttest(
                            acc_ent,
                            acc_gin,
                            dataset=ds,
                            cfg_a=f"entropy (n={n_est})",
                            cfg_b=f"gini (n={n_est})",
                        )
                        comparisons.append(
                            {
                                "comparison": f"Entropy_vs_Gini_(n={n_est})",
                                "dataset": ds,
                                "model": "rf",
                                "n_pairs": t_res["n_pairs"],
                                "mean_diff": t_res["mean_diff"],
                                "std_diff": t_res["std_diff"],
                                "t_stat": t_res["t_statistic"],
                                "p_value": t_res["p_value"],
                                "sig_005": t_res["significant_005"],
                                "cfg_a": str(t_res["cfg_a"]),
                                "cfg_b": str(t_res["cfg_b"]),
                            }
                        )

    comp_df = pd.DataFrame(comparisons)
    filename = f"{model_type}_statistical_comparisons.csv"
    comp_df.to_csv(out_path / filename, index=False)
    return comp_df


def compare_with_knn_baseline(
    eager_results_df: pd.DataFrame,
    knn_baseline_dir: str | Path,
    model_type: str,
    output_dir: str | Path,
) -> pd.DataFrame | None:
    """Compare best eager model configuration against best KNN baseline on matching folds."""
    baseline_path = Path(knn_baseline_dir)
    if not baseline_path.exists():
        raise FileNotFoundError(f"KNN baseline directory not found: {baseline_path}")

    # Search for KNN results file in baseline path
    knn_file = baseline_path / "all_experiments_results.csv"
    if not knn_file.exists():
        candidates = list(baseline_path.glob("results_*.csv")) + list(baseline_path.glob("*.csv"))
        if candidates:
            knn_file = candidates[0]
        else:
            raise FileNotFoundError(
                f"No results CSV found inside KNN baseline directory: {baseline_path}"
            )

    knn_df = pd.read_csv(knn_file)
    if "distance_metric" not in knn_df.columns:
        raise ValueError(
            f"CSV file '{knn_file.name}' does not appear to contain KNN experiment results."
        )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    comparisons: list[dict[str, Any]] = []

    for ds in eager_results_df["dataset"].unique():
        canonical_ds = resolve_dataset_dir(ds, "data")[1]
        ds_matches = [
            d for d in knn_df["dataset"].unique()
            if resolve_dataset_dir(d, "data")[1] == canonical_ds
        ]
        if not ds_matches:
            continue

        knn_ds_sub = knn_df[knn_df["dataset"] == ds_matches[0]]
        eager_ds_sub = eager_results_df[eager_results_df["dataset"] == ds]

        # Aggregate KNN to find best
        agg_knn = aggregate_experiment_results(knn_ds_sub)
        if agg_knn.empty:
            continue
        best_knn_row = agg_knn.iloc[0]
        best_knn_cfg = {
            "distance_metric": best_knn_row["distance_metric"],
            "k": int(best_knn_row["k"]),
            "voting_strategy": best_knn_row["voting_strategy"],
            "retention_policy": best_knn_row["retention_policy"],
        }

        # Filter KNN best per-fold accuracy
        knn_fold_mask = (
            (knn_ds_sub["distance_metric"] == best_knn_cfg["distance_metric"])
            & (knn_ds_sub["k"] == best_knn_cfg["k"])
            & (knn_ds_sub["voting_strategy"] == best_knn_cfg["voting_strategy"])
            & (knn_ds_sub["retention_policy"] == best_knn_cfg["retention_policy"])
        )
        knn_fold_sub = knn_ds_sub[knn_fold_mask].sort_values("fold")
        knn_accs = knn_fold_sub["accuracy"].to_numpy()

        # Aggregate eager to find best
        agg_eager = aggregate_eager_results(eager_ds_sub, model_type)
        if agg_eager.empty:
            continue
        best_eager_row = agg_eager.iloc[0]

        if model_type == "svm":
            best_eager_cfg = {
                "kernel": best_eager_row["kernel"],
                "C": float(best_eager_row["C"]),
                "gamma": str(best_eager_row["gamma"]),
            }
            eager_fold_mask = (
                (eager_ds_sub["kernel"] == best_eager_cfg["kernel"])
                & np.isclose(eager_ds_sub["C"].astype(float), best_eager_cfg["C"])
                & (eager_ds_sub["gamma"].astype(str) == best_eager_cfg["gamma"])
            )
        else:
            best_eager_cfg = {
                "n_estimators": int(best_eager_row["n_estimators"]),
                "criterion": best_eager_row["criterion"],
                "max_depth": best_eager_row["max_depth"],
            }
            eager_fold_mask = (
                (eager_ds_sub["n_estimators"] == best_eager_cfg["n_estimators"])
                & (eager_ds_sub["criterion"] == best_eager_cfg["criterion"])
            )

        eager_fold_sub = eager_ds_sub[eager_fold_mask].sort_values("fold")
        eager_accs = eager_fold_sub["accuracy"].to_numpy()

        if len(eager_accs) == len(knn_accs) and len(eager_accs) >= 2:
            t_res = compute_paired_arrays_ttest(
                eager_accs,
                knn_accs,
                dataset=ds,
                cfg_a=best_eager_cfg,
                cfg_b=best_knn_cfg,
            )
            comparisons.append(
                {
                    "dataset": ds,
                    "eager_model": model_type,
                    "eager_best_config": str(best_eager_cfg),
                    "eager_mean_accuracy": float(np.mean(eager_accs)),
                    "knn_best_config": str(best_knn_cfg),
                    "knn_mean_accuracy": float(np.mean(knn_accs)),
                    "mean_diff": t_res["mean_diff"],
                    "std_diff": t_res["std_diff"],
                    "t_statistic": t_res["t_statistic"],
                    "p_value": t_res["p_value"],
                    "significant_005": t_res["significant_005"],
                }
            )

    if comparisons:
        comp_df = pd.DataFrame(comparisons)
        comp_df.to_csv(out_path / "model_comparison.csv", index=False)
        return comp_df
    return None


def export_eager_evaluation_report(
    results_df: pd.DataFrame,
    model_type: str,
    output_dir: str | Path = "results",
    knn_baseline_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Export summary tables, top configurations, and statistical comparisons for eager models."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    agg_df = aggregate_eager_results(results_df, model_type)
    agg_df.to_csv(out_path / f"{model_type}_aggregated_results.csv", index=False)

    best_df = find_best_eager_configurations(agg_df, top_n=5)
    best_df.to_csv(out_path / f"{model_type}_best_configurations.csv", index=False)

    stat_df = generate_eager_statistical_comparisons(results_df, model_type, out_path)

    knn_comp_df: pd.DataFrame | None = None
    if knn_baseline_dir:
        knn_comp_df = compare_with_knn_baseline(
            results_df, knn_baseline_dir, model_type, out_path
        )

    return agg_df, best_df, stat_df, knn_comp_df
