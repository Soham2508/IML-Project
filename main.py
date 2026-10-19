from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import pandas as pd

from iml_project.experiments.evaluation import (
    aggregate_experiment_results,
    compute_paired_ttest,
    export_evaluation_report,
    find_best_configurations,
    get_next_experiment_dir,
)
from iml_project.experiments.runner import run_experiment_matrix


def run_dataset_pipeline(
    dataset_name: str,
    k_values: list[int],
    distances: list[str],
    voting_strategies: list[str],
    retention_policies: list[str],
    output_dir: str = "results",
) -> pd.DataFrame:
    print(f"\n========================================================")
    print(f"Starting KNN evaluation pipeline for dataset: {dataset_name}")
    print(f"Configurations per fold: {len(distances)} distances x {len(k_values)} k x {len(voting_strategies)} voting x {len(retention_policies)} retention = {len(distances)*len(k_values)*len(voting_strategies)*len(retention_policies)}")
    print(f"Total evaluations across 10 folds: {10 * len(distances)*len(k_values)*len(voting_strategies)*len(retention_policies)}")
    print(f"========================================================")

    t0 = time.perf_counter()
    df = run_experiment_matrix(
        datasets=[dataset_name],
        folds=range(10),
        distances=distances,
        k_values=k_values,
        voting_strategies=voting_strategies,
        retention_policies=retention_policies,
        data_root="data",
        output_dir=output_dir,
    )
    elapsed = time.perf_counter() - t0
    print(f"Completed {len(df)} evaluations for '{dataset_name}' in {elapsed:.2f} seconds.")

    # Aggregate results for this dataset
    agg = aggregate_experiment_results(df)
    print(f"\nTop 5 Configurations for {dataset_name}:")
    top5 = agg.head(5)
    for idx, row in top5.iterrows():
        print(
            f"  #{idx+1}: metric={row['distance_metric']:10s} k={row['k']} vote={row['voting_strategy']:16s} "
            f"ret={row['retention_policy']:3s} -> Acc: {row['mean_accuracy']*100:.2f}% "
            f"(std: {row['std_accuracy']*100:.2f}%) Eff: {row['mean_efficiency_ms']:.3f}ms Ret: {row['mean_retention_ratio']*100:.1f}%"
        )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="IML Work 2: KNN Experiment Framework")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["heart-statlog", "grid", "tao-grid", "all"],
        help="Dataset to run experiments for (default: all)",
    )
    parser.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=[1, 3, 5, 7, 9],
        help="List of k values (default: 1 3 5 7 9)",
    )
    parser.add_argument(
        "--distances",
        nargs="+",
        type=str,
        default=["euclidean", "manhattan", "canberra", "clark", "heom", "ivdm"],
        help="Distance metrics",
    )
    parser.add_argument(
        "--voting",
        nargs="+",
        type=str,
        default=["majority", "inverse_distance"],
        help="Voting strategies",
    )
    parser.add_argument(
        "--retention",
        nargs="+",
        type=str,
        default=["nr", "ar", "dc", "og"],
        help="Retention policies",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for CSV reports",
    )

    args = parser.parse_args()

    # Determine output directory (auto-increment results/experiment_results_{N})
    if args.output_dir == "results":
        out_dir = get_next_experiment_dir("results")
    else:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n========================================================")
    print(f"Output directory: {out_dir}")
    print(f"========================================================")

    target_datasets = (
        ["heart-statlog", "grid"]
        if args.dataset == "all"
        else [args.dataset]
    )

    all_dfs = []
    for ds in target_datasets:
        df_ds = run_dataset_pipeline(
            dataset_name=ds,
            k_values=args.k,
            distances=args.distances,
            voting_strategies=args.voting,
            retention_policies=args.retention,
            output_dir=out_dir,
        )
        all_dfs.append(df_ds)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.to_csv(out_dir / "all_experiments_results.csv", index=False)

    agg_df, best_df, stat_df = export_evaluation_report(combined_df, output_dir=out_dir)
    print(f"\nResults successfully saved to {out_dir}:")
    print(f"  - all_experiments_results.csv ({len(combined_df)} rows)")
    print(f"  - aggregated_results.csv ({len(agg_df)} rows)")
    print(f"  - best_configurations.csv ({len(best_df)} rows)")
    print(f"  - statistical_comparisons.csv ({len(stat_df)} tests)")


if __name__ == "__main__":
    main()
