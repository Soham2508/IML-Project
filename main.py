from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure src/ is on sys.path
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import pandas as pd

from iml_project.experiments.evaluation import (
    aggregate_eager_results,
    aggregate_experiment_results,
    evaluate_stage_d_comparison,
    export_eager_evaluation_report,
    export_evaluation_report,
    get_next_experiment_dir,
)
from iml_project.experiments.runner import (
    run_experiment_matrix,
    run_rf_experiments,
    run_svm_experiments,
)


def run_knn_pipeline(
    dataset_name: str,
    k_values: list[int],
    distances: list[str],
    voting_strategies: list[str],
    retention_policies: list[str],
    reduction_method: str | None = None,
    reduction_distance: str | None = None,
    output_dir: str | Path = "results",
) -> pd.DataFrame:
    print("\n========================================================")
    print(f"Starting KNN evaluation pipeline for dataset: {dataset_name}")
    if reduction_method:
        print(f"Instance Reduction: {reduction_method.upper()} (reduction distance: {reduction_distance})")
    print(
        f"Configurations per fold: {len(distances)} distances x {len(k_values)} k "
        f"x {len(voting_strategies)} voting x {len(retention_policies)} retention = "
        f"{len(distances)*len(k_values)*len(voting_strategies)*len(retention_policies)}"
    )
    print(
        f"Total evaluations across 10 folds: "
        f"{10 * len(distances)*len(k_values)*len(voting_strategies)*len(retention_policies)}"
    )
    print("========================================================")

    t0 = time.perf_counter()
    df = run_experiment_matrix(
        datasets=[dataset_name],
        folds=range(10),
        distances=distances,
        k_values=k_values,
        voting_strategies=voting_strategies,
        retention_policies=retention_policies,
        reduction_method=reduction_method,
        reduction_distance=reduction_distance,
        data_root="data",
        output_dir=output_dir,
    )
    elapsed = time.perf_counter() - t0
    print(f"Completed {len(df)} evaluations for '{dataset_name}' in {elapsed:.2f} seconds.")

    agg = aggregate_experiment_results(df)
    print(f"\nTop 5 Configurations for {dataset_name} (KNN by Pooled Accuracy):")
    top5 = agg.head(5)
    for idx, row in top5.iterrows():
        print(
            f"  #{idx+1}: metric={row['distance_metric']:10s} k={row['k']} vote={row['voting_strategy']:16s} "
            f"ret={row['retention_policy']:3s} -> Acc (pooled): {row['pooled_accuracy']*100:.2f}% "
            f"(fold mean: {row['mean_accuracy']*100:.2f}%) Eff: {row['mean_efficiency_ms']:.3f}ms Ret: {row['retention_ratio']*100:.1f}%"
        )
    return df


def run_svm_pipeline(
    dataset_name: str,
    kernels: list[str],
    C: float,
    gamma: str | float,
    random_state: int,
    reduction_method: str | None = None,
    reduction_distance: str | None = None,
    output_dir: str | Path = "results",
) -> pd.DataFrame:
    print("\n========================================================")
    print(f"Starting SVM evaluation pipeline for dataset: {dataset_name}")
    if reduction_method:
        print(f"Instance Reduction: {reduction_method.upper()} (reduction distance: {reduction_distance})")
    print(f"Configurations per fold: {len(kernels)} kernels (C={C}, gamma={gamma})")
    print(f"Total evaluations across 10 folds: {10 * len(kernels)}")
    print("========================================================")

    t0 = time.perf_counter()
    df = run_svm_experiments(
        datasets=[dataset_name],
        folds=range(10),
        kernels=kernels,
        C=C,
        gamma=gamma,
        random_state=random_state,
        reduction_method=reduction_method,
        reduction_distance=reduction_distance,
        data_root="data",
        output_dir=output_dir,
    )
    elapsed = time.perf_counter() - t0
    print(f"Completed {len(df)} evaluations for '{dataset_name}' in {elapsed:.2f} seconds.")

    agg = aggregate_eager_results(df, "svm")
    print(f"\nSVM Configurations for {dataset_name} (by Pooled Accuracy):")
    for idx, row in agg.iterrows():
        print(
            f"  #{idx+1}: kernel={row['kernel']:8s} C={row['C']} gamma={row['gamma']:6s} -> "
            f"Acc (pooled): {row['pooled_accuracy']*100:.2f}% (fold mean: {row['mean_accuracy']*100:.2f}%) "
            f"Fit: {row['mean_fit_time_s']*1000.0:.2f}ms Eff: {row['mean_efficiency_ms']:.3f}ms Ret: {row['retention_ratio']*100:.1f}%"
        )
    return df


def run_rf_pipeline(
    dataset_name: str,
    n_estimators_list: list[int],
    criteria_list: list[str],
    max_depth: int | None,
    random_state: int,
    n_jobs: int,
    reduction_method: str | None = None,
    reduction_distance: str | None = None,
    output_dir: str | Path = "results",
) -> pd.DataFrame:
    print("\n========================================================")
    print(f"Starting Random Forest evaluation pipeline for dataset: {dataset_name}")
    if reduction_method:
        print(f"Instance Reduction: {reduction_method.upper()} (reduction distance: {reduction_distance})")
    print(
        f"Configurations per fold: {len(n_estimators_list)} n_estimators x {len(criteria_list)} criteria = "
        f"{len(n_estimators_list)*len(criteria_list)}"
    )
    print(f"Total evaluations across 10 folds: {10 * len(n_estimators_list)*len(criteria_list)}")
    print("========================================================")

    t0 = time.perf_counter()
    df = run_rf_experiments(
        datasets=[dataset_name],
        folds=range(10),
        n_estimators_list=n_estimators_list,
        criteria_list=criteria_list,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs,
        reduction_method=reduction_method,
        reduction_distance=reduction_distance,
        data_root="data",
        output_dir=output_dir,
    )
    elapsed = time.perf_counter() - t0
    print(f"Completed {len(df)} evaluations for '{dataset_name}' in {elapsed:.2f} seconds.")

    agg = aggregate_eager_results(df, "rf")
    print(f"\nRandom Forest Configurations for {dataset_name} (by Pooled Accuracy):")
    for idx, row in agg.iterrows():
        print(
            f"  #{idx+1}: n_estimators={row['n_estimators']:3d} crit={row['criterion']:8s} -> "
            f"Acc (pooled): {row['pooled_accuracy']*100:.2f}% (fold mean: {row['mean_accuracy']*100:.2f}%) "
            f"Fit: {row['mean_fit_time_s']*1000.0:.2f}ms Eff: {row['mean_efficiency_ms']:.3f}ms Ret: {row['retention_ratio']*100:.1f}%"
        )
    return df


def _print_evaluation_summary(
    eval_dict: dict[str, Any],
    model_name: str,
    target_datasets: list[str],
    is_reduction: bool,
) -> None:
    """Print structured summary of single-dataset descriptive rankings or multi-dataset Friedman test."""
    print(f"\n========================================================")
    print(f"Statistical Evaluation Summary for {model_name.upper()}")
    print("========================================================")

    if len(target_datasets) == 1:
        ds = target_datasets[0]
        print(f"Mode: Single-Dataset Descriptive Leaderboards (dataset: '{ds}')")
        print("Note: Cross-validation folds are not independent datasets; fold-level Friedman inference is omitted.")

        if "top10_accuracy" in eval_dict:
            print(f"\nTop 5 Configurations by Accuracy for {ds}:")
            for idx, r in eval_dict["top10_accuracy"].head(5).iterrows():
                cid = r.get("config_id", f"config_{idx+1}")
                print(f"  #{idx+1}: {cid} -> Pooled Acc: {r['pooled_accuracy']*100:.2f}%, Eff: {r['mean_efficiency_ms']:.3f}ms")

        if "top10_efficiency" in eval_dict:
            print(f"\nTop 5 Configurations by Efficiency for {ds}:")
            for idx, r in eval_dict["top10_efficiency"].head(5).iterrows():
                cid = r.get("config_id", f"config_{idx+1}")
                print(f"  #{idx+1}: {cid} -> Eff: {r['mean_efficiency_ms']:.3f}ms, Pooled Acc: {r['pooled_accuracy']*100:.2f}%")

        if is_reduction and "top10_retention" in eval_dict:
            print(f"\nTop 5 Configurations by Storage / Retention Ratio for {ds}:")
            for idx, r in eval_dict["top10_retention"].head(5).iterrows():
                cid = r.get("config_id", f"config_{idx+1}")
                print(f"  #{idx+1}: {cid} -> Ret: {r['retention_ratio']*100:.1f}%, Pooled Acc: {r['pooled_accuracy']*100:.2f}%")

    else:
        print(f"Mode: Multi-Dataset Friedman Statistical Testing across {len(target_datasets)} datasets: {target_datasets}")

        if "accuracy_test" in eval_dict:
            t = eval_dict["accuracy_test"]
            print(f"\nFriedman Omnibus Test (Accuracy):")
            print(f"  D={t['D']} datasets, M={t['M']} configurations")
            print(f"  chi_F^2 (corrected) = {t['statistic_chi2']:.4f} (df={t['df_chi2']}, p={t['p_value_chi2']:.4e})")
            if not pd.isna(t.get("statistic_f")):
                print(f"  Iman-Davenport F_F = {t['statistic_f']:.4f} (df1={t['df_f1']}, df2={t['df_f2']}, p={t['p_value_f']:.4e})")
            if not pd.isna(t.get("cd_nemenyi_005")):
                print(f"  Nemenyi Critical Difference (alpha=0.05): CD = {t['cd_nemenyi_005']:.4f}")

        if "top10_accuracy" in eval_dict:
            print(f"\nTop 5 Configurations Across Datasets by Accuracy (lowest average rank first):")
            for idx, r in eval_dict["top10_accuracy"].head(5).iterrows():
                cid = r.get("config_id", f"config_{idx+1}")
                print(f"  #{idx+1}: {cid} -> Avg Rank: {r['average_rank']:.2f}")

        if "efficiency_test" in eval_dict:
            t = eval_dict["efficiency_test"]
            print(f"\nFriedman Omnibus Test (Efficiency):")
            print(f"  chi_F^2 (corrected) = {t['statistic_chi2']:.4f} (df={t['df_chi2']}, p={t['p_value_chi2']:.4e})")
            if not pd.isna(t.get("statistic_f")):
                print(f"  Iman-Davenport F_F = {t['statistic_f']:.4f} (df1={t['df_f1']}, df2={t['df_f2']}, p={t['p_value_f']:.4e})")

        if is_reduction and "retention_test" in eval_dict:
            t = eval_dict["retention_test"]
            print(f"\nFriedman Omnibus Test (Storage / Retention Ratio):")
            print(f"  chi_F^2 (corrected) = {t['statistic_chi2']:.4f} (df={t['df_chi2']}, p={t['p_value_chi2']:.4e})")

        print(f"\nMethodological Limitations:")
        print(f"  {eval_dict.get('accuracy_test', {}).get('limitations', 'Demsar rule of thumb: D>10, M>5.')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IML Work 2: Classification with Lazy Learning, SVM, and RF"
    )

    # 1. Singular Model Family Selector
    parser.add_argument(
        "--model",
        type=str,
        default="knn",
        choices=["knn", "svm", "rf", "all"],
        help="Model family to run ('knn', 'svm', 'rf', or 'all'; default: 'knn')",
    )

    # 2. Instance Reduction Options (Preprocessing Stage)
    parser.add_argument(
        "--reduction-method",
        "--instance-reduction",
        type=str,
        default=None,
        choices=["fcnn3", "rng2", "drop3"],
        help="Instance reduction algorithm applied at the preprocessing stage ('fcnn3', 'rng2', 'drop3'; default: None, ignored if not specified)",
    )
    parser.add_argument(
        "--reduction-distance",
        "--reduction-metric",
        type=str,
        default="euclidean",
        choices=["euclidean", "manhattan", "canberra", "clark", "heom", "ivdm"],
        help="Distance metric used strictly for instance reduction during preprocessing (default: 'euclidean')",
    )

    # 3. Dataset and Output Options
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["heart-statlog", "grid", "tao-grid", "all"],
        help="Dataset to run experiments for (default: all)",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        type=str,
        default="results",
        dest="output_dir",
        help="Base output directory for CSV reports (default: 'results')",
    )
    parser.add_argument(
        "--knn-baseline-dir",
        type=str,
        default=None,
        help="Path to an existing KNN experiment directory for statistical comparison (SVM/RF only)",
    )

    # 4. KNN Specific Arguments
    parser.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=None,
        help="List of k values for KNN (default: 1 3 5 7 9)",
    )
    parser.add_argument(
        "--distances",
        nargs="+",
        type=str,
        default=None,
        help="Distance metrics for KNN (default: euclidean manhattan canberra clark heom ivdm)",
    )
    parser.add_argument(
        "--voting",
        nargs="+",
        type=str,
        default=None,
        help="Voting strategies for KNN (default: majority inverse_distance)",
    )
    parser.add_argument(
        "--retention",
        nargs="+",
        type=str,
        default=None,
        help="Retention policies for KNN (default: nr ar dc og)",
    )

    # 5. SVM Specific Arguments
    parser.add_argument(
        "--svm-kernels",
        nargs="+",
        type=str,
        choices=["linear", "rbf", "poly", "sigmoid"],
        default=None,
        help="Kernels for SVM (default: linear rbf)",
    )
    parser.add_argument(
        "--svm-c",
        type=float,
        default=None,
        help="Regularization parameter C for SVM (strictly positive float; default: 1.0)",
    )
    parser.add_argument(
        "--svm-gamma",
        type=str,
        default=None,
        help="Kernel coefficient gamma for SVM ('scale', 'auto', or a float; default: scale)",
    )

    # 6. Random Forest Specific Arguments
    parser.add_argument(
        "--rf-n-estimators",
        nargs="+",
        type=int,
        default=None,
        help="List of n_estimators for Random Forest (default: 100 200)",
    )
    parser.add_argument(
        "--rf-criteria",
        nargs="+",
        type=str,
        choices=["gini", "entropy", "log_loss"],
        default=None,
        help="Split criteria for Random Forest (default: gini entropy)",
    )
    parser.add_argument(
        "--rf-max-depth",
        type=int,
        default=None,
        help="Max depth for Random Forest trees (optional integer, default: None)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state seed (default: 42)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs for Random Forest (default: -1)",
    )

    args = parser.parse_args()

    # Model-Specific Option Dispatches & Incompatibility Validation
    knn_explicit = {
        "--k": args.k,
        "--distances": args.distances,
        "--voting": args.voting,
        "--retention": args.retention,
    }
    svm_explicit = {
        "--svm-kernels": args.svm_kernels,
        "--svm-c": args.svm_c,
        "--svm-gamma": args.svm_gamma,
    }
    rf_explicit = {
        "--rf-n-estimators": args.rf_n_estimators,
        "--rf-criteria": args.rf_criteria,
        "--rf-max-depth": args.rf_max_depth,
    }

    if args.model == "knn":
        for opt, val in svm_explicit.items():
            if val is not None:
                parser.error(f"{opt} cannot be used when --model knn is selected.")
        for opt, val in rf_explicit.items():
            if val is not None:
                parser.error(f"{opt} cannot be used when --model knn is selected.")
        if args.knn_baseline_dir is not None:
            parser.error("--knn-baseline-dir is only applicable for --model svm or --model rf.")

    elif args.model == "svm":
        for opt, val in knn_explicit.items():
            if val is not None:
                parser.error(f"{opt} cannot be used when --model svm is selected.")
        for opt, val in rf_explicit.items():
            if val is not None:
                parser.error(f"{opt} cannot be used when --model svm is selected.")

    elif args.model == "rf":
        for opt, val in knn_explicit.items():
            if val is not None:
                parser.error(f"{opt} cannot be used when --model rf is selected.")
        for opt, val in svm_explicit.items():
            if val is not None:
                parser.error(f"{opt} cannot be used when --model rf is selected.")

    # Apply defaults
    k_values = args.k if args.k is not None else [1, 3, 5, 7, 9]
    distances = (
        args.distances
        if args.distances is not None
        else ["euclidean", "manhattan", "canberra", "clark", "heom", "ivdm"]
    )
    voting = (
        args.voting
        if args.voting is not None
        else ["majority", "inverse_distance"]
    )
    retention = (
        args.retention
        if args.retention is not None
        else ["nr", "ar", "dc", "og"]
    )

    svm_kernels = args.svm_kernels if args.svm_kernels is not None else ["linear", "rbf"]
    svm_c = args.svm_c if args.svm_c is not None else 1.0
    if svm_c <= 0:
        parser.error(f"--svm-c must be strictly positive, got {svm_c}")

    svm_gamma = args.svm_gamma if args.svm_gamma is not None else "scale"
    if svm_gamma not in {"scale", "auto"}:
        try:
            gamma_float = float(svm_gamma)
            if gamma_float <= 0:
                parser.error(f"--svm-gamma float must be > 0, got {gamma_float}")
        except ValueError:
            parser.error(
                f"--svm-gamma must be 'scale', 'auto', or a positive float, got '{svm_gamma}'"
            )

    rf_n_estimators = (
        args.rf_n_estimators if args.rf_n_estimators is not None else [100, 200]
    )
    for n in rf_n_estimators:
        if n <= 0:
            parser.error(f"--rf-n-estimators values must be strictly positive, got {n}")

    rf_criteria = (
        args.rf_criteria if args.rf_criteria is not None else ["gini", "entropy"]
    )
    rf_max_depth = args.rf_max_depth
    if rf_max_depth is not None and rf_max_depth <= 0:
        parser.error(f"--rf-max-depth must be strictly positive, got {rf_max_depth}")

    # Resolve output directory with auto-versioning
    if args.output_dir == "results":
        out_dir = get_next_experiment_dir("results")
    else:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    print("\n========================================================")
    print(f"Selected Model Family: {args.model.upper()}")
    print(f"Instance Reduction:    {args.reduction_method.upper() if args.reduction_method else 'None (Full Training Data)'}")
    if args.reduction_method:
        print(f"Reduction Distance:    {args.reduction_distance}")
    print(f"Output directory:      {out_dir}")
    print("========================================================")

    target_datasets = (
        ["heart-statlog", "grid"]
        if args.dataset == "all"
        else [args.dataset]
    )

    is_red = bool(args.reduction_method and str(args.reduction_method).lower() != "none")

    # Dispatch logic
    if args.model == "knn":
        all_dfs = []
        for ds in target_datasets:
            df_ds = run_knn_pipeline(
                dataset_name=ds,
                k_values=k_values,
                distances=distances,
                voting_strategies=voting,
                retention_policies=retention,
                reduction_method=args.reduction_method,
                reduction_distance=args.reduction_distance,
                output_dir=out_dir,
            )
            all_dfs.append(df_ds)

        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.to_csv(out_dir / "all_experiments_results.csv", index=False)

        agg_df, best_df, eval_dict = export_evaluation_report(combined_df, output_dir=out_dir)
        _print_evaluation_summary(eval_dict, "knn", target_datasets, is_red)

    elif args.model == "svm":
        all_dfs = []
        for ds in target_datasets:
            df_ds = run_svm_pipeline(
                dataset_name=ds,
                kernels=svm_kernels,
                C=svm_c,
                gamma=svm_gamma,
                random_state=args.random_state,
                reduction_method=args.reduction_method,
                reduction_distance=args.reduction_distance,
                output_dir=out_dir,
            )
            all_dfs.append(df_ds)

        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.to_csv(out_dir / "svm_results.csv", index=False)

        agg_df, best_df, eval_dict, knn_comp = export_eager_evaluation_report(
            combined_df,
            model_type="svm",
            output_dir=out_dir,
            knn_baseline_dir=args.knn_baseline_dir,
        )
        _print_evaluation_summary(eval_dict, "svm", target_datasets, is_red)
        if knn_comp is not None:
            print(f"\nModel Comparison vs KNN Baseline saved ({len(knn_comp)} rows).")

    elif args.model == "rf":
        all_dfs = []
        for ds in target_datasets:
            df_ds = run_rf_pipeline(
                dataset_name=ds,
                n_estimators_list=rf_n_estimators,
                criteria_list=rf_criteria,
                max_depth=rf_max_depth,
                random_state=args.random_state,
                n_jobs=args.n_jobs,
                reduction_method=args.reduction_method,
                reduction_distance=args.reduction_distance,
                output_dir=out_dir,
            )
            all_dfs.append(df_ds)

        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.to_csv(out_dir / "rf_results.csv", index=False)

        agg_df, best_df, eval_dict, knn_comp = export_eager_evaluation_report(
            combined_df,
            model_type="rf",
            output_dir=out_dir,
            knn_baseline_dir=args.knn_baseline_dir,
        )
        _print_evaluation_summary(eval_dict, "rf", target_datasets, is_red)
        if knn_comp is not None:
            print(f"\nModel Comparison vs KNN Baseline saved ({len(knn_comp)} rows).")

    elif args.model == "all":
        # Run all three model families and execute Stage D comparison
        knn_dfs, svm_dfs, rf_dfs = [], [], []
        for ds in target_datasets:
            knn_dfs.append(
                run_knn_pipeline(
                    dataset_name=ds,
                    k_values=k_values,
                    distances=distances,
                    voting_strategies=voting,
                    retention_policies=retention,
                    reduction_method=args.reduction_method,
                    reduction_distance=args.reduction_distance,
                    output_dir=out_dir,
                )
            )
            svm_dfs.append(
                run_svm_pipeline(
                    dataset_name=ds,
                    kernels=svm_kernels,
                    C=svm_c,
                    gamma=svm_gamma,
                    random_state=args.random_state,
                    reduction_method=args.reduction_method,
                    reduction_distance=args.reduction_distance,
                    output_dir=out_dir,
                )
            )
            rf_dfs.append(
                run_rf_pipeline(
                    dataset_name=ds,
                    n_estimators_list=rf_n_estimators,
                    criteria_list=rf_criteria,
                    max_depth=rf_max_depth,
                    random_state=args.random_state,
                    n_jobs=args.n_jobs,
                    reduction_method=args.reduction_method,
                    reduction_distance=args.reduction_distance,
                    output_dir=out_dir,
                )
            )

        knn_combined = pd.concat(knn_dfs, ignore_index=True)
        svm_combined = pd.concat(svm_dfs, ignore_index=True)
        rf_combined = pd.concat(rf_dfs, ignore_index=True)

        knn_combined.to_csv(out_dir / "all_experiments_results.csv", index=False)
        svm_combined.to_csv(out_dir / "svm_results.csv", index=False)
        rf_combined.to_csv(out_dir / "rf_results.csv", index=False)

        export_evaluation_report(knn_combined, output_dir=out_dir)
        export_eager_evaluation_report(svm_combined, model_type="svm", output_dir=out_dir)
        export_eager_evaluation_report(rf_combined, model_type="rf", output_dir=out_dir)

        # Stage D: Compare best configuration from each family
        stage_d_df, stage_d_stats = evaluate_stage_d_comparison(
            knn_combined, svm_combined, rf_combined, output_dir=out_dir
        )
        print("\n========================================================")
        print("Stage D: Best of Family Comparison (KNN vs SVM vs RF)")
        print("========================================================")
        for _, r in stage_d_df.iterrows():
            print(f"  {r['dataset']} | {r['model_family'].upper():3s} -> Acc: {r['pooled_accuracy']*100:.2f}%, Eff: {r['mean_efficiency_ms']:.3f}ms")


if __name__ == "__main__":
    main()
