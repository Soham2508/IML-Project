from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from iml_project.data.folds import get_dataset_fold_map
from iml_project.evaluation.runner import run_week1_experiments


def cmd_validate_data(data_root: str | Path) -> None:
    dataset_map = get_dataset_fold_map(data_root)
    for dataset_name, folds in dataset_map.items():
        print(f"{dataset_name}: {len(folds)} fold pairs found")
        for fold in folds:
            print(
                f"  fold {fold.fold_index:06d}: {fold.train_path.name} | {fold.test_path.name}"
            )


def cmd_week1(data_root: str | Path, output_dir: str | Path) -> None:
    results = run_week1_experiments(data_root=data_root, output_dir=output_dir)
    print(
        f"Saved {len(results['svm_fold'])} SVM rows and {len(results['random_forest_fold'])} Random Forest rows."
    )
    print(f"Outputs written to {Path(output_dir).resolve()}.")


def cmd_week2(data_root: str | Path) -> None:
    dataset_names = sorted(get_dataset_fold_map(data_root).keys())
    print(
        "Week 2 custom k-NN scaffold is not yet implemented in this repository version."
    )
    print(
        f"Configured datasets: {', '.join(dataset_names) if dataset_names else 'none found'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="IML project runner")
    parser.add_argument(
        "--experiment",
        choices=["validate-data", "week1", "week2"],
        default="validate-data",
    )
    parser.add_argument("--dataset", choices=["all"], default="all")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="results/week1")
    args = parser.parse_args()

    if args.dataset != "all":
        raise ValueError("Only --dataset all is supported in this scaffold.")

    if args.experiment == "validate-data":
        cmd_validate_data(args.data_root)
    elif args.experiment == "week1":
        cmd_week1(args.data_root, args.output_dir)
    else:
        cmd_week2(args.data_root)


if __name__ == "__main__":
    main()
