from pathlib import Path

import numpy as np
from scipy.io import arff
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DATA_DIRECTORY = REPOSITORY_ROOT / "data" / "heart-statlog"
REPORT_PATH = REPOSITORY_ROOT / "results" / "heart_statlog_svm_report.txt"
CLASS_LABELS = ("absent", "present")


def _load_split(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    records, _ = arff.loadarff(path)
    if records.dtype.names is None or "class" not in records.dtype.names:
        raise ValueError(f"Expected a 'class' attribute in {path}")

    feature_names = tuple(name for name in records.dtype.names if name != "class")
    features = np.column_stack([records[name] for name in feature_names]).astype(float)
    labels = np.asarray(
        [
            value.decode("ascii") if isinstance(value, bytes) else str(value)
            for value in records["class"]
        ]
    )
    return features, labels, feature_names


def build_report(data_directory: Path = DATA_DIRECTORY) -> str:
    test_paths = sorted(data_directory.glob("*.test.arff"))
    if not test_paths:
        raise FileNotFoundError(f"No test folds found in {data_directory}")

    fold_results: list[tuple[str, int, int, float]] = []
    all_true: list[str] = []
    all_predicted: list[str] = []

    for test_path in test_paths:
        fold_name = test_path.name[: -len(".test.arff")]
        train_path = data_directory / f"{fold_name}.train.arff"
        if not train_path.is_file():
            raise FileNotFoundError(f"Missing training fold: {train_path}")

        X_train, y_train, train_features = _load_split(train_path)
        X_test, y_test, test_features = _load_split(test_path)
        if train_features != test_features:
            raise ValueError(f"Feature mismatch between {train_path} and {test_path}")

        classifier = make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=1.0, gamma="scale"),
        )
        classifier.fit(X_train, y_train)
        predicted = classifier.predict(X_test)

        fold_results.append(
            (
                fold_name,
                len(y_train),
                len(y_test),
                float(accuracy_score(y_test, predicted)),
            )
        )
        all_true.extend(y_test.tolist())
        all_predicted.extend(predicted.tolist())

    fold_accuracies = np.asarray([result[3] for result in fold_results])
    accuracy = float(accuracy_score(all_true, all_predicted))
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true,
        all_predicted,
        labels=["present"],
        average="binary",
        pos_label="present",
        zero_division=0,
    )
    matrix = confusion_matrix(all_true, all_predicted, labels=CLASS_LABELS)
    correct_predictions = int(np.trace(matrix))

    lines = [
        "Heart Statlog SVM Classification Report",
        "=" * 40,
        "",
        f"Data: {data_directory}",
        f"Evaluation: {len(fold_results)} supplied train/test folds",
        "Model: StandardScaler followed by SVC(kernel='rbf', C=1.0, gamma='scale')",
        "Scaling was fit on each training fold only.",
        "",
        "Fold results:",
        "Fold                                      Train  Test  Accuracy",
    ]
    lines.extend(
        f"{name:<42} {train_size:>5} {test_size:>5}  {fold_accuracy:.4f}"
        for name, train_size, test_size, fold_accuracy in fold_results
    )
    lines.extend(
        [
            "",
            f"Total test examples: {len(all_true)}",
            f"Mean fold accuracy: {float(fold_accuracies.mean()):.4f}",
            f"Fold accuracy standard deviation (population): {float(fold_accuracies.std()):.4f}",
            f"Overall accuracy: {accuracy:.4f} ({correct_predictions}/{len(all_true)})",
            f"Precision for 'present': {float(precision):.4f}",
            f"Recall for 'present': {float(recall):.4f}",
            f"F1 score for 'present': {float(f1):.4f}",
            "",
            "Confusion matrix (rows=true, columns=predicted; absent, present):",
            str(matrix),
            "",
            "Per-class metrics:",
            classification_report(
                all_true,
                all_predicted,
                labels=list(CLASS_LABELS),
                target_names=list(CLASS_LABELS),
                digits=4,
                zero_division=0,
            ).rstrip(),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    report = build_report()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
