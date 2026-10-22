# IML Work 2: Classification Framework (Lazy & Eager Learning)

## Overview

This repository implements a modular, reproducible classification and empirical evaluation framework for **IML Work 2 (Course 2026–2027)**.
The framework provides:

1. **Lazy Learning (KNN from scratch):** Comparative evaluation of distance/similarity metrics (`euclidean`, `manhattan`, `canberra`, `clark`, `heom`, `ivdm`), voting strategies (`majority`, `inverse_distance`), and dynamic case-base retention policies (`nr`, `ar`, `dc`, `og`).
2. **Eager Learning (SVM & Random Forest):** Integrates scikit-learn's Support Vector Machines (`SVC`) and Random Forests (`RandomForestClassifier`) directly into the predefined 10-fold cross-validation pipeline.
3. **Comparative Evaluation & Statistical Testing:** Reports identical classification and efficiency metrics across all model families and performs paired two-tailed Student's $t$-tests ($df = 9$) across configurations and against prior KNN baseline runs (`--knn-baseline-dir`).
4. **Zero Data Leakage:** All data preprocessing (feature-type inference, normalization bounds $[0, 1]$, imputation medians/modes, one-hot encoding, and IVDM conditional probabilities) is strictly fitted on each training fold.

---

## Repository Structure

```
IML-Project/
├── data/
│   ├── grid/                       # TAO-Grid 10-fold CV ARFF files
│   └── heart-statlog/              # Heart-Statlog 10-fold CV ARFF files
├── results/                        # Auto-versioned results directories (experiment_results_{N})
│   └── experiment_results_1/       # Prior baseline KNN results
├── src/
│   └── iml_project/
│       ├── core/
│       │   ├── knn.py              # KNNClassifier & kNNAlgorithm from scratch
│       │   ├── svm.py              # svmAlgorithm wrapper using SVC
│       │   ├── random_forest.py    # rfAlgorithm wrapper using RandomForestClassifier
│       │   ├── voting.py           # Majority and Inverse Distance Weighted voting
│       │   └── retention.py        # CaseBase, NR, AR, DC, and OG retention policies
│       ├── experiments/
│       │   ├── runner.py           # Experiment grids for KNN, SVM, and RF
│       │   └── evaluation.py       # Aggregation, statistical tests & baseline comparison
│       └── utils/
│           ├── arff_reader.py      # ARFF table loading and fold alignment
│           ├── distance_metrics.py # Euclidean, Manhattan, Canberra, Clark, HEOM, IVDM
│           ├── folds.py            # Fold discovery and validation
│           └── preprocessing.py    # Leakage-free feature inference & normalization
├── tests/
│   ├── test_preprocessing.py       # Preprocessing and normalization tests
│   ├── test_data_pipeline.py       # Fold alignment, feature inference, leakage tests
│   ├── test_core_algorithms.py     # Hand-calculated metric tests, voting, retention
│   ├── test_experiments.py         # KNN experiment matrix and directory tests
│   └── test_eager_models.py        # SVM/RF algorithms, eager runner, CLI validation
├── main.py                         # Unified CLI entry point
├── CLASS_DESIGN.md                 # Condensed class & method reference
├── DATAFLOW.md                     # Architecture diagram and end-to-end dataflow
├── requirements.txt                # Dependencies
└── README.md
```

---

## Setup and Installation

1. Create and activate a Python 3.11 virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

---

## Running the Unit Tests

Execute the complete test suite (33 unit tests across 5 test modules):

```powershell
$env:PYTHONPATH="src"
python -m unittest discover tests
```

To run code quality and linting checks:

```powershell
ruff check src tests main.py
```

---

## Running Experiments via CLI

`main.py` requires selecting exactly one model family per invocation via `--model {knn, svm, rf}` (default: `knn`).

### 1. Run Lazy Learning (KNN Baseline)

Run the full parameter grid or custom subsets on `heart-statlog`, `grid`, or `all`:

```powershell
$env:PYTHONPATH="src"
python main.py --model knn --dataset heart-statlog --k 1 3 5 7 9
```

*Customizing KNN options:*

- `--distances`: Distance metrics (`euclidean`, `manhattan`, `canberra`, `clark`, `heom`, `ivdm`)
- `--k`: Neighbor counts (e.g. `--k 1 3 5 7 9`)
- `--voting`: Voting policies (`majority`, `inverse_distance`)
- `--retention`: Retention policies (`nr`, `ar`, `dc`, `og`)

### 2. Run Support Vector Machines (SVM)

Run SVM cross-validation across kernels with automatic comparison against prior KNN baseline:

```powershell
$env:PYTHONPATH="src"
python main.py --model svm --dataset heart-statlog --knn-baseline-dir results/experiment_results_1
```

*Customizing SVM options:*

- `--svm-kernels`: List of kernels (`linear`, `rbf`, `poly`, `sigmoid`; default: `linear`, `rbf`)
- `--svm-c`: Regularization parameter $C > 0$ (default: `1.0`)
- `--svm-gamma`: Kernel coefficient (`scale`, `auto`, or positive float; default: `scale`)

### 3. Run Random Forest (RF)

Run Random Forest cross-validation across tree counts and impurity criteria:

```powershell
$env:PYTHONPATH="src"
python main.py --model rf --dataset heart-statlog --knn-baseline-dir results/experiment_results_1
```

*Customizing Random Forest options:*

- `--rf-n-estimators`: Tree counts (default: `100 200`)
- `--rf-criteria`: Split criteria (`gini`, `entropy`; default: `gini`, `entropy`)
- `--rf-max-depth`: Maximum tree depth (optional, default: `None` for full depth)
- `--n-jobs`: Parallel worker threads (default: `-1` for all CPU cores)

### 4. Mutual Exclusivity & Safety

Passing options from another model family (e.g. `--k` with `--model svm` or `--svm-kernels` with `--model knn`) immediately produces an informative error:

```
main.py: error: --k cannot be used when --model svm is selected.
```

---

## Output Files & Directory Versioning

Each execution automatically creates a new sequentially numbered directory under `results/`:
`results/experiment_results_{N}/` (e.g. `results/experiment_results_1`, `results/experiment_results_2`, ...).

Depending on the model executed, the following CSV artifacts are exported:

| Artifact                                                                  | Description                                                                                                                                                                           |
| ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `all_experiments_results.csv` / `{model}_results.csv`                 | Raw fold-by-fold results containing all classification and timing metrics.                                                                                                            |
| `aggregated_results.csv` / `{model}_aggregated_results.csv`           | Mean and standard deviation across all 10 folds for every metric.                                                                                                                     |
| `best_configurations.csv` / `{model}_best_configurations.csv`         | Top performing hyper-parameter configurations ranked by accuracy and efficiency.                                                                                                      |
| `statistical_comparisons.csv` / `{model}_statistical_comparisons.csv` | Paired Student's $t$-tests across kernels, metrics, voting, or retention policies.                                                                                                  |
| `model_comparison.csv`                                                  | Paired Student's $t$-test comparing the best eager configuration against the best KNN baseline configuration on matching folds (generated when `--knn-baseline-dir` is provided). |

---

## Metrics Reported Across All Models

To ensure strict comparability between lazy and eager learners, all models record:

- **`accuracy`**: Fraction of correct predictions ($\frac{\text{Correct}}{\text{Total}}$).
- **`balanced_accuracy`**: Macro-average of recall across all classes ($\frac{1}{|C|} \sum_c \text{Recall}_c$), accounting for class imbalance.
- **`precision_macro`**: Macro-averaged precision score across classes.
- **`recall_macro`**: Macro-averaged recall score across classes.
- **`f1_macro`**: Macro-averaged F1 score ($\frac{2 \cdot P \cdot R}{P + R}$).
- **`fit_time`**: Model fitting latency in seconds (tracked for SVM, RF, and KNN instance base creation).
- **`mean_prediction_time`**: Average query inference latency in seconds.
- **`total_prediction_time`**: Cumulative test set prediction latency in seconds.
- **`retention_ratio`**: Ratio of final instance count to initial count ($\frac{|CB_{final}|}{|CB_{initial}|}$; equals $1.0$ for eager models).
