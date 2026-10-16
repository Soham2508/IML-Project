# IML Project - Week 1 scaffold

Group: 12

Team members:
- Student 1: [Full name]
- Student 2: [Full name]
- Student 3: [Full name]
- Student 4: [Full name]

Python requirement: Python 3.11
Operating system validation: Windows 11 / Windows PowerShell

This repository is the course project scaffold for Work 2: Classification with Lazy Learning, SVM and Random Forest. The repository currently contains the Week 1 implementation scaffold and the required dataset-validation utilities. The Week 2 custom k-NN experiment remains a planned extension, but the project structure and interfaces are in place for it.

## Dataset requirement mismatch notice

The selected datasets in the assignment table are TAO-Grid and Heart-Statlog, and the project is intentionally written to support both numeric-only and mixed-type data. However, the assignment table reports zero nominal attributes for this pair, which conflicts with a later statement that at least one dataset should contain both numeric and nominal attributes. The code does not silently swap datasets; instead it keeps the selected pair configurable and uses type-aware preprocessing structures that can accommodate nominal features if the TA changes the dataset choice.

This mismatch should be confirmed with the teaching team before final submission. Until then, the project documents the current pair as all-numeric benchmarking data only.

## Repository structure

```text
IML_Project/
├── AGENTS.md
├── README.md
├── requirements.txt
├── main.py
├── src/
│   └── iml_project/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── arff_reader.py
│       │   ├── folds.py
│       │   └── preprocessing.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py
│       │   └── runner.py
│       └── models/
│           ├── __init__.py
│           ├── distance_metrics.py
│           ├── knn.py
│           ├── retention.py
│           └── sklearn_models.py
├── data/
│   ├── grid/
│   └── heart-statlog/
├── results/
│   └── week1/
├── tests/
│   └── test_project_scaffold.py
└── .venv/
```

## Environment setup

```powershell
cd C:\Users\Soham Chakraborty\Documents\IML-Project
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Validate the data contract

```powershell
python main.py --experiment validate-data --dataset all
```

This command scans the `data/` directory and verifies that each dataset contains exactly ten train/test fold pairs with the expected fold naming pattern.

## Run the Week 1 experiment

```powershell
python main.py --experiment week1 --dataset all --output-dir results/week1
```

The script reads the supplied fold files, fits normalization parameters only on the training folds, evaluates SVM and Random Forest over the provided fold pairs, and writes the following CSV outputs:

```text
results/week1/
├── svm_fold_results.csv
├── svm_summary.csv
├── random_forest_fold_results.csv
└── random_forest_summary.csv
```

## Model configuration used

### SVM

- Model: `sklearn.svm.SVC`
- Kernels: `linear`, `rbf`
- C values: `0.1`, `1.0`, `10.0`
- Gamma: `scale` for RBF experiments
- Training: fit on each provided training fold; prediction on the matching test fold
- Timing: fit time and prediction time recorded separately using `time.perf_counter()`

### Random Forest

- Model: `sklearn.ensemble.RandomForestClassifier`
- `n_estimators`: `50`, `100`, `200`
- `criterion`: `gini`, `entropy`
- `random_state`: `42`
- Timing: fit time and prediction time recorded separately

## Data loading and preprocessing rules

- ARFF files are loaded with `scipy.io.arff.loadarff`.
- Byte-valued nominal labels are decoded to Python strings.
- Missing values are handled safely. Numeric features use train-only medians; constant numeric training features are normalized to `0` rather than dividing by zero.
- Numerical features are normalized to `[0, 1]` using the training fold min/max values only, then applied to the corresponding test fold.
- The current dataset pair is all-numeric; the code keeps metadata for future mixed-type datasets.

## Quality checks

```powershell
python -m unittest discover -s tests -v
```
