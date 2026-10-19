# IML Work 2: Classification with Lazy Learning Framework

## Overview
This repository implements the baseline K-Nearest Neighbors (KNN) classification framework from scratch for **IML Work 2 (2026–2027)**.
The framework performs a comparative study of distance/similarity metrics, voting schemes, retention policies, and cross-validation performance on two UCI benchmark datasets (`heart-statlog` and `grid`).

All code is implemented in Python 3.11 with zero data leakage: feature-type inference, normalization parameters, missing value statistics, and conditional probabilities are strictly fitted on each training fold.

---

## Repository Structure
```
IML-Project/
├── data/
│   ├── grid/                # TAO-Grid 10-fold CV ARFF files
│   └── heart-statlog/       # Heart-Statlog 10-fold CV ARFF files
├── results/                 # Generated experiment CSVs and statistical reports
│   ├── all_experiments_results.csv
│   ├── results_heart-statlog.csv
│   ├── results_grid.csv
│   ├── aggregated_results.csv
│   ├── best_configurations.csv
│   └── statistical_comparisons.csv
├── src/
│   └── iml_project/
│       ├── core/
│       │   ├── knn.py             # KNNClassifier & kNNAlgorithm from scratch
│       │   ├── voting.py          # Majority and Inverse Distance Weighted voting
│       │   └── retention.py       # CaseBase, NR, AR, DC, and OG retention policies
│       ├── experiments/
│       │   ├── runner.py          # Grid runner with fold isolation
│       │   └── evaluation.py      # Statistical paired t-tests & aggregation
│       └── utils/
│           ├── arff_reader.py     # ARFF table loading and fold alignment
│           ├── distance_metrics.py# Euclidean, Manhattan, Canberra, Clark, HEOM, IVDM
│           ├── folds.py           # Fold discovery and validation
│           └── preprocessing.py   # Data-driven feature inference & normalization
├── tests/
│   ├── test_preprocessing.py      # Legacy preprocessing tests
│   ├── test_data_pipeline.py      # Fold alignment, feature inference, leakage tests
│   ├── test_core_algorithms.py    # Hand-calculated metric tests, voting, retention
│   └── test_experiments.py        # Smoke tests and statistical comparison tests
├── main.py                        # CLI entry point to run experiments
├── requirements.txt               # Dependencies
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

Execute the comprehensive test suite (22 unit tests) with `PYTHONPATH` set to `src`:
```powershell
$env:PYTHONPATH="src"
python -m unittest discover tests
```

---

## Running Experiments

You can run experiments for `heart-statlog` and `grid` separately or together:

### 1. Run Heart-Statlog:
```powershell
$env:PYTHONPATH="src"
python main.py --dataset heart-statlog --k 1 3 5 7 9 --output_dir results
```

### 2. Run Grid:
```powershell
$env:PYTHONPATH="src"
python main.py --dataset grid --k 1 3 5 7 9 --output_dir results
```

### 3. Run Both Datasets:
```powershell
$env:PYTHONPATH="src"
python main.py --dataset all --k 1 3 5 7 9 --output_dir results
```

### Available CLI Options:
- `--dataset`: `heart-statlog`, `grid`, or `all`
- `--k`: List of neighbors, e.g. `--k 1 3 5 7 9`
- `--distances`: List of distance metrics (`euclidean`, `manhattan`, `canberra`, `clark`, `heom`, `ivdm`)
- `--voting`: List of voting strategies (`majority`, `inverse_distance`)
- `--retention`: List of retention policies (`nr`, `ar`, `dc`, `og`)
- `--output_dir`: Output directory for generated CSVs (default: `results`)

---

## Implemented Components & Mathematical Definitions

### 1. Distance Metrics
- **Euclidean** ($r=2$): $d(x, y) = \sqrt{\sum_{f} (x_f - y_f)^2}$
- **Manhattan** ($r=1$): $d(x, y) = \sum_{f} |x_f - y_f|$
- **Canberra**: $d(x, y) = \sum_{f} \frac{|x_f - y_f|}{|x_f| + |y_f|}$ (with $0/0 \to 0.0$)
- **Clark**: $d(x, y) = \sqrt{\sum_{f} \left(\frac{x_f - y_f}{x_f + y_f}\right)^2}$ (with $0/0 \to 0.0$)
- **HEOM** (Wilson & Martinez 1997): Heterogeneous Euclidean-Overlap Metric handling numeric normalization and nominal overlap ($\delta = 0$ if equal, $1$ otherwise).
- **IVDM** (Wilson & Martinez 1997): Discretizes continuous features and estimates conditional class probabilities $P(C=c \mid f=v)$ strictly on the training fold with Laplace smoothing.

### 2. Voting Policies
- **Majority Voting**: Each neighbor casts 1 vote.
- **Inverse Distance Weighted Voting**: Each neighbor casts $\frac{1}{d(q, x_c)^p}$ ($p=1$). Zero-distance neighbors take absolute precedence, preventing division by zero.
- **Tie-Breaking**: Nearest individual neighbor (1-NN), then class prior in training fold, then stable string order.

### 3. Retention Policies
- **Never Retain (NR)**: Training set remains fixed.
- **Always Retain (AR)**: Retains every solved query.
- **Different Class (DC)**: Retains query only when misclassified ($y_{pred} \neq y_{true}$).
- **Oblivion by Goodness (OG)**: Initial goodness $G_0 = 0.5$, $\alpha = 0.2$. Retrieved neighbors update goodness based on accuracy reward $r(c) \in \{0, 1\}$. Cases with $G_t(c) < 0.5$ are pruned immediately.
# IML Work 2: Classification with Lazy Learning Framework

## Overview
This repository implements the baseline K-Nearest Neighbors (KNN) classification framework from scratch for **IML Work 2 (2026–2027)**.
The framework performs a comparative study of distance/similarity metrics, voting schemes, retention policies, and cross-validation performance on two UCI benchmark datasets (`heart-statlog` and `grid`).

All code is implemented in Python 3.11 with zero data leakage: feature-type inference, normalization parameters, missing value statistics, and conditional probabilities are strictly fitted on each training fold.

---

## Repository Structure
```
IML-Project/
├── data/
│   ├── grid/                # TAO-Grid 10-fold CV ARFF files
│   └── heart-statlog/       # Heart-Statlog 10-fold CV ARFF files
├── results/                 # Generated experiment CSVs and statistical reports
│   ├── all_experiments_results.csv
│   ├── results_heart-statlog.csv
│   ├── results_grid.csv
│   ├── aggregated_results.csv
│   ├── best_configurations.csv
│   └── statistical_comparisons.csv
├── src/
│   └── iml_project/
│       ├── core/
│       │   ├── knn.py             # KNNClassifier & kNNAlgorithm from scratch
│       │   ├── voting.py          # Majority and Inverse Distance Weighted voting
│       │   └── retention.py       # CaseBase, NR, AR, DC, and OG retention policies
│       ├── experiments/
│       │   ├── runner.py          # Grid runner with fold isolation
│       │   └── evaluation.py      # Statistical paired t-tests & aggregation
│       └── utils/
│           ├── arff_reader.py     # ARFF table loading and fold alignment
│           ├── distance_metrics.py# Euclidean, Manhattan, Canberra, Clark, HEOM, IVDM
│           ├── folds.py           # Fold discovery and validation
│           └── preprocessing.py   # Data-driven feature inference & normalization
├── tests/
│   ├── test_preprocessing.py      # Legacy preprocessing tests
│   ├── test_data_pipeline.py      # Fold alignment, feature inference, leakage tests
│   ├── test_core_algorithms.py    # Hand-calculated metric tests, voting, retention
│   └── test_experiments.py        # Smoke tests and statistical comparison tests
├── main.py                        # CLI entry point to run experiments
├── requirements.txt               # Dependencies
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

Execute the comprehensive test suite (22 unit tests) with `PYTHONPATH` set to `src`:
```powershell
$env:PYTHONPATH="src"
python -m unittest discover tests
```

---

## Running Experiments

You can run experiments for `heart-statlog` and `grid` separately or together:

### 1. Run Heart-Statlog:
```powershell
$env:PYTHONPATH="src"
python main.py --dataset heart-statlog --k 1 3 5 7 9 --output_dir results
```

### 2. Run Grid:
```powershell
$env:PYTHONPATH="src"
python main.py --dataset grid --k 1 3 5 7 9 --output_dir results
```

### 3. Run Both Datasets:
```powershell
$env:PYTHONPATH="src"
python main.py --dataset all --k 1 3 5 7 9 --output_dir results
```

### Available CLI Options:
- `--dataset`: `heart-statlog`, `grid`, or `all`
- `--k`: List of neighbors, e.g. `--k 1 3 5 7 9`
- `--distances`: List of distance metrics (`euclidean`, `manhattan`, `canberra`, `clark`, `heom`, `ivdm`)
- `--voting`: List of voting strategies (`majority`, `inverse_distance`)
- `--retention`: List of retention policies (`nr`, `ar`, `dc`, `og`)
- `--output_dir`: Output directory for generated CSVs (default: `results`)

---

## Implemented Components & Mathematical Definitions

### 1. Distance Metrics
- **Euclidean** ($r=2$): $d(x, y) = \sqrt{\sum_{f} (x_f - y_f)^2}$
- **Manhattan** ($r=1$): $d(x, y) = \sum_{f} |x_f - y_f|$
- **Canberra**: $d(x, y) = \sum_{f} \frac{|x_f - y_f|}{|x_f| + |y_f|}$ (with $0/0 \to 0.0$)
- **Clark**: $d(x, y) = \sqrt{\sum_{f} \left(\frac{x_f - y_f}{x_f + y_f}\right)^2}$ (with $0/0 \to 0.0$)
- **HEOM** (Wilson & Martinez 1997): Heterogeneous Euclidean-Overlap Metric handling numeric normalization and nominal overlap ($\delta = 0$ if equal, $1$ otherwise).
- **IVDM** (Wilson & Martinez 1997): Discretizes continuous features and estimates conditional class probabilities $P(C=c \mid f=v)$ strictly on the training fold with Laplace smoothing.

### 2. Voting Policies
- **Majority Voting**: Each neighbor casts 1 vote.
- **Inverse Distance Weighted Voting**: Each neighbor casts $\frac{1}{d(q, x_c)^p}$ ($p=1$). Zero-distance neighbors take absolute precedence, preventing division by zero.
- **Tie-Breaking**: Nearest individual neighbor (1-NN), then class prior in training fold, then stable string order.

### 3. Retention Policies
- **Never Retain (NR)**: Training set remains fixed.
- **Always Retain (AR)**: Retains every solved query.
- **Different Class (DC)**: Retains query only when misclassified ($y_{pred} \neq y_{true}$).
- **Oblivion by Goodness (OG)**: Initial goodness $G_0 = 0.5$, $\alpha = 0.2$. Retrieved neighbors update goodness based on accuracy reward $r(c) \in \{0, 1\}$. Cases with $G_t(c) < 0.5$ are pruned immediately.