# IML Practice: Classification Framework (Lazy, Eager Learning & Instance Reduction)

## Team

- Soham Chakraborty
- Luis David Landicho Antonio
- Bhavan Mullur Gopal
- Vinay Narayanappa

## Environment

- **Python Version:** `3.11.9` (compatible with Python 3.11+)
- **Tested Operating Systems:**
  - Windows 11 (tested in PowerShell and Command Prompt)
  - macOS (tested in Zsh)
- **Hardware Requirements:** Standard multi-core CPU (e.g., Intel/AMD x86_64 or Apple Silicon ARM64). No GPU or specialized accelerator hardware is required.
- **Software Requirements:** Python 3.11+ and standard `pip` package manager. All external libraries are documented in `requirements.txt`.

---

## Project structure

The submission package follows the required organization:

```
submission.zip
├── report.pdf                      # Practice scientific and technical report (outside code/)
└── code/                           # Mandatory root directory containing all executable code
    ├── README.md                   # Operational guide and reproduction manual
    ├── requirements.txt            # Pinned package dependencies
    ├── main.py                     # Unified executable command-line entry point
    ├── data/                       # Predefined 10-fold cross-validation datasets
    │   ├── grid/                   # TAO-Grid 2D synthetic dataset (10 train/test ARFF pairs)
    │   └── heart-statlog/          # Heart-Statlog clinical dataset (10 train/test ARFF pairs)
    ├── results/                    # Auto-created output folder for experiment CSV artifacts
    │   └── experiment_results_1/   # Sample baseline KNN execution directory
    ├── src/
    │   └── iml_project/            # Main project package
    │       ├── core/               # Learning algorithms, reduction, voting, and retention
    │       │   ├── instance_reduction.py # InstanceReducer base class, FCNN3, RNG2, DROP3
    │       │   ├── knn.py          # KNNClassifier and kNNAlgorithm (from scratch)
    │       │   ├── svm.py          # svmAlgorithm (eager wrapper for scikit-learn SVC)
    │       │   ├── random_forest.py# rfAlgorithm (eager wrapper for RandomForestClassifier)
    │       │   ├── voting.py       # MajorityVoting and InverseDistanceWeightedVoting
    │       │   └── retention.py    # CaseBase container and retention policies (NR, AR, DC, OG)
    │       ├── experiments/        # Experiment execution, metrics, and statistical analysis
    │       │   ├── runner.py       # Cross-validation experiment matrices for KNN, SVM, and RF
    │       │   └── evaluation.py   # Aggregation, metric calculation, and statistical tests
    │       └── utils/              # Data parsing, feature preprocessing, and metrics
    │           ├── arff_reader.py  # ARFF file loading and fold pairing
    │           ├── distance_metrics.py # Vector and nominal distance functions (Euclidean, IVDM, etc.)
    │           ├── folds.py        # 10-fold CV discovery and structure validation
    │           └── preprocessing.py# Preprocessor (scaling, imputation, encoding, leakage-free reduction)
    └── tests/                      # Automated unit test suite (52 tests across 6 modules)
        ├── test_core_algorithms.py # Hand-calculated metric, voting, and retention tests
        ├── test_data_pipeline.py   # Fold discovery, validation, and ARFF parsing tests
        ├── test_eager_models.py    # SVM/RF wrappers and parameter validation tests
        ├── test_experiments.py     # Experiment grid execution and directory versioning tests
        ├── test_instance_reduction.py # FCNN3, RNG2/RNGE2, and DROP3 algorithm tests
        └── test_preprocessing.py   # Feature inference, scaling, and zero-leakage tests
```

### Class Design and Responsibilities

The codebase adheres to modular, object-oriented design principles with clear single-responsibility separation and explicit public interfaces:

1. **`InstanceReducer` (`src/iml_project/core/instance_reduction.py`)**:

   - *Abstract Base Class* for instance selection and editing algorithms.
   - *Public Interface:* `fit(X, y)` fits the reduction logic strictly on training fold data; `transform(X, y)` returns the filtered instances `(X_reduced, y_reduced)`; `fit_transform(X, y)` combines both.
   - *Subclasses:*
     - `FCNN3`: 1-NN consistent condensation initializing with the centroid representative of the majority class and iteratively incorporating the closest Voronoi enemy of the cell with the highest enemy count.
     - `RNG2` (alias `RNGE2`): Graph-based editing that constructs a Relative Neighbourhood Graph and prunes instances whose graph neighborhood is dominated by opposing classes.
     - `DROP3`: Decremental reduction algorithm performing an initial ENN noise filtering pass followed by an ordered associate-based pruning loop sorted by distance to nearest enemy.
   - *Factory:* `get_instance_reducer(method, distance_metric, **kwargs)`.
2. **`KNNClassifier` & `kNNAlgorithm` (`src/iml_project/core/knn.py`)**:

   - Non-parametric lazy learning classifier implemented from scratch without black-box nearest-neighbor dependencies.
   - *State Management:* Instantiated with `KNNClassifier(distance_metric, k, voting_strategy, retention_policy)`.
   - *Execution Lifecycle:* `fit(X, y)` populates an internal `CaseBase`; `predict(X)` evaluates test instances by querying pairwise distances, locating the $k$ nearest stored cases, delegating vote aggregation to the `VotingStrategy`, and conditionally updating the `CaseBase` according to the active `RetentionPolicy`.
3. **`CaseBase` & `RetentionPolicy` (`src/iml_project/core/retention.py`)**:

   - `CaseBase`: Stateful container storing active prototypes `(X, y)`. Supports dynamic additions (`add`), removals (`remove`), and vectorized distance queries.
   - `RetentionPolicy` (ABC): Implements incremental learning policies via `update(case_base, query_x, predicted_y, true_y, neighbors)`.
   - Policies include: `NeverRetain` (`nr`), `AlwaysRetain` (`ar`), `DifferentClassRetain` (`dc`, retains misclassified instances), and `OnlyGreaterRetain` (`og`, retains instances misclassified with high neighbor confidence).
4. **`VotingStrategy` (`src/iml_project/core/voting.py`)**:

   - *Abstract Base Class* defining `decide(neighbor_classes, neighbor_distances, class_priors)`.
   - `MajorityVoting`: Unweighted plurality vote with deterministic tie-breaking by class prior probabilities.
   - `InverseDistanceWeightedVoting`: Distance-weighted vote proportional to $1 / (d + \epsilon)$ to prioritize closer neighbors.
5. **`svmAlgorithm` & `rfAlgorithm` (`src/iml_project/core/svm.py`, `random_forest.py`)**:

   - Eager learning wrappers integrating `sklearn.svm.SVC` and `sklearn.ensemble.RandomForestClassifier` into the standardized practice pipeline.
   - Follow standard `fit(X, y)` and `predict(X)` conventions while recording training fit latency, prediction efficiency, and retention ratios.
6. **`Preprocessor` (`src/iml_project/utils/preprocessing.py`)**:

   - Enforces **zero data leakage**: feature-type inference, numeric min-max bounds $[0, 1]$, median imputation, categorical mode imputation/encoding, and IVDM conditional probability tables are computed strictly on `X_train`. Test folds (`X_test`) are transformed using training statistics only.
   - Coordinates instance reduction hooks: when enabled, instance reduction algorithms filter training instances *after* scaling and imputation but *before* model training.

---

## Installation

Follow these steps to set up the working environment and install dependencies on a clean machine:

### 1. Open a terminal in the code folder

Navigate to the root `code/` directory (where `main.py` and `requirements.txt` reside):

```bash
cd <path-to-submission>/code
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

*(On Windows PowerShell or Command Prompt, use `python -m venv .venv`)*

### 3. Activate the virtual environment

- **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```
- **Windows PowerShell:**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
- **Windows Command Prompt (CMD):**
  ```cmd
  .venv\Scripts\activate.bat
  ```

### 4. Install the dependencies listed in requirements.txt

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 5. Run the experiments described in the README while the environment is active.

### 6. Deactivate the environment after the experiments have finished

```bash
deactivate
```

---

## Execution order

- **Independent Experiments:** Every standard experiment (e.g., KNN without reduction, KNN with FCNN3/RNG2/DROP3, SVM, or Random Forest) can be executed in **any order**. Output directories and result files are automatically created by the runner if they do not exist.
- **Statistical Comparison Dependency (`--knn-baseline-dir`):**
  - If you run an eager learning experiment (SVM or RF) with the `--knn-baseline-dir` flag to calculate paired Student's $t$-tests against the KNN baseline on identical folds, the **KNN baseline experiment must be run first** so that its output directory exists.
  - *Example order:*
    1. Run KNN baseline:
       ```bash
       python main.py --model knn --dataset all --output-dir results/knn_baseline
       ```
    2. Run SVM or Random Forest comparing against the generated baseline:
       ```bash
       python main.py --model svm --dataset all --knn-baseline-dir results/knn_baseline
       ```

---

## Entry points

1. **`main.py` (Primary CLI Entry Point):**
   - **Path:** `main.py`
   - **Purpose:** Central executable script that parses command-line arguments, loads predefined cross-validation folds, executes training and inference grids, performs instance reduction, aggregates evaluation metrics, computes statistical tests, and exports structured CSV reports.
2. **Automated Test Suite Entry Point:**
   - **Command:** `python -m unittest discover -s tests -t .`
   - **Purpose:** Executes 52 automated unit tests verifying correctness across instance reduction algorithms, distance metrics, voting strategies, retention policies, eager models, and preprocessing pipelines.

---

## Parameters

All command-line arguments are introduced with explicit flags (no positional parameters are required).

### General and Execution Parameters

| Parameter                    | Type                 | Valid Values                                       | Default     | Purpose / Scope                                                                                                                                              |
| :--------------------------- | :------------------- | :------------------------------------------------- | :---------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--model`                  | Optional string      | `knn`, `svm`, `rf`, `all`                  | `knn`     | Model family to execute. `all` runs KNN, SVM, and RF sequentially.                                                                                         |
| `--dataset`                | Optional string      | `heart-statlog`, `grid`, `tao-grid`, `all` | `all`     | Target dataset(s). `all` executes both `heart-statlog` and `grid`.                                                                                     |
| `--output_dir--output-dir` | Optional string/path | Any valid directory path                           | `results` | Destination folder for CSV reports. Defaults to auto-incrementing directories: `results/experiment_results_{N}`. Output folders are created automatically. |
| `--knn-baseline-dir`       | Optional string/path | Existing directory path                            | `None`    | Path to prior KNN baseline results. Used by SVM and RF to produce `model_comparison.csv` with paired $t$-tests.                                          |
| `--random-state`           | Optional integer     | Integer $\ge 0$                                  | `42`      | Global pseudo-random seed ensuring deterministic results across SVM, RF, and tie-breaking.                                                                   |

### Instance Reduction Parameters (Preprocessing Stage)

| Parameter                                  | Type            | Valid Values                                                              | Default       | Purpose / Scope                                                                                                        |
| :----------------------------------------- | :-------------- | :------------------------------------------------------------------------ | :------------ | :--------------------------------------------------------------------------------------------------------------------- |
| `--reduction-method--instance-reduction` | Optional string | `fcnn3`, `rng2`, `drop3`                                            | `None`      | Instance reduction algorithm applied to training folds before model fitting. If omitted, full training folds are used. |
| `--reduction-distance--reduction-metric` | Optional string | `euclidean`, `manhattan`, `canberra`, `clark`, `heom`, `ivdm` | `euclidean` | Distance metric used strictly during instance reduction. Decoupled from classifier distance metrics.                   |

### KNN Specific Parameters (valid only when `--model knn`)

| Parameter       | Type                     | Valid Values                                                              | Default                       | Purpose / Scope                                            |
| :-------------- | :----------------------- | :------------------------------------------------------------------------ | :---------------------------- | :--------------------------------------------------------- |
| `--k`         | Optional list of ints    | Positive integers ($k \ge 1$)                                           | `3 5 7 9 11 13 15`          | Neighborhood sizes to evaluate.                            |
| `--distances` | Optional list of strings | `euclidean`, `manhattan`, `canberra`, `clark`, `heom`, `ivdm` | All 6 metrics                 | Distance functions evaluated for KNN query classification. |
| `--voting`    | Optional list of strings | `majority`, `inverse_distance`                                        | `majority inverse_distance` | Voting rules for neighbor class aggregation.               |
| `--retention` | Optional list of strings | `nr`, `ar`, `dc`, `og`                                            | `nr ar dc og`               | Dynamic case-base retention policies.                      |

### SVM Specific Parameters (valid only when `--model svm`)

| Parameter         | Type                     | Valid Values                               | Default        | Purpose / Scope                                               |
| :---------------- | :----------------------- | :----------------------------------------- | :------------- | :------------------------------------------------------------ |
| `--svm-kernels` | Optional list of strings | `linear`, `rbf`, `poly`, `sigmoid` | `linear rbf` | Kernel functions evaluated for Support Vector Classification. |
| `--svm-c`       | Optional float           | Float $C > 0$                            | `1.0`        | Regularization penalty parameter.                             |
| `--svm-gamma`   | Optional str or float    | `'scale'`, `'auto'`, or float $> 0$  | `'scale'`    | Kernel coefficient for RBF, poly, and sigmoid kernels.        |

### Random Forest Specific Parameters (valid only when `--model rf`)

| Parameter             | Type                     | Valid Values                        | Default          | Purpose / Scope                                                |
| :-------------------- | :----------------------- | :---------------------------------- | :--------------- | :------------------------------------------------------------- |
| `--rf-n-estimators` | Optional list of ints    | Integers $> 0$                    | `100 200`      | Number of decision trees in the ensemble.                      |
| `--rf-criteria`     | Optional list of strings | `gini`, `entropy`, `log_loss` | `gini entropy` | Impurity split criteria.                                       |
| `--rf-max-depth`    | Optional integer         | Integer $> 0$, or omitted         | `None`         | Maximum tree depth (defaults to full depth).                   |
| `--n-jobs`          | Optional integer         | Integer (e.g. `-1`)               | `-1`           | Number of parallel worker threads (`-1` uses all CPU cores). |

> [!NOTE]
> **Parameter Safety & Mutual Exclusivity:** Specifying arguments from an inactive model family (e.g., passing `--k` with `--model svm` or `--svm-kernels` with `--model knn`) triggers an immediate CLI error informing the user of the invalid option.

---

## Running Experiments

Every command below can be copied and executed directly from the root `code/` folder.

### Experiment 1: Baseline Lazy Learning (KNN on Full Training Sets)

- **Entry point:** `main.py`
- **Command:**
  ```bash
  python main.py --model knn --dataset all --k 1 3 5 7 9 --distances euclidean manhattan canberra clark heom ivdm
  ```
- **Output:** Executes 10-fold CV across 6 distance metrics, 5 values of $k$, 2 voting schemes, and 4 retention policies on both `heart-statlog` and `grid`. Automatically creates `results/experiment_results_{N}/` containing `all_experiments_results.csv`, `aggregated_results.csv`, `best_configurations.csv`, and `statistical_comparisons.csv`. Prints top-5 leaderboards and Friedman statistical test tables to the console.

### Experiment 2: Lazy Learning with Fast Condensation (KNN + FCNN3)

- **Entry point:** `main.py`
- **Command:**
  ```bash
  python main.py --model knn --dataset heart-statlog --k 1 3 5 --distances euclidean --reduction-method fcnn3 --reduction-distance manhattan
  ```
- **Output:** Evaluates FCNN3 condensation on `heart-statlog`. Employs Manhattan distance to build Voronoi enemy graphs during training fold condensation, and evaluates Euclidean 1/3/5-NN on the condensed case base. Saves CSV reports with storage retention ratios and inference latencies.

### Experiment 3: Support Vector Machines (SVM) with Baseline Comparison

- **Entry point:** `main.py`
- **Command:**
  ```bash
  python main.py --model svm --dataset all --svm-kernels linear rbf poly --knn-baseline-dir results/experiment_results_1
  ```
- **Output:** Executes 10-fold CV for Linear, RBF, and Polynomial SVMs on both datasets. Computes classification accuracy, macro F1, and training/inference timing. Compares fold-by-fold results against the prior KNN baseline specified in `--knn-baseline-dir` and generates `model_comparison.csv` with paired two-tailed Student's $t$-tests ($df=9$).

### Experiment 4: SVM with Graph-Based Neighborhood Editing (SVM + RNG2)

- **Entry point:** `main.py`
- **Command:**
  ```bash
  python main.py --model svm --dataset heart-statlog --svm-kernels linear rbf --reduction-method rng2 --reduction-distance euclidean
  ```
- **Output:** Prunes borderline and noisy training instances using 2nd-order Relative Neighbourhood Graph editing before training SVM classifiers. Records retention ratios and evaluates boundary classification performance.

### Experiment 5: Random Forest with Baseline Comparison

- **Entry point:** `main.py`
- **Command:**
  ```bash
  python main.py --model rf --dataset all --rf-n-estimators 100 200 --rf-criteria gini entropy --knn-baseline-dir results/experiment_results_1
  ```
- **Output:** Evaluates Random Forest ensembles across tree counts and impurity criteria. Produces fold-level and aggregated metrics, ranking reports, and statistical comparison against the KNN baseline.

### Experiment 6: Random Forest with Decremental Noise Filtering & Reduction (RF + DROP3)

- **Entry point:** `main.py`
- **Command:**
  ```bash
  python main.py --model rf --dataset heart-statlog --rf-n-estimators 100 --reduction-method drop3 --reduction-distance euclidean
  ```
- **Output:** Runs DROP3 two-stage reduction (ENN noise filtering + associate-based decremental pruning) on each training fold prior to fitting Random Forests. Demonstrates the impact of instance reduction on tree depth and training speed.

### Experiment 7: Comprehensive Multi-Model Pipeline (`--model all`)

- **Entry point:** `main.py`
- **Command:**
  ```bash
  python main.py --model all --dataset all
  ```
- **Output:** Runs KNN, SVM, and Random Forest across both datasets in a single execution. Exports combined multi-model leaderboards, efficiency comparisons, and omnibus Friedman ranking tests.

---

## Outputs

Each execution automatically writes timestamped and versioned CSV artifacts into `results/experiment_results_{N}/` (or the folder passed via `--output-dir`). Pre-creating the output directory is not required.

### Generated Files and Structure

| Output Artifact                 | Expected Location                   | Description and Contents                                                                                                                                                                                                     |
| :------------------------------ | :---------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `all_experiments_results.csv` | `results/experiment_results_{N}/` | Raw fold-by-fold results across all 10 folds containing individual accuracy, macro precision, macro recall, macro F1, balanced accuracy, fit time, inference latency, and retention ratio.                                   |
| `aggregated_results.csv`      | `results/experiment_results_{N}/` | Mean and standard deviation computed across the 10 folds for each metric, grouped by hyperparameter configuration and instance reduction method.                                                                             |
| `best_configurations.csv`     | `results/experiment_results_{N}/` | Top-performing configurations ranked by accuracy, efficiency (latency in milliseconds), and case-base retention ratio.                                                                                                       |
| `statistical_comparisons.csv` | `results/experiment_results_{N}/` | Within-family statistical hypothesis tests: paired two-tailed Student's $t$-tests across kernels, metrics, voting, retention, or reduction methods.                                                                        |
| `model_comparison.csv`        | `results/experiment_results_{N}/` | Paired Student's $t$-test comparing the best eager configuration (SVM or RF) directly against the best KNN baseline configuration on identical cross-validation folds (generated when `--knn-baseline-dir` is provided). |

### Console Output

In addition to exported CSV files, each run prints formatted tables directly to standard output:

- **Configuration Breakdown:** Number of configurations, total evaluations across 10 folds, and active reduction method.
- **Progress & Timing:** Execution duration per fold and cumulative pipeline runtime in seconds.
- **Descriptive Leaderboards:** Top 5 configurations ranked by pooled accuracy, inference efficiency (ms/query), and storage retention ratio (%).
- **Multi-Dataset Hypothesis Testing:** When multiple datasets are evaluated, prints Friedman omnibus test statistics ($\chi_F^2$), Iman-Davenport $F_F$ statistics with $p$-values, average model ranks, and Nemenyi Critical Differences ($\alpha = 0.05$).

---

## Running Unit Tests

The project includes an automated test suite with 52 unit tests verifying data pipeline isolation, distance calculations, eager models, lazy learners, and instance reduction algorithms.

To run all unit tests from the root `code/` folder:

```bash
python -m unittest discover -s tests -t .
```

To run a specific test suite (e.g., instance reduction tests):

```bash
python -m unittest tests/test_instance_reduction.py
```

To verify code quality and style using Ruff:

```bash
ruff check src tests main.py
```

To run a specific test suite (e.g., instance reduction tests):

```bash
python -m unittest tests/test_instance_reduction.py
```

To verify code quality and style using Ruff:

```bash
ruff check src tests main.py
```
