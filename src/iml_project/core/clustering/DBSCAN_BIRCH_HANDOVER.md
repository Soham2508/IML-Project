# DBSCAN and BIRCH handover (Luis → Soham)

DBSCAN (assignment task 2) and BIRCH (task 3) are Luis's part of the work plan (section 4). This note records what was built, the decisions taken and their implications, and the decisions that are still open for Soham. It covers the six handover items from plan section 10 and the checklist in plan section 4.3.

**Status (27 Nov 2026):** both wrappers and their tests are complete. Preliminary DBSCAN runs on grid and heart-statlog are summarised in section 7. **CMC has not been run yet**, because its data file is missing (O6).

## 1. Scope

- **Task 2 (DBSCAN):** use `sklearn.cluster.DBSCAN`.
  - Test at least two distance metrics.
  - Analyse the `algorithm` parameter (the neighbour search).
  - Vary `eps` and `min_samples`.
  - Report the effect on the number of clusters, the amount of noise and the clustering quality.
- **Task 3 (BIRCH):** use `sklearn.cluster.Birch`, and vary `threshold`, `branching_factor` and `n_clusters`.
- **Plan section 4:** exact public signatures, parameter and input validation, timing, the standard `ClusterResult` output, and unit tests. The algorithms themselves are scikit-learn's; they are **not** reimplemented.
- **Rule for both:** never use the class labels for clustering or for choosing parameters.

## 2. Files

| File | Contents |
|---|---|
| `src/iml_project/core/clustering/dbscan.py` | `DBSCANClusterer`, the helpers `compatible_algorithms(metric)` and `k_distance_curve(X, k, metric, metric_params)`, the shared input validator `_validate_feature_matrix(X)`, and the constants `METRIC_ALGORITHMS`, `EXCLUDED_METRICS`, `REQUIRED_METRIC_PARAMS`, `VALID_ALGORITHMS`, `TREE_ALGORITHMS`, `NOISE_LABEL` |
| `src/iml_project/core/clustering/birch.py` | `BIRCHClusterer`, which imports `_validate_feature_matrix` from `dbscan.py` |
| `src/iml_project/core/clustering/__init__.py` | Package marker; exports nothing |
| `tests/test_dbscan.py` | 27 `unittest` tests |
| `tests/test_birch.py` | 16 `unittest` tests |
| `src/iml_project/core/clustering/DBSCAN_BIRCH_HANDOVER.md` | This note |

The folder `src/iml_project/core/clustering/` replaces the plan's `src/work3/clustering/`, as agreed in the team discussion. Nothing else in the Work 2 package was changed.

## 3. Public interface

The signatures of `__init__`, `fit`, `fit_predict` and `get_result` match plan section 4 **exactly**: parameter names, order, type hints, defaults and return types. In both test files, `test_public_signature_matches_work_plan` checks this.

```python
DBSCANClusterer(
    eps: float = 0.5,
    min_samples: int = 5,
    metric: str = "euclidean",
    algorithm: str = "auto",            # "auto" | "ball_tree" | "kd_tree" | "brute"
    leaf_size: int = 30,
    metric_params: dict[str, Any] | None = None,
    n_jobs: int | None = None,
)

BIRCHClusterer(
    threshold: float = 0.5,
    branching_factor: int = 50,
    n_clusters: int | None = 3,
    compute_labels: bool = True,
    copy: bool = True,                  # kept for the plan's signature; no effect (D7)
)

.fit(X) -> Self
.fit_predict(X) -> NDArray[int64]       # shape (n_samples,); DBSCAN noise = -1
.get_result() -> ClusterResult          # deep copy; RuntimeError before fit
```

**Additions** (outside the plan; nothing in the plan's interface changed):
- `get_params()` and `__repr__` on both classes;
- the fitted attributes `labels_` (both), `core_sample_indices_` (DBSCAN) and `subcluster_centers_` (BIRCH);
- in `dbscan.py`:
  - `compatible_algorithms(metric)`, which lists the search algorithms that accept a metric, for building grids;
  - `k_distance_curve(X, k, metric, metric_params)`, which returns every sample's distance to its k-th nearest neighbour, sorted. It is the label-free way to choose `eps` (Ester et al., 1996);
  - the constants listed in section 2.

### ClusterResult

- **`algorithm`:** `"DBSCAN"` or `"BIRCH"`.
- **`labels`:** `int64`. **DBSCAN noise is always −1.**
- **`memberships`:** `None`.
- **`n_clusters`:**
  - DBSCAN: the number of clusters, not counting noise.
  - BIRCH: the number **actually produced**, which can be lower than requested (D9).
- **`n_noise`:** the number of −1 labels for DBSCAN; always 0 for BIRCH.
- **`execution_time`:** `time.perf_counter()` around the scikit-learn call only (D4). For BIRCH with `compute_labels=False`, it covers fit + predict (D8).
- **`metadata`:**
  - for both: `parameters` (every constructor argument), `backend` (with the scikit-learn version), `n_samples`, `n_features`, `cluster_sizes` and `warnings`;
  - DBSCAN also stores `noise_label` (−1), `n_core_samples` and `noise_ratio`;
  - BIRCH also stores `requested_n_clusters`, `n_subclusters`, `fewer_clusters_than_requested` and `labels_source` (`"fit"` or `"predict"`).

### Usage example

Run from the project root once `work3.contracts` exists:

```python
import sys
sys.path.insert(0, "src")

from sklearn.datasets import make_blobs

from iml_project.core.clustering.birch import BIRCHClusterer
from iml_project.core.clustering.dbscan import DBSCANClusterer, k_distance_curve

X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=0)

# DBSCAN: choose eps without labels, from the knee of the k-distance curve (k = min_samples).
curve = k_distance_curve(X, k=5)
dbscan = DBSCANClusterer(eps=0.5, min_samples=5, metric="manhattan")
labels = dbscan.fit_predict(X)  # only X; true labels are never passed
result = dbscan.get_result()
print("DBSCAN:", result.n_clusters, "clusters,", result.n_noise, "noise points")

# Metrics that need metric_params
DBSCANClusterer(eps=0.5, metric="minkowski", metric_params={"p": 3}).fit(X)

# BIRCH
birch = BIRCHClusterer(threshold=0.5, branching_factor=50, n_clusters=3)
birch.fit(X)
result = birch.get_result()
requested = result.metadata["requested_n_clusters"]
print(f"BIRCH: {result.n_clusters} clusters (requested {requested})")
```

## 4. How the implementation works

**Both wrappers** follow the same steps:

1. **`__init__` validates every parameter.**
   - Wrong types raise `TypeError`, and bad values raise `ValueError`.
   - `bool` is rejected where an int or float is expected, because `isinstance(True, int)` is True.
   - NaN and infinity are rejected, because `nan <= 0` is False and would slip through a plain range check.
   - NumPy scalars are accepted.
   - The values are stored as plain Python types.
2. **`fit(X)` first clears any previous result**, so a failed refit can't leave a stale one behind.
3. **`_validate_feature_matrix(X)` builds a new float64 copy of `X`**, so the caller's array is never modified.
   - Non-numeric input raises `TypeError`.
   - Input that is not 2-D, is empty, or contains non-finite values raises `ValueError`.
4. **The scikit-learn estimator is built and fitted.** Only that call is timed, and any scikit-learn warnings are captured into `metadata["warnings"]` instead of being printed.
5. **The labels are converted to `int64`, checked, and packed into a `ClusterResult`**, which `get_result()` returns as a deep copy.

**DBSCAN specifics.**
- **Supported metrics.** Every string metric that `sklearn.cluster.DBSCAN` accepts and that runs on a feature matrix in scikit-learn 1.9.1 is supported: **23 metrics**. `METRIC_ALGORITHMS` records which search algorithms accept each one:
  - **Any algorithm** (`auto`, `ball_tree`, `kd_tree`, `brute`): euclidean, l2, manhattan, cityblock, l1, chebyshev, minkowski.
  - **`auto`, `ball_tree` or `brute`** (KD-trees only support the Minkowski family): braycurtis, canberra, haversine, mahalanobis, seuclidean, hamming, dice, jaccard, rogerstanimoto, russellrao, sokalsneath.
  - **`auto` or `brute` only:** cosine, correlation, sqeuclidean, nan_euclidean, yule.
- **Incompatible pairs.** A metric/algorithm pair that doesn't work raises `ValueError` in `__init__`. It is never silently switched to `brute`. `"auto"` is always valid.
- **Rejected metrics:**
  - `precomputed`, because `X` must be a feature matrix, not a distance matrix;
  - `matching` and `wminkowski`, which DBSCAN lists but scikit-learn 1.9.1's neighbour search rejects;
  - callables, because the interface takes `metric: str`.
- **Metrics that need `metric_params`** (checked in `__init__`):
  - `minkowski` needs `{"p": p}` with p > 0, and p ≥ 1 for `ball_tree`/`kd_tree`. scikit-learn 1.9.1's DBSCAN fails without `p`.
  - `mahalanobis` needs `{"V": …}` or `{"VI": …}`.
  - `seuclidean` needs `{"V": …}`.
- **Other metric limits.** `haversine` needs exactly two features. The boolean metrics (dice, jaccard, rogerstanimoto, russellrao, sokalsneath, yule) are meant for binary data such as one-hot columns: scikit-learn treats every nonzero value as True, and warns about it only with `brute`.
- **`min_samples` counts the point itself**, as in scikit-learn. `eps` is a neighbourhood radius, not a bound on a cluster's diameter, so clusters can chain.
- **The `algorithm` parameter changes only run time and memory, never the labels.** A test checks that the labels are identical across all compatible algorithms for four metrics.
- **`k_distance_curve` counts the point itself as its first neighbour**, matching scikit-learn's DBSCAN, so call it with `k = min_samples`. Choose `eps` near the knee of the curve.

**BIRCH specifics.**
- **Merge test.** BIRCH builds a CF-tree in which a subcluster absorbs a point only if its radius stays below `threshold`.
- **Integer `n_clusters`.** scikit-learn then groups the leaf subclusters with a final `AgglomerativeClustering` step. If there are fewer leaf subclusters than requested, it produces fewer clusters and emits a `ConvergenceWarning`.
- **`n_clusters=None`** returns the leaf subclusters directly, so `threshold` and `branching_factor` alone decide the number of clusters.
- **`n_clusters` > `n_samples`** raises `ValueError` in `fit`, because the sample count is only known then.

## 5. Decision log

All decisions below were Luis's.

| # | Decision | Implication |
|---|---|---|
| D1 | **DBSCAN supports all 23 string metrics** that scikit-learn's DBSCAN accepts on a feature matrix in 1.9.1. This started as an 11-metric list and was widened | The `METRIC_ALGORITHMS` table must match scikit-learn. `tests/test_dbscan.py` re-checks every metric/algorithm pair against scikit-learn and fails if a different version changes them |
| D2 | **Return annotations without quotes** (`-> DBSCANClusterer`) | Equivalent to the plan's quoted form, because of `from __future__ import annotations` |
| D3 | **The helpers `compatible_algorithms` and `k_distance_curve` were added** | Not required by the plan, and they don't change the contract. `k_distance_curve` is the label-free way to choose `eps` |
| D4 | **`execution_time` covers only the scikit-learn call** | Timings compare the algorithms themselves, without the wrapper's own overhead |
| D5 | **Degenerate outcomes are recorded in `metadata["warnings"]`, not raised.** Examples: everything is noise, or a single cluster with no noise | Parameter grids keep those runs for the sensitivity tables. scikit-learn's own warnings are captured instead of printed |
| D6 | **`k` in `k_distance_curve` counts the point itself** | Call it with `k = min_samples`, matching scikit-learn's counting |
| D7 | **BIRCH keeps `copy` in the signature as a documented no-op**, recorded in metadata | Matches the plan's signature, although scikit-learn 1.9.1's `Birch` has no `copy` argument (it was deprecated in 1.6). The input is never modified whatever its value (see O4) |
| D8 | **With BIRCH `compute_labels=False`, the labels come from `predict(X)`** | The timing then covers fit + predict, so it isn't directly comparable with a `compute_labels=True` run. `metadata["labels_source"]` says which applies |
| D9 | **BIRCH records scikit-learn's warning plus the flag `fewer_clusters_than_requested`** | Filter degenerate runs on the flag. `n_clusters` is the number actually produced; `requested_n_clusters` and `n_subclusters` are in metadata |
| D10 | **`birch.py` imports the private `_validate_feature_matrix` from `dbscan.py`** | Both wrappers always validate `X` identically, but BIRCH depends on `dbscan.py` and on a private name (see O7) |
| D11 | **Extra fitted attributes** (`labels_`, `subcluster_centers_`, `core_sample_indices_`) | They are additions outside the contract, not requirements |
| D12 | **Defensive checks on the shape and sign of BIRCH's labels** | They can't trigger with current scikit-learn, so no test covers them |
| D13 | **`get_result()` returns a deep copy, and `fit()` clears the previous result first** | Callers can't corrupt the stored result, and a failed refit can't leave a stale one. This was a real bug, caught by a test |
| D14 | **Location `iml_project/core/clustering/`** instead of `work3/clustering/`, agreed by the team | `ClusterResult` is still imported from `work3.contracts`. If the contract ends up elsewhere, only that import line changes (O2) |

## 6. Tests and verification

**`tests/test_dbscan.py`: 27 tests.** They cover:
- synthetic blobs with outliers, and noise labelled −1;
- 5 `eps` × 4 `min_samples` values, and 4 metrics;
- identical labels across every compatible search algorithm, for euclidean, manhattan, canberra and cosine;
- every metric/algorithm pair checked against scikit-learn, and the required `metric_params`;
- invalid parameters and input, `get_result()` before `fit`, the input never being modified, the metadata, and the signatures.

**`tests/test_birch.py`: 16 tests.** They cover:
- synthetic blobs;
- a grid of 4 `threshold` × 3 `branching_factor` × 4 `n_clusters` values;
- fewer clusters than requested, `compute_labels=False`, and `n_clusters` > `n_samples`;
- invalid parameters and input, `get_result()` before `fit`, the input never being modified, the metadata, and the signatures.

**Verification:**
- All 43 tests pass. They were run against a temporary local copy of plan section 3's contract, which is not part of these files.
- Tested on Python 3.11.18 (macOS) with the versions pinned in `requirements.txt` (`numpy==2.4.6`, `scikit-learn==1.9.1`). There are no new dependencies.
- `ruff` is clean.
- The usage example above runs as written.
- Run the tests from the project root: `python -m unittest tests.test_dbscan tests.test_birch`.

**Plan 4.3 checklist:**

- [x] Exact public signatures and return types implemented.
- [x] Parameter and input validation included.
- [x] DBSCAN noise is always `-1`.
- [x] Runtime measured in seconds.
- [x] Tests cover valid input, invalid input, unfitted `get_result()`, and a small synthetic dataset.
- [x] At least two metrics and multiple parameter values tested for DBSCAN.
- [x] BIRCH threshold, branching factor, and cluster count varied.
- [x] Short usage example and note on metric/search restrictions supplied.
- [x] No edits to contracts, preprocessing, evaluation, or experiment runner.

## 7. Known limitations and preliminary findings (for the report)

**Limitations:**
- **DBSCAN uses a single density level.** One `eps` can't separate clusters of different density, and because `eps` is a radius, neighbouring clusters can chain together.
- **Metric choice.** L1 is the same as Manhattan and L2 the same as Euclidean, so choose the reported metrics from Euclidean, Manhattan and cosine. Cosine ignores magnitude, which makes it meaningless on grid's 2-D coordinates. Boolean metrics treat every nonzero value as True.
- **BIRCH results depend on the order of the input**, because it builds the CF-tree incrementally. It also favours round clusters, and `threshold` interacts with feature scaling. A requested `n_clusters` may not be reached.
- **One-hot data.** One-hot columns dominate Euclidean distances: two rows that differ in one category are already √2 apart. This matters for heart-statlog and CMC (O5).

**Preliminary findings (26 Nov).** These used grid and heart-statlog, with fold 0 train + test (numerically identical to the single files) and the Work 2 preprocessing:
- **DBSCAN on grid.** The k-distance curve is flat: the nearest neighbour is at 0.021 everywhere, because grid is a regular lattice. This gives two regimes: everything is noise below `eps` ≈ 0.021, and there is one cluster at or above it. grid's classes are black/white regions, not differences in density.
- **DBSCAN on heart-statlog (28 one-hot columns, `min_samples` = 5).** The knee of the curve is at `eps` ≈ 1.58 for Euclidean, 3.1 for Manhattan and 0.14 for cosine.
  - Below the knee there are 6 tiny clusters and 87% noise; at or above it, 1 cluster with about 10% noise.
  - **The 87%-noise runs have the *best* silhouette (up to 0.94) and Davies–Bouldin index (0.34).** This is a metric trap: always read the internal metrics together with the share of noise.
- **Purity's floor** is the share of the largest class, for example 42.7% on CMC.

## 8. Open decisions for Soham

| # | Open decision | Options | Luis's recommendation |
|---|---|---|---|
| O1 | **Accept the additions** (`get_params`, `__repr__`, fitted attributes, the two helpers and the constants) | Keep them, or remove them | Keep them. They don't change the plan's interface, and `k_distance_curve` is needed to choose `eps` without labels |
| O2 | **Where `ClusterResult` lives.** Both modules and their tests import `from work3.contracts import ClusterResult`, which doesn't exist yet, so they can't be imported until it does | Create `src/work3/contracts.py` as in the plan, or choose another location (then one import line per file changes) | Create `work3/contracts.py` as planned |
| O3 | **Who supplies the DBSCAN/BIRCH parameter grids.** Plan section 2 gives Luis the "parameter configurations", and section 7.3 puts the grids in Soham's runner | Luis writes the grid configuration and the runner executes it, or Soham writes both | Luis supplies it, as below |
| O4 | **BIRCH `copy`**, which has no effect in scikit-learn 1.9.1 | Keep it as a documented no-op, or remove it from the shared signature (a contract change) | Keep it. Removing it changes the plan's interface for no gain |
| O5 | **CMC encoding** (the preprocessing owner's call) | By default CMC becomes 24 columns, all one-hot. The alternative is 12 columns: the ordinal attributes kept as scaled numbers and the binary ones as single 0/1 columns | 12 columns, the same for every algorithm. Distance-based DBSCAN and BIRCH depend heavily on it |
| O6 | **Data files.** Only the fold folders of grid and heart-statlog are in `data/`, and `data/cmc.arff` is missing | Restore the single ARFF files | Restore `data/cmc.arff` (and ideally the single grid and heart-statlog files) before the final runs |
| O7 | **Shared input validation.** `birch.py` imports it from `dbscan.py`. `kmeans.py` has its own copy so that it never imports scikit-learn, and `fuzzy_cmeans.py` imports that copy | Keep it as it is, or move the checks into one scikit-learn-free `clustering/_validation.py` | Keep it until the deadline. Moving it later changes one import line per module |
| O8 | **Noise in the evaluation** (Bhavan's module): plan 5.2 counts noise (−1) as its own group for the external metrics and excludes it for DBI | Confirm this convention | Confirm it, and always report the noise share next to the internal metrics (section 7) |

**Suggested grid (O3):**
- **DBSCAN:**
  - metrics: Euclidean, Manhattan and cosine (but not cosine on grid);
  - `eps`: per metric and dataset, roughly 0.5× to 1.5× the knee of `k_distance_curve`;
  - `min_samples`: 3, 5, 10 and 20;
  - `algorithm`: every value from `compatible_algorithms(metric)`, for the runtime comparison only.
- **BIRCH:**
  - `threshold`: a few values around the scale of the data;
  - `branching_factor`: 20, 50 and 100;
  - `n_clusters`: None plus 2…10.
- **Report per run:**
  - `n_clusters`, the noise %, and `execution_time`;
  - for BIRCH, also `n_subclusters` and `fewer_clusters_than_requested`;
  - silhouette (↑) and DBI (↓) for tuning;
  - ARI, purity, F-measure and the confusion matrix only afterwards, for evaluation.

## 9. Next steps

1. Soham decides O1–O8.
2. Run DBSCAN and BIRCH on CMC once `data/cmc.arff` is restored.
3. The runner executes the grid (O3) on all three datasets with the same preprocessing as the other algorithms.
4. The report states:
   - the metrics and parameters used, and how `eps` was chosen (the k-distance knee);
   - the effect of `eps`, `min_samples` and `algorithm` (runtime only), and of `threshold`, `branching_factor` and `n_clusters`;
   - the limitations and the metric trap in section 7.

## 10. References

- Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. *Proceedings of KDD-96*, 226–231. AAAI Press.
- Zhang, T., Ramakrishnan, R., & Livny, M. (1996). BIRCH: An efficient data clustering method for very large databases. *Proceedings of the 1996 ACM SIGMOD International Conference on Management of Data*, 103–114. https://doi.org/10.1145/233269.233324
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
- Course material: Work 3 assignment (2026–2027, v2), tasks 2–3; work plan sections 4 and 10.
