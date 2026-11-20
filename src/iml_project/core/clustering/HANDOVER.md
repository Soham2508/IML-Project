# Handover: DBSCAN and BIRCH wrappers (Luis)

Work plan section 4. Branch: `user/luisdavidantonio/clustering-dbscan-birch`.

## 1. Source files

| File | Contents |
| --- | --- |
| `src/iml_project/core/clustering/dbscan.py` | `DBSCANClusterer`, plus helpers `compatible_algorithms(metric)` and `k_distance_curve(X, k, metric, metric_params)` |
| `src/iml_project/core/clustering/birch.py` | `BIRCHClusterer` |
| `src/iml_project/core/clustering/__init__.py` | Package marker (exports nothing) |

The location `src/iml_project/core/clustering/` replaces the plan's
`src/work3/clustering/`, as agreed in the team discussion. Nothing else in the
Work 2 package was changed.

## 2. Unit tests

| File | Tests | Covers |
| --- | --- | --- |
| `tests/test_dbscan.py` | 27 | Synthetic blobs with outliers, noise = `-1`, 5 `eps` × 4 `min_samples` values, 4 metrics, identical labels across all compatible search algorithms (euclidean, manhattan, canberra, cosine), every metric/algorithm pair checked against scikit-learn, required `metric_params`, invalid parameters and input, unfitted `get_result()`, no input mutation, metadata, signatures |
| `tests/test_birch.py` | 16 | Synthetic blobs, grid of 4 `threshold` × 3 `branching_factor` × 4 `n_clusters` values, fewer clusters than requested, `compute_labels=False`, `n_clusters` > `n_samples`, invalid parameters and input, unfitted `get_result()`, no input mutation, metadata, signatures |

Run from the project root (unittest, like the rest of the project):

```bash
python -m unittest tests.test_dbscan tests.test_birch
```

## 3. Runnable usage example

Run from the project root. It needs `work3.contracts` (see section 5).

```python
import sys

sys.path.insert(0, "src")

from sklearn.datasets import make_blobs

from iml_project.core.clustering.birch import BIRCHClusterer
from iml_project.core.clustering.dbscan import DBSCANClusterer, k_distance_curve

X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=0)

# DBSCAN: choose eps without labels from the knee of the k-distance curve.
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

## 4. Dependencies

No new dependencies. The modules use `numpy` and `scikit-learn`, both already
pinned in `requirements.txt` (`numpy==2.4.6`, `scikit-learn==1.9.1`). Tested
with Python 3.11.18 on macOS. The
metric table and the BIRCH notes below were verified against scikit-learn
1.9.1; `tests/test_dbscan.py` fails if a different version changes which
metric/algorithm pairs work.

## 5. Implementation decisions and known limitations

**Both wrappers**

- **Depends on `work3.contracts.ClusterResult` (Soham).** Both modules import
  `from work3.contracts import ClusterResult`, as in the plan. Until
  `src/work3/contracts.py` exists, the modules and their tests cannot be
  imported. If the contracts end up elsewhere, only that import line in each
  module and test file needs to change.
- **`execution_time`** is measured with `time.perf_counter()` around the
  scikit-learn call only; input validation is excluded.
- **Input checks:** `X` is converted to a new float64 array, so the caller's
  array is never modified. Non-numeric input raises `TypeError`; non-2-D,
  empty or non-finite input raises `ValueError`.
- **Parameter checks** run in `__init__`: wrong types raise `TypeError`, bad
  values raise `ValueError`. `bool` is rejected where an int or float is
  expected.
- **`get_result()`** raises `RuntimeError` before fitting and returns a deep
  copy, so callers cannot change the stored result. A failed re-fit clears
  the previous result.
- **Warnings** raised by scikit-learn during fit are stored in
  `metadata["warnings"]` instead of being printed.
- **`metadata`** contains `parameters` (every constructor argument),
  `backend` (with the scikit-learn version), `n_samples`, `n_features`,
  `cluster_sizes` and `warnings`, plus the algorithm-specific keys below.
- **Citations** (Ester et al., 1996; Zhang et al., 1996; Pedregosa et al.,
  2011) are in each module's docstring.

**DBSCAN**

- **Metrics:** every string metric that `sklearn.cluster.DBSCAN` accepts and
  that runs on a feature matrix in scikit-learn 1.9.1 is supported (23
  metrics). `METRIC_ALGORITHMS` lists which search algorithms accept each one,
  and an incompatible pair is rejected in `__init__`. Use
  `compatible_algorithms(metric)` to build grids; `"auto"` is always valid.
- **Rejected metrics:** `precomputed` (`X` must be a feature matrix, not a
  distance matrix), `matching` and `wminkowski` (DBSCAN lists them but
  scikit-learn 1.9.1 rejects them), and callables (the interface takes
  `metric: str`).
- **Metrics that need `metric_params`:** `minkowski` needs `{"p": p}` with
  `p > 0`, and `p >= 1` for `ball_tree`/`kd_tree` (scikit-learn 1.9.1's DBSCAN
  fails without `p`). `mahalanobis` needs `{"V": ...}` or `{"VI": ...}`.
  `seuclidean` needs `{"V": ...}`.
- **Other metric limits:** `haversine` needs exactly two features. dice,
  jaccard, rogerstanimoto, russellrao, sokalsneath and yule are boolean
  metrics, intended for binary data such as one-hot columns. scikit-learn
  treats every nonzero value as True and warns about it only with `brute`.
- **Diagnostics in `metadata`:** `noise_label` (-1), `n_core_samples`,
  `noise_ratio`.
- **`k_distance_curve`** is a label-free helper for choosing `eps` (sorted
  k-dist graph, Ester et al., 1996). It is not part of the plan's interface.

**BIRCH**

- **`copy` has no effect.** scikit-learn 1.9.1's `Birch` has no `copy`
  parameter. The argument is kept because it is in the agreed signature and
  is recorded in `metadata["parameters"]`; the input is never modified
  whatever its value.
- **`execution_time` with `compute_labels=False` covers fit + predict.** In
  that case scikit-learn does not label the training data, but
  `ClusterResult` needs labels, so they come from `predict(X)` and that call
  is included in the timing. `metadata["labels_source"]` is `"predict"`. With
  the default `compute_labels=True` the timing covers the fit only, as the
  plan specifies (`labels_source` is `"fit"`).
- **Actual vs requested clusters:** `n_clusters` in `ClusterResult` is the
  number actually produced, which can be lower than requested.
  `metadata["requested_n_clusters"]` holds the requested value,
  `metadata["fewer_clusters_than_requested"]` flags the difference, and
  scikit-learn's `ConvergenceWarning` is in `metadata["warnings"]`.
- **`n_clusters` > `n_samples`** raises `ValueError` in `fit`, because the
  sample count is only known then.
- **`n_clusters=None`** returns the CF-tree leaf subclusters directly
  (`metadata["n_subclusters"]`).
- `n_noise` is always 0 and `memberships` is always `None`.

## 6. Signature confirmation

The public signatures of `DBSCANClusterer` and `BIRCHClusterer` (`__init__`,
`fit`, `fit_predict`, `get_result`) match section 4 of the work plan exactly:
parameter names, order, type hints, defaults and return types.
`test_public_signature_matches_work_plan` in both test files checks this.
Extra public members are additions only: `get_params()` and `__repr__` on
both classes, the fitted attributes `labels_` (both), `core_sample_indices_`
(DBSCAN) and `subcluster_centers_` (BIRCH), and in `dbscan.py` the helpers
`compatible_algorithms` and `k_distance_curve` plus the constants
`METRIC_ALGORITHMS`, `EXCLUDED_METRICS`, `REQUIRED_METRIC_PARAMS`,
`VALID_ALGORITHMS`, `TREE_ALGORITHMS` and `NOISE_LABEL`.

## Checklist (work plan 4.3)

- [x] Exact public signatures and return types implemented.
- [x] Parameter and input validation included.
- [x] DBSCAN noise is always `-1`.
- [x] Runtime measured in seconds.
- [x] Tests cover valid input, invalid input, unfitted `get_result()`, and a small synthetic dataset.
- [x] At least two metrics and multiple parameter values tested for DBSCAN.
- [x] BIRCH threshold, branching factor, and cluster count varied.
- [x] Short usage example and note on metric/search restrictions supplied.
- [x] No edits to contracts, preprocessing, evaluation, or experiment runner.
