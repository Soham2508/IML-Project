# K-Means handover (Luis → Soham)

K-Means (task 4) was handed from Soham to Luis. This note covers the six handover items from plan section 10 and the decisions taken.

Branch: `user/luisdavidantonio/kmeans`, created from `user/luisdavidantonio/clustering-dbscan-birch` (`bfe1692`), so it contains the DBSCAN/BIRCH commits too. Merge the DBSCAN/BIRCH pull request first.

## 1. Files

| File | Contents |
|---|---|
| `src/iml_project/core/clustering/kmeans.py` | `KMeansClusterer`, the helper `elbow_curve`, the distances `squared_euclidean` and `manhattan`, and the centroid functions `mean_centroid` and `median_centroid` |
| `tests/test_kmeans.py` | 32 `unittest` tests |
| `src/iml_project/core/clustering/__init__.py` | Docstring updated to mention K-Means |
| `src/iml_project/core/clustering/KMEANS_HANDOVER.md` | This note |

## 2. Public interface (the plan gives none, so the runner must use this one)

```python
KMeansClusterer(
    n_clusters: int = 3,
    init: str | ArrayLike = "random",   # "random" (Forgy) or (n_clusters, n_features) centroids
    n_init: int = 1,
    max_iter: int = 300,
    metric: str = "sqeuclidean",        # or "manhattan" (extension)
    random_state: int | None = None,
)
.fit(X) -> KMeansClusterer
.fit_predict(X) -> NDArray[int64]       # shape (n_samples,)
.get_result() -> ClusterResult          # deep copy; RuntimeError before fit
.get_params() -> dict
```

The fitted attributes are additions outside the contract: `labels_`, `cluster_centers_` (K × d), `inertia_` and `n_iter_`.

`elbow_curve(X, k_values, seeds, metric="sqeuclidean", max_iter=300) -> dict` returns, for each K, the mean, the standard deviation (ddof=0) and the per-seed values of the objective. It never uses labels.

### ClusterResult

- `algorithm`: `"KMeans"` for squared Euclidean, or `"KMeans-manhattan"` for the extension, so result tables never mix the two.
- `memberships`: `None`.
- `n_noise`: 0.
- `n_clusters`: the number of non-empty clusters actually produced. It always equals K, except for the degenerate case described under decision 9.
- `execution_time`: covers the whole pure-Python algorithm, from converting X to lists through all `n_init` runs. Validation and result packaging are excluded.
- `metadata` holds:
  - the plan keys: `parameters`, `backend`, `n_samples`, `n_features`, `cluster_sizes`, `warnings`;
  - details of the run: `random_state_used`, `init_method`, `metric`, `centroid_update`, `objective_name`, `inertia`, `objective_history`, `n_iter`, `converged` and `empty_cluster_events`;
  - for restarts: `n_runs`, `best_run_index` and `run_objectives`;
  - and the final centroids as `cluster_centers`.

## 3. Usage example (from the project root, once `work3.contracts` exists)

```python
import sys
sys.path.insert(0, "src")

import numpy as np
from iml_project.core.clustering.kmeans import KMeansClusterer, elbow_curve

rng = np.random.default_rng(0)
X = np.vstack([rng.normal(c, 0.3, size=(100, 2)) for c in (0.0, 5.0, 10.0)])
curve = elbow_curve(X, k_values=[1, 2, 3, 4, 5], seeds=[0, 1, 2])
kmeans = KMeansClusterer(n_clusters=3, random_state=0)
labels = kmeans.fit_predict(X)          # only X; true labels are never passed
result = kmeans.get_result()
print(result.n_clusters, result.metadata["inertia"], result.execution_time)
```

The suggested experiment grid for the runner:

- K = 2…10, plus more for grid if the elbow isn't clear;
- 10 seeds per K, using `n_init=1`, so each seed is one run;
- report the mean ± std of SSE, silhouette and DBI per K;
- pick K from the elbow together with silhouette and DBI;
- use labels afterwards only, for ARI, purity and F-measure.

## 4. Dependencies

There are no new dependencies.

- **The algorithm itself** uses only the Python standard library: `random`, `math`, `statistics` and `time`.
- **NumPy** appears only at the boundary:
  - it validates and converts `X` once (`X.tolist()`);
  - it converts the final labels and centroids back to arrays, because the contract requires NumPy arrays.
- **scikit-learn** is not imported by `kmeans.py`, and a test checks this. Note that the package `iml_project/core/__init__.py`, from Work 2, imports scikit-learn-based classifiers. So *any* import from `iml_project.core.…` loads scikit-learn indirectly. The K-Means code never uses it.

## 5. Decision log

| # | Decision | Why / implication |
|---|---|---|
| 1 | **Pure Python, with no NumPy inside the algorithm.** NumPy is used only to convert data at the contract boundary | Tasks 4 and 5 say "not allowed to use sklearn library or any other library", while task 6 explicitly allows NumPy. So we follow the strict reading. Runtime is fine: about 0.1–0.5 s per run on a 1,473 × 24 matrix with K = 2–10. Execution times are **not comparable** with scikit-learn's compiled DBSCAN/BIRCH, which the report must say |
| 2 | **Own input validator** in `kmeans.py`, a copy of the checks used by `dbscan.py` | Importing it from `dbscan.py` would load scikit-learn. The error types and messages are the same as in DBSCAN/BIRCH |
| 3 | **Interface** as in section 2, mirroring `DBSCANClusterer`/`BIRCHClusterer`. The parameter names follow scikit-learn's `KMeans` | Task 5's text says a K-Means user specifies K, the initial centroids and the distance metric, so all three are parameters |
| 4 | **Forgy initialisation**: K *distinct* data rows, drawn without replacement with `random.Random(seed)`. An explicit centroid array is also accepted | This is the course's "select K points as initial centroids" (MacQueen/Forgy). Using distinct rows avoids identical starting centroids when the data have duplicates (CMC has 44 repeated rows). It raises `ValueError` if there are fewer than K distinct rows. K-Means++ is left for the improved version |
| 5 | **Restarts**: `n_init` is supported, keeping the run with the lowest objective. The default is `n_init=1` | With the default, one fit is one run, so the runner can loop over seeds and report the mean ± std, as the assignment requires. With an explicit `init`, `n_init > 1` would repeat the same run, so it is reduced to 1 with a warning |
| 6 | **Stopping**: stop when no point changes cluster, with a `max_iter` safety cap (300) | Unchanged labels mean the recomputed centroids are identical. That is slide 7's rule "until centroids do not change", with no floating-point tolerance. Hitting `max_iter` sets `converged=False` and adds a warning instead of raising |
| 7 | **Empty clusters**: the centroid moves to the point farthest from its own centroid, taken only from a cluster that keeps at least one other point. Each event is logged | The run keeps K clusters, which is deterministic, and the objective never increases. Several empty clusters in the same iteration get different points |
| 8 | **Ties** in the assignment step go to the lowest cluster index | Deterministic |
| 9 | **The reported `n_clusters`** is the number of non-empty clusters, and a warning is added if it is below K | This can only happen in degenerate data with duplicates. It hasn't been observed in the tests |
| 10 | **Extension: `metric="manhattan"`**, with the per-feature **median** as the centroid update (K-Medians) | The mean is only correct for squared Euclidean distance: it is the point where the gradient of Σ‖xᵢ − c‖² vanishes. For L1, Σ|xᵢⱼ − cⱼ| is minimised by the median of each feature, so the median update keeps the objective non-increasing. Cosine distance (spherical K-Means, using the normalised mean) was not implemented |
| 11 | **`elbow_curve` helper** for choosing K without labels | It gives the report a "systematic procedure". The objective always falls as K grows, so it must be read for the elbow, not minimised. Silhouette and DBI come from the evaluator, via `sklearn.metrics`, which is allowed for evaluation |
| 12 | **The seed used is recorded** in `random_state_used`, including when `random_state=None` | Any run can be reproduced |
| 13 | **`get_result()` returns a deep copy**, and **`fit()` clears the previous result first** | This is the same behaviour as DBSCAN/BIRCH, so a failed refit can't leave a stale result |

## 6. Known limitations, for the report

- **Local optima.** Forgy often starts two centroids in the same true cluster, and Lloyd's algorithm can't move them apart. With three clear blobs, 1–2 of 3 seeds ended in a worse local optimum (SSE ≈ 1,300 instead of ≈ 50). That is why the experiments need several seeds or `n_init > 1`, and it is what K-Means++ improves.
- **Assumptions about cluster shape.** K-Means assumes round clusters of similar size, and it is sensitive to scaling (hence min-max scaling) and to outliers.
- **One-hot data.** On one-hot columns (heart-statlog, CMC), a centroid is a vector of fractions, for example 0.3 for "occupation 2", which has no categorical meaning. One-hot columns also dominate Euclidean distances: two rows that differ in one category are already √2 apart. **The CMC encoding (24 columns, all one-hot, versus 12 columns with ordinals kept as numbers) is still Soham's open decision, and it strongly affects K-Means.**
- **Cost.** Each iteration costs O(n · K · d) in interpreted Python.

## 7. Confirmation

- All 32 tests in `tests/test_kmeans.py` pass, and the 43 DBSCAN/BIRCH tests still pass. They were run against a temporary local copy of plan section 3's contract, which is not committed.
- `ruff` is clean.
- No contracts, preprocessing, evaluation, runner or plot files were changed.
- K-Means was not yet run on the real datasets: CMC still has to be restored as `data/cmc.arff`, and the experiment grid belongs to the runner.
