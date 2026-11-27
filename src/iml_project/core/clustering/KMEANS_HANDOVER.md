# K-Means handover (Luis → Soham)

K-Means (assignment task 4) was moved from Soham to Luis. This note records what was built, the decisions taken and their implications, and the decisions that are still open for Soham. It covers the six handover items from plan section 10.

**Status (27 Nov 2026):** the implementation and its tests are complete. K-Means has **not yet been run on the real datasets**: that needs the data adapter, the CMC file and the experiment runner (see the open decisions in section 8).

## 1. Scope

- **Task 4:** implement our own K-Means, without scikit-learn "or any other library", using squared Euclidean distance. As an optional extension, add a second distance and explain whether the centroid update stays theoretically appropriate. Choose K appropriately and test different values.
- **Other assignment rules that apply:**
  - repeat random runs over several seeds and report the mean and variability;
  - use the same preprocessing as the other algorithms;
  - never use the class labels for clustering or for choosing parameters, including K;
  - K-Means is a candidate for the PCA part.
- **Not in scope:** the improved K-Means (K-Means++), which stays with Soham.

## 2. Branch, pull request and files

- **Branch:** `user/luisdavidantonio/kmeans`, created from `user/luisdavidantonio/clustering-dbscan-birch` (`bfe1692`), with one commit on top (`3b2eca1`).
- **Pull request:** opened with base `user/luisdavidantonio/clustering-dbscan-birch`, as a GitHub stack on top of **#12** (DBSCAN/BIRCH). Merge #12 first; the stack then retargets the K-Means pull request to `main`.

| File | Contents |
|---|---|
| `src/iml_project/core/clustering/kmeans.py` | `KMeansClusterer`, the helper `elbow_curve`, the distance functions `squared_euclidean` and `manhattan`, and the centroid functions `mean_centroid` and `median_centroid` |
| `tests/test_kmeans.py` | 32 `unittest` tests |
| `src/iml_project/core/clustering/__init__.py` | Docstring now mentions K-Means; no code |
| `src/iml_project/core/clustering/KMEANS_HANDOVER.md` | This note |
| `.gitignore` | Adds `.DS_Store`. Unrelated to K-Means; it went into the same commit |

## 3. Public interface

The plan gives no K-Means signature, so the runner must call this one. Please confirm it (open decision O1).

```python
KMeansClusterer(
    n_clusters: int = 3,
    init: str | ArrayLike = "random",   # "random" = Forgy, or an (n_clusters, n_features) array
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

- **Fitted attributes:** `labels_`, `cluster_centers_` (K × d), `inertia_` and `n_iter_`. They are additions outside the contract.
- **Helper:** `elbow_curve(X, k_values, seeds, metric="sqeuclidean", max_iter=300) -> dict`. It returns, aligned with `k_values`:
  - `objective_mean` and `objective_std` (population std, ddof=0, the same convention as Bhavan's aggregation);
  - `objective_per_seed`;
  - the `seeds`, the `metric` and the `objective_name`.

  It never uses labels.

### ClusterResult

- **`algorithm`:** `"KMeans"` for squared Euclidean, or `"KMeans-manhattan"` for the extension, so result tables never mix the two.
- **`labels`:** `int64`, values 0…K−1.
- **`memberships`:** `None`.
- **`n_noise`:** 0.
- **`n_clusters`:** the number of non-empty clusters actually produced. It equals K except in the degenerate case in decision D13.
- **`execution_time`:** covers the whole pure-Python algorithm, from converting X to lists through all `n_init` runs. Validation and result packaging are excluded.
- **`metadata`:**
  - the plan keys: `parameters`, `backend`, `n_samples`, `n_features`, `cluster_sizes`, `warnings`;
  - details of the run: `random_state_used`, `init_method`, `metric`, `centroid_update`, `objective_name`, `inertia`, `objective_history`, `n_iter`, `converged`, `empty_cluster_events`;
  - for restarts: `n_runs`, `best_run_index`, `run_objectives`;
  - the final centroids: `cluster_centers`.

### Usage example

Run from the project root once `work3.contracts` exists:

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

## 4. How the implementation works

The algorithm is Lloyd's algorithm, following the course's basic K-Means (session 2, slide 7): "select K points as initial centroids; repeat: form K clusters by assigning each point to its closest centroid, recompute the centroid of each cluster; until centroids do not change."

1. **Boundary.** `fit(X)` validates `X`: it must be a non-empty 2-D array of finite numbers, or a `TypeError`/`ValueError` is raised. It then converts `X` **once** to nested Python lists (`X.tolist()`). Everything after this step is plain Python.
2. **Forgy initialisation (`init="random"`).**
   - Duplicate rows are removed, keeping their order.
   - Then `random.Random(seed).sample(distinct_rows, K)` picks K different rows as the starting centroids.
   - Distinct rows matter because CMC has 44 duplicated rows: two identical starting centroids would leave one cluster empty.
   - If there are fewer than K distinct rows, a `ValueError` is raised.
   - The seed actually used is stored, including when `random_state=None`.
3. **Assignment.** For each point, the distance to each centroid is computed in a plain loop and the smallest wins. Ties go to the lowest cluster index (strict `<`).
   - For squared Euclidean, the distance is Σ(aⱼ − bⱼ)², never square-rooted, because the square root doesn't change which centroid is closest.
   - For Manhattan, it's Σ|aⱼ − bⱼ|.
4. **Update.** Each centroid becomes the per-feature **mean** of its points (`math.fsum(column) / n`), or the per-feature **median** (`statistics.median`) for Manhattan.
5. **Empty clusters.** If a cluster has no points, its centroid moves to the point farthest from its own centroid. That point must come from a cluster that keeps at least one other point. Each point is used once, and every event is logged with the iteration, cluster, point index and distance.
6. **Stopping.** After each update, the points are reassigned. If **no label changed**, the next centroids would be identical, so the run stops with `converged=True`. Otherwise it stops after `max_iter` iterations with `converged=False` and a warning.
7. **Restarts.** `n_init` runs use consecutive draws from the same seeded generator, and the lowest-objective run is kept. The default is `n_init=1`, so one fit is one run.
8. **Result.** The labels and centroids are converted back to NumPy arrays and packed into a `ClusterResult`.

**Why both steps never increase the objective.**

- Assignment can only lower each point's distance.
- For squared Euclidean, the mean is the exact minimiser of Σ‖xᵢ − c‖²: the gradient −2Σ(xᵢ − c) is zero at c = x̄.
- For L1, Σ|xᵢⱼ − cⱼ| is minimised by the per-feature median, so the **mean would be the wrong update** for Manhattan. That's why the extension uses the median (K-Medians). This answers the assignment's "is the centroid update still appropriate?"
- Refilling an empty cluster lowers the moved point's cost to 0.

So the objective never increases, and there are finitely many partitions, so every run converges, but only to a **local** optimum. A test checks that the objective history never increases.

**Cost.** Each iteration is O(n · K · d) in interpreted Python. On a random 1,473 × 24 matrix (CMC-sized), a run with K = 2, 5 or 10 took about 0.1–0.5 s.

## 5. Decision log

"Luis" means Luis chose it. "Default" means Luis accepted the recommended option without choosing.

| # | Decision | By | Implication |
|---|---|---|---|
| D1 | **Pure Python.** No NumPy inside the algorithm; NumPy only at the contract boundary (validating and converting `X`, and building the output arrays) | Luis | Tasks 4 and 5 forbid scikit-learn "or any other library", and only tasks 6 and PCA allow NumPy, so we follow the strict reading. Execution times are **not comparable** with scikit-learn's compiled DBSCAN/BIRCH, which the report must say. See O2 |
| D2 | **Own input validator** in `kmeans.py`, instead of importing it from `dbscan.py` | Luis | `dbscan.py` imports scikit-learn. The checks and error messages are the same as in the DBSCAN/BIRCH wrappers |
| D3 | **Interface** as in section 3, mirroring `DBSCANClusterer`/`BIRCHClusterer`; the parameter names follow scikit-learn's `KMeans` | Default | Task 5's text says a K-Means user specifies K, the initial centroids and the distance metric, so all three are parameters. Needs Soham's confirmation (O1) |
| D4 | **Forgy initialisation** with K distinct rows; an explicit centroid array is also accepted | Luis (Forgy), default (explicit array) | This is the course's MacQueen/Forgy seeding. K-Means++ is left for the improved version |
| D5 | **`n_init` supported, with default 1**, keeping the lowest-objective run | Luis | One fit is one run, so the runner loops over seeds and reports mean ± std. With an explicit `init`, `n_init > 1` is reduced to 1 with a warning, because the runs would be identical |
| D6 | **Stopping** when no label changes, with a `max_iter=300` cap | Luis | This is the slide's "centroids do not change", with no floating-point tolerance and no `tol` parameter. Hitting the cap adds a warning instead of raising |
| D7 | **Empty clusters** get the farthest point, from a cluster with more than one point, and the event is logged | Luis | The run keeps K clusters, it's deterministic, and the objective never increases |
| D8 | **Extension `metric="manhattan"`** with the per-feature median update | Luis | It gives the report its "is the update still appropriate" answer. Cosine (spherical K-Means) was not implemented |
| D9 | **`elbow_curve` helper** for choosing K without labels | Luis | It gives the report a systematic procedure. Silhouette and DBI come from the evaluator via `sklearn.metrics`, which is allowed for evaluation |
| D10 | **Branch** `user/luisdavidantonio/kmeans`, created from the DBSCAN/BIRCH branch; the pull request is stacked on #12 | Luis (name), default (base) | The K-Means pull request merges after #12 |
| D11 | **Algorithm names** `"KMeans"` and `"KMeans-manhattan"` | Default | Result tables can tell the two variants apart |
| D12 | **Ties** in the assignment step go to the lowest index | Default | Deterministic |
| D13 | **`n_clusters` in the result** counts the non-empty clusters; a warning is added if it's below K | Default | This only happens with degenerate duplicate data and hasn't been observed |
| D14 | **The seed used is always recorded** in `random_state_used` | Default | Any run, including `random_state=None`, can be reproduced |
| D15 | **`fit()` clears the previous result first**, and `get_result()` returns a deep copy | Default | The same behaviour as DBSCAN/BIRCH: a failed refit can't leave a stale result, and callers can't corrupt the stored one |

## 6. Tests and verification

- **`tests/test_kmeans.py`: 32 tests, all passing.** The 43 DBSCAN/BIRCH tests still pass too (75 in total). They were run with scikit-learn 1.9.1 and NumPy 2.5 against a temporary local copy of plan section 3's contract, which was **not committed**.
- **Coverage:**
  - three blobs are recovered;
  - every point is assigned to its nearest final centroid (both metrics);
  - converged centroids are the means, or the medians for Manhattan;
  - the objective history never increases (both metrics, 5 seeds each);
  - the inertia equals the recomputed SSE;
  - K = 1 gives the global mean, and K = n gives SSE 0;
  - the same seed gives identical results, and different seeds can differ;
  - a run with `random_state=None` can be reproduced from its recorded seed;
  - Forgy uses distinct rows;
  - an explicit `init` is used, and `n_init` with an explicit init warns;
  - `n_init` keeps the lowest objective;
  - empty clusters are refilled with the farthest point, including two at once;
  - hitting `max_iter` adds a warning;
  - invalid parameters and input are rejected;
  - NumPy integers are accepted as parameters;
  - `get_result()` raises before `fit`, and a failed refit clears the result;
  - the input array isn't modified, and `get_result()` returns a copy;
  - the signature, `__repr__` and metadata keys are as documented;
  - the module imports only the standard library, NumPy and `work3`;
  - `elbow_curve`'s values and argument checks.
- `ruff` is clean.
- The usage example above runs as written.
- No contracts, preprocessing, evaluation, runner or plot files were changed.

## 7. Known limitations (for the report)

- **Local optima.** Forgy can start two centroids in the same true cluster, and Lloyd's algorithm can't separate them. In two small three-blob examples, 1 of 3 seeds ended in a local optimum with an SSE about 50 times higher (2,575 instead of 53, and 3,970 instead of 60), so the *mean* elbow curve can hide the true K. Use several seeds (and look at the best run per K, or use `n_init > 1`). This is exactly what K-Means++ improves.
- **Assumptions about the data.** K-Means assumes round clusters of similar size, and it is sensitive to feature scaling (hence min-max scaling) and to outliers.
- **One-hot data.**
  - A centroid on one-hot columns is a vector of fractions, for example 0.3 for "occupation 2", which has no categorical meaning.
  - One-hot columns dominate Euclidean distance: two rows that differ in one category are already √2 apart.
  - This matters for heart-statlog (28 columns) and CMC (see O3).
- **Choosing K.** The objective always falls as K grows, so K must be chosen from the elbow plus silhouette and DBI, not by minimising the SSE.
- **Grid.** It is a regular lattice whose classes are black and white regions, so K-Means' round clusters are unlikely to match them. The report should discuss this rather than tune for it.
- **Timing.** Execution times are those of interpreted Python (D1).

## 8. Open decisions for Soham

| # | Open decision | Options | Luis's recommendation |
|---|---|---|---|
| O1 | **Confirm the K-Means interface** (section 3) for the runner and the configuration files | Accept it, or ask for changes before integration | Accept. It mirrors the DBSCAN/BIRCH wrappers and scikit-learn's parameter names |
| O2 | **NumPy at the K-Means boundary.** The contract forces `X` in and `labels` out as NumPy arrays, so `kmeans.py` imports NumPy only for that. Is this acceptable under "no other library"? | (a) Keep it, and say so in the report. (b) Luis can shrink NumPy to just building the output arrays, by validating the input with `math.isfinite` and similar. (c) Change the contract so custom algorithms return lists and the runner converts them | (a), with one sentence in the report. (c) would affect every algorithm and the evaluator |
| O3 | **CMC encoding** (the preprocessing owner's call). By default CMC becomes 24 columns, all one-hot, and the ordinal order is lost | 24 columns (all one-hot), or **12 columns**: the 3 ordinal attributes kept as scaled numbers and the 3 binary ones as single 0/1 columns | 12 columns. K-Means results depend heavily on this choice, and it must be the same for every algorithm |
| O4 | **Where `ClusterResult` lives.** All of Luis's modules import `from work3.contracts import ClusterResult`, which isn't pushed yet | Push `src/work3/contracts.py` as in the plan, or choose another location (then only one import line per module changes) | Push `work3/contracts.py` as planned |
| O5 | **Who runs the K-Means experiments, and with which grid** | The runner owns the grids (plan 7.3) | K = 2…10 (more for grid if the elbow is unclear); 10 seeds per K with `n_init=1`; report mean ± std of SSE, silhouette and DBI per K; choose K from the elbow plus silhouette and DBI; compute external metrics (ARI, purity, F-measure, confusion matrix) only afterwards. Luis can supply the parameter configuration file |
| O6 | **Whether to report the Manhattan extension** | Include it (`metric="manhattan"`, the median update and the theory paragraph from section 4), or leave it out for space | Include it briefly. It's the assignment's suggested extension and it's already tested |
| O7 | **K-Means++ reusing this code.** K-Means++ differs only in initialisation | Soham's `kmeans_plus_plus.py` can pass its seeded centroids to `KMeansClusterer(init=centroids)`, or reuse `_lloyd`, so both variants share the same assignment, update and stopping code. Or he writes it independently | Reuse it. The comparison between K-Means and K-Means++ then isolates the effect of initialisation |
| O8 | **The import chain loads scikit-learn.** `iml_project/core/__init__.py` (Work 2) imports classifiers built on scikit-learn, so *any* import from `iml_project.core.…`, including `kmeans`, loads scikit-learn indirectly, even though K-Means never uses it | Accept it and mention it, or make `core/__init__.py` import lazily, or move the Work 3 clustering code out of `iml_project.core` | Accept it and mention it. Changing the Work 2 package is risky this close to the deadline |
| O9 | **Data files.** Only the fold folders of grid and heart-statlog exist in `data/`; `data/cmc.arff` is missing | Restore the single ARFF files (the plan warns not to concatenate folds without checking) | Restore `data/cmc.arff` and the single grid and heart-statlog files before running experiments |
| O10 | **Merge order** | #12 (DBSCAN/BIRCH), then the K-Means pull request, which is stacked on it | Merge #12 on the website first. If it's squash-merged and the K-Means pull request isn't rebased automatically, run `git rebase --onto origin/main bfe1692 user/luisdavidantonio/kmeans` and force-push |
| O11 | **The `.gitignore` change** (adding `.DS_Store`) went into the K-Means commit | Keep it, or ask Luis to move it into its own commit | Keep it. It's harmless and stops macOS files being committed |

## 9. Next steps

1. Soham reviews and merges #12, then the K-Means pull request.
2. Soham decides O1–O4 and O9: the interface, NumPy, the CMC encoding, the location of the contract, and the data files.
3. The runner executes the K grid (O5) on all three datasets with the same preprocessing as the other algorithms.
4. The report states:
   - the pure-Python and NumPy-boundary choice (D1/O2);
   - Forgy initialisation and the use of seeds;
   - the procedure for choosing K (elbow + silhouette + DBI) and which K was best per dataset;
   - the Manhattan/median extension and why the mean isn't appropriate there;
   - the limitations in section 7.

## 10. References

- Lloyd, S. P. (1982). Least squares quantization in PCM. *IEEE Transactions on Information Theory*, 28(2), 129–137.
- MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability*, Vol. 1, 281–297.
- Forgy, E. W. (1965). Cluster analysis of multivariate data: efficiency versus interpretability of classifications. *Biometrics*, 21, 768–769.
- Jain, A. K. (2010). Data clustering: 50 years beyond K-means. *Pattern Recognition Letters*, 31(8), 651–666.
- Bradley, P. S., Mangasarian, O. L., & Street, W. N. (1997). Clustering via concave minimization. *Advances in Neural Information Processing Systems 9*, 368–374. (K-Median with the L1 norm and the median update.)
- Course material: Work 3 assignment (2026–2027, v2), tasks 4–6; support session 2, slides 6–10.
