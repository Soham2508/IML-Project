# Fuzzy C-Means handover (Luis → Soham)

Fuzzy C-Means (assignment task 7) is listed under Soham in the plan, and Luis has now implemented it. This note records what was built, the decisions taken and their implications, and the decisions that are still open for Soham. It covers the six handover items from plan section 10.

**Status (27 Nov 2026):** the implementation and its tests are complete. FCM has **not yet been run on the real datasets**: that needs the data adapter, the CMC file and the experiment runner (see section 8).

## 1. Scope

- **Task 7:** "Implement your own code of the fuzzy clustering algorithm… implement the original Fuzzy C-Means algorithm and apply it to three different datasets" (Bezdek, 1981).
  - Analyse the effect of the number of clusters C, the fuzzy exponent m, the initialisation and the stopping criterion.
  - Test several values of C and m, and justify the final configuration.
- **Other assignment rules that apply:**
  - standard FCM may be applied to a numerical representation of the data, and that choice must be justified;
  - repeat random runs over several seeds and report the mean and variability;
  - never use the class labels for clustering or for choosing parameters.
- **The report must answer:** "In the case of Fuzzy Clustering… you can optimize the C value. Have you done the optimization? Which are the results?…"

## 2. Files

| File | Contents |
|---|---|
| `src/iml_project/core/clustering/fuzzy_cmeans.py` | `FuzzyCMeansClusterer`, the helpers `fuzzy_validity_curve` and `validity_indices`, and the building blocks `squared_distances`, `update_memberships`, `update_centers` and `objective` |
| `tests/test_fuzzy_cmeans.py` | 37 `unittest` tests |
| `src/iml_project/core/clustering/__init__.py` | Docstring now mentions Fuzzy C-Means; no code |
| `src/iml_project/core/clustering/FCM_HANDOVER.md` | This note |

## 3. Public interface

The plan gives no FCM signature, so the runner must call this one. Please confirm it (O1).

```python
FuzzyCMeansClusterer(
    n_clusters: int = 3,                 # C
    m: float = 2.0,                      # fuzzifier, must be > 1
    init: str | ArrayLike = "random_memberships",   # or "random_centers", or (C, d) centres
    max_iter: int = 300,
    tol: float = 1e-5,
    stop_criterion: str = "centers",     # or "memberships", "objective"
    random_state: int | None = None,
)
.fit(X) -> FuzzyCMeansClusterer
.fit_predict(X) -> NDArray[int64]        # hard labels = argmax membership
.get_result() -> ClusterResult           # deep copy; RuntimeError before fit
.get_params() -> dict
```

- **Fitted attributes:** `labels_`, `memberships_` (n × C), `cluster_centers_` (C × d), `objective_` and `n_iter_`. They are additions outside the contract.
- **`validity_indices(X, U, V, m) -> dict`:** returns PC, PE and XB (section 4).
- **`fuzzy_validity_curve(X, c_values, seeds, m=2.0, init=…, stop_criterion=…, tol=…, max_iter=…) -> dict`:** for each C, it returns `<name>_mean`, `<name>_std` (ddof=0, ignoring NaN) and `<name>_per_seed` for `objective`, `partition_coefficient`, `partition_entropy` and `xie_beni`. It never uses labels. To study the fuzzifier, call it once per value of m.

### ClusterResult

- **`algorithm`:** `"FuzzyCMeans"`.
- **`memberships`:** the (n_samples, C) matrix. Its values are in [0, 1] and each row sums to 1, as the plan's contract asks for FCM.
- **`labels`:** `int64` hard labels, the argmax of each row (ties go to the lowest index), for the evaluator and the plots.
- **`n_clusters`:** the number of clusters that are the largest membership of at least one point. This can be below C, with a warning.
- **`n_noise`:** 0.
- **`execution_time`:** covers initialisation and all iterations. Validation and result packaging are excluded.
- **`metadata`:**
  - the plan keys: `parameters`, `backend`, `n_samples`, `n_features`, `cluster_sizes`, `warnings`;
  - details of the run: `random_state_used`, `init_method`, `stop_criterion`, `tol`, `objective`, `objective_history`, `change_history`, `n_iter`, `converged`;
  - events: `zero_distance_events`, `degenerate_cluster_events`;
  - `hard_labels`, `cluster_centers`;
  - the validity indices: `partition_coefficient`, `partition_entropy`, `xie_beni`.

### Usage example

Run from the project root once `work3.contracts` exists:

```python
import sys
sys.path.insert(0, "src")

import numpy as np
from iml_project.core.clustering.fuzzy_cmeans import FuzzyCMeansClusterer, fuzzy_validity_curve

rng = np.random.default_rng(0)
X = np.vstack([rng.normal(c, 0.3, size=(100, 2)) for c in (0.0, 5.0, 10.0)])
curve = fuzzy_validity_curve(X, c_values=[2, 3, 4, 5], seeds=[0, 1, 2], m=2.0)
fcm = FuzzyCMeansClusterer(n_clusters=3, m=2.0, random_state=0)
labels = fcm.fit_predict(X)          # only X; true labels are never passed
result = fcm.get_result()            # memberships: (300, 3), rows sum to 1
print(result.metadata["partition_coefficient"], result.execution_time)
```

## 4. How the implementation works

**Objective.** FCM minimises

J_m(U, V) = Σᵢ Σⱼ uᵢⱼᵐ ‖xᵢ − vⱼ‖²,  subject to uᵢⱼ ∈ [0, 1] and Σⱼ uᵢⱼ = 1 for every point i.

Here U is the n × C membership matrix, V holds the C centres, and m > 1 is the fuzzifier. Setting the derivatives of the Lagrangian (one multiplier per point's sum-to-one constraint) to zero gives the two updates:

- **Memberships:** uᵢⱼ = 1 / Σₖ (dᵢⱼ² / dᵢₖ²)^(1/(m−1)).

  A point's membership depends only on its *relative* distances to the centres. The code computes the equivalent form (dᵢⱼ²)^(−p) / Σₖ (dᵢₖ²)^(−p), with p = 1/(m−1), in **log space** (a softmax of −p·log d²). This avoids overflow and underflow when m is close to 1, where p is large.
- **Centres:** vⱼ = Σᵢ uᵢⱼᵐ xᵢ / Σᵢ uᵢⱼᵐ.

  Each centre is a weighted mean of *all* points, with weights uᵐ.

**Alternating optimisation** (session 2, slide 27): t ← t + 1; compute Uₜ from Vₜ₋₁; compute Vₜ from Uₜ; repeat until t = max_iter or the stopping criterion holds. Each update is the exact minimiser of J_m with the other block fixed, so **J_m never increases**. The objective history `J_m(U_t, V_t)` is recorded, and a test checks that it is monotone for both random initialisations and all three stopping criteria.

**Implementation details:**

1. **Distances.** Squared Euclidean distances are computed by broadcasting. The ‖x‖² + ‖v‖² − 2x·v expansion is avoided because it can turn an exact 0 into ±1e-16, and the zero-distance rule needs exact zeros.
2. **Initialisation.**
   - `random_memberships` (Bezdek's original, the default): a random U₀ with rows normalised to 1, then V₀ is computed from it.
   - `random_centers` (the slide's "guess initial cluster centers V₀"): C *distinct* data rows. This matters because CMC has duplicate rows.
   - Explicit centres: a (C, d) array.
   - All randomness uses `np.random.default_rng(seed)`, and the seed actually used is always recorded.
3. **Zero distances** (Bezdek's rule): if a point coincides with q centres, it gets membership 1/q in each of them and 0 elsewhere. The events are logged.
4. **A degenerate centre.** If a centre receives zero total weight, which needs every point to sit exactly on other centres, it keeps its previous position, with a logged warning.
5. **Stopping criteria.** Largest absolute changes are used, so `tol` has the same meaning for any n or C:
   - `centers` (slide): max|Vₜ − Vₜ₋₁| ≤ tol;
   - `memberships` (Bezdek et al., 1984): max|Uₜ − Uₜ₋₁| ≤ tol;
   - `objective`: |Jₜ₋₁ − Jₜ| / Jₜ₋₁ ≤ tol.

   The value tested at each iteration is stored in `change_history`. Hitting `max_iter` sets `converged=False` and adds a warning instead of raising.
6. **Hard labels:** argmax of U.

**Validity indices** (label-free, for choosing C and m):

- **Partition coefficient:** PC = (1/n) Σ uᵢⱼ², in [1/C, 1]. **Higher is better** (crisper).
- **Partition entropy:** PE = −(1/n) Σ uᵢⱼ ln uᵢⱼ, in [0, ln C]. **Lower is better.**
- **Xie–Beni:** XB = J_m / (n · min_{j≠k} ‖vⱼ − vₖ‖²). **Lower is better**, meaning compact, well-separated clusters. It is NaN when C = 1 or when two centres coincide.

PC and PE are biased: PC tends to fall and PE to rise as C grows. So they should be read together with XB, and with silhouette and DBI on the hard labels from the evaluator.

**What m does.** As m → 1 the memberships become crisp, so FCM behaves like K-Means. As m grows they tend towards 1/C. The tests check this on three blobs: PC was above 0.999 at m = 1.1, fell steadily as m increased, and was below 0.6 at m = 10. A common default is m = 2.

**Cost.** Each iteration is O(n · C · d), vectorised. On a random 1,473 × 24 matrix (CMC-sized), C = 2–10 took under 0.01 s per run.

## 5. Decision log

"Luis" means Luis chose it. "Default" means Luis accepted the recommended option.

| # | Decision | By | Implication |
|---|---|---|---|
| F1 | **NumPy for the array maths; no clustering or fuzzy library** (no scikit-fuzzy) | Luis | Task 7 says only "implement your own code". Unlike tasks 4 and 5, it doesn't add "no other library", and task 6 and PCA allow NumPy explicitly. So this differs from K-Means, which is pure Python under the stricter wording of tasks 4–5. The report should state both choices |
| F2 | **Interface** as in section 3, in the agreed clustering folder (not the plan's `work3/fuzzy/`) | Luis | The same shape as DBSCAN, BIRCH and K-Means. Needs Soham's confirmation (O1) |
| F3 | **Initialisation**: `random_memberships` (Bezdek's original) by default; `random_centers` (the slide) and explicit centres as alternatives | Luis ("follow task 7") | Task 7 asks for the *original* FCM **and** an analysis of the initialisation, so both random schemes are available for comparison |
| F4 | **Three selectable stopping criteria** (`centers` by default, `memberships`, `objective`), all using largest absolute changes and a `max_iter` cap | Luis (recommendation) | This supports the requested analysis of the stopping criterion. `change_history` makes convergence plots possible |
| F5 | **Zero distance**: Bezdek's exact rule (1/q membership on the q coinciding centres) | Luis | No epsilon distorts the other memberships. It can happen with duplicate rows (CMC) and `random_centers` |
| F6 | **The validity helper** reports J_m, PC, PE and XB per C (mean ± std over seeds) | Luis | It answers the report's "did you optimise C?" without labels. The indices are also stored per run in metadata |
| F7 | **m must be > 1** (TypeError/ValueError otherwise) | Default | At m = 1 the update divides by m − 1 = 0, and the limit is K-Means. The log-space update keeps m close to 1 (e.g. 1.001) stable |
| F8 | **The membership update is computed in log space** | Default | No overflow or underflow for any m > 1. A test checks it against the textbook formula to 1e-12 |
| F9 | **Distances are computed by broadcasting**, not by the dot-product expansion | Default | Exact zeros for the zero-distance rule. The memory needed is n × C × d floats, fine for these datasets |
| F10 | **Hard labels = argmax(U)**, with ties going to the lowest index; `n_clusters` counts the clusters that actually appear in the labels | Default | The evaluator and plots get ordinary labels, while the full U stays in `memberships` |
| F11 | **No `n_init` parameter**; one fit is one run | Default | The runner loops over seeds and reports mean ± std, the same as for K-Means |
| F12 | **Input checks reuse `_validate_feature_matrix` and `_check_int` from `kmeans.py`** | Default | The same checks and messages as the other modules, with no scikit-learn import. It depends on two private names in `kmeans.py` |
| F13 | **XB is NaN when undefined** (C = 1, or coinciding centres); the curve's mean and std ignore NaN, and are NaN if every value is NaN | Default | Matches Bhavan's aggregation convention (plan 5.3) |
| F14 | **`fit()` clears the previous result first**, and `get_result()` returns a deep copy | Default | The same behaviour as the other modules |

## 6. Tests and verification

- **`tests/test_fuzzy_cmeans.py`: 37 tests, all passing.** The 75 K-Means, DBSCAN and BIRCH tests still pass too (112 in total). They were run with scikit-learn 1.9.1 and NumPy 2.5 against a temporary local copy of plan section 3's contract, which is not part of these files.
- **Coverage:**
  - the building blocks:
    - the membership formula by hand (u = 0.8 / 0.2) and against the textbook definition for m = 1.3, 2 and 4;
    - the zero-distance rule (one and two coinciding centres);
    - no overflow at m = 1.001;
    - the weighted-mean centre update by hand, and the degenerate-centre fallback;
    - exact-zero distances;
    - PC, PE and XB by hand, and with C = 1;
  - the clusterer:
    - three blobs are recovered with every initialisation;
    - J_m never increases (2 random initialisations × 3 criteria × 3 seeds);
    - a converged solution is a fixed point of both updates;
    - the stored objective matches a recomputed one;
    - C = 1 gives the global mean;
    - m controls softness (PC falls monotonically from m = 1.1 to m = 10);
    - duplicate rows and a point lying on a centre are handled;
    - each stopping criterion is recorded and met;
    - hitting `max_iter` adds a warning;
    - seeds are reproducible, differ from each other, and a `random_state=None` run can be repeated from its recorded seed;
    - the initialisation method is recorded;
    - invalid parameters and input are rejected;
    - `get_result()` raises before `fit`, a failed refit clears the result, the input isn't modified, and `get_result()` returns a copy;
    - the signature, `__repr__` and metadata keys are as documented;
    - no clustering library is imported;
  - the curve:
    - PC, PE and XB all choose C = 3 on three blobs;
    - mean and std are computed correctly;
    - XB is NaN for C = 1;
    - invalid arguments are rejected.
- `ruff` is clean.
- The usage example above runs as written.
- No contracts, preprocessing, evaluation, runner or plot files were changed.

## 7. Known limitations (for the report)

- **Assumptions about cluster shape.** Like K-Means, FCM assumes round clusters (Euclidean distance) and is sensitive to scaling.
- **The sum-to-one constraint.** It forces outliers to share their membership among the clusters instead of getting a low membership everywhere (possibilistic C-Means removes this constraint).
- **High dimensions.** Distances become similar, so the memberships drift towards 1/C and PC towards its minimum. On a random 1,473 × 24 matrix the runs converged in 5–6 iterations with near-uniform memberships. Expect this with one-hot CMC and heart-statlog, and report PC/PE relative to 1/C and ln C.
- **One-hot data.** The centres are weighted means of the one-hot columns, so they are fractions with no categorical meaning. The task allows standard FCM on a numerical representation, but it must be justified (see O3).
- **Validity indices.** PC and PE are monotonically biased in C, and XB can favour small C. Use several indices and discuss any disagreement, as the report questions ask.
- **Local optima.** FCM is less sensitive to initialisation than K-Means, but it can still reach different local optima, so several seeds are needed.

## 8. Open decisions for Soham

| # | Open decision | Options | Luis's recommendation |
|---|---|---|---|
| O1 | **Confirm the FCM interface** (section 3) for the runner, and that Luis owns FCM (the plan lists it under Soham) | Accept it, or ask for changes before integration | Accept. It follows the DBSCAN, BIRCH and K-Means modules |
| O2 | **NumPy in FCM, but not in K-Means** | Keep it as it is, explained by the different wording of the tasks, or make them consistent | Keep it, and give the report one sentence on the task wording |
| O3 | **Representation for FCM.** The task asks for the representation to be justified | Standard FCM on the same numerical encoding as the other algorithms, or a mixed-data fuzzy algorithm (not implemented) | Standard FCM on the same encoding as everything else. The CMC encoding (24 one-hot columns or 12 with ordinals kept) is still open from the K-Means handover |
| O4 | **The experiment grid** (the runner owns the grids) | — | C = 2…10; m ∈ {1.5, 2, 2.5, 3}; both random initialisations; the three stopping criteria at one tol (e.g. 1e-5), and tol ∈ {1e-3, 1e-5, 1e-7} for the default criterion; 10 seeds each. Report mean ± std of J_m, PC, PE and XB per (C, m), plus silhouette and DBI on the hard labels, and iterations and time per stopping criterion. External metrics (ARI, purity, F-measure) only afterwards |
| O5 | **How to evaluate fuzzy results externally** | Hard labels (argmax) for ARI, purity and the confusion matrix, or also a fuzzy external index (not implemented) | Use the hard labels, and say so in the report. The memberships stay available for plots, for example colouring by max membership |
| O6 | **Where `ClusterResult` lives.** FCM imports `from work3.contracts import ClusterResult`, which doesn't exist yet | The same open item as in the K-Means handover | Create `work3/contracts.py` as planned |
| O7 | **Dependency on `kmeans.py`.** FCM imports `_validate_feature_matrix` and `_check_int` from `kmeans.py` | Keep it, or move the shared checks into a small `clustering/_validation.py` | Keep it for now. Refactoring them into a shared module is a one-line change per import |

## 9. Next steps

1. Soham decides O1–O7.
2. The runner executes the FCM grid (O4) on all three datasets with the same preprocessing as the other algorithms.
3. The report states:
   - the representation used and its justification;
   - the effect of C, m, the initialisation and the stopping criterion;
   - how C was chosen (PC, PE, XB, silhouette, DBI) and the best configuration per dataset;
   - the limitations in section 7.

## 10. References

- Bezdek, J. C. (1981). *Pattern Recognition with Fuzzy Objective Function Algorithms.* Plenum Press, New York.
- Bezdek, J. C., Ehrlich, R., & Full, W. (1984). FCM: The fuzzy c-means clustering algorithm. *Computers & Geosciences*, 10(2–3), 191–203.
- Dunn, J. C. (1973). A fuzzy relative of the ISODATA process and its use in detecting compact well-separated clusters. *Journal of Cybernetics*, 3(3), 32–57.
- Xie, X. L., & Beni, G. (1991). A validity measure for fuzzy clustering. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 13(8), 841–847.
- Pal, N. R., & Bezdek, J. C. (1995). On cluster validity for the fuzzy c-means model. *IEEE Transactions on Fuzzy Systems*, 3(3), 370–379.
- Course material: Work 3 assignment (2026–2027, v2), task 7 and the report questions; support session 2, slides 25–28.
