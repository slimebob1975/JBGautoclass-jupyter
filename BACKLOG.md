
# Backlog – JBGAutoClassification

## Current direction

- [ ] Run one or more realistic end-to-end classification projects outside the regression suite and let observed product/runtime issues drive the next patches.
- [ ] Treat `Regr. suite` primarily as a regression safety net after changes rather than continuing to expand it by default.
- [ ] Exercise the full real-world lifecycle where practical: configure data, train, review CV/holdout results, save model, reload model, and predict previously unknown rows.
- [ ] Improve `Repeat last` clarity by restoring the saved run values into all corresponding GUI setting widgets before execution, so the visible table/model/settings match the run that is about to be repeated.

## Known issues / correctness

- [ ] Sometimes: Conversion problem float64 to int 64 when running SMOTE with MLPC in GridSearchCV.
- [ ] Review scoring names/semantics for `Balanced F1 Micro/Macro/Weighted`. In single-label classification, micro-F1 closely tracks accuracy and can hide minority-class failure, as the realistic Återkrav run demonstrated.
- [ ] Review the NumPy compatibility/fallback path for overly broad `TypeError` handling that may mask estimator-internal errors.
- [ ] Review `execute_n_job` exception wrapping; generic wrapping may make outer `TypeError` fallback handling unreachable.

## ML methodology

- [ ] Revisit the train/validation/test methodology. Holdout scores are still visible during spot-checking even though model selection itself is now CV-only; consider a stricter final untouched test set for unbiased final reporting.
- [ ] Review whether `best_test_score` and related state names still communicate their now-diagnostic-only role clearly.
- [ ] Rename stale state/variables such as `candidate_success` where the current behavior no longer matches the name.
- [ ] Use realistic datasets with weaker signal, missing values, class imbalance, correlated/irrelevant features, and mixed categorical/numerical data to expose issues that benchmark fixtures may hide.
- [ ] Later, add a more realistic free-text project with overlapping vocabulary and non-trivial categories; do not add it to the regression suite until it proves useful as a stable regression fixture.

## Serialization / model persistence

- [ ] Generated `autoclassconfig_*.py` files currently persist `sql_password` in plaintext; remove credential serialization and inject credentials only at runtime.
- [ ] Evaluate whether the project can reduce or remove its dependency on `dill` in favor of standard `pickle` by making pipelines fully pickle-friendly.
- [ ] Replace local lambdas/functions embedded in `FunctionTransformer` steps with module-level pickle-friendly callables where practical.
- [ ] Define and test a supported model persistence contract: save, reload in a fresh process, retrain, and predict.
- [ ] Treat serialized model files as trusted input only; document the security implications of loading pickle/dill artifacts.

## Dependencies / packaging / code structure

- [ ] Clean up duplicate and/or unpinned requirements and make the supported Python/scikit-learn dependency set explicit.
- [ ] Reduce `sys.path` manipulation and direct-import coupling in favor of a clearer package/import structure.
- [ ] Review hard-coded flags/settings that should instead be configuration values.
- [ ] Remove or update stale tests/names such as the `Detector`/`Detecter` mismatch when encountered.
- [ ] Make settings and output paths less dependent on the current working directory.

## Runtime / server / operations

- [ ] Investigate intermittent widget-rendering failure after `Reclassification table for most mispredicted`, where the UI only shows `Error displaying widget` without an application crash or explanatory exception. Capture whether the failure originates in widget payload size/content, serialization, frontend rendering, or kernel/frontend state, and make the failure observable in logs.
- [ ] Investigate the Voilà `_xsrf` shutdown/reload 403 behavior.
- [ ] Review the local server/kernel communication setup; the current TCP transport has no encryption and should have an explicit trust/security model.
- [ ] Add possibility of using threads in `execute_n_jobs` when `PicklingError` occurs, but verify NaN handling and estimator thread-safety before enabling it.
- [ ] Revisit resource/file-handle warnings only if they reappear in current runtime logs.

## Regression suite – maintain, do not expand by default

Current coverage includes numeric binary/multiclass classification, 4/13/30-feature datasets, 15 broad model families, multiple preprocessors, NOR/PCA/RFE, Balanced Accuracy, Balanced F1 Macro, random over/undersampling, final GridSearch, Dark Numbers, and a targeted mixed text/category sparse-TF-IDF profile.

- [ ] Add new suite coverage only when a real bug needs a stable reproducer or a genuinely important untested branch is identified.
- [ ] Keep suite runtime bounded; prefer targeted profiles over multiplying the full model/preprocessor/reduction matrix.

## Solved / established

- [X] 046 — Fixed class-label normalization in `validate_dataset`: the previous `DataFrame.astype(...)` result was discarded, so numeric known labels could remain numeric despite the loader contract. Known labels are now normalized to strings while `None`/empty labels remain untouched so prediction rows are still recognized as unclassified; added targeted regression coverage.
- [X] 045 — Removed the repeatedly non-converging `LinearSVC(loss="hinge", dual=True)` branch from the grid. With the project's sparse-compatible `StandardScaler(with_mean=False)`, that branch produced 18 `ConvergenceWarning` messages during the Breast Cancer random-oversampling suite profile even at `max_iter=20000`; the six remaining squared-hinge combinations stay warning-free in targeted regression coverage.
- [X] 044 — Made Regression Suite own its lifecycle reporting: the final progress status now shows wall-clock time for the complete suite, per-profile completion emails are suppressed, and one final compact suite email reports the status of every completed/failed/missing run.
- [X] 043 — Capped TruncatedSVD `n_components` at the available input feature count, preventing the reduction floor of 100 from creating invalid TSVD candidates on compact text matrices such as the 93-feature regression dataset; added default and explicit-component regression tests.
- [X] 042 — Tightened sparse/text spot-check preflight: silently retain expected skipped candidates in the result table instead of printing one `SKIPPED ...` line per combination, skip sparse NOR/RFE + LDA before CV because LDA requires dense input, and cap Nystroem components to the smallest CV training fold to avoid repeated `n_components > n_samples` warnings.
- [X] 041 — Removed the numerically unsafe `alpha=0.0` branch from Multinomial/Bernoulli/Complement Naive Bayes grids. Current scikit-learn keeps zero smoothing unchanged by default, which produced repeated `log(0)` RuntimeWarnings and non-finite probabilities on sparse text data; the grids now use `0.01, 0.1, 1.0` and have targeted finite-probability/warning coverage.
- [X] 040 — Made the automatic text-categorization setting effective in `TextDataToNumbersConverter`; disabling it now keeps low-cardinality text on the text/TF-IDF path while explicitly forced categorical columns still remain categorical. The regression text profile now relies on auto-detection instead of forcing `channel`.
- [X] 039 — Made per-round RFE binary-search status transient by routing it through the existing progress label instead of permanent INFO output; the GUI now updates one status line while genuine RFE stop warnings remain persistent.
- [X] 038 — Regularized the QDA spot-check baseline with the grid's minimum `reg_param=0.1`, preventing rank-deficient covariance matrices from rejecting QDA before its already-regularized grid search can run; added a targeted collinearity regression test.
- [X] 037 — Fixed the Nearest Centroid grid for current scikit-learn: corrected `euclidian` to `euclidean` and replaced invalid `shrink_threshold=0.0` with `None`; regression coverage now fits every configured NCT grid combination.
- [X] 035 — Guard evaluation-time probability collection for estimators without `predict_proba()`: emit one capability warning, then use classification-report precision as the fallback confidence without per-row label-key warnings (including numeric class labels).
- [X] 034 — Added a persistent `Repeat last` action beside `Regr. suite`. It stores the most recent manual classifier settings without SQL credentials and can recreate the run after a GUI/kernel restart using the current login; data is deliberately fetched again.
- [X] 032 — Skip Min-Max preprocessing during spot-check preflight when converted features are sparse; avoids known `MinMaxScaler` CV failures without unsafe automatic densification.
- [X] 031 — Fixed sparse PCA with fractional variance targets on current scikit-learn by using `covariance_eigh`; also fixed spot-check failure propagation so failed CV candidates cannot be treated as successful or have their root error overwritten by validation.
- [X] Persistent timestamped logging and structured warning/error output.
- [X] CV-only winner selection with stdev tie-break; holdout is diagnostic only.
- [X] Final training uses the winning feature-reduction state.
- [X] Modernized several scikit-learn estimator grids and removed known invalid parameter combinations.
- [X] Added multi-dataset regression suite with failure isolation and targeted sampling/text profiles.
- [X] Preserved sparse text features as SciPy CSR at estimator boundaries and removed sparse-to-dense warning noise.
- [X] Guarded Dark Number correction factors against non-finite/non-positive values.
- [X] Removed NumPy arrays from serialized algorithm-grid Enum values to make affected model metadata pickle-safe.
