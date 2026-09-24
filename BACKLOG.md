
# Backlog – JBGAutoClassification

## Current direction

- [X] Run one or more realistic end-to-end classification projects outside the regression suite and let observed product/runtime issues drive the next patches. This revision used repeated targeted real-data runs to drive fixes 031–047.
- [ ] Treat `Regr. suite` primarily as a regression safety net after changes rather than continuing to expand it by default.
- [ ] Exercise the full real-world lifecycle where practical: configure data, train, review CV/holdout results, save model, reload model, and predict previously unknown rows.
- [ ] Improve `Repeat last` clarity by restoring the saved run values into all corresponding GUI setting widgets before execution, so the visible table/model/settings match the run that is about to be repeated.

## Known issues / correctness

- [X] 053 — Normalize feature input to float64 before interpolation-based SMOTE-family oversampling, preventing integer truncation of synthetic samples and the observed float64-to-int64 failure path in MLPC/GridSearchCV. Sparse matrices stay sparse; random/categorical-only oversampling keeps its prior dtype behavior. Runtime integration is verified on Breast Cancer with an `IMP -> FLT -> SMOTE -> ... -> MLPC` GridSearchCV pipeline; that dataset was already float64, so the original integer-input path remains covered by the targeted regression reproducer rather than a real-data rerun.
- [ ] Review scoring names/semantics for `Balanced F1 Micro/Macro/Weighted`. In single-label classification, micro-F1 closely tracks accuracy and can hide minority-class failure, as the realistic Återkrav run demonstrated.
- [ ] Review the NumPy compatibility/fallback path for overly broad `TypeError` handling that may mask estimator-internal errors.
- [X] 054 — Make `execute_n_job` exception-safe: preserve original exception types/tracebacks for caller-specific handling, retry only the existing resource/pickling exception classes with reduced worker counts, re-raise the original failure at one worker, and treat a negative global worker setting as unlimited rather than accidentally overriding an explicit positive `n_jobs_desired`. The normal parallel path is runtime-verified on a broad Breast Cancer run through spot-check, GridSearchCV, final fit, evaluation and retraining; the exceptional branches remain covered by targeted regression tests.
- [X] 055 — Mark Bernoulli/Complement/Multinomial Naive Bayes as RFE-incompatible so the existing spot-check compatibility gate skips those candidates before RFE attempts to read `coef_`/`feature_importances_`. This removes the avoidable RFE failures seen in the broad Breast Cancer stress run.
- [ ] Add non-negative-feature compatibility preflight for Multinomial/Complement Naive Bayes after preprocessing/reduction; the broad stress run still produced failures when transformations introduced negative values.
- [ ] Investigate the Keras MLP wrapper/grid shape error (`Cannot convert '100' to a shape`) observed in the broad Breast Cancer stress run.

## ML methodology

- [ ] Revisit the train/validation/test methodology. Holdout scores are still visible during spot-checking even though model selection itself is now CV-only; consider a stricter final untouched test set for unbiased final reporting.
- [ ] Review whether `best_test_score` and related state names still communicate their now-diagnostic-only role clearly.
- [ ] Rename stale state/variables such as `candidate_success` where the current behavior no longer matches the name.
- [ ] Use realistic datasets with weaker signal, missing values, class imbalance, correlated/irrelevant features, and mixed categorical/numerical data to expose issues that benchmark fixtures may hide.
- [ ] Later, add a more realistic free-text project with overlapping vocabulary and non-trivial categories; do not add it to the regression suite until it proves useful as a stable regression fixture.

## Dark Numbers / label-noise estimation

- [X] 049 — Froze the published linear Dark Number formula as a compatibility contract and strengthened correctness/observability around it: the one-vs-rest formula is regression-tested, the FP=0/FN>0 alpha bug is fixed, injected-noise fraction is configurable with 20% as the compatibility default, correction factors are estimated separately for the CV and retrained model/data pairings, and `D_cv_test`, `D_cv_full`, and `D_retrained_full` are reported separately while the old combined full-data result is retained as an explicit legacy comparison. Runtime-verified on the Breast Cancer project: both model-specific corr estimations and all three named estimates completed without application exceptions.
- [X] 050 — Added a standalone validation harness without changing the Dark Number formula: inject known positive-to-negative label noise, compute the known hidden-positive share, run D_cv_test/D_cv_full/D_retrained_full with model-specific correction factors, measure envelope coverage and truth position, summarize correction-factor stability across repeated runs, and quantify probability-ranked enrichment relative to random selection.
- [X] 051 — Robustified correction-factor estimation without changing the Dark Number formula: non-finite/zero-recovery and synthetic insufficient-sample fallback values are excluded from regression, regression requires enough real finite observations, corr provenance is reported as `direct`, `regressed`, or `fallback/unestimable`, and the validation harness uses the same fallback semantics. Runtime-verified on Wine after 052: all six class/model correction factors were finite `direct` estimates without regressor/fallback use.
- [ ] Refine the Dark Number correction fallback strategy by failure cause. Keep sample-size regression primarily for resource-constrained execution of the same target estimator (for example memory pressure), rather than treating zero recovery/statistically unestimable correction as a sample-size problem. For statistically unestimable cases, validate an explicit fallback correction classifier in the 050 harness before selecting a default; FUT Voting is an initial candidate because it has been used successfully before. Always report the target model, the model actually used to estimate `corr`, and provenance such as `direct`, `regressed_same_model`, or `fallback_model:<name>`; never present another model's correction factor as if it belonged to the target estimator.
- [ ] Add dedicated Dark Numbers GUI/config controls and decouple calculation from `Display mispredicted`: a separate enable/disable control plus a mutually exclusive Linear/Non-linear method choice; later expose advanced settings such as injected-noise fraction and alpha variant where useful, and persist/restore them through generated config files and `Repeat last`.

## Serialization / model persistence

- [X] Generated config/model artifacts no longer persist `sql_password`; generated configs use runtime `JBG_SQL_PASSWORD` when needed and loaded models receive the current runtime credentials.
- [ ] Evaluate whether the project can reduce or remove its dependency on `dill` in favor of standard `pickle` by making pipelines fully pickle-friendly.
- [X] 055 — Replace the no-op oversampling, undersampling, preprocessing and reduction `FunctionTransformer` lambdas with a shared module-level identity callable so ordinary pipelines no longer depend on dill serializing local lambdas.
- [X] 055 — Define and test the supported model persistence contract: new `.sav` files use a versioned artifact envelope with required fields and atomic main-file replacement, legacy six-item artifacts remain readable, config loading uses the same parser, fixture regeneration writes the current format, and a fresh-process regression test covers reload, predict, retrain and predict. The contract remains dependency-version-sensitive and trusted-input-only.
- [X] Documented that serialized pickle/dill model files are trusted-input-only and must not be loaded from untrusted sources.

## Dependencies / packaging / code structure

- [ ] Pin the supported Python/scikit-learn dependency set explicitly; the duplicate `matplotlib` requirement was removed in 048.
- [ ] Reduce `sys.path` manipulation and direct-import coupling in favor of a clearer package/import structure.
- [ ] Review hard-coded flags/settings that should instead be configuration values.
- [ ] Remove or update stale tests/names such as the `Detector`/`Detecter` mismatch when encountered.
- [ ] Make settings and output paths less dependent on the current working directory.

## Runtime / server / operations

- [X] 052 — Hardened FastICA dimensions before broad runs: component counts are capped to the effective input rank and the smallest CV training fold instead of relying on sklearn auto-capping, sparse input is preflight-skipped, initialization is reproducible, and the iteration budget is raised from 200 to 1000 without changing FastICA's algorithm or tolerance. Runtime-verified on Wine: all FICA candidates used 13 components and the old `n_components is too large` warnings disappeared.
- [ ] FastICA convergence remains open after 052. The Wine verification removed all dimension warnings but still produced repeated `FastICA did not converge` warnings even with `max_iter=1000`, while several FICA candidates remained among the strongest models. Identify the exact preprocessing/fold patterns and evaluate a targeted retry/tolerance/algorithm strategy rather than globally disabling FastICA or merely increasing iterations again.
- [ ] Investigate intermittent widget-rendering failure after `Reclassification table for most mispredicted`, where the UI only shows `Error displaying widget` without an application crash or explanatory exception. Capture whether the failure originates in widget payload size/content, serialization, frontend rendering, or kernel/frontend state, and make the failure observable in logs.
- [ ] Investigate the Voilà `_xsrf` shutdown/reload 403 behavior.
- [ ] Review the local server/kernel communication setup; the current TCP transport has no encryption and should have an explicit trust/security model.
- [ ] Add possibility of using threads in `execute_n_jobs` when `PicklingError` occurs, but verify NaN handling and estimator thread-safety before enabling it.
- [ ] Investigate PyTorch estimator concurrency/checkpoint isolation. The broad Breast Cancer stress run produced JSON parse errors, `PytorchStreamReader` read failures, and shape mismatches consistent with parallel workers sharing temporary/checkpoint files. Give each fit/fold an isolated artifact path before considering broader PyTorch parallelism safe.
- [ ] Preflight-skip `SelfTrainingClassifier` when the training set contains no unlabeled samples; both the broad stress run and the 054 Breast Cancer verification repeatedly emitted `y contains no unlabeled samples`, so the current candidate adds work/noise without exercising semi-supervised behavior.
- [ ] Trace the remaining broad-run convergence warnings after 045. The 054 Breast Cancer verification confirms that some scikit-learn `MLPClassifier` spot-check instances still report the default `max_iter=500` despite the GUI/configured maximum being 20000; identify the instantiation path that is not receiving the configured value. Liblinear warnings also still occur in some SVC/stacked-SVC paths.
- [ ] Preflight feature-selection combinations that can select zero features; the broad stress run emitted `No features were selected` warnings and should skip or clearly classify those candidates before downstream fitting.
- [ ] Investigate the broad-run joblib warning `A worker stopped while some jobs were given to the executor`; determine whether it is timeout/resource pressure or an estimator-specific leak before changing worker settings.
- [ ] Revisit resource/file-handle warnings only if they reappear in current runtime logs.

## Regression suite – maintain, do not expand by default

Current coverage includes numeric binary/multiclass classification, 4/13/30-feature datasets, 15 broad model families, multiple preprocessors, NOR/PCA/RFE, Balanced Accuracy, Balanced F1 Macro, random over/undersampling, final GridSearch, Dark Numbers, and a targeted mixed text/category sparse-TF-IDF profile.

- [ ] Add new suite coverage only when a real bug needs a stable reproducer or a genuinely important untested branch is identified.
- [ ] Keep suite runtime bounded; prefer targeted profiles over multiplying the full model/preprocessor/reduction matrix.

## Solved / established

- [X] 054 — `execute_n_job` now preserves original exception types and tracebacks, adds concise parallel-execution context without generic wrapping, scales down workers only for the existing MemoryError/SystemError/PicklingError retry path, re-raises the original failure when one worker still fails, and correctly interprets negative global worker limits as unlimited.
- [X] 053 — Interpolation-based SMOTE-family samplers now receive float64 feature matrices before resampling, preserving fractional synthetic values and avoiding the integer-cast failure path seen with MLPC/GridSearchCV; sparse inputs remain sparse and non-interpolating samplers are unchanged.
- [X] 052 — FastICA now caps `n_components` against both input dimensions and the smallest CV training fold, skips sparse input before CV, uses deterministic initialization, and gets a larger iteration budget while preserving sklearn's default ICA algorithm/tolerance.
- [X] 051 — Correction-factor regression now excludes non-finite/zero-recovery observations and synthetic insufficient-sample fallbacks, requires enough valid points for extrapolation, reports corr provenance (`direct`/`regressed`/`fallback-unestimable`) in Dark Numbers output, and applies the same semantics in the validation harness without changing the published Dark Number formula.
- [X] 050 — Added a standalone Dark Number validation harness for controlled known-noise experiments, including three-estimate coverage/position, model-specific correction-factor stability, and probability-ranked enrichment versus random selection; production Dark Number calculations are unchanged.
- [X] 049 — Preserved the published linear Dark Number formula while separating CV-test/CV-full/retrained-full estimates and model-specific correction factors, made the 20% injected-label-noise level configurable, fixed alpha handling for FN-only misclassifications, retained Combined as a labelled legacy comparison, and added focused regression coverage.
- [X] 048 — Removed SQL passwords from generated Python configs and serialized model metadata, injects current credentials when loading saved models, supports `JBG_SQL_PASSWORD` for command-line runtime injection, documents pickle/dill trust requirements, and removed the duplicate `matplotlib` requirement.
- [X] 047 — Kept generated persistence-test `.sav` files out of version control: `tests/fixtures/*.sav` is now ignored, the fixture directory documents the local/trusted-artifact policy, and the regeneration helper creates the directory when needed. The final broad Breast Cancer run after 046 completed spot-check, GridSearch, evaluation, retraining, misprediction handling, and Dark Numbers without candidate exceptions; the older accumulated FastICA warning storm is explicitly deferred above.
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
