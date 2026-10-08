# Revision log – JBGAutoClassification

## 122 — Scale original training folds before numerical SMOTE

- Correct the pipeline order for numerical neighbor-based interpolation: imputation -> float conversion -> selected scaling -> oversampling -> undersampling -> reduction -> estimator. Apply this to SME/ADA/BRD/KMS/SVM. Scaling stays inside the fitted/CV pipeline and learns statistics from original fold rows, so numerical units affect neither SMOTE neighbors nor synthetic-row statistics through the previous unscaled path. No separate global fit is introduced. NOS remains a no-op; categorical/random/no-oversampling paths retain their existing behavior pending dedicated review. Sampling strategy, neighbors, estimator/grid/scoring defaults, enum values and persistence format are unchanged.
- Existing fitted artifacts remain untouched on loading; prediction and cloning/retraining retain their stored step order. Newly constructed training pipelines use 122. Record the 2026-10-06 Windows normal-path 121 check without claiming the empty-selection branch was triggered, and keep the convergence/timing-calibration observations open separately.
- Validation: 124 focused checks pass, comprising 27 new scaling/resampling checks, existing 121 guards, timing calibration, feature importance, model persistence/source layout and two existing sampler tests. Real SMOTE input/synthetic samples are invariant to a numerical-unit change; all five numerical sampler families fit (the KMeansSMOTE test uses an explicit test-only permissive cluster threshold); scalar statistics match original training rows in serial/process CV; NaN/imputation, float conversion, sparse integer input, all four NOS/STA/MAX/MIX choices, PCA/RFE/Tomek ordering, prediction without fitting/resampling, GridSearch/refit, fresh full-data retraining and fresh-process current/legacy artifact predictions are covered. Local verification omits unused SQL/TensorFlow/JAX imports; the container's missing worker-PID /proc view requires disabling loky's optional RSS monitor in the verification harness and interrupting process-pool interpreter cleanup after the passing test summaries. Application workers/retries are not modified. Windows/Voilà and real dataset performance remain pending.

## 121 — Guard empty SLSV feature selections inside training folds

- Reproduce the open zero-feature warning with nonconstant, low-amplitude input: SLSV's L1 SelectFromModel drops all columns, emits `No features were selected`, then its downstream LinearSVC raises a generic zero-feature error. Use a clone-compatible SelectFromModel subclass to check the fitted support mask before transformation/downstream fitting. A dedicated EmptyFeatureSelectionError gives the selector, input count, threshold and practical preprocessing/model/data guidance. The CV table/CSV records this expected failure as a concise `UNUSABLE` result; unsuccessful candidates cannot win and other candidates continue.
- Fit the selector within the existing CV training folds. Add no full-data selection probe or training fits, forced features, smoothing, threshold/grid changes or holdout access. Nonempty selection uses sklearn's original transform, including sparse output; ordinary fitting/retraining also receive the guard. Existing fitted models with ordinary sklearn SelectFromModel remain readable and retain their predictions; retraining an old artifact retains its stored selector type. Algorithm/enum values and the original JBGMeta.SelectFromModel import remain compatible.
- Validation: 89 focused feature-selection, completion-mail/task lifecycle, source-layout, model-persistence and GridSearch-calibration checks pass (7.03 s). The 17 new regression checks include real dense/sparse SLSV, unchanged support/threshold/predictions against sklearn, clone parameters, unfitted/prefitted paths, process-worker exception propagation, a full-data-valid but CV-fold-empty reproduction, exactly three fold-local selector fits, pickle/dill reload of current and historical selectors, actionable CV reporting/candidate continuation, unrelated-error preservation, GridSearch and ordinary/fresh-retraining guards. Production sklearn/application modules were exercised with verification-only adaptations for unavailable SQL Server and unused TensorFlow/JAX bootstrap imports. Syntax/whitespace checks, unchanged class/enum metadata, clean revision-120 patch application/reversal and exact changed-file comparisons also pass. SMTP delivery and Windows/Voilà runtime remain pending.

## 120 — Prediction-only completion-mail configuration

- Preserve the current runtime `Config.Mail` object when prediction-only startup merges a saved model configuration. Model artifacts intentionally keep runtime mail settings out of persisted metadata, so the previous merge silently replaced GUI/CLI notification settings with the sanitized defaults before the completion task ran. SQL/runtime and model semantics are otherwise unchanged.
- Treat an absent recipient, invalid recipient, or absent SMTP server as an explicit notification-disabled/misconfigured state. These cases now skip email without labelling the successful classification/prediction as an error, and a missing (`None`) environment value is handled safely. SMTP connection/send failures remain reported as actual mail errors.
- Added focused regression coverage for runtime-mail preservation and the missing/invalid recipient paths. Runtime verification should include a fresh-kernel prediction-only reload with a configured recipient, plus one run with notifications deliberately left blank.

## 119 — Flat source tree and repository-root logs

- Move every tracked application module/resource from `src/JBGclassification/` directly into `src/`. Update the root notebook's import, pytest source path, resource/test paths, runner commands, fixture instructions and benchmark documentation. Keep SQL/config/GUI resources, CSVs, timing history, model files and Keras sidecars at their corresponding locations inside the flattened `src/` tree; model/algorithm behavior and stored enum values are unchanged.
- Route default application, validation/sensitivity-runner and PowerShell server logs to `<repo-root>/logs/`, based on the source checkout rather than the current working directory. Preserve explicit Python log-directory overrides and update ignore rules. The launcher template invokes the dependency-free local migration before starting Voilà; customized launchers need their log destination updated separately.
- Provide `scripts/migrate_source_layout.py` with a no-write `--dry-run`, whole-plan collision checks, verified copies before original removal, identical-copy recovery and repeatable no-op completion. Git handles tracked files; the helper moves all residual local/ignored content, including configs, saved models/sidecars, output/checkpoint companions, timing JSON and historical/interim logs. Empty directories are retained, differing versions are never overwritten, and incomplete copies retain originals. Stop running Voilà/kernels before the structural move. Branch switches do not relocate ignored runtime files; use separate worktrees for parallel old/new-layout execution. Existing sensitivity code-version guards remain intact, so earlier-revision checkpoint resume is still rejected.
- Preserve historical qualified pickle/dill/joblib module globals using aliases to the canonical application modules, sharing class/function/enum identities without retaining the old directory. Normalize only saved/configured paths under the old source directory to the current checkout; external custom paths remain unchanged. Current `src.*` imports also use canonical objects rather than duplicate module definitions.
- Validation: 64 focused source-layout/migration/logging/model-persistence/calibration checks pass (5.41 s), including generated-config loading, collision-before-mutation, interrupted-copy recovery, fresh-process legacy-global pipeline prediction/retraining, canonical enum identity and unchanged GridSearch fit counts/results with cached reuse. Real scikit-learn and CPU Torch models created by the pre-119 snapshot with old qualified Config/Helpers/Meta/Transformers globals reload with identical predictions in a fresh process under 119, normalize saved paths, retrain, resave and reload. Local verification omits unused TensorFlow/JAX execution and SQL Server access, with compatibility adaptations for the legacy test mock annotation/OS username. The broader pre-existing widget/config suite has the same 4 failures and 17 fixture errors before/after the patch (outdated widget expectations, invalid legacy ngram fixture and an existing legacy-control check); these were not folded into the structural change. Notebook canonical-import and all three CLI help/import smoke checks pass using a placeholder environment file. Syntax/notebook-JSON/ignore/whitespace checks, clean-baseline patch application/reversal and 99 tracked-file byte comparisons pass; migration on that applied snapshot preserves all seeded local artifacts and removes the old directory. Windows/Voilà migration, customized-launcher log routing and saved-model prediction remain pending.

## 118 — Shared GridSearch forecast calibration and measured cost diagnostics

- Recent Windows searches show opposing forecast errors: the Breast Cancer/TOTA searches took about 42 s versus estimates of 9–10 s, while the larger Återkrav/SMOTE/TOSS search took 101.14 s at 63.4% of its estimate. Add learned correction only for comparable completed searches across all model families using the common GridSearch path. Keep the original selected-CV formula as the uncalibrated baseline; do not add training probes, change the search grid/concurrency/scoring, or apply a universal PyTorch multiplier.
- Persist a bounded numeric JSON history at output/grid_search_timing.json under the application script directory. Retain up to five observations per profile from the last 30 days and at most 64 profiles. Match hashed data-source/table/target, training shape/input container/schema/class counts, cloned unfitted pipeline and complete search parameters, scorer, fold count, resolved search and observed CV workers, coarse factor-of-two CV fold-work/dispatch-cost bands, host, Python/framework versions and thread environment. The timing bands avoid carrying a startup-dominated cold-CV ratio into a materially different warm baseline. This is comparable workload metadata, not a fingerprint of dataset contents; changed data values, randomization, startup and contention can still alter costs.
- Multiply the current baseline by the median actual/base ratio of matching observations. Log the uncalibrated/calibrated basis, sample count and descriptive factor range, retaining the existing unrounded actual/estimate percentage comparison. Always record successful observations against the raw baseline, avoiding repeated correction feedback. First/unmatched/expired/unsupported histories use the original estimate. Missing telemetry, corrupt or inaccessible history cannot discard a trained model; failed searches and ordinary-fit fallback never teach calibration. Use atomic JSON replacement, exclude runtime history/temp files from git, and persist no rows, labels, credentials or fitted state.
- Add completion diagnostics from GridSearchCV's existing candidate mean fit/scoring times and measured refit_time_. Report the remaining CV/dispatch phase; do not claim that its startup, scheduling and contention components have been isolated. These diagnostics support later calibration review without extra fits or model changes.
- Close the pending 116/117 Windows verification: the 13:57 Breast Cancer run improved scaled TOSS results, and the 14:17 Återkrav run used the exact new TOSS grid, selected rate 0.02, and completed search/refit and Dark Numbers. Its holdout Ja precision was only 4.31% despite 83.93% recall, so classification quality remains a separate concern. After a new kernel started at 14:41:14, the 14:45 prediction-only run loaded model TOSS, predicted 1000 unknown rows (443 Ja/557 Nej), and inserted 1000 prediction rows. Completion-mail failure and the pre-restart shutdown 403 remain separate backlog items; CUDA/resource profiling remains open.
- Validation: 115 focused timing/calibration/estimator-input checks pass (9.63 s) using production modules/method bodies with unrelated SQL/text/TensorFlow bootstrap imports bypassed. Real sklearn searches confirm exactly six CV fits plus one refit per run, unchanged predictions/scores, and learned second-run reporting. Checks cover upward/downward/median calibration, immutable raw denominators, scope/timing-regime separation, bounded/expired/corrupt history, unsupported objects, failed-search exclusion, read/write failure isolation and measured-refit diagnostics. Fresh-process CPU Torch repeat searches reuse persisted matching observations with real parallel fitting; sklearn checks exercise matched reuse and safe misses when timing bands change. Syntax/whitespace and clean-revision-117 patch application/reversal/byte checks pass. Windows repeat-run forecast accuracy remains pending; calibration is an approximation, not a confidence guarantee.

## 117 — Evidence-supported Sigmoid+SGD profile

- Runtime-verified 116's Windows/Voilà training path on the 2026-10-05 13:07 Breast Cancer run: all 36 PyTorch/scaler/reduction candidates completed, STA/PCA/TOTA won, its six-combination/ten-fold final search and refit completed, and within-run saved-model loading, full-data retraining and final Dark Number exports succeeded without application warnings/errors. Holdout accuracy was 0.95614 and malignant-class recall 0.92857. Fresh-kernel prediction-only reload remains pending; the logs do not identify/verify CUDA behavior. Record the short GridSearch forecast discrepancy (about 9 s estimated, 41.94 s actual, 470.2% of estimate) without applying an unsupported blanket multiplier.
- Review weak scaled TOSS performance using a paired real CPU depth/rate study on the same dataset family. Reserve 20% before fitting and leave it unused, fit preprocessing within three fixed outer training folds, and repeat five profiles under three initialization seeds for STA/PCA and STA/NOR (90 fits). The shallow SGD/0.1 profile improves AUC over deep SGD/0.02 in all 18 matched fold/seed/reduction comparisons: mean AUC 0.52150 → 0.98941 with PCA and 0.57929 → 0.98399 without reduction. Malignant-class recall improves from zero to 0.89028/0.81064. Raising the rate alone leaves malignant recall at zero; reducing depth alone improves AUC but leaves weak NOR recall. Record validation-loss diagnostics and the dataset/CPU limitations in docs/verification/117_pytorch_sigmoid_sgd.md, with an optional reproducible benchmark outside the automatic regression suite.
- Change only the operational TOSS profile to one hidden stage (zero additional stages), initial SGD rate 0.1, search rates 0.02/0.05/0.1 and widths 48/100. Retain six search combinations, 50 epochs, dropout/internal-validation policy, all six identities and the other five profiles. Keep the base constructor's depth/defaults and existing fitted states unchanged. Preserve every stored enum value/order, including revision-116 TOSS, by supplying the current TOSS grid through its runtime parameters getter; saved enum-by-value artifacts still resolve. Retrain to obtain the new profile.
- Validation: 57 focused real CPU Torch/skorch, model-persistence and fallback-execution checks pass (21.27 s), including new factory/grid consistency and actual shallow architecture, revision-116 TOSS enum compatibility with pickle/dill, and fresh-process shallow-model artifact reload after checkpoint cleanup. Syntax/whitespace, stored-enum/helper source comparison and clean-revision-116 patch application/reversal/byte checks pass. The local harness bypasses unrelated SQL/text/TensorFlow bootstrap imports while using production bodies and real fitting. Windows verification of 117, an independent Windows fresh-kernel prediction reload and CUDA profiling remain pending.

## 116 — PyTorch fit isolation, real optimizer variants and bounded searches

- Code review confirms all PyTorch fits wrote `params.pt`/optimizer/history to the same `output/nn_checkpoints` directory. Allocate a unique temporary child directory for every fit and restore its best checkpoint before cleanup; failures clean only their own temporary files and cannot remove unrelated historical files. The older stress-run errors are consistent with this collision, but their individual causes are not all proven by source review alone.
- Fix ignored optimizer selection: the unused module-local optimizer never trained the network, while NeuralNetClassifier received no optimizer and defaulted to SGD. Pass Adam/SGD and the configured learning rate to skorch, and remove the redundant module-local optimizer. Fix prediction-time dropout by using the module's training/evaluation state. Preserve Softmax + explicit NLLLoss, class encoding, and the historical additional-hidden-layer architecture semantics.
- Add ClassifierMixin, return self from fit, validate dense float32 inputs/feature counts and fitted state, and discard failed fit state. Replace the lambda checkpoint monitor requiring simultaneous train/validation improvements with the standard validation-best monitor, or training-best when internal validation is disabled. Preserve checkpoint restoration without adding actual early stopping or changing the experimental framework fallback's sequential policy.
- Keep all six named variants and fix their activation/optimizer through final tuning. Replace the operational 1944-combination shared search with six combinations per variant: Adam rates 0.0003/0.001/0.003 or SGD 0.01/0.02/0.05, widths 48/100, fixed 50 epochs/dropout 0.1/two additional hidden stages/internal validation. Initial learning rates are 0.001/0.02; the default epoch budget changes from 20 to 50 independently of the GUI iteration setting. Batch size remains the skorch default 128, now explicit. These are bounded starting profiles rather than universal optimization claims. Preserve every existing enum value/order and the unused historical PYNN member; new grid members are appended and selected by the runtime search_params property so old enum-by-value artifacts still resolve.
- Local PyTorch 2.14.1+cpu/skorch 1.4.0/dill 0.4.1 testing also reproduced a fitted-optimizer dill failure involving typing_extensions.sentinel, while standard pickle succeeded. Encode only the skorch component with standard pickle inside the wrapper's existing dill artifact, retain legacy plain-net state reading, and verify fresh-process model reload after checkpoint cleanup. No global serializer or model-envelope change is made.
- Validation: 54 focused real CPU Torch/skorch, model-persistence and fallback-execution checks pass (19.08 s). They exercise all six variants/optimizers, stochastic training versus deterministic inference, private checkpoint success/failure/refit handling, parallel heterogeneous feature/class shapes, parallel ROC-AUC CV and grid search, standard-pickle/dill roundtrips, legacy state/enum resolution and fresh-process artifact reload. Source comparison also confirms every original Algorithm/Grid enum value and order is unchanged. Syntax/whitespace and clean-revision-115 patch application/reversal/byte checks pass. The isolated local harness loads production class/enum bodies while bypassing unrelated SQL/text/TensorFlow bootstrap dependencies. Windows/Voilà end-to-end and CUDA verification remain pending; GPU resource policies and explicit epoch/device/seed controls remain in BACKLOG.

## 115 — Actionable low-rediscovery warning

- Keep the existing direct-correction warning trigger unchanged: finite positive mean recovery below 5% (corresponding to corr >20). The revised printed message reports model/target, rediscovery rate and correction factor, notes possible statistical instability, and explicitly recommends considering another pipeline combination and investigating the correctness and quality of the training dataset. It prompts investigation rather than declaring the dataset flawed.
- Retain direct correction values, provenance, Dark Number formulas, interval/order reporting, the sensitivity runner's warning policy and the experimental fallback policy. No threshold change, clamp, rejection, extra estimation or automatic fallback is introduced.
- Runtime-verified 114 in the 2026-10-05 11:47 Windows/Voilà run: both ten-fold STA/NYS candidates used ten workers and observed two native threads, with FUTV/MLPC CV wall times 40.68/11.84 s and no retained MLP iteration-cap hits. The pipeline finished through final Dark Number exports. This closes the worker issue; 4580 rows/201 features versus the prior 14407/321 prevents attributing a paired speedup to 114. Direct corrections 96 and 34.2857 had recovery 1.04% and 2.92%, so both would emit the revised warning. Experimental fallback was not activated and its live meter/performance verification remains pending.
- Validation: 93 focused checks pass using production handler method bodies and runner/calculator/checkpoint modules in an isolated harness that bypasses unrelated SQL/GUI bootstrap imports. They cover warning text, the unchanged strict 5% boundary, missing/non-finite recovery, unchanged direct factors/provenance and no warning-driven fallback. Syntax/whitespace and clean-revision-114 patch application/reversal/byte checks pass. Windows/Voilà verification of the revised warning text remains pending.

## 114 — Bound spot-check CV workers by the number of folds

- Runtime-verified 113 on the 2026-10-05 MAX+STA/NYS/MLPC+FUTV run: 40 successful fold measurements and the runtime CSV download were generated; final training, feature importance, retraining and all four Dark Number scopes/exports completed. Nystroem fit/transform averaged 0.34–0.55 s and represented only 0.04–0.44% of mean fold fit time. All variants used about 20338 post-SMOTE rows/321 dense features (~49.8 MiB), native threads=1, and retained MLP iterations 162–605 against a 20000 cap. Other fit work includes sampling/preprocessing/classifier fitting and is not MLP-only.
- STA/NYS/FUTV used 298.79 s CV wall time versus MAX's 1660.11 s; STA/NYS/MLPC used 169.22 s versus MAX's 1964.47 s. STA had higher mean CV scores too (0.982560/0.982647 versus MAX 0.977266/0.958961). Record STA-only NYS/MLP as a narrower profile to try on this dataset, retaining MAX as a user-selectable option. No global candidate removal, scaling/kernel change or iteration-budget reduction is introduced; stochastic seeds and kernel differences prevent a general paired causal proof.
- Fixed the observed `Workers=24` for ten-fold CV. `min(fold_count, STANDARD_DESIRED_N_JOBS)` turned the default global `-1` into an all-CPU request, overriding the intended fold task bound. Request the positive fold count instead; the unchanged executor applies CPU and positive global caps and retains resource/pickling worker reductions. Successful CV timing and runtime diagnostics consequently report the bounded/reduced pool. Ten tasks cannot use 24 workers concurrently. Configured None/zero/negative settings now follow the executor's existing no-positive-cap convention consistently.
- GridSearch and other executor callers retain their separate policies. Smaller CV pools may change joblib's automatic native-thread allowance; the runtime CSV measures the observed value and no additional thread cap is forced. Do not claim a measured real-data speedup until a rerun verifies it. Model parameters, folds, sampling and model-selection scores/logic are unchanged; stochastic and numerical execution differences remain possible with changed parallelism.
- Validation: 109 focused checks pass in the isolated production-method harness, including real sklearn/SciPy/joblib score equivalence, serial/process NYS profiling, and nine new checks for negative/None/zero/positive caps, CPU limits, fold counts, bounded resource retry provenance and unchanged non-CV all-CPU behavior. Syntax/whitespace and clean-revision-113 application/reverse checks verified. Windows/Voilà runtime verification of the new CV pool is pending.

## 113 — Measure expensive Nystroem/MLP candidate fits before changing limits

- Reviewed the attached cross-validation CSVs: in the 14407-row run on 2026-10-01, MAX/NYS/FUTV used 1654.62 s and MAX/NYS/MLPC 1368.49 s, versus STA/NYS times 221.31 s and 164.19 s. Both classifiers contain an MLP. The later 4580-row runs had different bottlenecks, with MAX/NYS/FUTS taking 431.99 s and 236.02 s. These are total candidate measurements from different/stochastic runs, not proof of which pipeline stage or scaling policy caused the cost.
- Added `JBGNystroemRuntime.py`: instrument only disposable NYS/MLPC, MLP2, FUTV and FUTS CV clones. Time Nystroem fit/transform and scoring transforms; collect post-sampling dimensions/output-buffer bytes, retained fitted MLP `n_iter_`/configured budgets/cap hits, and native numerical thread counts. The existing scorer runs once and returns its original score together with scalar diagnostics through sklearn's callable multi-metric CV interface. No fitted fold estimators are returned to the parent process.
- Log one compact summary per completed relevant candidate and export raw fold measurements alongside the normal CV CSV as `<crossval stem>_nystroem_runtime.csv`. Preserve original-pipeline provenance for selected-candidate GridSearch timing and leave the normal CV table/CSV schema unchanged. Distinguish wall time from concurrent fold work and identify remaining fit work as sampling/preprocessing/classifier work, not MLP-only. Stacking's discarded inner-fold iterations cannot be inspected; cap hits are not asserted to prove non-convergence.
- Diagnostics preparation/collection/export failures do not trigger extra training, change successful scores or abort a successful run. Reset transient rows per spot-check so Repeat Last cannot export stale measurements. Custom NYS subclasses, non-default output containers and cached pipelines retain their original uninstrumented path. Original candidates, parameters, component/iteration limits, CV/sampling, scoring/winner policy, worker settings, final searches and saved Nystroem types remain unchanged. This is the measurement step; runtime optimization remains open pending real-data evidence.
- Record 112's successful Windows/Voilà verification: the 2026-10-02 12:25 MAX/NOR/HIST run used 14407 labelled rows/321 converted features, logged guarded Dark Number predictions and dense-input reuse, produced both direct corrections (12.1736 and 6.22222), all four scopes and final exports, and emitted the ordered range `[0.00475303, 0.0150731, 0.0211068, 0.0352375]`. The normal experimental fallback was not activated, so its live progress/speed verification remains separate.
- Validation: 100 focused checks pass using real sklearn/SciPy/joblib and production handler/timing/input method bodies in an isolated bootstrap harness. Tests cover exact score equivalence for all four profile algorithms with serial/dense and process/sparse CV, unchanged parameters, no extra MLP fits/scorer calls, original timing provenance, plain-Nystroem persistence roundtrip, unrelated CV/scorer failures, optional diagnostics/export failure, scope exclusions and per-run reset. Full optional application bootstrap/SMOTE and Windows/Voilà profiling are pending; no real-data speedup is claimed. Syntax/whitespace and clean-revision-112 patch application verified.

## 112 — Carry confirmed dense input through Dark Number calculations

- The 2026-10-02 mixed-data run confirms 111's GaussianNB skips and guarded dense retries during CV, GridSearch, evaluation, retraining and misprediction analysis. The winning MAX/NOR/HistGradientBoosting pipeline then failed at 09:39:27, before Dark Number correction estimation: scope caching still called `predict`/`predict_proba` directly on sparse features. This was an omitted production path in 111.
- Extend the same narrow sklearn rejection guard to Dark Number scope predictions. Retain the successful input for the accompanying probability call and remember the confirmed dense requirement separately for each model during this calculation. Later scopes reuse that requirement with a logged shape/dtype/buffer estimate before allocation; sparse-capable models retain sparse input.
- Convert each model's correction input once when its dense requirement was confirmed. The cross-trained model still uses the CV-training partition and the retrained model uses full data. Direct correction fits, sample-size regression and experimental perturbed clones all receive this compatible input. Zero-FP targets still bypass correction work and correction-input allocation. No formula, correction ownership, label perturbation, estimator setting, split or interval ordering is changed.
- Validation: reproduced the exact fatal sparse-X error with revision 111's original scope method and a real MaxAbsScaler/HistGradientBoosting pipeline. 133 focused checks pass (55 Dark Number checks and 78 existing estimator/timing checks), using production method bodies and real sklearn/SciPy/joblib in isolated bootstrap harnesses. Sparse and dense runs match all four scopes, all five formulas, confusion matrices, correction factors and fallback metadata apart from elapsed time; the real fixture exercises direct correction and perturbed-clone fallback. Additional checks force the regression/perturbed branches to verify input forwarding, preserve the zero-FP shortcut and model-specific sparse behavior, reject unrelated TypeErrors, and stop after an allocation failure. Full application/SMOTE/Windows/Voilà execution remains to be verified in the user's environment.

## 111 — Confirm sparse compatibility failures before retrying dense input

- Preflight-skip GaussianNB on sparse-preserving NOR/RFE paths before scaler/probe/CV work, matching the existing LDA policy. Dense GNB and pipelines with PCA/TSVD/Nystroem remain eligible. The two sparse-to-dense warnings in the completed 14407-row run match MAX/NOR/GNB and STA/NOR/GNB; their successful candidate timings sum to roughly 34 s. The long Nystroem/FUTV/MLP candidate costs are tracked separately in BACKLOG.
- Added `JBGEstimatorInput.py` with a narrow sparse-X rejection guard: require SciPy sparse prepared input, the specific sklearn TypeError text, and sklearn's `_ensure_sparse_format` validation frame locally or in a joblib `_RemoteTraceback`. Support current/older rejection wording and POSIX/Windows worker traceback paths. Reject unrelated TypeErrors, matching messages without validation origin, sparse-y rejection and errors raised on already dense feature input.
- Retry only once with `toarray()` on the actual prepared SciPy input, retaining feature order/dtype. Before allocating, log operation context, pipeline name, input type/shape/dtype and dense-buffer bytes/MiB; identify additional worker/estimator memory as separate. Preserve retry failure type/traceback and attach context where supported. A failed dense allocation or fit is not retried again.
- Replace the generic CV TypeError -> NumPy -> arbitrary-failure serial-CV sequence with the confirmed sparse retry. Existing `execute_n_job` resource/pickling reductions and worker policy remain unchanged. CV timing records the successful worker count and the complete call's elapsed time, including any failed sparse attempt; failure clears stale timing and never emits a success comparison.
- Apply the shared policy to normal/fresh-retrain fitting, GridSearch fitting, validation fitting/scoring/AUC probabilities, prediction-only calls and active misprediction/probability analysis. Preserve optional-probability fallback behavior. Each model's misprediction call now retries independently, so a second-model failure does not repeat a successful first-model prediction. Calculations, defaults, estimator hyperparameters, sparse-capable paths, winner ranking and Dark Number formulas/display are unchanged. Unconfirmed DataFrame-related TypeErrors now surface instead of automatically changing representation; the three existing fallback fixtures were updated to assert this stricter contract. The unused historical `most_mispredicted_old` implementation is not invoked by the production task path and is outside this change.
- Validation: 78 focused checks passed in an isolated harness using the real shared module, production handler/helper/timing method bodies and real sklearn/SciPy/joblib. Enum identifiers and optional bootstrap surfaces were lightweight stand-ins. Covers genuine local/process sparse rejection, Windows traceback parsing, spoofed/unrelated/wrong-input errors, allocation/retry failure, GNB preflight and dense reduction viability, exact CV-score equivalence, fit/GridSearch/scoring compatibility, legacy dense-GNB prediction, optional-probability behavior, independent model failure, changed original contracts, selected timing provenance, resource worker retries and estimate/actual completion summaries. Syntax/whitespace and clean-revision-110 application verified. Full application bootstrap remains unavailable locally; actual Windows/Voilà behavior and optional framework compatibility are pending.

## 110 — Display the four-point Dark Number range below the calculation table

- Implemented the presentation item recorded in 109. Immediately after the Dark Number Calculations table, show `[D_re_full, D_cv_full, D_comb, D_cv_test] = [values]` using mathematical subscripts/serif type and a target/formula label. Values use the existing table proportions with six significant digits. Native HTML requires no new widgets or MathJax dependency; both logger paths display the output with verbose off and retain readable plain-text output/logging.
- Build one range per target/formula from the existing result rows, including blank continuation model labels and duplicate row indices. Retain the requested scope order and compare original unrounded values, allowing ties and explicitly identifying adjacent inversions. Missing/non-finite/ambiguous estimates or unresolved correction provenance produce an incomplete check; `fallback/unestimable` is displayed as unestimable instead of presenting its neutral numeric placeholder as an estimate. Valid `not_needed_zero_fp` zero results remain visible with their provenance. Additional model scopes do not replace any of the requested four points.
- No new prediction, correction fitting, recalculation, sorting, model/threshold decision or CSV change. Escape target/formula labels for notebook HTML and identify the range as heuristic. Implemented shared reporting functions in `JBGDarkNumberReporting.py` and inserted the display before existing download links.
- Reviewed the completed 14407-row Återkrav/SMOTE/ROC-AUC run: MAX/NOR/FUTV won from 84 candidates (mean CV AUC 0.918128); holdout AUC 0.928887, accuracy 0.981957, majority baseline 0.980569, balanced accuracy 0.710747 and MCC 0.474503. GridSearch/refit took 410.12 s, 98.0% of estimate (-2.0%). Both factors were direct (cross-trained 5.6619007569, retrained 5.04), so this run does not exercise the 107 fallback meter or 108 zero-FP skip. The reported four-point sequence is [0.0029057252, 0.0050656710, 0.0090620397, 0.0290838200], correctly ordered; all FP counts are nonzero (holdout 20, cross-full 26, retrained-full 29, combined 49).
- Recorded the long sparse-to-dense retry warnings and the `No such comm` event at the same time as a server kernel-connection restoration in the existing open BACKLOG issues. The run continued to final exports. These are follow-up observations, not fixes or conclusive attribution in this patch.
- Validation: 22 focused checks passed using the real reporting module and production handler/logger method bodies in an isolated harness. Covers submitted-run values, all formula/target groups, blank model labels, fixed scope order, rounding-hidden inversions, ties, valid zero-FP results, missing/ambiguous/non-finite/unestimable/provenance cases, HTML escaping, additional models, exact display placement and unchanged CSV inputs, quiet notebook output in both loggers and terminal/plain logging. Syntax/whitespace and clean-revision-109 patch application checked; static HTML preview visually reviewed. Full application bootstrap remains unavailable locally and actual Windows/Voilà rendering is pending.

## 109 — Track final Dark Number range and ordering display

- Added a BACKLOG presentation item for a labelled four-estimate range directly below the Dark Number Calculations table in the requested order `[D_re_full, D_cv_full, D_comb, D_cv_test]`, per target/formula. Map the shortened names to the existing retrained-full and combined legacy scopes.
- Planned order checks allow ties, compare unrounded values, explicitly identify adjacent inversions without sorting them away, and handle missing/non-finite/unestimable estimates without presenting a completed order check. The display should distinguish valid `not_needed_zero_fp` zeros from unresolved correction placeholders and identify the range as heuristic rather than a confidence interval.
- Documentation only. No GUI, formula, prediction or correction changes. Whitespace and clean-revision-108 patch application verified.

## 108 — Skip corrections that cannot affect zero-FP Dark Number outputs

- Cache the existing model/scope predictions and confidence values once before correction planning. Estimate a model/target correction only when it can affect at least one reported result; inspect every dependent CV-test/CV-full/retrained/additional-model scope and the combined legacy report under its existing correction ownership. Do not infer that the cross-trained factor is unused from CV-full alone: holdout or combined false positives can still require it.
- Added a conservative zero-FP dependency guard for all five formula variants and `all`. Require exactly zero observed FP, both target and rest support, aligned complete labels and valid finite confidence values for alpha variants. No epsilon/rounding threshold, count smoothing, changed classification threshold, new model fitting or altered formula is introduced. Prediction-only classifiers still produce only their confusion matrices.
- Skip the entire direct/regression/experimental correction lifecycle for unused model/target pairs. Preserve matrix ordering/content, alpha calculations, the existing combined predictions and all zero Dark Number rows. Explicitly report `corr_source=not_needed_zero_fp`; numeric `corr=1.0` is a neutral placeholder rather than an estimate. Log each affected scope's FP/FN counts and warn if the published formula gives zero despite observed false negatives. Standalone correction/sensitivity/validation experiments retain their existing estimation purpose.
- Recorded the complete 106 mixed Återkrav run: the four target-Ja matrices contain zero FP and FN counts 7/12/6/15. The independently decoded misprediction export contains 15 Ja-to-Nej errors and no Nej-to-Ja errors. The cross-trained fallback took 29 min 19 s, accepted 5/5 factors [8, 5.052632, 5.189189, 6.4, 6.193548] with median 6.193548 and CV 0.171801; retrained direct estimation gave corr 120 from 0.83% mean recovery. Both correction estimates multiplied a zero FP term, so all four exported Dark Numbers were zero. Source/CSV provenance is consistent. This is completed pre-107 evidence, not verification of 107's speed or live meter.
- Validation: 46 focused checks passed in an isolated harness executing the real source calculator/helper/handler method bodies. Coverage includes exact reproduction of all four submitted matrices with no correction fits; all formulas; a single FP; holdout/combined correction dependencies; combined ownership when the first model has no probabilities; selective multiclass targets; sparse prediction input; invalid/undefined scope guards; source placeholders/warnings; existing target/model-specific behavior; and exact nonzero-output comparisons with the actual unmodified 107 handler across all formula variants and combined on/off. Full application bootstrap remains unavailable locally; ordinary Windows/Voilà verification is pending. Updated two existing scope fixtures to contain actual FP so their selected-target/model-specific correction assertions continue exercising required estimates.

## 107 — Progress and bounded parallel execution for experimental Dark Number fallback

- The five-shadow-clone fallback previously forced all fold/repeat fits through `n_jobs=1`. Normal production now passes the configured worker policy, resolving CPU/config/fit caps to at most eight independent process fits. Clones remain sequential, while their folds/repeats run concurrently. One native numerical thread per fit and capped explicit nested `n_jobs` avoid oversubscription; large arrays may be memmapped. Framework/checkpoint estimators and older joblib without streaming retain sequential execution with fit-level progress. The standalone sensitivity caller retains its explicit sequential default.
- Added a parent-kernel inline `Shadow fits` meter, upfront clone/fit/worker work summary, per-clone completion/status/duration and a final completed/skipped/elapsed summary. No widget/logger is sent to fit workers. Streamed results advance the meter as fits return while statistical reduction preserves submission order. A failed clone accounts for skipped work explicitly; 100% describes work accounting rather than statistical acceptance. Interrupted execution leaves progress incomplete.
- Resource retries reduce workers and retain already returned results; serialization/backend failures fall back to sequential unfinished fits without concurrently sharing estimator training state. Reduced worker counts carry into later clones. Added planned/completed/skipped fits, initial/final worker counts and elapsed seconds to existing aggregate experimental-fallback telemetry. Direct correction, resource-regression precedence, injected-noise fraction, bootstrap/flip seeds, all five clones and the minimum-three/CV <= 0.50 acceptance rules remain unchanged. No early statistical stopping, model simplification or new training data is introduced.
- Recorded 106 runtime verification: numeric Breast Cancer/RFE/LRN (seven repeats, 211 scores, 1.04 s, MCC 0.908547) and mixed text/category Återkrav/MAX/TSVD/FUTV (47 source inputs / 202 converted features, five repeats, 236 scores, 29.52 s, ROC AUC 0.874802) produced both feature-importance CSVs. Final search comparisons reported +7.8%, +12.0% and -18.7% on the three submitted ordinary runs. The latest Återkrav log ends at the cross-trained target-Ja `zero_recovery` fallback start; no eventual fallback outcome or final exports can be inferred from that partial log.
- Validation: 39 focused checks passed in an isolated harness using real sklearn/joblib correction/execution modules and the original production handler method bodies. Tests cover serial/process equivalence, exact actual unmodified-106 correction results, sparse TSVD/voting pipelines, acceptance guards, seed/bootstrap preservation, CPU/config caps, parent-only callbacks, ordered reductions, retained results after retry, skipped work, interruption and direct/fallback/regression precedence. Full application collection remains blocked locally by optional/bootstrap dependencies; Windows/Voilà live meter and actual heavy-fallback speed remain pending.

## 106 — Timing comparison and optional held-out feature importance

- Final GridSearch now reports approximate estimated duration alongside measured search-fit/refit duration, actual/estimate percentage and signed deviation. Percentages use unrounded seconds and the actual interval uses a monotonic clock. Missing/invalid estimates are explicitly unavailable; failed searches and ordinary-fit fallback cannot produce a successful GridSearch percentage.
- Added an optional `Feature analysis` section with `Feature importance` (default off) and `Repeats` (default 5, range 2–30). Original input columns are preserved only when requested, then permuted across all final holdout rows using the selected scorer and the already fitted converter/pipeline. Whole original columns keep meaningful names for numeric/category/text inputs and continue through PCA/TSVD/RFE and the production classifier. Analysis follows final evaluation and precedes fresh full-data retraining; model selection, saved fitted models, holdout predictions, Dark Number formulas/defaults and ordinary disabled-mode training remain intact.
- The top 20 inputs show mean score decrease and repeat standard deviation. Two CSVs retain all input ranks, raw repeat scores and method/model/scorer/baseline/row/class-count/seed/timing settings. Negative importance is retained; repeat standard deviation is described as shuffle variation, not a confidence interval. The report measures this fitted model's reliance, is not causal, and can understate correlated inputs. It never fits an auxiliary model or removes features automatically. Progress tracks the known number of score evaluations, sequential execution preserves sparse conversion, and optional analysis failures warn without stopping retraining. The original-column snapshot is released after analysis.
- Persisted the flag/repeat count through generated Python configs, saved-model normalization, GUI model restoration and Repeat Last. Older configs/artifacts/local GUI settings default off/five repeats; old widget layouts gain the new section without duplicates. Controls follow training/enable state even when ordinary restoration observers are locked. Prediction-only runs and regression suites omit the analysis.
- The generated-config round-trip exposed three existing parser/template defects: unquoted NgramRange names, an omitted mail section required by the loader, and positional Mail/Debug arguments in the wrong order. Generated names are now quoted, the loader accepts enum or string NgramRange values and omitted-mail defaults while preserving explicit mail settings, and Config construction uses named fields. Existing files with bare enum names need regeneration/manual quoting; source files that cannot import cannot be repaired by a loader.
- Validation: 72 focused checks passed in an isolated harness executing real source timing/handler/task/config/converter/widget-control methods with real sklearn CV/GridSearch/permutation importance and joblib; optional bootstrap enums and the widget observer surface were lightweight stand-ins. Coverage includes known signal/constant rankings, repeat statistics, deterministic results, no model/converter refit or input/model mutation, original names with encrypted TF-IDF plus categorical conversion and sparse prediction, decision-function scoring, full holdout capture/alignment, task ordering/opt-out/suite behavior, CSV exports, optional failures, config/template/legacy-artifact/Repeat Last persistence, widget migration/locked state, and unrounded/missing/failed-search timing comparisons. Full application collection is blocked locally by missing IPython/bootstrap dependencies. Real Voilà/Windows/SQL integration, actual widget rendering and sampling/Keras pipelines remain pending; syntax, JSON, whitespace and clean-revision-105 patch application checks passed.

## 105 — Track estimate-versus-actual timing summary

- Added a small BACKLOG presentation item to show final GridSearch/refit estimated and actual duration together with actual/estimate percentage and a signed deviation after completion. The planned comparison uses unrounded durations and matching measurement scopes, with explicit handling of unavailable estimates and failed/fallback searches.
- Recorded 104's real Windows/Voilà verification on the weekly-2025-41 Återkrav run: the estimate was displayed before FUTV GridSearch (40 CV fits plus one refit), forecast ~14 min versus 589.96 s actual final training, and the run completed evaluation, retraining and direct target-Ja Dark Number exports. The estimate conservatively used 10 workers while the search requested 24; one successful forecast/reporting run is not treated as broad calibration evidence.
- Documentation only: no timing calculations, progress widgets, search/training behavior or model selection changes. Validated whitespace and clean-revision-104 patch application.

## 104 — Estimate final GridSearch wall-clock time

- Added an approximate time budget before final GridSearchCV fit, visible in the existing progress label and application log even with verbose off. The output includes parameter combinations, folds, CV fit count, one final refit and explicit timing/concurrency assumptions; short searches retain seconds while longer searches use minutes/hours (for example `~2 h 35 min`).
- Spot-check CV now uses sklearn `cross_validate` to collect fit/scoring durations while returning the same `test_score` array to existing selection/reporting code. Measurements follow the exact winning pipeline/RFE round, are reset for new searches, and stay transient on the handler/state rather than being attached to or saved with the model/config. CV splits, selected scorer, fit parameters, error handling, worker policy, final grid, holdout isolation and refit behavior are preserved.
- The budget uses concurrent batches of measured fold fit+scoring work, one observed startup/dispatch allowance and a sequential refit scaled by the approximate row ratio k/(k-1). Search concurrency uses resolved CPU/config caps and is conservatively bounded by actual CV concurrency, including reduced workers after retries. No extra benchmark fit or optimistic speedup beyond observed concurrency is introduced. Missing/nonfinite telemetry logs an unavailable reason and fit count without preventing GridSearch. Parameter-dependent work, nonlinear refit cost, contention and retries remain limitations rather than promises of an exact finish time.
- Recorded 103's real Breast Cancer/MCC evaluation verification: MAX/PCA/LSVC, mean CV MCC 0.958519, holdout accuracy 0.973684, majority baseline 0.631579, balanced accuracy 0.964286 and MCC 0.944155. Dark Numbers were correctly skipped for the non-probabilistic LinearSVC; live F1 tooltip/Repeat Last and the creditcard scoring comparison remain pending.
- Validation: 37 focused checks passed in an isolated harness executing the real timing/handler/helper method bodies with real sklearn CV/GridSearch and joblib. Coverage includes unchanged CV scores on dense/sparse input and serial/parallel execution, winner-timing provenance, worker retries, serial fallback, reduced folds, partial batches, refit/overhead accounting, duration formatting, invalid telemetry, and pre-fit logging with verbose off (also training without telemetry). Application-wide collection remains blocked by missing optional/bootstrap dependencies, first `IPython`; actual Voilà/Windows ETA verification remains pending. Syntax, whitespace and clean-revision-103 patch application checks passed.

## 103 — Explain scoring and expose imbalanced evaluation performance

- Corrected F1 display labels to `F1 Micro`, `F1 Macro` and `F1 Weighted`. Existing enum names/value dictionaries, sklearn callables/kwargs, defaults and saved config/model compatibility are retained because enum pickles resolve by value. Micro-F1 still uses exactly the same scorer; no selection objective is silently changed.
- Added a dynamic scoring tooltip explaining candidate/GridSearch ranking and the selected metric's averaging. It states that micro-F1 equals accuracy for single-label classification over all classes, macro-F1 weights each class equally, and weighted-F1 weights by class support. It updates on both manual selection and programmatic/locked settings restoration.
- Both logger variants now include evaluation-majority baseline accuracy, balanced accuracy and MCC in the existing evaluation table, computed from the same held-out confusion matrix. The baseline is a descriptive evaluation-majority share, not a separately trained competitor; these diagnostics do not influence CV winner selection or reuse evaluation data for tuning.
- Recorded the completed revision-102 Windows `creditcard_fraud` study: 36/36 direct cells, no fallback, 15% corr CV 0.0693 versus 0.1232 at 20%, and about 9.9% higher mean corr/D_cv_full at 20%. Normal checkpoint initialization and completion are runtime-verified; Windows reload/resume remains unexercised. Sensitivity work is parked, 20% remains the production default, and seed stability is not treated as truth-accuracy validation.
- Validation: 24 focused checks passed in an isolated harness executing the original enum/widget/logger methods with real sklearn metrics. Additional smoke coverage loaded actual baseline-102 F1 enum pickles under 103, confirmed all 17 enum values/order are unchanged, and matched diagnostics against sklearn on 120 binary/multiclass matrices. Corrected a stale existing score-option count assertion (16 versus the actual 17). Syntax and clean-baseline patch apply checks passed. Full application test collection remains blocked locally by unavailable application dependencies. GUI/ordinary SQL-run validation remains pending.

## 102 — Preserve and resume interrupted Dark Number sensitivity studies

- The validation-only sensitivity CLI now creates a checkpoint automatically and writes each completed target×fraction×seed cell through a flushed atomic replacement. The original fetched data/source-pipeline snapshot is saved before training, avoiding a new randomized SQL selection/application shuffle on resume. Experiment metadata is persisted before fixed-model training; the original fitted pipeline and fixed predictions are stored in a checksummed companion so resuming does not retrain the fixed model. The cell interrupted before persistence is rerun, and complete checkpoints can regenerate final outputs.
- Resume checks dataset values/order/feature schema, split, full seeded model hyperparameters, source artifact checksum and last-run settings fingerprint, the complete study grid, formula/fallback settings and perturbation guardrails, source code and dependency versions. Mismatches, invalid progress and missing/damaged model bundles are refused without replacing checkpoint data. OS locks prevent concurrent writers and release on forced process exit; JSON safely round-trips NaN and positive/negative infinity without losing diagnostics or numeric target labels.
- Added `--checkpoint` for explicitly named/new studies, automatic per-model checkpoint defaults, resume progress logs, and README instructions. The interrupted revision-100 `creditcard_fraud` run (10/36 cells, target support 422/25,134) remains incomplete in BACKLOG; its log-only results cannot seed a checkpoint. Production defaults, correction/fallback formulas and ordinary model artifacts are unchanged.
- Validation: 52 targeted checkpoint and sensitivity tests passed, run with real correction/sensitivity code and isolated SQL/logger bootstrap imports; they cover interruption before training, interruption mid-cell, exact resumed-vs-uninterrupted results/summary, no-refit/no-SQL-fetch CLI replay, input-snapshot checksums, experiment mismatches, damaged checkpoints, atomic-write failure, nonfinite diagnostics and abrupt subprocess exit. A separate MLP/FunctionTransformer smoke test forcibly exited after cell 1, restored the original fitted model in a new process, and matched all results/summary from an uninterrupted run. The repository-wide conftest cannot load locally because application dependencies (first missing: IPython) are unavailable. Windows/real-dataset runtime verification is pending.

## 101 — Track GridSearch runtime estimation

- Added a BACKLOG item to estimate final GridSearchCV wall-clock duration before execution from the already observed k-fold training time and the known number of grid combinations/fits. The planned UI/log output should present an approximate hours/minutes duration, fit count and relevant assumptions, accounting for effective parallelism where the measured timing supports it.
- Documentation only: no training, GridSearchCV, scheduling or runtime behavior changes in this revision.

## 100 — Dark Number sensitivity evidence and runner observability

- Runtime-validated revision 099 on the realistic Återkrav/FUTV `Target=Ja` case with one fixed dataset/split/model across nine seeds per fraction. At 5% every run was `insufficient_sample` (two planned positive flips per correction fit). At 10/15/20%, all runs were direct and fallback-free; corr CV was about 0.234/0.192/0.190 respectively. The 15% and 20% levels were therefore similarly stable, while 20% produced the higher mean corr (about 5.14 versus 4.40 at 15%).
- Added the Breast Cancer control study (569 rows, targets B/M) as a second sensitivity datapoint. For the operationally interesting `M` class all 5/10/15/20% cells were direct with no fallback or low-recovery warning; corr CV was about 0.065/0.038/0.044/0.041 while mean corr increased monotonically from about 1.18 to 1.38 as the perturbation increased. This shows that flip fraction can shift the correction-factor level even when stability remains good.
- Methodology decision: retain the production 20% compatibility default for now. Treat 15% as the leading lower-perturbation candidate because it was close to 20% in stability on Återkrav and Breast Cancer, but do not change the global default until the same fixed-basis study has been run on another genuinely imbalanced real dataset (roughly 2-10% target support). Do not introduce an adaptive fraction rule without cross-dataset evidence.
- Improved the validation runner's observability for long studies: every completed fraction×seed cell now reports `[n/N]`, resolved target classes are logged explicitly, and both the loaded source pipeline and the deterministic fixed sensitivity pipeline are logged with step/final-estimator identity. Full source/fixed model identity, including estimator repr, is also persisted in the metadata JSON. This makes it clear which model was studied and whether an unset target expanded a 36-cell run to multiple target classes.
- Added focused regression coverage for pipeline/final-estimator identity formatting and updated the runner documentation. A standalone identity-format smoke and `py_compile` pass; `git diff --check` and clean `git apply --check` against revision 099 pass. Targeted pytest collection remains blocked before the test module by the existing missing `dill` dependency.

## 099 — Reproducible Dark Number correction-noise sensitivity runner

- Added `JBGDarkNumberNoiseSensitivityRunner.py`, a validation-only real-data experiment that reads the latest Repeat Last state/model, fetches the dataset once, creates one deterministic train/test split and one fixed cross-trained pipeline, then varies only correction flip fraction and correction seed. Defaults are 5%, 10%, 15%, 20% and nine seeds. Production Dark Number defaults and model artifacts are not changed.
- The detailed CSV records target support, correction status, reconstructed hard flipped/recovered counts, mean/pooled recovery, recovery spread, direct corr, low-recovery-warning state, fallback eligibility/activation/acceptance, final corr provenance and configured-formula `D_cv_full`. Unestimable 1.0 sentinels remain visible in detailed output but are excluded from corr/Dark Number stability summaries.
- Added dataset and split SHA-256 fingerprints plus fixed split/model seeding so every fraction/seed cell in one study can be audited as sharing the same data, split and fitted cross-trained model. The runner defaults to the configured revision-095 Dark Number target; if the target is unset it studies all observed labels.
- Preserved `dark_number_target` when reconstructing the validation-only `Config` from `.jbg_last_run.json`; the older validation runner previously omitted that post-095 field.
- Added focused regression coverage for the requested fraction grid/nine-seed defaults, target resolution, hard recovery-count reconstruction (including planned flip counts when preflight skips before recovery), exclusion of unestimable sentinels from stability summaries, experiment fingerprints, and output artifacts. A direct synthetic smoke completed a fixed-dataset/fixed-split/fixed-model 10%/20% grid and produced distinct recovery/corr stability summaries as expected. `py_compile` and `git diff --check` pass; targeted pytest collection remains blocked by the existing missing `dill` dependency. Runtime-verified on 2026-09-29 with the fixed Återkrav/FUTV `Target=Ja` study: 5% was unestimable in all nine seeds, while 10/15/20% were fully direct and fallback-free with corr CV about 0.234/0.192/0.190. The current 20% production default remains deliberately unchanged pending further cross-dataset evidence.

## 098 — Track Dark Number label-noise sensitivity study

- Added a methodology backlog item to validate the correction estimator's configured target-positive label-noise fraction before changing the current 20% compatibility default. On rare target classes, 20% may be an overly strong perturbation, while much smaller fractions may yield too few flipped/recovered observations for a stable correction estimate.
- The planned experiment keeps model/dataset pairs fixed and compares 5%, 10%, 15%, and 20% across repeated seeds, recording absolute flipped/recovered counts, recovery fraction, correction-factor stability, fallback activation, and downstream Dark Number behavior. A later adaptive percentage-plus-absolute-count rule is only a candidate if the sensitivity evidence supports it.
- Documentation only: no Dark Number formula, correction routing, default noise fraction, GUI behavior, or model runtime behavior changes in this revision.

## 097 — Dark Number target-first layout and low-recovery warning

- Runtime verification of 095: a real FUTV/Återkrav run with `Dark Number target = Ja` estimated correction factors only for `Ja` (`4.92308` cross-trained and `5.33333` retrained, both direct), emitted only `Ja` Dark Number rows, and still retained both classes in the diagnostic confusion matrices.
- Reordered the Dark Numbers controls to `Estimate -> Target -> Method -> Alpha -> Failure`, including migration of locally persisted widget-section order.
- Added direct-correction recovery diagnostics to `DarkNumberCorrectionFactorEstimator` (`recovery_results_` and `mean_recovery_`).
- Added a warning-only guard for very low but non-zero direct recovery: below 5% mean recovery (`corr > 20`) the log now warns that the direct estimate may be statistically unstable. The corr value remains `direct` and is neither clamped nor redirected to the experimental fallback.
- Added regression coverage for target-first widget order, persisted-settings migration, recovery diagnostics, and warning-without-behavior-change semantics.
- Runtime verification on the 2026-09-29 Återkrav/FUTV Repeat Last run: `Target=Ja` persisted correctly and direct corr remained moderate (`5.64706` cross-trained, `4.8` retrained), so no low-recovery warning was expected or emitted. The ordinary non-warning direct path is therefore runtime-verified; the actual `<5%` warning branch still awaits a real selected-target high-corr case.
- Static validation: `py_compile`, JSON parse, and `git diff --check` pass. Targeted pytest collection is blocked in this container by the existing missing `dill` dependency (`ModuleNotFoundError: No module named 'dill'`). Clean `git apply --check` against revision 096 is verified during patch packaging.

## 096 — Dark Number control-order backlog item

- Added a presentation-only backlog item to move `Target` directly after the `Estimate` checkbox in the Dark Numbers panel, with the intended order `Estimate -> Target -> Method -> Alpha -> Failure`. No runtime or calculation code is changed in this revision.

## 095 — Target-specific Dark Number estimation

- Added a `Target:` radio group to the Dark Numbers panel. It is populated from the selected class column and offers `All classes` plus each observed non-empty class label; `All classes` remains the backwards-compatible default.
- A specific target now limits the expensive Dark Number path itself: correction-factor estimation, recovery-level `perturbed_same_model` fallback, provenance and Dark Number rows are produced only for that target. Confusion matrices deliberately retain all classes as model diagnostics. A persisted target that no longer exists in the known labels fails explicitly rather than silently reverting to another class.
- Persisted the target through generated configs, saved-model config normalization and Repeat Last. Older configs/artifacts/local GUI settings default to `All classes`, while Repeat Last can restore a specific target before the class-distribution widget has been repopulated. The configuration summary now reports `Dark Number target`.
- Updated the live formula card with the current target and widened the control side from 68% to 72% (equation card 24%) so the additional radio group stays with Method/Alpha/Failure without reintroducing the wrapping fixed in 093.
- Motivation from the realistic `Training_data_weekly_2025_44` runs: RFCL produced an extreme direct correction for the negative `Nej` class while the positive `Ja` target required the experimental fallback; a subsequent FUTV-only run produced moderate direct corrections for both classes. Because the operational Dark Number question is target-specific, unused negative-class correction behavior should not drive or consume the estimate.
- Added focused regression coverage for config/default/legacy persistence, dynamic target options, Repeat Last restoration, equation-card target annotation, calculator target restriction, TaskRunner propagation, and handler-level proof that an `M` target never requests correction for `B`. Static validation: changed Python files compile and the standalone calculator target smoke passes. Project pytest collection remains blocked in this container by missing `dill` in `tests/conftest.py`; real-GUI/runtime verification is pending.

## 094 — Track Keras/NumPy terminal-only deprecation warnings

- BACKLOG-only follow-up from real runtime feedback: repeated Keras/TensorFlow NumPy 2 `__array__(copy=...)` `DeprecationWarning` messages are visible in the launching terminal but are not captured in the `jbg-server` log.
- Expanded the existing dependency-compatibility backlog item to cover both parts of the issue: establish a tested NumPy/TensorFlow/Keras/SciKeras version contract before suppressing the warning, and investigate third-party warning/stderr routing so the same diagnostics are observable in server logs without duplicating normal application logging.
- No runtime or model behavior changes in this revision.

## 093 — Compact Dark Number panel layout

- Runtime feedback on 092: the dynamic equation card worked technically in the real GUI, but the initial 60/36 split compressed the Method/Alpha/failure radio controls enough that option labels and radio markers wrapped awkwardly.
- Increased the Dark Number control area to 68% and reduced the equation card to 28%, with a 28px left gap so the formula sits visibly farther to the right. Method and Alpha now have explicit 150px/145px minimum widths and the failure group 210px, preventing the equation card from squeezing the radio controls.
- Shortened the visible failure-policy copy to `Failure:` with `No fallback` / `Experimental`; the persisted boolean semantics are unchanged (`False` still means controlled immediate unestimable handling and `True` still means the experimental perturbed fallback). Existing local settings are normalized to the compact labels while preserving the saved value.
- Reduced the equation-card typography and copy: `Formula` replaces `Current equation`, alpha notes are abbreviated, the symbol legend is condensed, non-linear root text becomes `root=3`, and the card allows horizontal overflow rather than forcing surrounding controls to collapse. Calculation/formula selection remains unchanged.
- Updated focused UI tests for the new widths, labels and compact equation text. Static validation: `py_compile` and `git diff --check` pass. Runtime visual verification: confirmed in the real Voilà GUI before 094/095.

## 092 — Dynamic Dark Number equation card

- Runtime status for 091: the new `Estimate` label and `Controlled failure` / `Experimental fallback` radio control were confirmed working in the real GUI before this revision.
- Added a read-only `HTMLMath` equation card on the right side of the existing Dark Numbers panel. The control group stays left-aligned at 60% width and the equation card uses 36%, keeping Method/Alpha/failure controls together while using the previously reserved right-side space.
- The card updates immediately when Method, Alpha or correction-failure policy changes. It shows the selected Dark Number equation, alpha semantics, non-linear root degree when applicable, and the active correction-failure policy; when `Estimate` is off the card remains in place but is visibly dimmed and marked `Estimate is off`.
- Moved Method+Alpha-to-calculator-branch resolution into `DarkNumberCalculator.resolve_calculation_type()` and made `Config.get_dark_number_calculation_type()` use that same resolver. Formula LaTeX is also exposed from `JBGDarkNumbers.py`, so the GUI does not maintain an independent method/alpha branch map or freehand formula set.
- Added focused regression coverage for all five implemented formula branches, rejection of unsupported Non-linear+Separated, panel width/alignment, dynamic Method/Alpha updates, failure-policy annotation and Estimate-off dimming. Static validation: `py_compile`, `git diff --check`, and direct formula-resolution smoke checks passed. GUI runtime smoke in this container is blocked by missing `ipywidgets`; normal project pytest remains dependency-sensitive and should be run in the project environment.


## 091 — Dark Number correction-failure radio control

- Replaced the revision-089 `Experimental perturbed fallback` checkbox with an explicit failure-policy `RadioButtons` control: `Controlled failure` maps to the existing `False` config value and immediate `fallback/unestimable`, while `Experimental fallback` maps to `True` and remains the default. The underlying config/persistence contract stays boolean, so production correction routing is unchanged.
- Made the new failure-policy radio compact (`width=auto`) with a small left margin so it sits next to the Alpha controls instead of visually drifting across the Dark Numbers row. The main Dark Numbers enable checkbox now carries the requested `Estimate` label.
- Added migration for revision-089/local settings that still contain `dark_number_perturbed_fallback_checkbox`, preserving the saved True/False value while converting the visible control and classifier-section membership to `dark_number_failure_mode`. Repeat Last/config loading continues to restore the same persisted policy.
- Changed configuration summary wording from a boolean `Experimental perturbed fallback` row to a human-readable `Correction failure handling` row showing either `Controlled failure` or `Experimental fallback`.
- Added the proposed dynamic-equation display to BACKLOG as a separate follow-up: keep controls on the left and render the currently selected Dark Number formula on the right side of the same panel, driven by the implementation rather than duplicated freehand UI formulas.
- Validation: focused GUI/config tests cover the two radio values, experimental default, compact placement, dependency disabling, legacy-checkbox migration and no-control legacy default. `py_compile`, `git diff --check`, default-settings/config-mapping smoke checks, legacy-migration smoke checks, and exact apply/apply-check against revision 090 passed. Targeted pytest collection remains blocked locally by missing `dill` in `tests/conftest.py`.

## 090 — Dark Numbers `Estimate` label backlog item

- Documentation-only follow-up recording the requested cosmetic change: give the primary Dark Numbers enable checkbox the visible label `Estimate`. Revision 091 implements the label together with the clearer correction-failure radio control.

## 089 — Default-on experimental perturbed Dark Number fallback

- Runtime-verified 088 on the 160-row `text_regression` reproduction with nine paired seeds. `perturbed_same_model` activated for 100% of correction attempts for both targets and correction-factor stability remained strong (`corr_cv` CV about 0.090 for `incident` and 0.064 for `request`). `D_cv_full` truth MAE improved versus the unestimable-sentinel baseline but retained a clear negative bias, so the method remains explicitly experimental rather than being treated as a validated standard estimator.
- Promoted the already validated shadow-clone mechanism into the normal Dark Number correction path behind a dedicated `Experimental perturbed fallback` checkbox. The checkbox is checked by default, is disabled whenever Dark Numbers are disabled, persists through generated configs/model config loading/Repeat Last, and can be unchecked to restore the previous immediate `fallback/unestimable` behavior. Older configs/artifacts that predate the setting default to enabled.
- Production hard/direct correction still has absolute precedence. Recovery-level statistical failures (`zero_recovery`, `nonfinite`, `nan_recovery`) may invoke five same-model shadow clones with deliberately perturbed random-state parameters and class-stratified bootstrap samples; at least three finite clone correction factors and coefficient of variation <= 0.50 are required, after which the median is used with provenance `perturbed_same_model`. `insufficient_sample` remains unestimable and resource/execution failures retain the existing `regressed_same_model` path. No alternate estimator is introduced.
- Added explicit `EXPERIMENTAL` logging for every attempted/accepted/rejected shadow-clone fallback and aggregate structured telemetry in `dark_number_experimental_fallback_*`. The telemetry contains model/target, direct failure status, clone correction factors, valid-clone count, median/mean/std/CV/range, acceptance reason, final correction and provenance; it does not store source rows or raw text. Normal Dark Number output continues to expose `corr_model` and `corr_source`.
- Added focused regression coverage for config defaults/opt-out, generated-config persistence, Repeat Last GUI restoration/dependency state, and production zero-recovery routing into `perturbed_same_model`. Validation: `py_compile`, `git diff --check`, direct shadow-clone helper smoke coverage, GUI-settings smoke coverage and exact apply-check against the 088 baseline passed. Targeted pytest collection remains blocked locally by missing `dill`. Runtime production verification: **verified** on a normal 160-row `text_regression` training run after 090. The default-on setting was visible in configuration, all four direct correction attempts ended in `zero_recovery`, the experimental fallback accepted all five shadow clones in every case with CV 0.074-0.181, `corr_source=perturbed_same_model` propagated into the Dark Number table, and `dark_number_experimental_fallback_*` telemetry was emitted.

## 088 — Perturbed same-model robustness evidence

- Runtime-verified 087 on the 160-row `text_regression` zero-recovery reproduction. Perturbed shadow clones activated for every correction attempt for both targets, with accepted clone-factor coefficient of variation roughly 0.15-0.37. The candidate improved `D_cv_full` truth MAE for both targets, materially worsened `D_cv_test`, and left `D_retrained_full` unchanged; it therefore remains validation-only and is not promoted to production.
- Strengthened the 087 comparison summary with estimable-only truth-error profiles for each Dark Number output. Rows whose correction source is `fallback/unestimable` are excluded so the harness's internal 1.0 sentinel cannot be mistaken for an observed estimate. Each profile reports estimable count/rate, MAE, median absolute error, p90 absolute error, mean signed error (bias) and signed-error standard deviation.
- Added separate correction-factor stability profiles for CV and retrained corrections, restricted to accepted `perturbed_same_model` sources. The summary now records activation count/rate, mean, median, standard deviation, coefficient of variation, IQR and range.
- Marked `D_cv_full` as the primary full-data diagnostic for this robustness experiment while keeping `D_cv_test` and `D_retrained_full` visible as separate outcomes. The standalone runner now defaults to nine paired seeds instead of three and writes `dark_number_perturbed_same_model_robustness_*` artifacts. Production Dark Number behavior and persisted model artifacts remain unchanged.
- Added focused regression coverage proving unestimable sentinel rows are excluded from robust truth-error metrics, correction stability uses only accepted perturbed sources, and the runner defaults to nine paired seeds. Validation: `git diff --check`, `py_compile`, direct robustness-helper smoke coverage, and exact apply-check against the 087 baseline passed. Targeted pytest collection remains blocked locally by missing `dill`. Runtime/empirical robustness validation: **completed** on `text_regression` with nine paired seeds; activation was 100% for both targets, correction-factor CV was low, `D_cv_full` improved but remained negatively biased, so 089 exposes the method only as an explicitly experimental default-on fallback.

## 087 — Perturbed same-model shadow-clone validation candidate

- Runtime-verified 086 on the 160-row `text_regression` zero-recovery reproduction. For both `incident` and `request`, every lower-noise hard-recovery probe at 10%, 12.5%, 15% and 17.5% remained `zero_recovery` across all paired runs. Adaptive activation was therefore 0%, candidate unestimability stayed 100%, and the conservative fallback correctly emitted no estimate. `adaptive_noise_same_model` is not promoted to production.
- Added validation-only `perturbed_same_model`. Hard/direct correction still has absolute precedence. Only recovery-level statistical failures may create local shadow clones of the same target pipeline; estimator family and ordinary hyperparameters are preserved, while random-state parameters are deliberately reseeded and each clone receives a class-stratified bootstrap sample with unchanged class counts. Correction estimation still uses hard predictions at the configured target noise.
- A perturbed fallback is accepted only when at least three shadow clones independently produce valid finite correction factors and their coefficient of variation is at or below 0.50 by default. The robust median correction factor is then reported with provenance `perturbed_same_model`; otherwise the result remains `fallback/unestimable`. Defaults are configurable in the validation runner with `--perturbation-clones`, `--perturbation-min-valid` and `--perturbation-max-cv`.
- The perturbed correction candidate itself uses hard `predict()` recovery, but the standalone real-data harness continues to require `predict_proba()` for its separate probability-ranked enrichment diagnostics. The gate message now states that reason instead of referring to the rejected soft fallback. Output artifacts now use `dark_number_perturbed_same_model_validation_*`.
- Added focused regression coverage for deliberate perturbation seeding, reproducible class-stratified bootstrap sampling, direct/insufficient-sample precedence, stable-clone acceptance and unstable-clone rejection. Production Dark Number behavior and persisted model artifacts remain unchanged. Runtime/empirical validation: **promising but not production-ready** on `text_regression`; 100% activation, stable accepted clone factors, improved `D_cv_full` MAE for both targets, worse `D_cv_test`, unchanged `D_retrained_full`.

## 086 — Adaptive-noise same-model validation candidate

- Recorded the completed 083/084 real-data experiment on the 160-row `text_regression` zero-recovery reproduction. `soft_same_model` activated for 100% of correction attempts and eliminated formal unestimability, but its truth error was substantially worse: `D_cv_test` MAE rose to 2.579925/5.275031 for `incident`/`request`, `D_cv_full` MAE rose to 0.318511/0.499714, and `D_retrained_full` did not improve. The baseline MAE values use the harness 1.0 sentinel and are not treated as valid competing estimates. The soft candidate is therefore retained only as a documented rejected validation experiment and is not promoted to production.
- Added a validation-only `adaptive_noise_same_model` candidate. Hard/direct correction still runs first and always wins when valid. Only recovery-level unestimable outcomes may retry the same estimator with the same hard class decisions at lower correction-noise fractions (50%, 62.5%, 75% and 87.5% of the configured target). At least three valid lower-noise recovery points are required; recovery rate is linearly extrapolated to the target noise and must remain finite and within `(0, 1]`, otherwise the candidate remains `fallback/unestimable`. No alternate estimator is introduced.
- Corrected the validation summary's misleading coverage delta. Estimability coverage is now derived from correction-source provenance (`fallback/unestimable` versus an observed candidate source), while Dark Number interval-coverage delta is reported separately. This fixes the observed 0.0 coverage delta when the soft experiment actually moved from 100% unestimable to 0% unestimable.
- Reused the production `prepare_estimator_input()` helper in the real-data validation runner so pandas SparseDtype feature frames are converted to SciPy CSR before sklearn fitting. This removes the repeated sparse-DataFrame-to-dense warnings seen in the 084 real-data run and avoids an unnecessary dense-memory copy on larger text datasets.
- Switched the standalone validation runner from the rejected soft candidate to the adaptive-noise candidate; output artifacts are now named `dark_number_adaptive_noise_validation_*`. Production Dark Number calculation and persisted model artifacts remain unchanged.
- Added focused regression coverage for adaptive extrapolation, the minimum-three-point guard, corrected estimability coverage semantics and sparse-frame CSR conversion. Validation: `py_compile`, `git diff --check`, direct adaptive-noise smoke coverage and clean apply-check against exact 085 baseline; full pytest collection remains blocked locally by missing `dill`. Runtime/empirical validation: **completed** on `text_regression`; all lower-noise probes remained `zero_recovery`, adaptive activation was 0%, and the candidate correctly stayed `fallback/unestimable`.

## 085 — Fix NumPy dataset truth-value check in 084 runner

- Fixed the standalone Dark Number validation runner after its first real-data execution fetched all 160 `text_regression` rows successfully but crashed before validation with `ValueError: The truth value of an array with more than one element is ambiguous`. `JBGHandler.get_dataset()` returns a NumPy array, so the runner now checks `data is None or len(data) == 0` instead of evaluating the array as a boolean.
- Added focused regression coverage that passes a non-empty NumPy dataset through `load_validation_inputs()` and verifies it reaches dataset loading, saved-converter reuse and pipeline return without triggering ambiguous truth-value evaluation.
- Production classifier/Dark Number behavior is unchanged. Runtime verification: **verified** by the corrected 085 run, which passed data preparation, completed both target-class comparisons and wrote the paired CSV/JSON outputs.
- Validation: `py_compile`, `git diff --check`, direct NumPy-array smoke coverage and clean `git apply --check` against the exact 084 baseline. Targeted pytest collection remains subject to the existing local dependency blocker (`dill`).

## 084 — Real-dataset runner for soft same-model validation

- Added `JBGDarkNumberValidationRunner.py`, a standalone validation-only command that reads the project-local `.jbg_last_run.json`, resolves runtime SQL credentials without persisting them, fetches the same labelled dataset again, loads the saved `.sav` artifact and reuses its fitted text/category converter before invoking the 083 paired comparison. The normal classifier, persisted model and production Dark Number path are not modified.
- By default the runner validates every observed class as a one-vs-rest target, with optional repeated `--target` filters. It reports baseline unestimable rate, `soft_same_model` activation rate, candidate unestimable rate, coverage-rate delta, the known injected-noise truth and baseline/candidate MAE for `D_cv_test`, `D_cv_full` and `D_retrained_full`.
- Added paired CSV and JSON-summary artifacts under the existing `output/csvs` directory, and a dedicated persistent `jbg-dark-number-validation_*` log so empirical runs can be audited separately from ordinary training logs.
- Strengthened the 083 paired contract for stochastic estimators: validation clones now fill only otherwise-unset `random_state` parameters with the paired seed while preserving explicitly configured seeds. This prevents MLP/random-forest initialization noise from being mistaken for a fallback-method effect. Production estimator settings remain untouched.
- The most recent ordinary `text_regression` run after 083 re-confirmed that production remains isolated: all four hard/direct correction attempts still ended at `fallback/unestimable`; no `soft_same_model` path appeared in normal training output.
- Added regression coverage for estimator seeding, all-observed-target runner orchestration, unknown-target rejection and output artifact creation. Validation: `py_compile` and focused direct runner/harness smoke checks pass; targeted pytest collection remains blocked in this environment by missing project dependency `dill`. Runtime verification: **verified after 085** on the real `text_regression` reproduction; both targets completed and dedicated CSV/JSON artifacts were written. The run also exposed the coverage-delta reporting bug and sparse-densification warnings fixed in 086.

## 083 — Validation-only soft same-model correction fallback

- Added an opt-in `enable_soft_same_model_fallback` path to the standalone 050 Dark Number validation harness only; production Dark Number calculation is unchanged.
- Hard/direct correction always runs first. A valid direct factor returns immediately and the soft path is not instantiated. Only recovery-level statistically unestimable direct outcomes (`zero_recovery`, `nonfinite`, `nan_recovery`) may try a second correction-factor estimate with the same target estimator/data/noise setup and `predict_mode="predict_proba"`; successful provenance is `soft_same_model`. `insufficient_sample` stays unestimable because changing the recovery metric cannot repair inadequate sample support.
- If the estimator lacks `predict_proba()` or the soft estimate is also invalid/unestimable, the harness retains `fallback/unestimable`. Resource/execution failures still follow 082's `regressed_same_model` path rather than being reclassified as statistical fallback cases.
- Added a paired comparison helper that runs baseline and soft-candidate validation on identical random seeds and reports source activation/unestimable rates plus mean absolute error for `D_cv_test`, `D_cv_full`, and `D_retrained_full` against the known injected-noise truth. This makes candidate adoption depend on accuracy/coverage evidence rather than merely obtaining a finite factor.
- Added focused coverage for a deterministic same-model classifier with zero hard recovery but non-zero target probability mass: baseline remains unestimable while the soft path yields a finite factor; separate guards prove direct estimates take precedence and that a failed soft attempt remains unestimable. No alternate-model/FUTV fallback is introduced.
- 082 runtime verification: the 160-row `text_regression` reproduction with stop words and uni/bi/tri-grams produced `zero_recovery` for both targets in both cross-trained and retrained models. Every case logged that sample-size regression was not applicable and used `fallback/unestimable`, while `corr_model` identified the target model correctly.
- Validation: Python compile and a direct end-to-end harness smoke test passed; the deterministic case produced baseline `(1.0, fallback/unestimable)` and soft `(4.0, soft_same_model)`, and paired runs used identical seeds with 100% soft activation. Targeted pytest collection remains blocked by missing `dill` in `conftest`. Empirical result after 085: **rejected for production in its current form**. On `text_regression`, soft activated for all correction attempts but materially worsened MAE for CV-based estimates and did not improve the retrained estimate.

## 082 — Dark Number correction fallback taxonomy

- Separated statistical unestimability from execution/resource failure in correction-factor estimation. Direct outcomes with `zero_recovery`, `nonfinite`, `nan_recovery`, or `insufficient_sample` now use the explicit 1.0 `fallback/unestimable` path without attempting sample-size regression.
- Restricted sample-size regression to `MemoryError`/`SystemError` resource/execution failures from the same target estimator and renamed its provenance from ambiguous `regressed` to `regressed_same_model`. Other unclassified direct failures no longer trigger regression automatically.
- Added `corr_model` to Dark Number results alongside `corr_source`, and include the correction model in per-target logs. Current direct/regressed paths therefore identify the same target estimator explicitly; a future alternate fallback model can be reported without masquerading as the target model.
- Updated the standalone 050 validation harness to use the same failure taxonomy and provenance contract. Added focused regression coverage proving zero recovery does not invoke the regressor, resource failure does, and model-specific Dark Number rows carry the correction-model identity.
- 081 runtime verification: a broad Breast Cancer ROC-AUC run produced a 108-candidate spot-check table containing CV metrics only, then performed FUTV GridSearchCV, then exposed the final holdout for the first time in `Evaluate trained model`. Full-data retraining, misprediction analysis and Dark Numbers completed normally; all four observed correction factors were direct estimates.
- Validation: `py_compile`, `git diff --check`, and a direct fallback-taxonomy smoke test passed. Pytest collection is blocked in this environment by missing project dependencies (`dill` first). Runtime verification: **verified** on the 160-row `text_regression` reproduction documented in 083; all four statistically unestimable hard/direct cases bypassed sample-size regression and reported `fallback/unestimable` with explicit `corr_model`.

## 081 — Untouched final-holdout training contract

- Removed final-holdout scoring from spot-check candidates. Candidate comparison is now based solely on cross-validation over the training partition; `X_validation`/`Y_validation` are not fitted against or scored during spot-checking.
- Removed the holdout column from the spot-check table/CSV so final-evaluation information is not exposed during model development.
- Removed the repeated train/validation split inside the candidate loop. The configured holdout is created once by the normal dataset-separation task and remains unchanged until `Evaluate trained model`.
- Final GridSearchCV continues to use only `X_train`/`Y_train`; the selected/tuned model is evaluated against the holdout for the first time in the existing final-evaluation task, after which the existing fresh full-data retraining contract remains unchanged.
- Updated focused regression coverage for the CV-only report schema/state and added a guard proving a successful spot-check candidate does not invoke the legacy holdout-evaluation path.
- Validation: Python compile and `git diff --check` passed; direct report-matrix smoke checks passed. Targeted pytest collection remains blocked in this environment by missing project dependencies (`dill`, `langdetect`, then `skorch` when temporary import stubs were supplied). Runtime verification: **verified** on the broad Breast Cancer ROC-AUC run documented in 082; spot-check output contained no holdout information, final evaluation occurred after GridSearchCV, and the subsequent full-data/Dark Numbers lifecycle completed normally.

## 080 — FUTS bounded-grid runtime verification

- Runtime-verified 077 on a FUTS-only Breast Cancer run with MAX + PCA and Accuracy.
- The final GridSearchCV used exactly `{'FUTS__cv': [5], 'FUTS__passthrough': [False, True], 'FUTS__final_estimator__C': [0.1, 1.0, 10.0]}` and logged 6 parameter combinations x 10 folds = 60 fits.
- The bounded search completed successfully and produced a fitted `StackingClassifier(cv=5, final_estimator=LogisticRegression())`; final evaluation also completed normally.
- Runtime behavior: unchanged; documentation/verification-only revision.

## 079 — FUT runtime verification status

- Runtime-verified 075's holdout-diagnostic contract on Breast Cancer: FUTS produced the stronger diagnostic holdout score (0.9912 versus FUTV 0.9737), but FUTV correctly remained ranked first because its mean CV score was higher (0.9759 versus 0.9737).
- Runtime-verified 078's bounded FUTV search: the final GridSearchCV logged exactly 4 parameter combinations x 10 folds = 40 fits, and the selected VotingClassifier retained `voting='soft'`.
- FUTS completed the same real-data spot-check successfully, but 077's bounded final GridSearchCV remains pending because FUTV won model selection and therefore was the only estimator sent to final grid search.
- Runtime behavior: unchanged; documentation/verification-only revision.

## 078 — Bounded FUT Voting grid search

- Replaced FUTV's empty/default search with four explicit soft-voting weight profiles: equal/default weighting plus one profile each emphasizing MLP, Random Forest or AdaBoost.
- Keep `voting='soft'` fixed in the estimator rather than searching hard/soft voting, preserving `predict_proba()` for probability diagnostics and Dark Numbers.
- Constituent MLP/RandomForest/AdaBoost grids remain outside FUTV, bounding the final GridSearchCV at 4 parameter combinations / 40 outer fits with the application's current 10-fold search.
- Added focused regression coverage for the exact four-profile grid, base-estimator ordering, absence of constituent-estimator parameters, and fixed soft-voting behavior.
- Runtime verification: **verified** on Breast Cancer after 078. The final search logged exactly 4 parameter combinations x 10 folds = 40 fits, and the selected VotingClassifier retained `voting='soft'`.

## 077 — Bounded FUT Stacking grid search

- Replaced FUTS' Cartesian product of the full MLP, Random Forest and AdaBoost grids plus `cv=(5, 10, 20)` with an ensemble-level grid only: fixed `cv=5`, `passthrough=[False, True]`, and logistic meta-estimator `final_estimator__C=[0.1, 1.0, 10.0]`.
- The final FUTS GridSearchCV is therefore bounded at 6 parameter combinations / 60 outer fits with the application's current 10-fold search instead of 31,104 combinations / 311,040 outer fits. Each outer stacking fit still performs its own five-fold internal stacking work.
- Added focused regression coverage for the exact six-combination contract, exclusion of constituent-estimator grids, and parameter routing into `StackingClassifier` and its logistic final estimator.
- FUTV remains unchanged in this revision; the four-profile soft-voting weight grid documented in 076 stays in BACKLOG.
- Runtime verification: **verified** on a FUTS-only Breast Cancer run after 079. The final search logged exactly 6 parameter combinations x 10 folds = 60 fits, completed successfully, and produced a fitted `StackingClassifier(cv=5, final_estimator=LogisticRegression())`.

## 076 — FUT ensemble grid-search backlog design

- Documented the reproduced FUTS search-space explosion: inheriting the full MLPC, Random Forest and AdaBoost grids plus stacking `cv=(5, 10, 20)` yields 31,104 parameter combinations and 311,040 outer fits at 10 folds.
- Added a bounded default design to BACKLOG: FUTV should search four soft-voting weight profiles only; FUTS should fix internal stacking CV at five folds and search passthrough plus three logistic meta-estimator `C` values, for six combinations total.
- Explicitly keep constituent-estimator grids out of FUTV/FUTS by default; any future joint tuning should use a small curated list of coherent profiles rather than a Cartesian product.
- Runtime behavior: unchanged; planning/documentation-only revision.

This file is the append-only history of numbered JBG development revisions. `BACKLOG.md`
tracks open work and prioritization; this file records what each completed patch changed and
how it was verified. Runtime verification is updated by the next patch after a real application
run when applicable.

Historical entries below are condensed from the existing BACKLOG/patch history. Revision
numbers for which no surviving documented patch entry is available are intentionally omitted
rather than reconstructed from guesswork.

## 075 — Holdout-diagnostic model-selection contract

- Renamed spot-check `test_score`/`best_test_score` state to explicit holdout terminology so
  internal names match the existing CV-only winner-selection behavior.
- Relabelled the spot-check report column from `Test data` to `Holdout (diagnostic)` and
  centralized report construction for both logger implementations.
- Result tables now sort only by mean CV score and CV standard deviation; holdout performance
  remains visible for diagnostics but can no longer act as a hidden third tie-breaker in the
  displayed ranking. Stable sorting preserves evaluation order for exact CV ties.
- Added focused regression coverage for the diagnostic column label, CV-only ordering and
  CV-standard-deviation tie-break behavior.
- Runtime verification: **verified** on Breast Cancer after 078. FUTS had the stronger diagnostic holdout score (0.9912 versus 0.9737), but FUTV remained first because its mean CV score was higher (0.9759 versus 0.9737), directly confirming that holdout no longer influences ranking.

## 074 — Runtime verification and completion-mail backlog

- Marked 073 runtime-verified across both supported data-layout lifecycles: a separated
  Breast Cancer train/predict table pair and the mixed Iris table with labelled and unlabelled
  rows in the same source table. Both fresh-process prediction runs restored the persisted
  Keras/SciKeras pipeline and completed prediction/write-back without reload fallback.
- Added a BACKLOG item for completion-mail recipient resolution after successful prediction-only
  runs logged `Error sending completion mail: no valid recipient(s).`. The current evidence does
  not yet distinguish a prediction-specific configuration bug from an absent/invalid
  `DEFAULT_NOTIFICATION_EMAIL`, so the notification path should be diagnosed before changing
  runtime behavior.
- Runtime verification: documentation-only revision; observations are verified by the Breast
  Cancer and Iris prediction-only logs following 073.

## 073 — Keras wrapper-metadata persistence

- Preserve the fitted SciKeras wrapper in the main model artifact and detach only its native
  Keras `model_` before serialization. This retains the fitted class/target/feature metadata
  required for inference without trying to pickle the TensorFlow model itself.
- On reload, load the `.keras` sidecar and reattach it to the persisted fitted wrapper, so a
  prediction-only process can restore a Keras pipeline before any dataset has been fetched.
- Keep the pre-073 restore path for older artifacts when labelled initialization data is
  available, but fail early with an explicit retrain requirement when an old Keras artifact is
  opened prediction-only instead of allowing the later `NoneType.pipeline` crash.
- Added focused persistence-helper regression coverage proving that detach does not mutate the
  live estimator, fitted wrapper metadata survives, and reattach restores the external model.
- Runtime verification: **verified** across both supported data-layout lifecycles. Breast Cancer was trained from a dedicated labelled table and, after process restart, reloaded for prediction from a separate unlabelled table; Iris was trained and reloaded from a mixed table containing labelled and unlabelled rows. Both prediction-only runs restored the persisted Keras/SciKeras pipeline and wrote predictions without reload fallback.

## 072 — Keras 3 native sidecar persistence

- Save Keras/TensorFlow model sidecars as `<artifact>.KERA.keras`, which satisfies the Keras 3 native-format extension contract instead of the rejected extensionless `<artifact>.KERA` path.
- Reload now resolves the native `.keras` sidecar first while retaining legacy sidecar discovery for backwards-compatibility diagnostics.
- Added focused persistence-path regression coverage for native path construction, legacy path construction and resolver preference/fallback.
- Runtime verification: **partially verified** on the targeted 30-feature Breast Cancer Keras
  run. Native `.keras` saving no longer emitted the Keras 3 invalid-extension error. A subsequent
  prediction-only restart exposed that the older artifact layout had discarded fitted SciKeras
  wrapper metadata and still required live data during restore; that cross-process gap is fixed
  by 073.

## 071 — Bounded Keras grid and explicit defaults

- Reduced the Keras MLP grid from 24 combinations to 4: 50/100 epochs, Adam only,
  learning rates 0.001/0.01, batch size 32 and `verbose=0`. With the current 10-fold final
  GridSearchCV this bounds the Keras search from 240 fits to 40 fits.
- Changed the base `MLPKerasClassifier` default from 200 to 50 epochs and made the baseline
  training parameters explicit: one 100-unit hidden layer, Adam, learning rate 0.001, batch
  size 32 and quiet output.
- Added regression coverage for the exact grid size and default estimator contract.
- Kept NumPy/Keras `__array__(copy=...)` deprecation warnings and TensorFlow retracing warnings
  visible; both are tracked in BACKLOG instead of being filtered away.
- Runtime verification: **verified** on the 30-feature Breast Cancer run after 071. The final Keras grid logged exactly 4 parameter combinations x 10 folds = 40 fits, KERA produced finite spot-check scores without shape exceptions, and the selected pipeline retained the expected explicit defaults/grid-selected values.

## 070 — Keras input-shape contract

- Fixed `MLPKerasClassifier._keras_build_fn()` to pass a one-dimensional tuple `(n_features,)`
  to `keras.layers.Input` instead of the scalar `n_features`.
- This addresses the broad-run Keras failure `Cannot convert '100' to a shape` without changing
  the Keras grid, hidden-layer configuration, optimizer choices, or training methodology.
- Added regression coverage asserting that 100 input features build a Keras model with
  `input_shape == (None, 100)`.
- Runtime verification: **verified** on the 30-feature Breast Cancer run after 071. KERA completed spot-checking and final GridSearchCV with 30 input features and no recurrence of the scalar-shape construction failure.

## 069 — SelfTraining preflight and revision-log contract

- Added a preflight guard for `SelfTrainingClassifier`: supervised training sets with no `-1`
  unlabeled target are skipped before cross-validation instead of repeatedly fitting a
  semi-supervised wrapper that has no unlabeled samples to consume.
- Kept the SelfTraining path available when an actual `-1` unlabeled target is present.
- Added targeted regression coverage for both the skip and allow branches.
- Established `CHANGELOG.md` as the append-only numbered revision history; `BACKLOG.md`
  remains the source for open work and prioritization.
- Runtime verification: **verified** on the Breast Cancer run after 069. All six selected
  SelfTraining/preprocessor candidates were preflight-skipped with the explicit `-1` unlabeled
  sample requirement and zero elapsed fitting time; the previous sklearn no-unlabeled warning did
  not recur.

## 068 — ROC-AUC capability contract

- Binary ROC AUC now resolves to sklearn `roc_auc`, allowing margin-based estimators with
  `decision_function()` to score without requiring `predict_proba()`.
- Ordinary SVC exposes probabilities, preserving multiclass AUC and probability-based
  diagnostics; multiclass AUC preflight-skips estimators that cannot provide probabilities.
- Runtime verification: **verified** on the Breast Cancer run after 068. SVC candidates produced
  finite ROC-AUC results with an empty exception column, and the selected SVC pipeline used
  `probability=True`.

## 067 — Non-negative Naive Bayes preflight

- Added compatibility preflight for MultinomialNB and ComplementNB after preprocessing/reduction.
- Signed reductions and remaining negative estimator input are skipped before CV; valid
  non-negative paths remain available and BernoulliNB is unaffected.
- Runtime verification: targeted code/regression coverage completed; the immediately following
  real-data run did not include MultinomialNB/ComplementNB, so the specific runtime branch remains
  unexercised there.

## 066 — MLP maximum-iteration contract

- Removed the hidden `MLP2(max_iter=500)` limit; both scikit-learn MLP variants now honor the
  configured maximum-iteration value.
- Runtime verification: **verified** with `max_iter=20000` visible in the selected MLP2 pipeline.

## 065 — Repeat Last placeholder cleanup / Categorizer label

- Removed stale `N/A` choices from restored Class, Unique id and Data controls once real saved
  options are available.
- Renamed the text-processing checkbox `Categorize` to `Categorizer` with legacy-label migration.

## 064 — Dark Number control state / GUI cleanup

- Kept Dark Number Method/Alpha disabled when Dark Numbers is off even in the post-Continue
  observer-lock state.
- Removed the framed Progress wrapper and redundant `Text:` prefixes from text controls.

## 063 — GUI section polish

- Added reusable titled/framed horizontal GUI sections, simplified Mode labels and removed the
  redundant Dark Numbers checkbox caption while preserving action-row alignment.

## 062 — Fresh full-data retraining / unique-id contract

- Full-data retraining now clones the selected fitted pipeline before fitting all known rows,
  preventing warm-start state reuse.
- Added global uniqueness/non-null checks for selectable ids, runtime duplicate-id rejection and
  configured-id prediction joins.
- Runtime verification: **verified** on the text dataset, including Bagging warm-start selection.

## 061 — Dark Number default alpha / action layout

- Made `Separated` the default Dark Number alpha adjustment for new configurations while
  preserving legacy fallback behavior.
- Kept Repeat Last left-aligned and Regression Suite right-aligned on the action row.

## 060 — Dark Number method and alpha controls

- Made method and alpha independent first-class controls with backwards-compatible persistence.

## 059 — Dedicated Dark Numbers GUI/config controls

- Decoupled Dark Number calculation from misprediction display and persisted the enable/method
  controls through configs, saved models and Repeat Last.

## 058 — Repeat Last observer-lock fix

- Prevented transient `N/A` class-column state from issuing SQL callbacks while Repeat Last
  restores a saved GUI state.

## 057 — Repeat Last visible-state restoration

- Restored persisted run settings into the visible GUI before execution while preserving current
  SQL credentials and suppressing model-selection observers during restoration.

## 056 — Prediction-only loading fixes / Dark Number wording

- Capped prediction-only row fetches to eligible unclassified rows and fixed all-NaN detection
  against the full feature matrix rather than an empty training partition.
- Clarified injected-label-noise logging as target-positive labels flipped to non-target labels.

## 055 — Persistence contract / Naive Bayes RFE compatibility

- Replaced no-op lambda transformers with a module-level identity callable and introduced a
  versioned model-artifact envelope with backwards-compatible legacy loading.
- Marked Bernoulli/Complement/Multinomial Naive Bayes as RFE-incompatible before feature-importance
  lookup failures can occur.

## 054 — Parallel execution exception contract

- Preserved original exception types/tracebacks in `execute_n_job`, restricted worker-reduction
  retries to known resource/pickling failures and fixed negative global worker-limit semantics.

## 053 — SMOTE float64 normalization

- Converted interpolation-based SMOTE-family input to float64 before resampling while preserving
  sparse structure and leaving non-interpolating samplers unchanged.

## 052 — FastICA dimensional safety

- Capped FastICA components to effective dimensions/CV folds, skipped sparse input, made
  initialization deterministic and raised its iteration budget. Convergence warnings remain a
  separate open BACKLOG item.

## 051 — Dark Number correction-factor robustness

- Excluded non-finite/zero-recovery and synthetic insufficient-sample observations from correction
  regression, required enough finite samples and made correction provenance explicit.

## 050 — Dark Number validation harness

- Added controlled known-noise validation for coverage, correction-factor stability and
  probability-ranked enrichment without changing production formulas.

## 049 — Dark Number correctness contract

- Preserved the published linear formula while separating CV-test/CV-full/retrained estimates,
  fixing alpha handling and making injected-noise fraction configurable.

## 048 — Secret handling / serialization trust

- Removed SQL passwords from generated configs/model metadata, added runtime secret injection,
  documented pickle/dill trust requirements and removed the duplicate matplotlib requirement.

## 047 — Persistence-fixture hygiene

- Ignored generated `.sav` persistence fixtures and documented their local/trusted-artifact policy.

## 046 — Class-label normalization

- Fixed discarded class-label string conversion while preserving null/empty prediction labels.

## 045 — LinearSVC convergence-grid cleanup

- Removed the repeatedly non-converging hinge-loss branch and retained the squared-hinge grid.

## 044 — Regression-suite lifecycle reporting

- Moved runtime/progress/email reporting to the suite level and suppressed per-profile completion
  messages in favor of one final summary.

## 043 — TruncatedSVD component cap

- Capped `n_components` to available input features, preventing invalid compact-text candidates.

## 042 — Sparse/text preflight tightening

- Retained expected skips in result tables without per-candidate output spam, preflight-skipped
  sparse LDA paths and capped Nystroem components to the smallest CV training fold.

## 041 — Naive Bayes alpha safety

- Removed `alpha=0.0` from Multinomial/Bernoulli/Complement Naive Bayes grids to avoid non-finite
  probability paths on sparse text data.

## 040 — Text categorization setting

- Made automatic categorization enable/disable behavior effective while preserving explicitly
  forced categorical columns.

## 039 — Transient RFE progress

- Routed RFE binary-search round status through the transient progress label rather than emitting
  permanent INFO rows.

## 038 — QDA regularized baseline

- Started QDA spot-checking at the grid's minimum regularization to avoid rank-deficient baseline
  failures before grid search.

## 037 — Nearest Centroid grid compatibility

- Corrected the `euclidean` metric spelling and replaced invalid `shrink_threshold=0.0` with `None`.

## 035 — Probability-less estimator evaluation fallback

- Guarded evaluation for estimators without `predict_proba()` and used classification-report
  precision as the fallback confidence without per-row warning spam.

## 034 — Persistent Repeat Last action

- Added a saved Repeat Last action that recreates the most recent manual classifier run without
  persisting SQL credentials; data is fetched again on execution.

## 032 — Sparse MinMax preflight

- Preflight-skipped MinMaxScaler for sparse converted features instead of allowing known CV failure
  or implicit densification.

## 031 — Sparse PCA / spot-check failure propagation

- Switched sparse fractional PCA to `covariance_eigh` and fixed failed-candidate propagation so
  CV failures cannot be treated as successful or have their root cause overwritten.
