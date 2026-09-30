# Revision log – JBGAutoClassification

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
