- [x] Revision 163 follow-up: repeated Train/Rerun/Repeat Last behavior was live-verified on
  Windows/Voilà after the reclassification widget lifecycle hardening; no `Error displaying widget`
  recurrence was reported. Reopen only if browser/server evidence shows a new rendering failure.
- [ ] Revision 159 follow-up: identify the Voilà/frontend version responsible for missing XSRF headers on shutdown; compare upstream issue #1458; verify actual live-kernel counts separately before implementing cleanup. Never bypass CSRF or kill running training jobs.
- [ ] Revision 158: review Voilà server runtime versions and shutdown 403 diagnostics; test repeated reload/disconnect and Script output widget lifecycle on Windows. Do not disable XSRF; source-level diagnostics alone do not fix the browser shutdown request.
- [ ] After revision 157: validate memory guard and process-tree heartbeat on Windows/Voilà; assess GaussianProcessClassifier large-data cost; Feature modelling concurrency was separated/automated in revision 169; continue investigating missing Script output and Voilà kernel shutdown _xsrf failures.
## Revision 155 — JSON export and Feature modelling button help

Transitional versioned JSON export with legacy Python config compatibility; corrected action tooltips. See `docs/configuration_155.md`.

# JBG AutoClassification — Active BACKLOG (Revision 152)

This is the **active**, prioritized maintenance roadmap for the general-purpose JBG ML framework.
The complete historical backlog (including completed revisions and research experiments)
is preserved without deletion in [`docs/backlog_archive_pre_152.md`](docs/backlog_archive_pre_152.md).
Revision 152 only reorganizes documentation. It does not implement, close, or re-validate
runtime features. A checked box below is a **documentation disposition**, not new test evidence.

## Decisions and scope

- [x] The prior private-dataset investigation is concluded at the user's request. No more dataset-specific experiments
  are planned in the main product backlog. Historical findings remain in the archive.
- [x] Revision 151 policy: native CatBoost is retained for **Feature modelling**, and is not an
  ordinary exhaustive-pipeline candidate. Legacy artifact loading is intended to remain supported;
  native CatBoost integration into ordinary search is a separately scoped future refactor.
- [x] Revisions 147–150 introduced generally applicable, opt-in diagnostics (threshold/OOF,
  group-aware validation, predictor exclusion and benchmarking). Keep them available but optional;
  assess independently whether their ongoing maintenance is warranted.
- [ ] Avoid dataset-specific logic in core training, GUI defaults, persistence and validation.

## P1 — Correctness, reliability and compatibility

- [x] **Saved-model/data compatibility (revisions 173/173a/173b):** new model artifacts carry a privacy-safe
  source fingerprint, required SQL type-family contract, feature-role signature, fitted converter-input
  dtype families and configured/fitted text-representation roles. Prediction-only reuse verifies the
  selected source before loading saved settings; incompatible required-column/schema changes fail closed,
  while physical source-provenance changes, same-family SQL type changes and ODBC-driver changes remain
  explicit unverified warnings. Legacy model artifacts
  remain loadable with an explicit warning that these checks are unavailable. Saved Feature modelling
  candidates now record only investigated predictor SQL types and must match source identity, selected
  schema and saved converter representation before explicit application; they remain candidate evidence,
  never silently promoted to independently validated settings. Revision 173a also allows configured predictors
  that were globally all-NaN and deliberately dropped before converter fit, while unexpected fitted inputs still fail closed.
  Revision 173b treats the exact physical SQL source as provenance rather than a hard compatibility key, so
  a separate prediction table is allowed when every fitted-model-required column remains schema-compatible;
  extra unused source columns are ignored and required-column/type mismatches still fail closed.
- [x] **Prediction-only all-NULL target restoration (revision 168):** when reusing a saved model,
  keep that model's exact configured class column selectable if it exists in the chosen source table,
  even when generic training discovery excludes it because the current table has fewer than two observed
  target values. Keep the ordinary training cardinality rule unchanged and do not expose unrelated empty
  columns as class candidates.
- [x] **Windows/Voilà regression:** revision 163 was live-verified after the reclassification
  lifecycle hardening with no recurrence of `Error displaying widget`. Keep widget rendering failures
  distinct from successful model training and reopen with browser/server evidence if it returns.
- [x] **Documentation privacy guard (revision 164):** scan committed Markdown for common high-risk
  identifiers and support a local git-ignored private blocklist for organization-specific person,
  table, database and project names. Never echo private blocklist values in checker output.
- [x] **Git-history privacy audit tooling (revision 165):** extend the documentation privacy checker
  with a fail-closed `--history` mode that requires the local private blocklist and a complete non-shallow
  clone, scans reachable Markdown history plus commit messages, and never echoes matched values or source
  lines. The audit is read-only and must never rewrite history automatically.
- [x] **Pre-publication history privacy remediation (revisions 166/166a/166b):** the authoritative repository
  was audited with the local private blocklist, the affected BACKLOG/CHANGELOG history was rewritten locally with
  a verified external bundle backup, and the rewritten `dev` branch was published with an explicit
  `--force-with-lease`. Local and reachable Git history subsequently passed the private-blocklist audit; `main`
  and `prod` were left unchanged. Keep the pre-rewrite bundle private because it intentionally contains the old
  history.
- [x] **Persistent-log privacy baseline (revision 170):** keep full SQL and row-level dataset previews
  available to the active operator UI, but omit them from persistent run logs by default. Provide an explicit
  local-only `JBG_LOG_SENSITIVE_DATA=1` opt-in for trusted diagnostics, and route generated/misprediction SQL
  through the same privacy path.
- [x] **Broader application runtime-log privacy audit (revisions 172/172a/172b):** persistent application logs now
  summarize or omit local paths, configuration matrices, schema/statistics tables, class-labelled evaluation
  matrices, row-level prediction/reclassification details, split class labels/signatures, feature-importance
  column names and embedded download payloads by default. The active GUI keeps the operator-visible details,
  and `JBG_LOG_SENSITIVE_DATA=1` remains the explicit trusted-local opt-in for full persistent diagnostics.
  Revision 172a preserves valid HTML closing tags while the path redactor removes real local paths; revision 172b fixes the legacy formatted-info markup so bold status messages emit a complete `</b>` closing tag before redaction.
- [ ] **Voilà server-log privacy without breaking cleanup:** the separate launcher/server log can still contain
  local runtime paths, including the exact connection-directory path used by the guarded kernel-cleanup tooling.
  Do not redact that path until cleanup has a replacement machine-readable source of the same evidence; treat
  server logs as sensitive local diagnostics in the meantime.
- [ ] **Server/kernel lifecycle:** investigate Voilà `_xsrf` 403 during reload/shutdown and verify
  clean kernel/session termination without weakening CSRF protections.
- [ ] **Pause/Resume concurrency:** verify the revision-140 candidate-level checkpoint/resume flow
  with automatic parallelism on Windows (set the shared global cap to two workers for a reproducible test),
  including forced interruption, deterministic recovery and failure
  reporting. Investigate `A worker stopped while some jobs were given to the executor` if reproduced.
- [x] **Sampler correctness (revision 174):** numerical neighbor-based oversamplers retain the
  fold-local scaler-before-sampler contract and now validate that ordering at pipeline construction.
  Categorical SMOTE no longer relies on dtype auto-detection after ordinal conversion: SMOTE-NC receives
  explicit categorical feature indices from the fitted converter, while SMOTE-N is accepted only when
  every converted predictor is explicitly categorical. Ambiguous/missing categorical roles fail closed,
  and regression tests verify that synthetic categorical values remain discrete.
- [ ] **Compatibility smoke tests:** verify ordinary train → save → fresh-process reload → predict,
  as well as Repeat Last, after future persistence/configuration refactors.

## P2 — Efficient, understandable pipeline and feature search

- [x] **Feature modelling panel cleanup (revision 169):** remove the user-facing `Search workers`
  control and derive candidate parallelism from JBG's shared logical-CPU/global-cap policy plus the
  existing pre-search memory guard. Put `Search rows` on its own row; order pipeline controls by execution
  (`Scaling` → `Oversampling` → `Reduction` → `Model`) and keep fit/search-budget controls visually
  separate. Candidate estimators remain single-threaded and the existing spawn/checkpoint safety is retained.
  The revised layout was live-confirmed on Windows/Voilà. Revision 170 adds a persistent start diagnostic that
  records the selected worker count and its CPU/global-cap/search-budget/memory-guard constraints; pause/resume
  and forced worker-failure behavior remain to be smoke-tested with the shared global cap set to two workers.
- [ ] **Search budget and cost:** measure fit counts, elapsed times, memory and candidate-family
  costs on representative general-purpose datasets; improve Quick/Standard/Extended/Custom estimates
  and safe preflight controls. Avoid inflating default budgets without evidence.
- [ ] **High-dimensional feature search:** assess subset-size coverage, adaptive refinement limits,
  class imbalance, and reproducible nested-CV behavior for the original-column search.
- [ ] **Resource handling:** investigate costly Nystroem/MLP combinations using the revision-113
  instrumentation; distinguish estimator time from transform time, monitor memory and parallel
  oversubscription before changing global concurrency defaults.
- [ ] **Optional fold checkpoints:** evaluate cost/benefit of fold-level resume for especially
  expensive feature-search candidates; keep candidate-level atomic checkpointing intact.
- [ ] **Native CatBoost ordinary-search integration (deferred):** consider only as a general
  pipeline-local preprocessing architecture, with candidate-specific encoders, correct CV isolation,
  comparable metrics and serialization tests. Otherwise retain revision-151 separation.
- [ ] **fastai tabular classifier integration (deferred; P2):** evaluate fastai's `TabularPandas`/
  `tabular_learner` as an opt-in neural tabular candidate rather than forcing it into the existing sklearn
  pipeline contract. Prototype a JBG estimator adapter with fold-local preprocessing, deterministic validation
  splits, class/probability mapping, bounded PyTorch workers, save → fresh-process reload → predict/retrain,
  and comparable CV/scoring semantics. Prefer fastai's model/state persistence where appropriate instead of
  pickling a whole live `Learner`; do not add the dependency until the adapter and resource cost show a clear
  benefit over the existing sklearn/PyTorch models.
- [ ] **XGBoost classifier integration (deferred; P2, after reliability/resource priorities):**
  add `XGBClassifier` as an opt-in ordinary pipeline-search candidate using JBG's existing
  fold-local numeric preprocessing; subsequently consider Feature modelling integration.
  Keep estimator-level threading bounded (e.g. `n_jobs=1` under parallel search) to avoid
  oversubscription; evaluate memory/fit cost before enabling broader search budgets.
  Test class-label encoding, multiclass behavior, imbalance options, scoring/CV isolation,
  model save -> fresh-process reload -> predict, Repeat Last, and dependency compatibility.
  Defer native categorical handling and early stopping until their data/CV contracts are verified.
  Do not prioritize ahead of the remaining P1 stability and live Windows/Voila checks.

## P2 — Configuration, packaging and architecture

- [ ] **Versioned JSON configuration migration:** replace generated executable
  `autoclassconfig_*.py` output with schema-validated, versioned JSON, while preserving legacy
  reading, saved settings, Repeat Last and runtime-only SQL credentials. Plan migration before
  deleting the Python-config reader.
- [ ] **Explicit dependency support matrix:** pin/test supported Python, scikit-learn, NumPy,
  TensorFlow, Keras and SciKeras versions; maintain a reproducible Windows setup.
- [ ] **Module boundaries and paths:** reduce `sys.path` mutation and working-directory dependence;
  incrementally split oversized `JBGMeta.py` behind backward-compatible import facades so existing
  serialized models continue loading.
- [ ] **Model serializer migration (revision 167 implementation):** new model artifacts now prefer
  standard-library `pickle`; `dill` is optional and used only for legacy loading or as a compatibility fallback
  when a current model is not pickle-compatible. The loader reports an actionable optional-dependency error when
  dill is required but unavailable, and tests no longer import dill unconditionally. Complete Windows validation
  with existing legacy `.sav` files plus representative sklearn, PyTorch/skorch, Keras and Flax models before
  closing this item. Keep the trusted-artifact security boundary unchanged.
- [ ] **P2 — GUI framework architecture spike (Panel / NiceGUI):** evaluate whether moving away from
  JupyterLab/Voilà/ipywidgets is worth the migration cost. First separate configuration, job lifecycle,
  progress events, results and ML core behind stable application interfaces. Prototype only configuration,
  Script output and progress; compare lifecycle/reconnection, resources, UX, maintenance and packaging.
  No full rewrite without a demonstrable benefit over stabilized Voilà.

## P3 — Usability and diagnostics

- [x] **README cleanup (revision 171):** `README.md` now describes the current product, prerequisites/setup, configuration, startup, core workflows, model persistence, Feature modelling, runtime-log privacy, testing and troubleshooting. Revision-by-revision material and stale historical links were removed; development history remains in `CHANGELOG.md`. Generic privacy-safe examples are retained.
- [ ] Rename GUI section `Model selection` → `Machine learning pipeline selection` and label
  `Algorithm:` → `Model:`; migrate legacy saved label keys if required.
- [ ] Restrict integer class-column choices to source-table columns with at most five distinct
  non-null values, without hiding those columns from feature or unique-ID selectors; efficiently
  inspect cardinality, avoid enormous widget options and explain incompatible saved choices.
- [ ] Extend data limits to explicit counts (`1000`), percentages (`25%`) and fractions (`0.25`);
  define rounding and boundaries and preserve representation through Repeat Last/configuration.
- [ ] Improve GUI feedback and actionable error messages for estimator incompatibilities and
  candidate skips, while keeping routine progress compact.
- [ ] **Reclassification preview widget density:** replace comm-backed widgets for read-only table
  cells with a lighter HTML/table rendering approach and retain interactive widgets only where editing is
  required. Preserve the revision-163 lifecycle cleanup and fail-safe fallback; prefer reducing widget
  count over merely increasing the current frontend widget budget, and make any truncation notice explain
  the limiting factor clearly. Revision 174a moved the static style/heading/truncation decorations off
  ipywidget comm models after a live `model not found` recurrence and was live-verified without the
  frontend error. Revision 174b raises the interim bounded grid-widget budget from 240 to 640 so wide
  previews show a useful number of rows while the lighter read-only HTML/table redesign remains the
  preferred long-term fix.

## Research / optional work — not ordinary release blockers

- [ ] External independent feature-recommendation validation/approval, if product requirements
  justify reintroducing an advanced cohort workflow. Keep exploratory evidence distinct from
  approval, and require schema/semantic compatibility. Earlier related 133-stage tasks were
  superseded or narrowed by revisions 133F/133G; see archive.
- [ ] Dark Number methodology and injected-label-noise research are **parked**, not declared
  validated. Reopen only for a new, separately justified project with a clear estimand,
  ascertainment mechanism and defensible external/independent ground truth.
- [ ] Treat the regression suite chiefly as a safety net; add scenarios when they protect
  new changes or reproduce real failures, rather than expanding it routinely.

## Definition of done for future revisions

1. Small, reviewable Git patch against the immediately preceding revision.
2. Documented default behavior and backward-compatibility/migration impact.
3. Focused automated checks plus a reported status for Windows/Voilà live verification.
4. Any experimental methods are opt-in, generic, and separately reversible.
5. Update this active backlog and append the revision to CHANGELOG.md; do not erase history.

## Revision 152 inventory limitations

This is a documentation-based prioritization, **not** a full code audit or a fresh end-to-end
runtime test. Open items are drawn from the pre-152 backlog and reported prior issues. The
historical backlog contains duplicates, superseded GUI proposals and mixed completion states;
its unchecked entries should not all be read as currently actionable requirements. Future
revisions must verify current code behavior before implementing any listed item.

## Revision 153 implementation note

- [x] Added a bounded interactive misprediction preview, guarded rendering and Python exception logging.
- [ ] Confirm in live Windows/Voilà that `Error displaying widget` no longer occurs; investigate client-side failures that cannot be caught in Python.

## Revision 154 status

- [x] Parallel Feature modelling candidate batches now emit diagnostic lifecycle events and preserve worker error context.
- [ ] Verify two-worker pause, resume and forced worker failure on Windows/Voilà; inspect any joblib worker-termination warning separately (the Feature modelling executor uses `concurrent.futures`, not joblib).
- [ ] Measure time and memory of parallel searches before tuning default worker counts.

## Revision 163 status

- [x] Reclassification reruns now detach and close stale comm-backed preview widgets before a new render.
- [x] The preview is bounded by both row count and a total grid-widget budget; over-wide results fail closed to a plain-text notice without invalidating the completed ML run.
- [x] Post-run file diagnostics distinguish render, fallback and unexpected Python-side failure paths.
- [x] Repeated Train/Rerun/Repeat Last behavior was live-verified on Windows/Voilà after revision 163 with no reported `Error displaying widget` recurrence.

## Revision 164 status

- [x] Reconstructed revisions 160–162d in CHANGELOG from the supplied original patch files without copying runtime-specific identifiers.
- [x] Added a dependency-free Markdown privacy checker plus an optional local, git-ignored private blocklist.
- [x] Removed the revision-161 test dependency on the absent `docs/` directory; the architecture-spike item now lives in this active BACKLOG.
- [ ] The local private blocklist remains an operator responsibility because arbitrary person/table names cannot be recognized reliably by syntax alone.

- [ ] Revision 156a: correlate file-based startup stage markers with GUI and SQL progress; investigate Voilà kernel lifecycle separately.

- Revision 157a: Corrected negative global worker-cap handling in GridSearchCV and Dark Numbers; verify live Windows parallel execution and continue investigating resource pressure.
- [ ] **Threads versus processes (investigation only):** benchmark joblib `threading` versus `loky` / process-based CV and pipeline search on representative CPU-bound and NumPy/scikit-learn workloads; measure GIL behavior, memory sharing, nested BLAS/OpenMP parallelism, estimator thread safety, isolation after native crashes, and reproducibility. Do not change the default backend without Windows/Voila evidence.

## Revision 165 status

- [x] Added a read-only reachable-history mode to the documentation privacy checker.
- [x] History mode requires the local private blocklist and refuses shallow repositories.
- [x] Historical Markdown blobs and commit messages are scanned without printing source lines or matched private values; sensitive paths are redacted.
- [x] The audit was run on the authoritative local Git repository with the real private blocklist. After local-only untracked notes were removed, the tracked working tree passed and the remaining findings were confined to historical root `BACKLOG.md`/`CHANGELOG.md` content.

## Revision 166 status

- [x] Added a fail-closed, local-only remediation script for private blocklist matches in historical root `BACKLOG.md` and `CHANGELOG.md`; all other file paths, filenames, commit messages and built-in-pattern findings abort remediation.
- [x] Default privacy scanning in a Git repository now checks tracked Markdown only; `--include-untracked` explicitly opts into local-note scanning.
- [x] Remediation dry-run is read-only. Apply mode requires a clean, non-shallow, single-worktree repository with attached HEAD, refuses annotated tags/signed commit metadata, creates and verifies an external pre-rewrite bundle, rewrites local refs, and requires the history audit to pass afterward.
- [ ] Run revision 166 dry-run/apply on the authoritative repository, verify the rewritten history, and coordinate remote history replacement if the affected commits were already pushed. No fetch/push/force-push or local object pruning is automated.

## Revision 166b status

- [x] Signed-commit/merge-tag preflight is limited to commits that must actually receive new object IDs; unaffected signed ancestors no longer block remediation.
- [x] Untouched commits are mapped to their original object IDs without reconstruction.
- [ ] Re-run dry-run/apply on the authoritative repository. If an affected signed or merge-tag-bearing commit remains, decide explicitly whether to preserve the signed history and abandon rewrite or perform a separately reviewed signature-dropping remediation.
