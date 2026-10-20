- [ ] Revision 159 follow-up: identify the Voilà/frontend version responsible for missing XSRF headers on shutdown; compare upstream issue #1458; verify actual live-kernel counts separately before implementing cleanup. Never bypass CSRF or kill running training jobs.
- [ ] Revision 158: review Voilà server runtime versions and shutdown 403 diagnostics; test repeated reload/disconnect and Script output widget lifecycle on Windows. Do not disable XSRF; source-level diagnostics alone do not fix the browser shutdown request.
- [ ] After revision 157: validate memory guard and process-tree heartbeat on Windows/Voilà; assess GaussianProcessClassifier large-data cost and separate Feature modelling concurrency; continue investigating missing Script output and Voilà kernel shutdown _xsrf failures.
## Revision 155 — JSON export and Feature modelling button help

Transitional versioned JSON export with legacy Python config compatibility; corrected action tooltips. See `docs/configuration_155.md`.

# JBG AutoClassification — Active BACKLOG (Revision 152)

This is the **active**, prioritized maintenance roadmap for the general-purpose JBG ML framework.
The complete historical backlog (including completed revisions and research experiments)
is preserved without deletion in [`docs/backlog_archive_pre_152.md`](docs/backlog_archive_pre_152.md).
Revision 152 only reorganizes documentation. It does not implement, close, or re-validate
runtime features. A checked box below is a **documentation disposition**, not new test evidence.

## Decisions and scope

- [x] <private-identifier> investigation is concluded at the user's request. No more <private-identifier>-specific experiments
  are planned in the main product backlog. Historical findings remain in the archive.
- [x] Revision 151 policy: native CatBoost is retained for **Feature modelling**, and is not an
  ordinary exhaustive-pipeline candidate. Legacy artifact loading is intended to remain supported;
  native CatBoost integration into ordinary search is a separately scoped future refactor.
- [x] Revisions 147–150 introduced generally applicable, opt-in diagnostics (threshold/OOF,
  group-aware validation, predictor exclusion and benchmarking). Keep them available but optional;
  assess independently whether their ongoing maintenance is warranted.
- [ ] Avoid dataset-specific logic in core training, GUI defaults, persistence and validation.

## P1 — Correctness, reliability and compatibility

- [ ] **Saved-model/data compatibility:** enforce source SQL schema, semantic feature definitions,
  column/type compatibility and converter-setting equivalence when applying saved feature
  recommendations or loading/reusing model artifacts. Explicitly describe unverified assumptions;
  never silently promote exploratory recommendations to validated status.
- [ ] **Windows/Voilà regression:** reproduce and diagnose intermittent `Error displaying widget`
  after the most-mispredicted reclassification table; capture exceptions and payload/size details.
  Distinguish widget rendering errors from model-training failures.
- [ ] **Server/kernel lifecycle:** investigate Voilà `_xsrf` 403 during reload/shutdown and verify
  clean kernel/session termination without weakening CSRF protections.
- [ ] **Pause/Resume concurrency:** verify the revision-140 candidate-level checkpoint/resume flow
  with two workers on Windows, including forced interruption, deterministic recovery and failure
  reporting. Investigate `A worker stopped while some jobs were given to the executor` if reproduced.
- [ ] **Numerical sampler correctness:** runtime-verify fold-local scaler-before-numerical-SMOTE
  behavior (revision 122), and separately examine categorical SMOTE-NC/SMOTE-N dtype semantics;
  do not interpolate categorical codes as continuous values.
- [ ] **Compatibility smoke tests:** verify ordinary train → save → fresh-process reload → predict,
  as well as Repeat Last, after future persistence/configuration refactors.

## P2 — Efficient, understandable pipeline and feature search

- [ ] **Search budget and cost:** measure fit counts, elapsed times, memory and candidate-family
  costs on representative *non-<private-identifier>* datasets; improve Quick/Standard/Extended/Custom estimates
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
- [ ] **Serialization review:** assess whether `dill` dependence can be reduced with module-level
  callables, without changing the trusted-artifact security boundary.

## P3 — Usability and diagnostics

- [ ] Rename GUI section `Model selection` → `Machine learning pipeline selection` and label
  `Algorithm:` → `Model:`; migrate legacy saved label keys if required.
- [ ] Restrict integer class-column choices to source-table columns with at most five distinct
  non-null values, without hiding those columns from feature or unique-ID selectors; efficiently
  inspect cardinality, avoid enormous widget options and explain incompatible saved choices.
- [ ] Extend data limits to explicit counts (`1000`), percentages (`25%`) and fractions (`0.25`);
  define rounding and boundaries and preserve representation through Repeat Last/configuration.
- [ ] Improve GUI feedback and actionable error messages for estimator incompatibilities and
  candidate skips, while keeping routine progress compact.

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

- [ ] Revision 156a: correlate file-based startup stage markers with GUI and SQL progress; investigate Voilà kernel lifecycle separately.

- Revision 157a: Corrected negative global worker-cap handling in GridSearchCV and Dark Numbers; verify live Windows parallel execution and continue investigating resource pressure.
- [ ] **Threads versus processes (investigation only):** benchmark joblib `threading` versus `loky` / process-based CV and pipeline search on representative CPU-bound and NumPy/scikit-learn workloads; measure GIL behavior, memory sharing, nested BLAS/OpenMP parallelism, estimator thread safety, isolation after native crashes, and reproducibility. Do not change the default backend without Windows/Voila evidence.
