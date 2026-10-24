# JBG AutoClassification — Active BACKLOG (Revision 176c)

This is the **active**, prioritized maintenance roadmap for the general-purpose JBG ML framework.
The complete historical backlog (including completed revisions and research experiments)
is preserved without deletion in [`docs/backlog_archive_pre_152.md`](docs/backlog_archive_pre_152.md).
This file contains only currently relevant product work plus concise completion status for recent
revisions. Historical/superseded backlog material belongs in the archive/CHANGELOG rather than in the
active priority queue. A checked box indicates implemented work; live verification is stated explicitly
where it matters.

## Decisions and scope

- [x] The prior private-dataset investigation is concluded at the user's request. No more dataset-specific experiments
  are planned in the main product backlog. Historical findings remain in the archive.
- [x] Native CatBoost is retained for **Feature modelling** and is not an ordinary exhaustive-pipeline
  candidate. Native CatBoost integration into ordinary search remains a separately scoped future refactor.
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
  explicit unverified warnings. Revision 176 intentionally ends pre-176 model-artifact support; old models
  must be retrained. Saved Feature modelling
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
- [ ] **Legacy Voilà runtime containment:** verify whether reload/shutdown `_xsrf` 403 responses actually
  leave orphan kernels/processes or create resource leakage. Keep the current runtime stable without weakening
  CSRF protections, avoid further Voilà-specific architecture investment, and keep the separate launcher/server
  log treated as sensitive local diagnostics until cleanup no longer depends on path evidence.
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
- [ ] **Revision-176 compatibility smoke tests:** with old configs/models removed, verify fresh JSON
  config generation/load, GUI Repeat Last, and ordinary train → save → restart/fresh-process reload →
  predict using a newly trained current artifact.
- [x] **Development-only SQL credential prefill (revision 176b):** production/default GUI startup keeps SQL
  username/password fields empty. Trusted local development may explicitly opt in with
  `JBG_GUI_PREFILL_SQL_CREDENTIALS=1` and developer-supplied `JBG_SQL_USERNAME`/`JBG_SQL_PASSWORD`
  values in the untracked environment. JBG must never generate, persist or log the credential values.

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
- [x] **Imbalance-safe default scoring (revision 176c):** use Matthews Correlation Coefficient
  (MCC) as the default spot-check/model-selection metric in both backend configuration and the
  GUI. Keep Accuracy and all existing metrics available as explicit user choices, and migrate
  legacy local GUI-default settings from Accuracy to MCC without overriding restored run/config values.
- [ ] **Robust multi-metric model-selection objective:** add an optional composite selection mode
  alongside single-metric scoring. Rank candidates independently per selected metric and aggregate
  ranks rather than averaging raw metric values with incompatible scales/baselines. Start with MCC,
  Balanced Accuracy and ROC-AUC where estimator/target semantics support them; gate probability-based
  metrics explicitly, define multiclass averaging semantics, and retain single-metric selection.
- [ ] **High-dimensional feature search:** assess subset-size coverage, adaptive refinement limits,
  class imbalance, and reproducible nested-CV behavior for the original-column search.
- [ ] **Resource handling:** investigate costly Nystroem/MLP combinations using the existing
  instrumentation; distinguish estimator time from transform time, monitor memory and parallel
  oversubscription, and benchmark joblib threading versus process-based execution on representative
  workloads before changing global concurrency defaults.
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

**Architecture execution order:** after direct P1 defects and stability issues, prefer the sequence
**CSV/file I/O abstraction → source/package and `JBGMeta.py` refactor → Azure FastAPI GUI spike**.
Treat this as the default dependency order unless a blocking production issue requires otherwise: the CSV layer
should first remove SQL Server as a hard runtime assumption, then the backend should be modularized behind stable
application/data/job interfaces, and only then should the new web GUI prototype be coupled to those interfaces.

- [x] **Versioned JSON configuration migration (revision 176):** generated runtime configuration is now
  schema-validated JSON only. Python config generation/import has been removed; schema-v1 transitional JSON
  and pre-176 model artifacts are intentionally unsupported. The schema serializes every current configuration
  dataclass field, keeps SQL passwords runtime-only, supports CLI loading without executing code, and preserves
  GUI Repeat Last through its separate JSON snapshot. Old Python configs/models should be deleted/retrained.
- [ ] **CSV data source and result export (development-friendly I/O):** add a file-based execution mode
  that can load input data from CSV instead of SQL Server and export prediction/result datasets back to CSV.
  Keep the ML/configuration pipeline independent of the physical source/sink behind the existing data-layer
  boundary so SQL Server remains the production path while CSV can support local development, demos and
  reproducible tests without database access. Define schema/type inference, class/ID column selection,
  encoding/delimiter handling, output naming/overwrite behavior and compatibility metadata for file-backed
  runs; do not persist secrets or silently mix SQL and CSV source identities. Treat this as the first enabling
  architecture step before the larger source/package refactor and FastAPI GUI spike.
- [ ] **Explicit dependency support matrix:** pin/test supported Python, scikit-learn, NumPy,
  TensorFlow, Keras and SciKeras versions; maintain a reproducible Windows setup.
- [ ] **Source/package structure, module boundaries and paths:** reorganize the flat `src/` layout into
  cohesive subpackages by responsibility (for example configuration, data I/O, modelling/search, persistence,
  GUI/application orchestration and shared utilities) so ownership and dependency direction are easier to follow.
  Split oversized modules where they mix unrelated responsibilities, with `JBGMeta.py` as the primary target:
  first identify and extract stable domain/configuration/runtime interfaces, then move implementations behind
  those interfaces rather than merely relocating the existing monolith. Reduce `sys.path` mutation and
  working-directory dependence at the same time, preserve stable public import facades during the migration,
  and add focused tests for import direction/cycles and package layout. Revision 176 removes the old-model
  compatibility constraint, so future refactors no longer need to preserve pre-176 serialized module paths.
  Complete the main boundary cleanup before the FastAPI GUI spike so the new frontend targets stable application
  interfaces instead of today's internal module coupling.
- [x] **Current model serializer contract (revisions 167/176):** current artifacts prefer standard-library
  `pickle`; `dill` is an optional fallback only when a newly generated current artifact contains state that
  pickle cannot serialize. Revision 176 increments the artifact envelope and rejects pre-176 payloads rather
  than carrying legacy readers. Keep the trusted-artifact security boundary unchanged and smoke-test representative
  sklearn, PyTorch/skorch, Keras and Flax models as those families are exercised.
- [ ] **High-priority GUI replacement architecture track — Azure FastAPI first:** after direct defects/stability,
  the CSV/file I/O abstraction and the main source/`JBGMeta.py` boundary refactor, prioritize evaluating a move
  away from JupyterLab/Voilà/ipywidgets. The first technology candidate must be the organisation's already
  established Azure web-service stack: FastAPI/Python
  backend with HTML/JavaScript frontend, following the existing organisational visual/design standard and, where
  applicable, established deployment, authentication, observability and operational conventions. Treat reuse of
  an already accepted platform and support model as a major decision criterion, not just raw GUI capability.
  First separate configuration, data-source access, job lifecycle, progress events, results and ML core behind
  stable application interfaces. Then build a deliberately narrow FastAPI prototype covering configuration,
  Script output, progress, long-running job control/reconnection and representative result presentation. Evaluate
  multi-user/session isolation, asynchronous/background job handling, streaming progress (for example SSE/WebSocket),
  file/CSV workflows, model/config persistence, security, deployment/packaging and maintainability before deciding
  on a full migration. Panel/NiceGUI may be retained as secondary comparison candidates only if useful; do not
  undertake a full rewrite until the FastAPI-based prototype demonstrates a clear migration path and operational
  benefit over the stabilized Voilà application.

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
2. Documented default behavior and any migration/breaking-change impact.
3. Focused automated checks plus a reported status for Windows/Voilà live verification.
4. Any experimental methods are opt-in, generic, and separately reversible.
5. Update this active backlog and append the revision to CHANGELOG.md; do not erase history.

## Recent live-verification notes

- [x] Reclassification rendering is live-verified through revisions 174a/174b: static decorations no longer create
  unnecessary comm models, `model not found` did not recur, and the wider bounded preview budget is useful.
- [x] Revision 175 application/build identity is live-verified in the GUI.
