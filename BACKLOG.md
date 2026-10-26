## Revision 216 - CSV prediction integration smoke

- [x] Test a trusted serialized sklearn estimator with registered CSV input and scoped prediction result publication through the existing CSV composition facade.
- [ ] Full JBG model artifact reload, common PredictionsHandler and TaskRunner execution, plus Windows/Voilà remain unverified; keep CSV GUI disabled.

## Revision 215 - CSV sklearn prediction scalars

- [x] Normalize NumPy scalar labels and confidence rates at CSV artifact publication without modifying the shared prediction handler or SQL persistence.
- [ ] Verify actual saved-estimator probabilities and end-to-end CSV execution before enabling the GUI.

## Revision 214 - CSV task sequence allowlist

- [x] Reject manual CSV task sequences that omit required prediction steps, reorder tasks, duplicate steps or invoke training/diagnostics, before executing anything.
- [ ] Validate complete saved-model prediction and class/probability mapping end to end before GUI activation; SQL task dispatch remains unchanged.

## Revision 213 - guarded shared task runner entry

- [x] Allow only prevalidated CSV prediction data-layer instances to enter the common task runner; SQL keeps its existing path.
- [ ] CSV training, regression and GUI execution remain disabled until end-to-end lifecycle and model compatibility checks are complete.

## Revision 211 - opt-in CSV prediction data-layer facade

- [x] Adapt the existing CSV prediction reader and result sink to the shared get_dataset/save_prediction_data calls without changing SQL or GUI execution.
- [ ] Wire the facade to the shared classifier only after the remaining SQL-dependent task/lifecycle paths are audited and tested end-to-end.

## Revision 210 - CSV prediction I/O composition

- [x] Add an opt-in boundary that pairs a saved CSV model reader with a scoped file prediction writer after identity and column-role checks.
- [ ] Connect the shared task runner only after removing SQL-only assumptions; keep CSV GUI execution disabled until end-to-end tests pass.

## Revision 207 – scoped CSV prediction result sink

- [x] Implement SQL-independent CSV prediction persistence matching the existing data-layer save method; use private scoped row artifacts and a manifest published last.
- [ ] Connect CSV runtime and results sink to task runner only after removing remaining SQL assumptions and adapting prediction ID coercion.
- [ ] Review safe output logging: avoid row contents, base64 download links and SQL-identifying fields in shared logs.
- [ ] Add run history/retrieval contract and authentication before multi-user deployment.

## Revision 205a

- Fixed the welcome logger handling of the internal data-source identity. Continue validating the complete SQL training/reload workflow in the Windows environment.

## Revision 199 – shared JSON configuration (migration phase 1)
- [x] Add optional backend identity to the established JBG JSON configuration; legacy SQL JSON files remain valid.
- [x] Keep dataset identity separate from model column roles; retain source identity when making model-safe configurations.
- [ ] Integrate shared JBG JSON into CSV model save/load; retain historical CSV configuration artifacts and a legacy reader until safely migrated.
- [x] Remove duplicate CSV GUI configuration controls (201), retaining historic configuration artifacts and readers.
- [ ] Retire the CSV-only serializer only after safe migration, model-artifact integration and equivalence tests.
- [ ] Preserve SQL training and existing saved-model compatibility; CSV training remains gated.

- [x] **Revision 198:** GUI save/list/load for scoped CSV model role and dialect configurations; execution remains disabled. Training and preprocessing settings remain a future requirement.
- [x] **Revision 197:** Persist a validated CSV delimiter, quote character and encoding per private model configuration, and use them when preparing a run. Older configurations remain supported. Preprocessing/training options and GUI wiring remain open.
- [x] **Revision 196:** Validate stored CSV bytes against private model roles and link configuration ID to prepared run manifest; CSV ML execution remains disabled.
- [x] **Revision 195:** Scoped immutable CSV model-role configuration artifacts with independent configuration IDs; reuse one registered dataset across different column-role assignments. This is planning-only: validate column existence/preprocessing and attach configuration to run manifests before enabling CSV training.
- [ ] Connect configuration identity to run publication and enforce actual CSV schema checks, model compatibility, and secure file result persistence before CSV ML activation.
- [x] **Revision 194b:** Preserve the saved dataset identity and filename on reload; avoid accidental duplicate registration. Explicit dataset version creation remains future work.
## Revision 194a — erase saved CSV

- [x] Explicit two-click deletion of a scoped registered CSV and its display metadata; unrelated datasets preserved.
- [ ] Web authorization, retention and concurrent multi-user deletion policies remain prerequisites for HTTP exposure.

- [x] **Revision 194:** Prepare SQL-independent CSV run artifact publication using opaque run IDs, scoped immutable manifest and optional aggregate metrics; no ML execution or GUI activation.

- [x] **Revision 193:** Human-readable scoped saved-CSV labels (uploaded filename + UTC registration time) and SQL-independent CSV run context; CSV ML execution remains disabled. Older entries retain opaque-ID fallback.
- [x] **Revision 192:** List and stage previously registered CSV datasets in the local project without re-upload; require explicit revalidation. No web authentication or CSV execution.
- [x] Revision 190: Explicit local CSV registration from the GUI after selecting model and roles; stable local project scope, no SQL/ML activation. Web authorization and retrieval remain open.
- [x] **Revision 189:** Opt-in validated CSV dataset registration into scoped file artifacts; no GUI or training activation. Windows tests pending.
- [ ] Bind a trusted scope and persisted CSV dataset references to the GUI; implement source-aware training and file-based run result publication before enabling CSV execution.
- Revision 188f: CSV now follows Data source -> Models (New model) -> Dataset columns; SQL model choices are restored separately, CSV execution remains disabled.
- Revision 188e: Clear Models selection when switching data sources; SQL/CSV execution separation remains in place.
# JBG AutoClassification — Active BACKLOG (Revision 188c)

This is the **active**, prioritized maintenance roadmap for the general-purpose JBG ML framework.
The complete historical backlog (including completed revisions and research experiments)
is preserved without deletion in [`docs/backlog_archive_pre_152.md`](docs/backlog_archive_pre_152.md).
This file contains only currently relevant product work plus concise completion status for recent
revisions. Historical/superseded backlog material belongs in the archive/CHANGELOG rather than in the
active priority queue. A checked box indicates implemented work; live verification is stated explicitly
where it matters.

- [x] **Revision 191:** Remove redundant Source label, position run history after Dataset columns, and retire Regr. Suite from visible actions; regression internals remain unchanged.

## Decisions and scope

- [x] **Revision 188d:** Show original CSV filename instead of temporary server path; enable CSV Dataset columns roles only after validation, keep SQL role editability on source switch and CSV execution disabled. Live Windows/Voilà verification pending.


- [x] **Revision 188c:** Conservative CSV delimiter and text-qualifier detection on upload, with manual override and private status messages. Windows/Voilà verification pending.


- [x] **Revision 188b:** Frame the CSV settings panel consistently with the SQL Server settings panel; presentation-only change.

- [x] **Revision 188a:** Browser CSV picker with temporary server-side inspection path displayed alongside; CSV-only filename filter and validation. Full upload lifecycle/retention and service-backed dataset storage remain future work.


- [x] **Revision 188:** Populate shared Dataset columns from validated CSV schema while preserving SQL role selections and preventing CSV-view SQL execution. CSV modelling remains unavailable; live Voilà testing pending.


- [x] **Revision 186:** Define opaque tenant/project/run/dataset/artifact identifiers and backend-neutral read/write interfaces for future web access; no storage implementation or authentication yet.
- [x] **Revision 187:** Implement a scoped, immutable local file artifact repository with atomic no-overwrite publication, logical lookup and tests.
- [ ] Add retention/cleanup policy, crash recovery, full run/dataset repositories, and an authenticated per-request authorization layer before web exposure; identifiers alone are not authorization.
- [ ] Expose user-authorized dataset and run catalogs, preview and downloads through future API without exposing host paths.

- [x] **Revision 185:** SQL-independent CSV column-role discovery and fail-closed role contract; not yet bound to Dataset columns GUI or model execution.
- [ ] Bind the CSV schema to shared Dataset columns widgets, with source-aware observers and SQL execution guards; test in Voilà before enabling CSV modelling.


- [x] **Revision 184d:** Fix Data source section construction with supported Box type; Windows/Voilà verification pending.

- [x] **Revision 184c:** Exclusive SQL/CSV settings display under Data source; class and ID roles stay in Dataset columns. CSV remains inspection only. Windows/Voilà verification pending.
- [ ] Bind CSV headers to shared Dataset columns roles before allowing CSV training; do not reuse SQL selections silently.


- [x] **Revision 184a:** Address CSV pandas parser dtype compatibility; Windows verification pending.

- [x] **Revision 184:** CSV inspection controls in GUI (path, delimiter, encoding, ID/class roles, private validation counts); SQL-only execution remains unchanged. Live Windows/Voilà verification pending.
- [ ] Connect CSV to the shared read contract and classifier execution, including source identity, persistence checks and prediction output, before enabling CSV training.


- [x] The prior private-dataset investigation is concluded at the user's request. No more dataset-specific experiments
  are planned in the main product backlog. Historical findings remain in the archive.
- [x] Native CatBoost is retained for **Feature modelling** and is not an ordinary exhaustive-pipeline
  candidate. Native CatBoost integration into ordinary search remains a separately scoped future refactor.
- [x] Revisions 147–150 introduced generally applicable, opt-in diagnostics (threshold/OOF,
  group-aware validation, predictor exclusion and benchmarking). Keep them available but optional;
  assess independently whether their ongoing maintenance is warranted.
- [ ] Avoid dataset-specific logic in core training, GUI defaults, persistence and validation.

## P1 — Correctness, reliability and compatibility

- [x] **Spot-check output resilience (revision 177):** retain rich output/complete CSV; add a compact plain-text top-ten and repeat the selected-pipeline metric/score after model save. No general Voilà/WebSocket reconnect work. Live Windows/Voilà validation pending.

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
- [x] **Revision 179 worker-failure regression:** test that a failed parallel candidate future reports failure and does not commit the failed candidate. This is a deterministic unit test, not a Windows process-crash simulation.
- [x] **Lifecycle-counter accuracy (revision 181):** GUI-only transitions omit unavailable candidate/fit counts instead of displaying misleading zeros; backend-provided counters remain unchanged. Windows/Voilà verification pending.
- [ ] **Pause/Resume concurrency:** Revisions 180/180a add privacy-minimal GUI lifecycle logging to the actual JBG application log (pause request, checkpoint pause, resume, completion/failure); the operator manually confirmed a pause/refresh/resume-to-result flow, but the specific two-worker Windows process lifecycle and forced-crash recovery remain unverified.
   verify the revision-140 candidate-level checkpoint/resume flow
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
- [ ] **Search budget and cost:** revision 178 labels uncalibrated GridSearch timing estimates as uncertain planning references after a cold-start underestimate in the revision-177 smoke test; measure fit counts, elapsed times, memory and candidate-family
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
**dual SQL Server/CSV I/O abstraction → source/package and `JBGMeta.py` refactor → Azure FastAPI GUI spike**.
Treat this as the default dependency order unless a blocking production issue requires otherwise: first make SQL
Server and CSV explicit peer data-source modes behind the same application/data interfaces, then modularize the
backend behind stable application/data/job boundaries, and only then couple the new web GUI prototype to those
interfaces. SQL Server remains a fully supported first-class source; CSV removes the current hard dependency on
database access for every workflow rather than replacing SQL Server.

- [x] **Versioned JSON configuration migration (revision 176):** generated runtime configuration is now
  schema-validated JSON only. Python config generation/import has been removed; schema-v1 transitional JSON
  and pre-176 model artifacts are intentionally unsupported. The schema serializes every current configuration
  dataclass field, keeps SQL passwords runtime-only, supports CLI loading without executing code, and preserves
  GUI Repeat Last through its separate JSON snapshot. Old Python configs/models should be deleted/retrained.
- [x] **Source selection foundation (revision 183):** explicit typed SQL/CSV construction boundary, SQL default, lazy SQL imports, isolated regression tests. GUI routing and common read/write contracts are still pending.
- [x] **CSV input foundation (revision 182):** add a standalone, explicit CSV reader with validated ID/class roles, configurable delimiter/encoding and string-preserving import; SQL runtime untouched. GUI selection, schema/type inference, source identity/persistence, and output export remain open.
- [ ] **Dual SQL Server/CSV data source and result export:** make the physical source/sink an explicit runtime
  choice (`SQL Server` or `CSV`) behind a common data-layer/application interface. Keep SQL Server as a fully
  supported first-class mode while adding CSV input plus CSV prediction/result export for local development,
  demos, reproducible tests and future web-service file-upload workflows. Define schema/type inference, class/ID
  column selection, encoding/delimiter handling, output naming/overwrite behavior and compatibility metadata for
  file-backed runs; keep SQL and CSV source identities explicit and never silently cross them. For SQL Server,
  separate shared deployment configuration from user/session-specific connection choices and credentials so the
  future web service does not depend on per-user `.env` files. Local `.env` may remain a development convenience;
  the web architecture must instead expose explicit request/session/application configuration and an appropriate
  secret/credential mechanism without persisting plaintext credentials in JSON, model artifacts or logs. Treat
  this as the first enabling architecture step before the larger source/package refactor and FastAPI GUI spike.
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
  stable application interfaces. The web layer must model data-source selection and user/session-specific settings
  explicitly rather than assuming each user can provide a private `.env`; `.env` remains only a local-development
  or shared deployment mechanism. Then build a deliberately narrow FastAPI prototype covering configuration,
  SQL Server/CSV source selection, CSV upload/download, Script output, progress, long-running job control/reconnection
  and representative result presentation. Evaluate multi-user/session isolation, credential/secret handling,
  asynchronous/background job handling, streaming progress (for example SSE/WebSocket), model/config persistence,
  security, deployment/packaging and maintainability before deciding on a full migration. Panel/NiceGUI may be retained as secondary comparison candidates only if useful; do not
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

### Follow-up after revision 200
- Restore all training and preprocessing options in the CSV GUI using source-safe widget updates.
- Migrate old role-only artifacts and remove redundant fields/adapters only after verified migration.
- Verify CSV ML execution and separate result persistence end to end.

- After 202: Route CSV run execution through the shared ML pipeline only after model persistence and file-backed result output are ready.

- [x] **203:** Opt-in shared DataFrame conversion for SQL/CSV row providers, no changes to training or SQL query behavior.
- [ ] **204:** Compare SQL/CSV type/null semantics and benchmark memory before any coercion or chunking.

- [x] **204:** Added opt-in aggregate SQL/CSV DataFrame diagnostics for null versus empty-string counts, dtype distributions and pandas deep memory estimates; no type coercion, ML, SQL or CSV retrieval changes.
- [ ] **205+:** Compare representative SQL/CSV source samples under the same roles and row-selection conditions before defining any explicit type/null normalization. Benchmark peak RSS and throughput in a controlled environment if performance work is justified; pandas `memory_usage(deep=True)` does not measure process peak memory.

- [x] **205:** Verify existing file-based model persistence and protect CSV model reloading using dataset/artifact identity, without changing legacy SQL loading.
- [ ] **206+:** Integrate source-aware model selection into CSV GUI and shared training pipeline; preserve scoped authorization and verify sidecar handling and output persistence.

## Revision 208 - prediction identifiers

- [x] Preserve CSV string IDs (including leading zeros) in shared result construction; keep legacy SQL integer keys.
- [ ] Before enabling CSV execution, integration-test prediction payloads, class label serialization and SQL result-table compatibility.
- [ ] Review SQL result-table key types separately with backwards compatibility.

## Revision 209 - CSV prediction payload preflight

- [x] Validate model label membership, unique class labels and optional probability-vector length/range before publishing CSV prediction artifacts.
- [ ] Wire shared classifier and CSV sink only after run lifecycle and result retrieval have been integrated and tested end to end.
- [ ] Verify real estimator probability serialization and class order during first gated CSV training/prediction smoke tests.

## Revision 212 - model reload source boundary

- [x] Preserve SQL schema compatibility checks on SQL reload; CSV reload requires the prepared scoped CSV data-layer facade and avoids SQL metadata calls.
- [ ] Verify end-to-end CSV task execution and artifact publication before enabling GUI execution.
