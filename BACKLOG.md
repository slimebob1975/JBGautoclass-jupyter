# JBG AutoClassification - Active BACKLOG (revision 272)

This is the **current, agreed development order**. The detailed revision notes below
are retained for traceability, not as a second active priority queue. Earlier
instructions saying that all CSV execution is blocked are historical; **the
complete configurable CSV ML pipeline is not yet enabled in the GUI**, even though
restricted CSV training and saved-model prediction already work.

## Confirmed baseline

- [x] CSV dataset registration, source-specific validation, restricted native
  classifier training, saved-model inspection/reuse and prediction through the
  shared ML engine (257-266). This does **not** demonstrate feature parity with SQL.
- [x] Scoped CSV prediction export and guarded deletion of CSV models, prediction
  runs and datasets (267-269c). These controls do not imply SQL model erasure works.
- [x] Full Windows regression checkpoint after 270f1: 761 unittest cases
  (758 passed, 3 skipped) plus 70 pytest cases passed. Three skipped symlink
  checks remain unverified on this host.
- [x] Revision 272: fast fail-closed regression check at normal local launch;
  separate explicit full regression before commit/release. Both modes passed
  in the local Windows environment. Do not silently weaken either gate.
- [x] Version 0.19.0 and the 271-series status/roadmap update were prepared;
  individual commit/push status must be checked in Git, not inferred here.

## Binding development sequence (decided after revision 272)

**Phase 1 - Complete and verify the ML engine for both data sources (P1).**

- [ ] **273a: CSV GUI capability audit.** Build a source-by-source capability
  matrix: already working, GUI-blocked, adapter work needed, and unverified.
  Trace controls and execution gates through the shared configuration,
  AutomaticClassifier/TaskRunner, converters, handlers, persistence and results.
  Document source-specific restrictions and define safe, incremental patches.
- [ ] **273b+: Enable the complete CSV ML pipeline in the current GUI.** Aim for
  functional parity with supported SQL features where semantically applicable:
  numeric/categorical/text processing, preprocessing, feature selection and RFE,
  class imbalance/sampling, Dark Numbers, algorithm and hyperparameter choice,
  parallelism, validation/CV, model comparison, feature importance, diagnostics,
  save/reload/predict and results/export. Do not assume that currently visible
  controls actually reach the CSV runtime. Preserve one shared ML implementation;
  use source-aware adapters rather than copying SQL-specific code or bypassing
  identity, compatibility, scope and publication checks.
- [ ] **SQL regression and known defect.** Use the authorized, privacy-safe
  protocol in `tests/SQL_REGRESSION_PROTOCOL_248.md` to confirm train -> save ->
  fresh-process reload -> predict -> result persistence, including error handling.
  **SQL `Erase models` is reported not to work and needs diagnosis and repair.**
  Do not change SQL behavior merely to accommodate CSV. Record findings without
  private schema/table names or row-level data.
- [ ] **End-to-end acceptance for both sources.** Test representative numeric,
  categorical and text workloads, supported feature-selection and validation
  options, saved-model compatibility, failures/retries, source switching and
  result publication from the current Windows/Voila GUI. Tests and synthetic
  fixtures complement, but do not replace, authorized SQL live acceptance.
  Treat parity as an acceptance objective, not a claim of current completion.

**Phase 2 - Major backend refactor, after functional acceptance (P1 architecture).**

- [ ] Refactor the oversized metadata/algorithm registry, especially
  `JBGMeta.py`, and other mixed-responsibility modules incrementally. Separate
  algorithm registration, capability metadata, validated parameter definitions
  and GUI presentation metadata where justified by real dependencies.
- [ ] Establish cohesive package boundaries, stable import/public facades,
  dependency direction, runtime configuration and model serialization contracts.
  First add behavioral/compatibility tests; refactor in small reversible patches
  and verify saved current-model reload. Avoid a disruptive all-at-once rewrite.
- [ ] Preserve SQL and CSV as first-class peers in the same ML engine; retain
  data-source-specific I/O and source identity without exposing local storage
  paths or credentials to future clients.

**Phase 3 - GUI-independent application layer, BEFORE building the new GUI.**

- [ ] Define a Python application/service interface independent of Voilà,
  ipywidgets and FastAPI: source/config validation, capabilities/parameters,
  job submission and lifecycle, progress/events, model/run catalogs, prediction,
  results/export and controlled deletion. Implement testable typed contracts
  and adapters, with no ML business rules in GUI callbacks or HTTP handlers.
- [ ] Keep long-running ML jobs decoupled from web request lifetimes. Define
  cancellation, failure handling, concurrency, resumability and persistence
  semantics before selecting transport or deployment technology.
- [ ] Migrate one complete vertical workflow to this layer before broadening it;
  verify that the current GUI and a future HTTP adapter can call the same API.

**Phase 4 - New web GUI (after application-layer contracts are stable).**

- [ ] Use **FastAPI** as HTTP adapter above the application layer, not as the
  ML business-logic layer. Build the HTML/CSS/JavaScript interface using the
  organization's existing Azure web applications, sample code and visual
  conventions when those references are supplied.
- [ ] Start with source selection, validation, configuration, train/status,
  results and saved-model reuse; expand incrementally after acceptance. Keep
  the existing Voilà GUI available during migration until replacement is proven.
- [ ] Allow a future GUI to be replaced without modifying ML engine or application
  contracts. Avoid hidden GUI-specific state as a source of truth.

**Phase 5 - Azure deployment, identity and multi-user security (deferred).**

- [ ] Use **Microsoft Entra ID** for authentication when the Azure deployment
  phase begins. Azure App Service authentication alone does not enforce
  per-project/per-artifact authorization: design server-side authorization,
  tenant/project isolation, credential handling, session context, retention,
  concurrency and audit rules before any multi-user rollout.
- [ ] Validate Azure deployment, secrets and storage policies against actual
  hosting architecture. Do not add speculative cloud coupling to today's
  standalone ML engine.

## Cross-cutting rules and deferred work

- [ ] Keep the fast local start gate and run full unittest + pytest regression
  before relevant commits/releases. Run focused tests for each small patch;
  retain clear failures and explicit skips.
- [ ] Scan tracked Markdown and proposed commits with the documentation privacy
  checker and local private blocklist. Do not commit runtime logs, datasets,
  SQL text, credentials, personal identifiers or local launcher settings.
- [ ] Keep optional modelling research, GUI cosmetics, performance tuning and
  new estimator families (such as XGBoost/fastai) in the detailed backlog, but
  do not let them displace the agreed ML-parity -> refactor -> service -> GUI order
  without an explicit priority decision.
- [ ] Revisit automatic project naming and the lightweight reclassification
  table; these remain usability follow-ups, not prerequisites for new GUI work.

## Historical revision notes and detailed roadmap

The notes below preserve prior decisions and open questions as recorded at the
respective revisions. Their unchecked boxes are **not automatically active**;
compare them against the binding phases above before scheduling work. In
particular, older CSV-execution prohibitions, full-suite-at-every-start requests
and descriptions of unfinished CSV model deletion have been superseded.

- [ ] CSV Models: add an "Erase model" button beside the Models dropdown, following the CSV "Erase saved" confirmation pattern. Delete only the selected saved checkpoint and its registration, preserve dataset and other models, refresh options immediately, disable for New model, reject wrong project/dataset, never touch SQL artifacts; add synthetic regression tests.
- [ ] Auto-name projects when the user leaves the default project name unchanged and presses Dataset actions / Continue. Derive a sanitized, descriptive name from the selected dataset/source (CSV or SQL) and class role; preserve user-entered names and avoid accidental collisions or silent overwrites of model/configuration artifacts. Use only synthetic names in tests and documentation.
- [ ] Local startup test gate: run the full Python test suite before starting JBG in a local installation; report passed, failed and skipped counts separately. Abort startup on required test failures, and ensure tests use isolated synthetic fixtures without production SQL connections or modifying persisted models. Define an explicit opt-in or scheduled mode for slow ML tests if startup latency becomes unacceptable.
- [x] Revision 256: Fail closed on CSV retraining when the saved-model checkpoint cannot be reloaded; preserve the SQL in-memory fallback. Four focused Windows tests passed.
- [x] Revision 255: Verify fitted synthetic CSV-model save/reload/predict equivalence and guard CSV save failures. Four focused Windows tests passed; this is not full native handler training.
- [ ] Revision 257+: Exercise full native CSV fitting through AutomaticClassifier/TaskRunner with real handlers, scoped model save/reload, and SQL access interception. Keep CSV GUI execution blocked until verified.
- [ ] Post-254: Run actual Iris-like native CSV ML fit with artifact save/reload in a scoped temporary directory, verify SQL interception and no partial ready model on failure before unlocking GUI.
- [ ] CSV: run full native AutomaticClassifier training with a scoped synthetic Iris fixture, verify model artifact persistence/reload and intercept SQL access before enabling GUI (post-253).
- 252: Introduce an opt-in, read-only registered CSV training data layer and a strict CSV-training TaskRunner dispatch gate. Still required before live CSV: native AutomaticClassifier bootstrap, full real model training/saving, SQL-blocked end-to-end checks and GUI source/model controls.
- 251: Keep CSV artifact timestamps in UTC; show saved CSV catalog labels in local time by default, with an explicit timezone hook for future browser-based UI. CSV training activation still requires full production bootstrap and tests.
- 250: CSV numeric-coded class labels need an explicit, validated class-role override; automatic candidates exclude numeric measurements. Preserve semantic action states during HTML/FastAPI migration. Live CSV execution remains blocked.
- 249 SQL REGRESSION: Verify successful persisted SQL prediction against eligible unlabeled rows, including database write counts and failure reporting. Fix GUI saved-model Continue/Target state. Keep CSV live gate closed.
- 248 SQL REGRESSION GATE: Execute the privacy-safe SQL regression protocol in tests/SQL_REGRESSION_PROTOCOL_248.md after the 247 commit. Block live CSV and CSV GUI until SQL baseline, saved-model reload, prediction and result checks pass; fix any regression first.
- 247: Add opt-in repository-backed CSV registration check before prediction composition; source identity and roles must match persisted configuration. Full classifier bootstrap and SQL regression gate still pending.
- 246: CSV prediction composition exposes and requires a pinned correction snapshot (including the unmodified-original case). Remaining: ordinary saved-config restore and genuine classifier bootstrap without fixture overrides.
- ACCEPTANCE GATE: Before the first live CSV pipeline run, perform a substantial end-to-end SQL regression run using authorized data; fix and re-test any SQL regressions before enabling or running CSV live. Keep private names, records and identifiers out of tracked docs and logs.
- 245: CSV composition-to-application run integration tested with native prepared handlers and SQL blockers; replace fixture-provided I/O builder/config restore/handler state with genuine bootstrap and check correction snapshot before GUI activation.
- 244: AutomaticClassifier.run() SQL-blocked integration uses prepared native-handler fixtures; unassisted runtime bootstrap, production configuration restoration, correction-revision provenance and full SQL interception remain acceptance gates.
- 243: CSV prediction now has an explicit application-composition entry point using the validated CSV facade and AutomaticClassifier. Full live classifier.run() with production Config/Logger and SQL interception is still pending; GUI remains disabled.
- 242: Native CSV model reload and prediction must remain SQL-free at ODBC, helper and SQL DataLayer boundaries. Test still uses synthetic handler state; full application bootstrap, run configuration restoration, source-scoped overlays and SQL-blocked CSV training remain open.
- 241: Native handler + versioned model-reload integration probe added; full SQL-intercepted application bootstrap remains open.
- 240: Full application bootstrap, real model/configuration restoration, and SQL connection interception remain required after native constructor probe.
## Revision 239 - fail-closed CSV task execution

- [x] Ensure CSV TaskRunner stops when abort_cleanly returns after task failure; treat CSV prediction publication errors as failed execution.
- [ ] Exercise full initialized handler/bootstrap path with SQL connection attempts explicitly blocked; current tests still isolate initialization.

## Revision 238 - source-aware integration fixture maintenance

- [x] Update historical versioned-model and semantic-review integration fixtures to CSV v2, keeping the real SQL/CSV source boundary active.
- [ ] Run fully bootstrapped SQL-blocked CSV TaskRunner integration and complete source-aware GUI model selection.

## Revision 237 - source-specific model persistence

- [x] Select CSV v2 or SQL v1 compatibility metadata at the real model-save boundary, without CSV-to-SQL fallback.
- [ ] Complete fully initialized CSV training, SQL-blocked TaskRunner prediction and source-aware model dropdown before GUI activation.

## Revision 236 - hard SQL/CSV model source boundary

- [x] Enforce SQL/CSV isolation on model reload using selected source, saved configuration, and metadata schema. Legacy SQL v1 is SQL-only.
- [ ] Filter saved-model dropdown by verified source identity before enabling CSV model selection; CSV execution remains disabled.
- [ ] Integrate CSV-native compatibility metadata into real model saving and validate full SQL-blocked runtime.

## Revision 235 - CSV-native compatibility groundwork

- [x] Define a CSV-native schema contract and fail-closed semantic review without altering SQL v1 model metadata.
- [ ] Wire validated CSV metadata generation into actual model saving and source selection. Derive and verify source semantic families from validated data and preprocessing; do not assume CSV strings imply numeric columns.
- [ ] Test complete model/handler bootstrap and SQL-blocked end-to-end CSV workflow before enabling GUI.

## Revision 234 - compatibility review checkpoint

- [x] Use production model semantic compatibility checker for synthetic model reload and TaskRunner prediction tests; reject tampered roles, missing converter schema and unsupported metadata versions.
- [ ] Define CSV-native source type and model compatibility contract (current metadata builder is SQL-shaped), verify genuine Config/handler bootstrap with all SQL access blocked, and only then activate CSV prediction GUI.

## Revision 233 - CSV model reload integration checkpoint

- Completed synthetic versioned JBG artifact reload and TaskRunner CSV prediction coverage (model-compatibility semantics stubbed); retain full live ModelHandler bootstrap, source compatibility and SQL-blocked application smoke test as open requirements.

## Revision 232 - combined CSV TaskRunner and handler execution

- [x] Exercise production TaskRunner dispatch, DatasetHandler ingestion, PredictionsHandler inference and scoped result writer in one synthetic CSV execution.
- [ ] Full ModelHandler model-artifact reload and complete application bootstrap with SQL blocked remain unverified; CSV GUI execution stays disabled.

## Revision 231 - CSV dataset to prediction handler bridge

- [x] Test production DatasetHandler ingestion and separation together with the existing prediction-result builder and scoped CSV writer, using synthetic CSV rows.
- [ ] Full JBG class initialization, saved model reload through ModelHandler, TaskRunner and SQL-blocked end-to-end execution remain unverified.

## Revision 230 - CSV identifier preservation during full dataset validation

- [x] Preserve opaque CSV row identifiers before generic numeric/date column validation; numeric feature conversion and SQL identifier behavior remain unchanged.
- [ ] Full live TaskRunner, ModelHandler and PredictionsHandler initialization with SQL blocked remains unverified; GUI CSV execution stays disabled.

## Revision 229 - staged CSV correction context lifecycle

- [x] Bind only an explicitly validated CSV correction context to the current project scope and registered dataset; invalidate it on source/file/catalog changes.
- [ ] Real CSV TaskRunner must supply the trusted run context; GUI dropdown and CSV execution remain disabled until a full run is verified.

## Revision 228 - explicit CSV GUI reclassification context

- [x] Validate project, dataset, source artifact, column roles and allowed classes before CSV corrections; block legacy SQL reclassification callback when CSV is selected.
- [ ] Bind a trusted active CSV run to the interactive widget and add explicit confirmation/download controls after the execution lifecycle is verified; no CSV GUI ML activation.

## Revision 227 - explicit CSV correction and download boundary

- [x] Provide a confirmed, scoped CSV correction call and a corrected-CSV download payload using the immutable original plus correction overlay. No SQL fallback.
- [ ] Connect the widget only after a verified active CSV run provides repository, scope, configuration and allowed classes; preserve the existing SQL correction route.
- [ ] Add user-visible download controls, run-level correction provenance, and multi-writer consistency checks before CSV GUI activation.

## Revision 226 - pin CSV correction head for each adapter

- [x] Capture the latest correction revision when preparing a scoped CSV reader, then apply only that revision to every subsequent read. Readers prepared before the first edit remain on original bytes.
- [ ] Propagate the pinned correction revision into run/model provenance and export/download metadata; complete real-handler execution and SQL-free runtime tests before GUI activation.
- [ ] Keep concurrent correction writers and reclassification GUI binding gated until explicit consistency and authorization contracts are tested.

## Revision 225 - linked class-correction overlay for registered CSV

- [x] Replace revision 224 full-CSV rewrite with small immutable correction events beside the original dataset. Replay is deterministic and historical correction revisions remain addressable.
- [ ] Wire correction-overlay reads into training, prediction and export; do not activate GUI CSV ML yet.
- [ ] Add cross-process single-writer/concurrency controls, retention/compaction and edit audit/authorization before multi-user service exposure.

## Revision 224 - immutable CSV reclassification foundation

- [x] Add a scoped, SQL-free single-row class correction that produces a new registered CSV dataset version and preserves the source artifact and opaque IDs.
- [ ] Wire the reclassification dropdown to an explicitly selected CSV version, with user confirmation and safe model/configuration rebinding. The original local upload is not overwritten.
- [ ] Define a deliberate export/replace workflow if users need to update a physical CSV file; keep GUI CSV execution disabled until full handler integration.

## Revision 223 - CSV misprediction SQL separation

- [x] Keep shared positional misprediction ranking and suppress SQL inspection-query generation for CSV sources; retain SQL reporting.
- [ ] CSV misprediction export remains a local file and its editable reclassification widget is SQL-oriented; verify scoped publication and GUI behavior before CSV training activation.
- [ ] Complete real-handler CSV TaskRunner and SQL-blocked end-to-end testing before GUI execution.

## Revision 222a - deterministic CSV labelled-first ordering

- [x] Rank labelled and unlabelled rows explicitly before sorting CSV IDs; cover null and empty labels across pandas versions.
- [ ] Windows-verification and full SQL-isolated TaskRunner lifecycle remain pending; CSV GUI execution stays disabled.

## Revision 222 - source-aware DatasetHandler row IDs

- [x] Preserve opaque CSV row IDs through DatasetHandler ordering and indexing while retaining SQL integer IDs.
- [ ] Verify complete handler lifecycle with real JBG configuration, conversion and TaskRunner before GUI activation.

## Revision 221 - shared CSV prediction handler integration

- [x] Exercise actual JBGHandler/PredictionsHandler/ModelHandler save and result-construction methods with scoped CSV results.
- [ ] Full DatasetHandler parsing/conversion and complete TaskRunner lifecycle still need a SQL-blocked integration test before GUI activation.

## Revision 220 - bounded CSV TaskRunner execution smoke

- [x] Exercise production task dispatch and task methods with scoped CSV I/O, a serialized estimator and controlled handler adapters.
- [ ] Full JBG DatasetHandler/ModelHandler/PredictionsHandler lifecycle, SQL isolation and GUI activation remain unverified.

## Revision 219 - CSV read/write data-layer identity guard

- [x] Require the same prepared CSV data-layer facade for TaskRunner reads and JBGHandler prediction writes before task dispatch.
- [ ] Full TaskRunner end-to-end execution and GUI activation are still pending; the guard does not prove the full pipeline works.

## Revision 218 - real configuration restore integration checks

- [x] Add Windows-targeted tests that run the real Config model-file loader through scoped CSV prediction preparation; keep CSV GUI execution disabled.
- [ ] Full TaskRunner execution and production JBG classification remain unverified.
- [ ] In environments lacking JBG bootstrap dependencies, integration tests skip explicitly rather than passing vacuously.

## Revision 217 - real JBG artifact and shared prediction handler

- [x] Integration-test the actual JBG model artifact read/write contract, the shared prediction-result constructor, registered CSV input and scoped result publication.
- [ ] Full Config loading, TaskRunner lifecycle, GUI execution and SQL isolation under real runtime startup remain pending.

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

**Current architecture decision:** the binding order is defined at the top of this
file: finish CSV/SQL functionality and acceptance -> refactor `JBGMeta.py` and
backend/package boundaries -> build a GUI-independent Python application layer
-> build a FastAPI-based web GUI -> address Azure/Entra/multi-user deployment.
The detailed items below are retained as technical reference. Older priority
wording here must not override the current sequence.

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
- [ ] **Future GUI replacement architecture track — FastAPI after application-layer extraction:** after full CSV/SQL ML acceptance,
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

- [ ] After revision 255, verify full native CSV fitting through AutomaticClassifier/TaskRunner with saved model and reload in Windows ML environment; release CSV GUI only after the full flow and source/identity checks pass.

- [ ] GUI cosmetic: when restoring a saved CSV model, auto-scroll multi-select pipeline widgets to their selected options when supported by ipywidgets/Voilà; do not change selection values or SQL behavior.
