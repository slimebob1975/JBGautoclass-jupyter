# Revision 273a — CSV/SQL ML-chain audit and acceptance matrix

This is a source-grounded planning baseline, not a claim of live SQL acceptance or full CSV GUI parity. Use synthetic data in automated checks; do not paste private SQL, row identifiers, credentials, or runtime paths into this document.

## Current implementation and gaps

| Capability | CSV source path | SQL source path | Required next evidence |
| --- | --- | --- | --- |
| Dataset selection, schema and column roles | Implemented for registered CSV; validated roles required | Established GUI path | Live source switching and role restoration |
| Basic numerical/text preprocessing and estimator choices | Selectable in CSV train-only GUI | Established GUI path | End-to-end samples for numeric, text, categorical, and mixed roles; check converter equivalence |
| Training and checkpoint save | Dedicated CSV classifier via shared `AutomaticClassifier` | Established common classifier path | Live CSV fit and fresh-process reload; SQL smoke |
| Saved-model prediction | Dedicated CSV handler with scoped model inspection | Existing SQL prediction | Verify known/unlabelled split, class mapping, publication, and restart |
| CSV run export / deletion | Scoped exports and guarded deletion implemented | SQL result and model lifecycle differs | Live CSV export/retrieval, failure recovery and isolation |
| Train + predict combined run | CSV GUI forces `predict=False` during training | Supported subject to SQL configuration | Identify shared task sequence and define CSV-safe output destination before enabling |
| Misprediction UI/reclassification | CSV GUI forces `mispredicted=False` for training; correction-context API exists but is not generally wired to live CSV run | Existing SQL workflow | Validate run-bound correction context, safe artifact publication and UI behavior |
| Metadata (`metas`) | CSV GUI forces off | Existing GUI control | Audit SQL dependencies and converter behavior before enabling |
| Dark Numbers | CSV GUI forces off; controls locked | Existing GUI options | Audit task dependencies, output sinks and diagnostics |
| Feature importance | CSV GUI forces off; controls locked | Existing GUI options | Verify estimator compatibility, privacy-safe output and artifact handling |
| Feature modelling / assisted selection / profile reuse | CSV GUI resets mode to `manual` and deactivates section | Existing Feature modelling workflow | Audit SQL-only source assumptions, profile identity, and job handling |
| Cross-validation, sampling, reduction, model settings | Selected CSV controls editable; complete parity unverified | Existing GUI settings | Behavioral matrix with synthetic data and representative estimators |
| Erase saved models | CSV two-click scoped deletion implemented | Operator reports SQL `Erase models` malfunction | Reproduce and fix SQL separately; preserve CSV boundary |

## Source evidence (revision 270+)

- `src/GUI/Widgets.py`: `_CSV_TRAINING_EDITABLE_FIELDS`, `_CSV_TRAINING_LOCKED_FIELDS`, `_csv_set_training_controls`, `_csv_training_readiness`, `_csv_prepare_training_runtime`, `_csv_start_saved_prediction`, `_csv_start_training`.
- `src/GUI/Widgets.py`: `EventHandler.start_button_was_clicked` dispatches separate CSV train and saved prediction paths; other SQL-oriented actions are guarded.
- `src/Storage/CSVGUITrainingConfig.py`: train-only config preparation enforces `mode.train=True`, `mode.predict=False`, `mode.mispredicted=False`, and source-scoped checkpoint storage.
- `src/Storage/CSVTrainingApplication.py`: validates registered CSV training source, roles, and pinned correction snapshot before common classifier execution.
- `src/GUIHandler.py`: `run_csv_classifier`, `run_csv_saved_prediction`, and CSV correction-context entrypoints.

## Implementation order after 273a

1. **273b — GUI capability gates:** write focused tests for every currently forced-off CSV mode and preserve SQL behavior. Inventory shared task dependencies before changing an enablement flag.
2. **273c onward — vertical enablement:** implement one capability at a time (start with safe train/predict workflow, then metrics/feature importance, followed by more complex Feature modelling and Dark Numbers when source contracts permit). Add source-specific adapters only where essential; do not duplicate the ML engine.
3. **SQL regression:** reproduce the reported SQL saved-model erase problem independently; run the authorized SQL acceptance protocol without persisting private names or SQL text.
4. **Joint end-to-end acceptance:** run both data-source paths, model save/reload, results, failure recovery and source switching; use the full regression gate prior to commit/release.
5. **Only then:** major backend / `JBGMeta.py` refactoring, GUI-independent application layer, FastAPI frontend, and later Entra ID/Azure deployment controls.

## Exit criteria for the CSV/SQL phase

- Each intended pipeline option has an explicit supported/rejected contract for CSV and SQL.
- Supported CSV choices execute the shared ML core, produce correct scoped artifacts, and do not access SQL.
- SQL existing training, reload, prediction and outputs remain unchanged except for explicitly repaired defects.
- Both automated regression and operator-run Voilà/SQL acceptance have recorded pass/fail evidence, without disclosing sensitive content.
