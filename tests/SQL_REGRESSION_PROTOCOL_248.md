# SQL regression gate — revision 248

This is an operator checklist for the **existing SQL workflow**, performed before any live CSV run or CSV GUI activation. Use an authorized SQL dataset and an approved environment. **Do not paste real schema/table names, person identifiers, row contents, credentials, local connection strings, or private log excerpts into the repository.** Record only anonymized outcomes (pass/fail, counts, duration bands, revision and failure category) in tracked documentation.

## Preconditions

- [ ] Working tree is clean, with revisions through 247 committed; note commit SHA locally.
- [ ] Use the established SQL source and existing SQL-only model selection. Do not switch on CSV training or CSV prediction in the GUI.
- [ ] Verify connection permissions and choose a representative *authorized* dataset, with a separate safe test environment for reclassification/write operations.
- [ ] Take an approved baseline/backup before any operation that writes to SQL. Know how to roll back the chosen test rows.
- [ ] Verify privacy screening for documentation (`python scripts/check_documentation_privacy.py`) and use the secure channel, not Git, if detailed diagnostic logs are needed.

## Test execution (record each step as PASS, FAIL, BLOCKED or NOT RUN)

| ID | Step | Acceptance evidence (sanitized) |
| --- | --- | --- |
| SQL-01 | Start JBG in the existing Jupyter/Voilà SQL mode; load previous settings if supported. | GUI opens; SQL remains the selected source and previously supported widget settings behave correctly. |
| SQL-02 | Select authorized SQL source, then load data with mixed numeric, categorical and text features (when present). | Read completes; row counts and ID uniqueness agree with authorized source checks; null handling is intentional. |
| SQL-03 | Run preprocessing and feature modelling with representative options, including feature selection if used operationally. | Job completes or reports a legitimate configuration error; selected feature counts, data splitting and progress are coherent. |
| SQL-04 | Train a representative set of algorithms, using established parallelism and validation settings. | Training completes; metrics and candidate model selection are present and plausible; no CSV adapter is invoked. |
| SQL-05 | Save the selected SQL model and configuration. | Versioned model artifact is created; source is SQL and compatible historical SQL v1 metadata remain supported. |
| SQL-06 | Restart the JBG process, restore the saved SQL model/configuration, and run SQL prediction. | Model reload succeeds, prediction IDs correspond to input IDs, results and progress complete without CSV fallback. |
| SQL-07 | Review result statistics, confusion/misprediction and any enabled analysis/export features. | Counts and tables agree with prediction output; no unexplained widget rendering failures. |
| SQL-08 | In an approved **write-safe** test target only, reclassify a known test row, then verify its result and rollback. | Only the intended row is modified; operation is auditable, reversible and SQL-specific. Mark BLOCKED if no safe write target is available. |
| SQL-09 | Select a saved model while SQL is active and inspect available model choices. | CSV-labelled models are not offered; selecting an incompatible CSV model is rejected by backend even if bypassing GUI. |
| SQL-10 | Validate errors/cancellation and recovery on a non-destructive scenario, then rerun a successful job. | Failure does not publish partial/incorrect results and JBG recovers without leaking SQL resources. |

## Evidence and decision

Keep raw logs, configuration dumps and data extracts **outside Git** in the approved secure environment. Before sharing diagnostic snippets, redact identifiers, physical source names, connection strings, credentials and any row values; retain stack traces, exception types, revision numbers and anonymized settings relevant to the failure.

For each ID record locally: `ID | PASS/FAIL/BLOCKED/NOT RUN | duration | observation | issue reference`. Compare where available with known SQL-only baselines (counts, metrics within expected tolerance, performance, GUI behavior). Do not claim exact metric equality when randomness, parallel execution or library versions differ.

**PASS gate:** All applicable critical checks SQL-01 through SQL-07, SQL-09 and SQL-10 pass, and SQL-08 passes if a safe write target is approved. If SQL-08 cannot be performed safely, record BLOCKED and require explicit risk acceptance before declaring the full SQL gate complete. Any actual regression means **STOP**: diagnose, patch, rerun the affected tests and the representative SQL workflow, then reassess. **No live CSV run or CSV GUI activation before this gate is approved.**

## After SQL approval

Next separately verify full CSV prediction bootstrap with registered configuration, pinned correction revision, real model load, SQL connection interception and result publication. CSV training and GUI enablement have their own acceptance checks; do not infer they are complete from the SQL result.
