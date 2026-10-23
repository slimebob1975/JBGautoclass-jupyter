# JBG AutoClassification

JBG AutoClassification is a Windows-oriented machine-learning application for rapid classifier
spot-checking, pipeline comparison, training, evaluation and prediction from SQL Server data. The
primary interface is a Jupyter/Voilà GUI backed by the Python modules in `src/`.

JBG is designed for mixed tabular datasets containing numerical, categorical and text features. It
combines preprocessing, resampling, dimensionality reduction, model selection, cross-validation,
final search, saved-model reuse and diagnostic tooling in one workflow.

## Main capabilities

- Read training and prediction data from Microsoft SQL Server through ODBC.
- Work with numerical, categorical and free-text input columns.
- Compare a broad classifier set from scikit-learn and imbalanced-learn, plus native CatBoost,
  PyTorch/skorch, Keras and Flax model families.
- Configure scaling, over/undersampling, dimensionality reduction, scoring and final model search.
- Train and evaluate on a deterministic train/holdout split with cross-validation inside the
  training partition.
- Save trained models and reuse them for later prediction or retraining.
- Use a separate prediction table with every target value set to `NULL`; when a saved model is
  selected, JBG can restore the model's expected target column.
- Run Feature selection modelling to investigate original-column subsets with Linear, Non-linear,
  Tree, Combination or CatBoost investigative model families.
- Run permutation-based feature importance on the final holdout for a freshly trained model.
- Inspect mispredictions and use the reclassification workflow for supported database-backed runs.
- Optionally run Dark Number diagnostics and validation utilities.
- Persist generated settings and use Repeat Last for supported workflows.
- Show the controlled application version and Git repository provenance below the GUI title and at the start of each application log; non-Git deployments fall back safely when repository metadata is unavailable.
- Record application and server diagnostics while keeping the persistent application log privacy-safe by
  default; SQL, row-level data and other identifying runtime details require explicit local opt-in.

`CHANGELOG.md` contains development history. `BACKLOG.md` contains planned and deferred work.

## Requirements

JBG is primarily developed and verified in a Windows environment. A typical installation needs:

- Python with `venv` support; an Anaconda-based interpreter can also be used.
- Access to Microsoft SQL Server.
- A compatible Microsoft SQL Server ODBC driver.
- A browser for the Voilà interface.
- Git if you maintain the checkout from source control.

Install the Python dependencies from the repository root:

```powershell
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
python -m pip install --upgrade ipykernel jupyterlab voila
```

The main requirements include the scientific Python stack, scikit-learn, imbalanced-learn,
TensorFlow/Keras, PyTorch/skorch, JAX/Flax, CatBoost, pyodbc and the Jupyter/Voilà UI dependencies.
A GPU is not required for ordinary use.

### Optional serializer fallback

`dill` is not required for a normal installation. Current JBG model artifacts prefer Python's
standard-library `pickle`. If a newly generated model contains state that pickle cannot serialize,
install the optional serializer fallback:

```powershell
python -m pip install -r .\requirements-optional-dill.txt
```

Pre-revision-176 model artifacts are intentionally unsupported and should be retrained. Only load
`.sav` model files from trusted sources: both pickle and dill can execute code while loading a
serialized object.

## Configuration

Copy the environment template and edit the local copy:

```powershell
Copy-Item .\.env_template .\.env
```

The main settings are:

| Setting | Purpose |
| --- | --- |
| `DEFAULT_HOST` | SQL Server host name or address. |
| `DEFAULT_ODBC_DRIVER` | ODBC driver used for SQL Server connections. |
| `DEFAULT_DATA_CATALOG` | Default source database/catalog. |
| `DEFAULT_DATA_TABLE` | Default source schema/table. |
| `DEFAULT_CLASSIFICATION_CATALOG` | Default destination database/catalog. |
| `DEFAULT_CLASSIFICATION_TABLE` | Default classification-result schema/table. |
| `DEFAULT_SMTP_SERVER` | Optional notification SMTP server. |
| `DEFAULT_NOTIFICATION_EMAIL` | Optional notification recipient. |
| `DEFAULT_N_JOBS_DESIRED` | Shared parallelism cap; `-1` allows JBG to use the available logical CPUs subject to workload and memory limits. |
| `JBG_LOG_SENSITIVE_DATA` | `0` by default. Set to `1` only for trusted local diagnostics when full SQL and row-level previews are required in persistent logs. |

SQL credentials are runtime credentials. Enter them in the GUI when using the notebook interface.
Generated configuration/model metadata deliberately does not persist the SQL password. JBG writes
schema-versioned JSON configuration files under `src/config/`; generated Python configuration files
are no longer supported. For command-line workflows that need a runtime password, JBG can use the
`JBG_SQL_PASSWORD` environment variable.

Do not commit a local `.env` file or credentials to source control.

## Starting the GUI

The repository contains `start_application_template.ps1`. Make a local copy and edit the paths in
the configuration block at the top of the script:

```powershell
Copy-Item .\start_application_template.ps1 .\start_application.ps1
```

At minimum, set the project root, base Python interpreter, virtual-environment name and any desired
Voilà port. Then run the local launcher:

```powershell
.\start_application.ps1
```

The template can create/update the virtual environment, install requirements, register the Jupyter
kernel, run the source-layout migration helper when needed, start Voilà and write a separate server
log. Its default Voilà port is `8866`.

If the environment is already prepared, Voilà can also be started directly from the repository
root:

```powershell
voila .\JBG_SML_GUI.ipynb --port 8866
```

## Core workflow

### 1. Connect to the data source

Enter the SQL username and password, select the source catalog/table, then select the target/class
column and unique ID column. JBG uses the unique ID when it needs stable row identity for splits,
diagnostics and database-backed follow-up actions.

Training rows need a known target value. Rows with a missing target can be treated as prediction
rows in workflows that support mixed labelled/unlabelled data.

### 2. Configure the machine-learning pipeline

Choose the preprocessing and model options in the GUI. Depending on the selected workflow, JBG can
apply scaling, over/undersampling, dimensionality reduction and classifier search inside the
training/CV lifecycle.

For imbalanced classification, inspect class-specific metrics rather than relying only on overall
accuracy. The selected scoring metric controls model comparison and several downstream diagnostics.

### 3. Train and evaluate

Start the classifier run. JBG separates the fetched labelled data into training and holdout
partitions, performs model/CV work on the training side, evaluates the chosen fitted pipeline on the
holdout and can then retrain as configured for final use.

The output area reports the selected pipeline, metrics, diagnostics and generated artifacts. Long
searches may also expose progress and runtime estimates.

### 4. Save and reuse models

Saved `.sav` files use a versioned JBG artifact envelope. New artifacts use pickle when possible and
fall back to optional dill only when necessary. Keras and other framework-specific state may use
additional sidecar/state handling managed by JBG.

The supported persistence contract is save -> restart/fresh process -> load -> predict/retrain in a
compatible JBG/Python/ML dependency environment. It is not a promise of portability across arbitrary
Python or third-party-library versions.

When reusing a newly saved model, JBG compares a privacy-safe fingerprint of the physical SQL source
as provenance and verifies the required SQL column/type families, the saved feature-role definition and
the fitted converter representation before prediction. Training and prediction may use different physical
tables: a source change is reported as a warning, while missing required model columns or incompatible SQL
type families block reuse. Extra source columns that the fitted model does not require are ignored. A
same-family SQL type change or ODBC-driver change is also reported as an explicit warning because semantic
equivalence cannot be proven automatically. The fitted converter checks predictor dtype families before
transforming prediction rows.

Pre-revision-176 model artifacts are intentionally rejected. Retrain and save a new model with the current JBG version instead of relying on legacy deserialization paths.


## Feature selection modelling

The **Feature selection modelling** panel investigates subsets of original source columns without
silently applying them to the main classifier configuration. Search rows are sampled from training
data, and candidate evaluation remains isolated from the ordinary holdout.

The available investigative families are:

- Linear
- Non-linear
- Tree
- Combination
- CatBoost

Pipeline controls are shown in execution order: **Scaler -> Oversampling -> Reduction -> Model type**. Candidate-process parallelism is automatic. JBG bounds it by the logical CPU count, the
shared `DEFAULT_N_JOBS_DESIRED` cap, the candidate/search budget and the memory guard; candidate
estimators remain internally single-threaded to avoid nested oversubscription.

The persistent log records the resolved worker decision in a line beginning with
`Feature search parallelism:`. Pause/resume and checkpoint support are available for the supported
investigation flow.

Saved Feature modelling recommendations remain **candidate** evidence. Before a saved candidate can be
explicitly applied, JBG checks the recorded source identity, predictor SQL schema and text/converter
representation against the current selection. Applying a compatible saved candidate restores both its
selected columns and the configurable text representation used by the investigation; it does not turn
the recommendation into independently validated settings.

The command-line diagnostic entry point is also available:

```powershell
python .\scripts\feature_search.py --help
```

## Feature importance

Enable **Feature importance** before a fresh training run to compute holdout permutation importance
for the fitted pipeline. JBG shuffles each original input column while keeping the fitted conversion
and model pipeline fixed, then reports the score decrease across repeats.

Use this as model inspection, not as proof of causality. Correlated inputs can hide each other's
importance, and a weak fitted model can assign low importance to useful variables.

## Dark Number diagnostics

Dark Number functionality is optional diagnostic/research tooling for estimating unseen or
misclassified cases under JBG's configured methodology. It has dedicated validation and sensitivity
utilities in `src/` and should be treated separately from ordinary classifier accuracy.

Use it only when its assumptions fit the problem being studied. The active backlog deliberately
keeps broader Dark Number methodology research separate from routine release work.

## Command-line classifier runs

Generated JSON classifier configurations can be executed from `src` with `AutomaticClassifier.py`:

```powershell
Set-Location .\src
python .\AutomaticClassifier.py -f .\config\<generated-config-file>.json
```

Only schema-current JSON configuration files are accepted. Runtime SQL credentials still need to be
supplied separately when required; configuration files never contain the SQL password.

## Files and runtime output

The main repository locations are:

| Path | Contents |
| --- | --- |
| `JBG_SML_GUI.ipynb` | Root notebook used by the Voilà GUI. |
| `src/` | Application and ML modules. |
| `src/config/` | Locally generated schema-versioned JSON configs. |
| `src/model/` | Saved models and framework sidecars. |
| `src/output/` | CSV exports, checkpoints, timing history and other generated output. |
| `logs/` | Application and server logs. |
| `scripts/` | Maintenance, privacy, diagnostics and feature-search utilities. |
| `tests/` | Automated regression tests. |

Generated models, configs, logs and output can contain operational or data-derived information.
Treat them according to the sensitivity of the source dataset and do not publish them by default.

### Persistent log privacy

Normal **application** logs keep the active GUI informative while omitting or summarizing fields that can
identify the local environment or source data. By default this includes full SQL, row-level dataset previews,
configuration values, schema/statistics matrices, class-labelled evaluation matrices, prediction and
reclassification payloads, split labels/signatures, feature-importance column names, embedded download
payloads and absolute local filesystem paths. Aggregate operational diagnostics such as row counts, timings,
worker decisions and generic error types remain available where they can be logged without the omitted values.

Only for trusted local diagnostics, enable sensitive persistent application logging before starting JBG:

```powershell
$env:JBG_LOG_SENSITIVE_DATA = "1"
```

The separate Voilà/server log has a different safety constraint: the guarded kernel-cleanup workflow currently
uses its exact connection-directory path as evidence. That server log can therefore still contain local paths
and should be treated as sensitive local diagnostic material rather than as publication-safe output.

Unset it, or return it to `0`, after troubleshooting. Generated artifacts, the separate server log and
unexpected third-party exception text are not made publication-safe by this switch; continue treating runtime
diagnostics as potentially sensitive and do not publish them by default.

## Testing

Run the test suite from the repository root in the same environment used by JBG:

```powershell
python -m pytest -q
```

For the documentation privacy check:

```powershell
python .\scripts\check_documentation_privacy.py
```

A local organization-specific `.jbg_doc_privacy_blocklist` can be used by the privacy tooling. Keep
that file private and out of source control. See `tests/README.md` for additional test and validation
commands.

## Troubleshooting

### SQL connection fails

Check the host, ODBC driver, database/catalog permissions and runtime SQL credentials. The `.env`
file supplies defaults, not a persisted SQL password.

### Voilà starts but the GUI does not behave correctly

Check the server log under `logs/` and the current application log. Ensure the notebook is using the
same Python environment where `requirements.txt`, Jupyter and Voilà were installed. If a stale
kernel/session is suspected, use the supplied diagnostic/cleanup scripts rather than terminating
unknown Python processes blindly.

### A current saved model requires dill

Install `requirements-optional-dill.txt` and retry. If the model still cannot load, verify that it
was produced by the current JBG artifact format in a compatible Python/library environment. Old
pre-revision-176 artifacts are not supported.

### Feature modelling uses fewer workers than expected

Find the `Feature search parallelism:` line in the application log. It reports the selected worker
count and the CPU, global-cap, candidate-budget and memory-guard constraints that produced it.

### Third-party warnings clutter a command-line run

Python warning filters can be used when you have confirmed a warning is irrelevant. For example:

```powershell
python -Wi .\AutomaticClassifier.py -f .\config\<generated-config-file>.json
```

Do not suppress warnings merely to hide a failing model or incompatible dependency.

## Development records

- `CHANGELOG.md` records completed numbered revisions and their verification status.
- `BACKLOG.md` tracks active priorities, deferred integrations and validation work.
- `tests/README.md` documents test-specific workflows.

## Acknowledgments

The project was partly inspired by Jason Brownlee's Machine Learning Mastery material.
