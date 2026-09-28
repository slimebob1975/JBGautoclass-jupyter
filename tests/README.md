# Tests

## Running tests

Run the test suite from the repository root:

```text
pytest
```

Useful variations:

- Single file: `pytest tests/<test-file>.py`
- Single test function: `pytest tests/<test-file>.py::<test_func>`
- Single test class: `pytest tests/<test-file>.py::<test_class>`
- Single function in a test class: `pytest tests/<test-file>.py::<test_class>::<test_func>`

Test-related files:

- `tests/`
- `.coverage`
- `.coveragerc`
- `pyproject.toml`

## Coverage

- All modules: `pytest --cov-report html:coverage --cov=./`
- One module: `pytest --cov-report html:coverage --cov=Config`

## Fixtures

Fixtures live in `tests/fixtures/`.

If `test_config.py` or `test_handler.py` fail after a deliberate serialization
change, inspect the fixtures first. The fixture regeneration helper currently
lives at `src/JBGclassification/regenerate-fixtures.py`.

## Dark Number real-dataset validation runner

Revisions 084-087 keep fallback experiments outside production and expose a separate empirical runner. The current
runner validates `perturbed_same_model`; revision 088 adds estimable-only error/stability robustness metrics and a nine-seed default, while the earlier `soft_same_model` and `adaptive_noise_same_model` experiments remain documented rejected candidates.
After an ordinary training run has produced `.jbg_last_run.json` and the corresponding model artifact, run from
the repository root:

```text
python .\src\JBGclassification\JBGDarkNumberValidationRunner.py --sql-username <username> --runs 9
```

Use `--target <class>` one or more times to restrict the one-vs-rest targets. The SQL password is read from
`JBG_SQL_PASSWORD` when available or requested interactively. The command creates a dedicated validation log and
paired CSV/JSON outputs; it does not modify the saved model or production Dark Number settings.

### 089 experimental perturbed Dark Number fallback

The production Dark Number path now has a default-on `Experimental perturbed fallback` checkbox. Focused tests cover its config default and opt-out, generated-config/Repeat Last persistence, dependency on the main Dark Numbers control, and routing of recovery-level direct failures into `perturbed_same_model` while preserving the older fallback when the checkbox is disabled. Structured fallback-event telemetry contains aggregate clone statistics only.
