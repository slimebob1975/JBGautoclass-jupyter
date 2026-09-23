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
