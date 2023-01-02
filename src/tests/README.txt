# README

## Running tests

To run tests: pytest (in root, src or tests)

### Variations (if you don't want to run all tests)
- Single file: pytest <path-to-test-file>
- Single test function: pytest <path-to-test-file>::<test_func>
- Single test class: pytest <path-to-test-file>::<test_class>
- Single function in test class: pytest <path-to-test-file>::<test_class>::<test_func>


Files related to tests:
- src/tests/*
- .coverage
- .coveragerc
- pyproject.toml

## Coverage

Shows how much code is tested.

- Show all modules: pytest --cov-report html:coverage --cov=./ 
- Show specific module: pytest --cov-report html:coverage --cov=Config (replace Config with module)

## Important notes

If test_config.py or test_handler.py fail, the first suspected culprit are the fixtures.
Check `regenerate_fixtures.py` in the `src` directory on what needs to be done to regenerate.