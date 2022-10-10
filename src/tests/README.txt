# README

- To run test: pytest (in root, src or tests)

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