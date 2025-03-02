# API Tests

This directory contains tests for the Vector Inference API module.

## Test Files

- `test_client.py` - Tests for the `VecInfClient` class and its methods
- `test_models.py` - Tests for the API data models and enums
- `test_examples.py` - Tests for the API example scripts

## Running Tests

Run the tests using pytest:

```bash
pytest tests/vec_inf/api
```

## Test Coverage

The tests cover the following areas:

- Core client functionality: listing models, launching models, checking status, getting metrics, shutting down
- Data models validation: `ModelInfo`, `ModelStatus`, `LaunchOptions`
- API examples: verifying that API example scripts work correctly

## Dependencies

The tests use pytest and mock objects to isolate the tests from actual dependencies.
