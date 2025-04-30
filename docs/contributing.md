# Contributing to Vector Inference

Thank you for your interest in contributing to Vector Inference! This guide will help you get started with development, testing, and documentation contributions.

## Development Setup

### Prerequisites

- Python 3.10 or newer
- [uv](https://github.com/astral-sh/uv) for dependency management

### Setting Up Development Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/VectorInstitute/vector-inference.git
   cd vector-inference
   ```

2. Install development dependencies:
   ```bash
   uv sync --all-extras --group dev
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

!!! tip "Using Virtual Environments"
    If you prefer using virtual environments, you can use `uv venv` to create one:
    ```bash
    uv venv
    source .venv/bin/activate
    ```

## Development Workflow

### Code Style and Linting

We use several tools to ensure code quality:

- **ruff** for linting and formatting
- **mypy** for type checking

You can run these tools with:

```bash
# Linting
uv run ruff check .

# Type checking
uv run mypy

# Format code
uv run ruff format .
```

!!! note "Pre-commit Hooks"
    The pre-commit hooks will automatically run these checks before each commit.
    If the hooks fail, you will need to fix the issues before you can commit.

### Testing

All new features and bug fixes should include tests. We use pytest for testing:

```bash
# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=vec_inf
```

## Documentation

### Documentation Setup

Install the documentation dependencies:

```bash
uv sync --group docs
```

### Building Documentation

Build and serve the documentation locally:

```bash
# Standard build
mkdocs build

# Serve locally with hot-reload
mkdocs serve
```

### Versioned Documentation

Vector Inference uses [mike](https://github.com/jimporter/mike) to manage versioned documentation. This allows users to access documentation for specific versions of the library.

#### Available Versions

The documentation is available in multiple versions:

- `latest` - Always points to the most recent stable release
- Version-specific documentation (e.g., `0.5.0`, `0.4.0`)

#### Versioning Strategy

Our versioning strategy follows these rules:

1. Each release gets its own version number matching the package version (e.g., `0.5.0`)
2. The `latest` alias always points to the most recent stable release
3. Documentation is automatically deployed when changes are pushed to the main branch

#### Working with Mike Locally

To preview or work with versioned documentation:

```bash
# Build and deploy a specific version to your local gh-pages branch
mike deploy 0.5.0

# Add an alias for the latest version
mike deploy 0.5.0 latest

# Set the default version to redirect to
mike set-default latest

# View the deployed versions
mike list

# Serve the versioned documentation locally
mike serve
```

#### Automatic Documentation Deployment

Documentation is automatically deployed through GitHub Actions:

- On pushes to `main`, documentation is deployed with the version from `pyproject.toml` and the `latest` alias
- Through manual trigger in the GitHub Actions workflow, where you can specify the version to deploy

!!! info "When to Update Documentation"
    - When adding new features
    - When changing existing APIs
    - When fixing bugs that affect user experience
    - When improving explanations or examples

## Pull Request Process

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** and add appropriate tests
3. **Ensure tests pass** and code meets style guidelines
4. **Write clear documentation** for your changes
5. **Submit a pull request** with a clear description of the changes

!!! important "Checklist Before Submitting PR"
    - [ ] All tests pass
    - [ ] Code is formatted with ruff
    - [ ] Type annotations are correct
    - [ ] Documentation is updated
    - [ ] Commit messages are clear and descriptive

## Release Process

1. Update version in `pyproject.toml`
2. Update changelogs and documentation as needed
3. Create a new tag and release on GitHub
4. Documentation for the new version will be automatically deployed

## License

By contributing to Vector Inference, you agree that your contributions will be licensed under the project's [MIT License](https://github.com/VectorInstitute/vector-inference/blob/main/LICENSE).
