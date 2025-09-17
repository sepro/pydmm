# Developer Documentation

This document provides information for developers who want to contribute to pyDMM or modify its functionality.

## Running Tests

The package includes a comprehensive test suite with 91% code coverage:

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=pydmm --cov-report=term-missing

# Run specific test file
pytest tests/test_core.py

# Run specific test
pytest tests/test_core.py::TestDirichletMixture::test_clustering_performance
```

## Development Setup

```bash
# Clone repository
git clone <repository-url>
cd pyDMM

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install in development mode with test dependencies
pip install -e ".[dev]"

# Install GSL (if not already installed)
sudo apt-get install libgsl-dev  # Ubuntu/Debian

# Run tests to verify installation
pytest
```

## Modifying C Code

After making changes to the C extension:

```bash
# Reinstall to recompile the C extension
pip install -e .

# Run tests to ensure changes work correctly
pytest tests/test_c_extension.py
```

## Code Architecture

- **`src/pydmm/core.py`**: Main Python interface and result handling
- **`src/pydmm/wrapper.c`**: NumPy C API wrapper with data layout conversion
- **`src/pydmm/dirichlet_fit_standalone.c`**: Core C implementation with GSL
- **`setup.py`**: Build configuration for C extension
- **`tests/`**: Comprehensive test suite

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass: `pytest`
6. Submit a pull request

## Development Requirements

- Python >= 3.7
- NumPy >= 1.19.0
- pandas >= 1.0.0
- GSL (GNU Scientific Library)
- pytest >= 6.0 (for testing)
- pytest-cov >= 2.10 (for coverage reports)

## Examples for Testing

The `docs/examples/` directory contains several example scripts that can be used for testing:

- `reference_counts.py`: Complete workflow with model selection and evaluation
- `probability_comparison.py`: Compare probability computations between C and Python implementations
- `classification_comparison.py`: Compare classification decisions between C and Python implementations

Run an example to test your changes:
```bash
python docs/examples/reference_counts.py
```