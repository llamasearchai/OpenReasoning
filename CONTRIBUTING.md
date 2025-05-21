# Contributing to OpenReasoning

We love your input! We want to make contributing to OpenReasoning as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

### Pull Requests

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Pull Request Requirements

Before submitting a pull request, please ensure:

1. Your code follows the style guidelines of this project
2. You have added tests that show your feature works or your bugfix resolves an issue
3. The existing tests pass with your changes
4. You've updated the documentation to reflect any changes

## Code Style

This project uses:

- [Black](https://black.readthedocs.io/) for Python code formatting
- [isort](https://pycqa.github.io/isort/) for import sorting
- [mypy](https://mypy.readthedocs.io/) for static type checking

To ensure your code meets our style requirements, run:

```bash
make lint
```

## Testing

We use pytest for testing. To run the tests:

```bash
make test
```

## Project Structure

The repository is organized as follows:

```
.github/
.pytest_cache/
docs/
libs/
  colorful-cli/
notebooks/
openreasoning/  <-- This is the code previously in OpenReasoning/OpenReasoning
openreasoning.egg-info/
tests/
.gitignore
CONTRIBUTING.md
Dockerfile
LICENSE
Makefile
README.md
requirements.txt
pyproject.toml
run_openreasoning.sh
test_installation.py
setup.py
```

## Environment Setup

We recommend using a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"   # Install package in development mode with dev dependencies
```

## Apple Silicon Optimization

If you're contributing code related to the M3 optimizations:

1. Test on both Apple Silicon and non-Apple Silicon machines if possible
2. Make optimizations optional/gracefully degrade on unsupported hardware
3. Document clearly which optimizations are applied and what dependencies are required

## Documentation

When adding new features, please update the relevant documentation files:

- Add docstrings to your functions and classes
- Update README.md if necessary
- Add usage examples to the appropriate documentation file
- Consider creating or updating a Jupyter notebook demonstrating the feature

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License. 