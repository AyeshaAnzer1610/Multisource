# Contributing to NeuroFormer

We welcome contributions to NeuroFormer! This document provides guidelines for contributing.

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/neuroformer.git
   cd neuroformer
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   pip install -e .
   ```

## Code Style

We follow PEP 8 style guidelines. Please ensure your code:
- Uses 4 spaces for indentation
- Has maximum line length of 100 characters
- Includes docstrings for all functions and classes
- Has type hints for function arguments and returns

Run formatting and linting:
```bash
black neuroformer/
flake8 neuroformer/
mypy neuroformer/
```

## Testing

All new features should include unit tests:
```bash
pytest tests/
pytest tests/ --cov=neuroformer --cov-report=html
```

## Pull Request Process

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit:
   ```bash
   git add .
   git commit -m "Description of changes"
   ```

3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Open a Pull Request on GitHub

5. Ensure CI passes and address any review comments

## Reporting Issues

When reporting issues, please include:
- Python version
- PyTorch version
- Operating system
- Complete error traceback
- Minimal code to reproduce the issue

## Questions?

Open an issue or contact the maintainers directly.
