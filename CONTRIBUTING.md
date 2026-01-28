# Contributing to AlignJuice

Thank you for your interest in contributing to AlignJuice!

## Development Setup

```bash
# Clone the repository
git clone https://github.com/Leeelics/alignjuice.git
cd alignjuice

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev,notebook]"
```

## Code Quality

We use the following tools:

- **ruff**: Linting and formatting
- **mypy**: Type checking
- **pytest**: Testing

Run all checks:

```bash
ruff check alignjuice/
ruff format alignjuice/
mypy alignjuice/
pytest
```

## Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Code Style

- Follow PEP 8
- Use type hints
- Write docstrings for public functions
- Keep functions focused and small
- Add tests for new features

## Reporting Issues

When reporting issues, please include:

- Python version
- AlignJuice version
- Steps to reproduce
- Expected vs actual behavior
- Error messages and stack traces
