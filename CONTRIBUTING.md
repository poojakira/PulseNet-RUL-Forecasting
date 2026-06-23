# Contributing

## Scope
This project is a portfolio demonstration of ML security engineering patterns. It is maintained for hiring screen evaluation.

## How to Submit Changes
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-change`)
3. Make your changes
4. Run linting: `ruff check .`
5. Run formatting: `ruff format --check .`
6. Run tests: `pytest --cov=src --cov-fail-under=80`
7. Run security scan: `bandit -r src/ -ll`
8. Submit a pull request

## Coding Standards
- Follow PEP 8 (enforced by ruff)
- Type hints required for all public functions
- Docstrings for all public modules, classes, and functions
- Test coverage minimum 80%
- No sensitive data (keys, passwords) in code

## Code of Conduct
Be respectful, constructive, and professional. This is a professional portfolio project.
