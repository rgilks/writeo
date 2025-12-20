# Agent Rules

This file contains rules for AI agents working on this codebase.

## Python Tooling

- **ALWAYS** use `uv` for Python package management and virtual environments.
- **NEVER** use `pip` directly. Use `uv pip ...` or `uv sync`.
- **NEVER** use `venv` or `virtualenv` directly. Use `uv venv`.
- **EXCEPTION**: The Modal SDK uses a method named `.pip_install()` for defining container images. This is part of the Modal API and should be preserved.
