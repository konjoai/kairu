---
paths:
  - "**/*.py"
---
# Python Conventions

- No bare `except:` or `except Exception:` — catch specific exceptions; log with `logging.warning` and re-raise
- No mutable default arguments — use `None` and assign inside the function body
- No implicit `None` returns from functions with documented return types
- `ruff check` and `ruff format` must be clean before every commit
- `mypy --strict` must be clean (each `# type: ignore` requires a justification comment)
- No dead code — `vulture` zero tolerance; remove unused functions, variables, and imports
- `radon cc -n C` zero functions above grade C (cyclomatic complexity > 10)
- `interrogate --fail-under 100` on all public API surfaces
- `bandit -ll` zero high-severity security issues
- Prefer `pathlib.Path` over `os.path` for file operations
- Use `logging` not `print` in production code
- `__all__` must be defined for every public module
