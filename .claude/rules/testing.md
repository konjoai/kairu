---
paths:
  - "**/test_*.py"
  - "**/*_test.py"
  - "**/tests/**"
---
# Testing Rules

A sprint is NEVER complete until all tests pass.
100% coverage is the floor — every code file needs a corresponding test file.

**Unit:** deterministic, isolated functions.
**Integration:** module interactions, DB boundaries, API handoffs.
**E2E:** full pipeline end-to-end — no mocking the core logic under test.
**CLI:** new flags must be tested for expected output and failure modes.

Anti-mocking rule: E2E tests must test reality. Never mock the DB or network in E2E tests.
Never commit with known failing tests. `python -m pytest` must be green before `git push`.
