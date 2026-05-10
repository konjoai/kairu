---
name: konjo-quality
description: Konjo Code Quality Framework — all gate definitions, thresholds, tools, and enforcement points. Auto-load when writing tests, reviewing code quality, refactoring, or when quality gate failures are mentioned. Applies the Three-Wall framework to prevent AI slop.
user-invocable: true
---

# Konjo Quality Framework — Agent Reference

## Why This Exists

AI-assisted code produces **1.7× more logical and correctness bugs** than traditional development (CodeRabbit 2026). AI agents change tests so broken code passes instead of fixing the code. AI self-review is architecturally circular — it checks code against itself, not against intent. This framework provides external ground truth via three independent walls that cannot be reasoned past.

The Konjo Critic (Wall 3) uses `claude-opus-4-6` in a separate session. The builder has blind spots from the construction process; the critic comes in cold with a different capability profile to reduce correlated failures.

---

## The Three Walls

| Wall | When | What | Blocks |
|------|------|------|--------|
| **Wall 1** | Pre-commit hook | Format, lint, dead-code scan, DRY (staged only), TODO scan | The commit |
| **Wall 2** | CI / GitHub Actions | Coverage, mutation, complexity, size, docs, review | The merge |
| **Wall 3** | Local only (disabled in CI) | Claude Opus adversarial review against 10 mandatory questions | The merge (when enabled) |

---

## Hard Quality Thresholds (all enforced by CI)

| Metric | Hard Block | Target | Tool |
|--------|-----------|--------|------|
| Line coverage | ≥ 80% | ≥ 95% | pytest-cov |
| Mutation survival (changed files) | ≤ 10% | 0% | mutmut |
| Cognitive complexity per function | ≤ 15 | ≤ 10 | radon |
| Lint violations | 0 | 0 | ruff |
| Dead code warnings | 0 | 0 | vulture |
| Undocumented public APIs | 0 | 0 | interrogate |
| Function body length | ≤ 50 lines | ≤ 30 lines | radon |
| File length | ≤ 500 lines | ≤ 300 lines | wc |
| DRY violations (>10L, >85% similar) | 0 | 0 | dry_check.py |
| Silent error swallowing | 0 | 0 | grep / ast check |
| Known CVEs in dependencies | 0 | 0 | safety / pip-audit |

---

## The Ten Review Questions (Wall 3 will ask all of these)

1. **Q1 Correctness** — Does this code actually do what it claims?
2. **Q2 Coverage Blind Spots** — What inputs would cause silent failure the tests won't catch?
3. **Q3 Dead Code** — Any unreachable code, unused variable, commented-out block?
4. **Q4 Documentation** — Every public API documented? Does it match the implementation?
5. **Q5 Error Handling** — Any errors swallowed? Any bare except? Fallbacks that mask real failures?
6. **Q6 DRY** — Any block of logic appearing >once at >85% similarity over >10 lines?
7. **Q7 Complexity** — Any function >50 lines, >15 cognitive complexity, any file >500 lines?
8. **Q8 Security** — Prompt injection? Logging sensitive data? Missing validation?
9. **Q9 Performance** — Any O(n²) where O(n log n) is obvious? Blocking I/O? Unnecessary allocation?
10. **Q10 Konjo Standard** — Is this seaworthy under 10,000 concurrent requests for 30 days?

---

## Running the Gates Locally

```bash
# Wall 1 equivalent:
ruff check . && ruff format --check .
python -m pytest tests/ --cov=. --cov-fail-under=80
vulture . --min-confidence 80

# DRY check:
python3 .konjo/scripts/dry_check.py --staged-only

# Wall 3 preview (requires ANTHROPIC_API_KEY):
git diff HEAD~1 | python3 .konjo/scripts/konjo_review.py
```
