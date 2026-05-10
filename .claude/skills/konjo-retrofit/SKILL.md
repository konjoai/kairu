---
name: konjo-retrofit
description: Retrofit the Konjo Quality Framework onto an existing repo that predates it. Use when asked to add konjo quality gates, improve code quality, audit an existing codebase, or run a quality sprint on any repo. Provides the step-by-step migration plan and triage protocol.
user-invocable: true
---

# Konjo Retrofit — Existing Repo Quality Migration

## The Problem With Retrofitting Blind

Installing hard quality gates on an existing codebase without measuring first causes one of two outcomes:
1. **Gates fail on day 1** — blocks all work, team disables the gates in frustration
2. **Gates are set too loose** — they pass everything, provide no value

The Retrofit Protocol solves this by measuring before gating, then ratcheting up incrementally.

---

## Step 1 — Baseline Audit (measure everything, fix nothing yet)

```bash
# Coverage baseline
python -m pytest --cov --cov-report=json > coverage_baseline.json

# Lint baseline
ruff check --output-format json > ruff_baseline.json

# Dead code
vulture . --min-confidence 60 > vulture_baseline.txt

# Complexity
radon cc . -n C > complexity_baseline.txt

# DRY
python3 .konjo/scripts/dry_check.py --json > dry_baseline.json

# File sizes
find . -name "*.py" | grep -v __pycache__ | xargs wc -l | sort -n | tail -20 > large_files.txt

# Mutation testing (sample)
mutmut run --paths-to-mutate src/ 2>&1 | tail -50 > mutation_sample.txt
```

---

## Step 2 — Triage

| Priority | Category | Definition | Handle |
|----------|----------|------------|--------|
| P0 | **CRITICAL** | Security issues, data corruption bugs | Fix immediately |
| P1 | **DEBT** | Coverage < 60%, bare except, undocumented public APIs | Fix in first 2 sprints |
| P2 | **STYLE** | Length violations, moderate duplication | Fix incrementally |

---

## Step 3 — Coverage Ratchet

| Sprint | Coverage Gate |
|--------|-------------|
| 0 (install) | current - 2% |
| 1 | current |
| 2 | current + 5% |
| N | 80% (hard floor) |
| Long-term | 95% (target) |

---

## The Shipbuilder's Checklist (Final Verification)

- [ ] The test suite runs in < 5 minutes
- [ ] A PR that deletes a function without updating its callers would fail CI
- [ ] A PR that introduces a bare `except:` would fail CI
- [ ] A PR that drops coverage below 80% would fail CI
- [ ] A PR is reviewed by the Konjo Critic before merge
- [ ] The code can be read and extended by someone who wasn't there when it was written
