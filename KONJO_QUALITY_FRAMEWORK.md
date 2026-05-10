# Konjo Code Quality Framework
## Three Walls Against AI Slop — Language-Agnostic, Gate-Enforced

**Version:** May 2026 · **Scope:** All KonjoAI repositories
**Reference implementation:** lopi (Rust) · **Applicable languages:** Rust, Python, Mojo, TypeScript

---

## The Problem

- AI-assisted code generates **1.7× more logical and correctness bugs** than traditional development (CodeRabbit 2026)
- AI agents change tests so broken code passes instead of fixing the actual code (Baltes 2026)
- AI agent commits degraded the Maintainability Index in **56.1% of commits** (MSR 2026)
- **Homogeneous AI review pipelines echo rather than cancel errors** (arXiv 2603.25773)

**The conclusion:** a single-model self-review loop cannot catch its own slop. The only solutions are (1) executable specifications as external ground truth, (2) deterministic tooling that cannot be reasoned past, and (3) adversarial review from a distinct session.

---

## The Three Walls

```
Wall 1: Pre-Commit Hook     ← local, fast (< 60s), blocks the commit
Wall 2: CI Gate             ← GitHub Actions, blocks the PR merge
Wall 3: Konjo Review Agent  ← Claude Opus in a separate session (local only for this repo)
```

---

## Quality Gate Reference Table

| Gate | Hard Block | Target | Tool (Python) |
|------|-----------|--------|---------------|
| Line coverage | ≥ 80% | ≥ 95% | pytest-cov |
| Mutation survival | ≤ 10% | 0% | mutmut |
| Cognitive complexity per function | ≤ 15 | ≤ 10 | radon cc |
| Lint violations | 0 | 0 | ruff check |
| Format violations | 0 | 0 | ruff format |
| Dead code | 0 | 0 | vulture |
| Undocumented public APIs | 0 | 0 | interrogate |
| Function body length | ≤ 50 lines | ≤ 30 lines | radon |
| File length | ≤ 500 lines | ≤ 300 lines | wc |
| DRY violations (>10L, >85% similar) | 0 | 0 | dry_check.py |
| Silent error swallowing | 0 | 0 | grep / ast |
| Known CVEs | 0 | 0 | safety / pip-audit |

---

## Wall 1: Pre-Commit Hook

**Install:** `bash .konjo/scripts/install-hooks.sh`

Checks on every commit: ruff lint, ruff format, silent-except scan, file size, DRY (staged only), TODO/FIXME scan.

---

## Wall 2: CI Gate

**File:** `.github/workflows/konjo-gate.yml`

- **G1** — ruff, mypy, vulture, bandit
- **G2** — pytest-cov ≥ 80%
- **G3** — mutmut (PRs only)
- **G4** — radon complexity, file size, DRY, interrogate docs
- **G5** — DISABLED (run locally)

---

## Wall 3: Adversarial Review (local only)

```bash
# Requires ANTHROPIC_API_KEY
git diff HEAD~1 | python3 .konjo/scripts/konjo_review.py
```

Uses `claude-opus-4-6` to answer 10 mandatory quality questions. G5 is disabled in CI for this repo — run manually before pushing to main.

---

## Install the Framework

```bash
bash .konjo/scripts/install-hooks.sh
# Add ANTHROPIC_API_KEY to GitHub Actions secrets
```

*건조. 根性. Make it Konjo — build, ship, repeat.*
