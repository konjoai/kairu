---
name: konjo-boot
description: Boot a Konjo session for kairu. Produces a Session Brief, runs Discovery, identifies the next sprint. Use at the start of any work session or when invoked with /konjo.
user-invocable: true
---
# Konjo Session Boot — kairu

## Step 1 — Read
Read in order: CLAUDE.md, README.md, CHANGELOG.md, PLAN.md, docs/ (if it exists).
Do not skip. Do not assume contents.

## Step 2 — Session Brief
```
REPO         kairu — real-time inference optimizer for LLMs (speculative decoding, KV cache, benchmarking)
LAST SHIPPED [most recent meaningful change from CHANGELOG.md]
OPEN WORK    [stated next steps from PLAN.md]
BLOCKERS     [failing tests, broken modules, open issues]
HEALTH       [Green / Yellow / Red — one line]
```
Unknown is stated as unknown. Fabricated state is a lie to the next session.

## Step 3 — Discovery (कोहजो)
Before executing any sprint, ask:
- What has shipped in LLM inference optimization since this repo last moved?
- Are there new papers on speculative decoding, early-exit, or KV cache strategies?
- What would a researcher building inference optimizers know today that kairu doesn't reflect?

Search: arXiv (speculative decoding, LLM inference, KV cache), GitHub (inference optimization frameworks), HuggingFace (relevant models/benchmarks).

## Step 4 — Identify Work
If PLAN.md exists: load it, validate against codebase, flag drift.
If no plan: run Discovery Protocol → propose a sprint.

## Invocation Keywords
- `konjo`
- `konjo kairu`
- `kairu konjo`
- `read KONJO_PROMPT.md and begin`
