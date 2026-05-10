#!/usr/bin/env python3
"""
Konjo Adversarial Review Agent — Wall 3.

Critic model: claude-opus-4-6
The builder has blind spots from the construction process; the critic comes in
cold from a distinct session with a different capability profile.

Exit codes:
  0 — APPROVED or WARNING-only
  1 — BLOCKER found
  2 — API error (treat as soft failure)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

CRITIC_MODEL = "claude-opus-4-6"
MAX_DIFF_CHARS = 80_000
MAX_TOKENS = 4096
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0

SYSTEM_PROMPT = """\
You are the Konjo Adversarial Reviewer — an independent critic whose role is to
find flaws that the builder missed. You were NOT involved in writing this code.
You have no loyalty to the implementation choices made. Your only loyalty is to
the ten quality standards below.

THE TEN MANDATORY REVIEW QUESTIONS — answer each explicitly:

Q1  CORRECTNESS — Does this code actually do what it claims? Logical errors, off-by-ones, race conditions?
Q2  COVERAGE BLIND SPOTS — What inputs would cause silent failure the test suite won't catch?
Q3  DEAD CODE — Any unreachable code, unused variable, commented-out block?
Q4  DOCUMENTATION — Every public API documented? Does it match the implementation?
Q5  ERROR HANDLING — Errors swallowed? Bare except? unwrap outside tests? Fallbacks masking failures?
Q6  DRY VIOLATION — Any block >10 lines at >85% similarity appearing more than once?
Q7  COMPLEXITY AND SIZE — Any function >50 lines, >15 cognitive complexity, any file >500 lines?
Q8  SECURITY — Prompt injection? Logging sensitive data? Missing validation? Hardcoded secrets?
Q9  PERFORMANCE — O(n²) where O(n log n) is obvious? Blocking async? Unnecessary allocation?
Q10 KONJO STANDARD — Seaworthy under 10,000 requests for 30 days? What would you cut?

VERDICT RULES:
- BLOCKER: any violation of Q1, Q3, Q5, Q8, any function >100 lines, any public API undocumented
- WARNING: Q2 partial coverage, Q6 minor duplication, Q7 approaching limits, Q9 cold-path perf
- APPROVED: all ten questions pass without reservation

OUTPUT FORMAT — respond ONLY with valid JSON:
{
  "verdict": "APPROVED" | "WARNING" | "BLOCKER",
  "summary": "one honest sentence",
  "questions": {
    "Q1": {"verdict": "PASS"|"WARN"|"BLOCK", "finding": "..."},
    "Q2": {"verdict": "PASS"|"WARN"|"BLOCK", "finding": "..."},
    "Q3": {"verdict": "PASS"|"WARN"|"BLOCK", "finding": "..."},
    "Q4": {"verdict": "PASS"|"WARN"|"BLOCK", "finding": "..."},
    "Q5": {"verdict": "PASS"|"WARN"|"BLOCK", "finding": "..."},
    "Q6": {"verdict": "PASS"|"WARN"|"BLOCK", "finding": "..."},
    "Q7": {"verdict": "PASS"|"WARN"|"BLOCK", "finding": "..."},
    "Q8": {"verdict": "PASS"|"WARN"|"BLOCK", "finding": "..."},
    "Q9": {"verdict": "PASS"|"WARN"|"BLOCK", "finding": "..."},
    "Q10": {"verdict": "PASS"|"WARN"|"BLOCK", "finding": "..."}
  },
  "blockers": ["file:line — specific blocking issue with recommended fix"],
  "warnings": ["file:line — non-blocking improvement"],
  "approved_aspects": ["what was genuinely well done"]
}
"""


def _load_anthropic():
    try:
        import anthropic
        return anthropic
    except ImportError:
        print("ERROR: pip install anthropic", file=sys.stderr)
        raise


def _call_api(diff_text: str, anthropic_module) -> dict:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set.")

    client = anthropic_module.Anthropic(api_key=api_key)

    if len(diff_text) > MAX_DIFF_CHARS:
        diff_text = diff_text[:MAX_DIFF_CHARS] + "\n\n[DIFF TRUNCATED]"

    user_content = (
        "Review this pull request diff against the ten Konjo quality standards.\n\n"
        f"<diff>\n{diff_text}\n</diff>"
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=CRITIC_MODEL,
                max_tokens=MAX_TOKENS,
                system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user_content}],
            )
            raw = response.content[0].text.strip()
            usage = response.usage
            print(
                f"[konjo-review] tokens: input={usage.input_tokens} output={usage.output_tokens}",
                file=sys.stderr,
            )
            return json.loads(raw)
        except (anthropic_module.RateLimitError, anthropic_module.APIStatusError) as exc:
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2**attempt)
                print(f"[konjo-review] retrying in {delay:.0f}s...", file=sys.stderr)
                time.sleep(delay)
            else:
                raise
        except json.JSONDecodeError as exc:
            raise ValueError(f"Non-JSON response: {raw}") from exc

    raise RuntimeError("Exhausted retries")


def _render_human(result: dict) -> str:
    lines = ["# Konjo Adversarial Review Report\n"]
    verdict = result.get("verdict", "UNKNOWN")
    emoji = {"APPROVED": "✅", "WARNING": "⚠️", "BLOCKER": "\U0001f6ab"}.get(verdict, "❓")
    lines.append(f"## Verdict: {emoji} {verdict}\n")
    lines.append(f"**Summary:** {result.get('summary', '')}\n")
    for b in result.get("blockers", []):
        lines.append(f"- \U0001f6ab {b}")
    for w in result.get("warnings", []):
        lines.append(f"- ⚠️ {w}")
    lines.append("\n## Question-by-Question\n")
    emoji_map = {"PASS": "✅", "WARN": "⚠️", "BLOCK": "\U0001f6ab"}
    for q, data in result.get("questions", {}).items():
        v = data.get("verdict", "?")
        lines.append(f"**{q}** {emoji_map.get(v, '?')} {v} — {data.get('finding', '')}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Konjo Adversarial Review — Wall 3")
    diff_group = parser.add_mutually_exclusive_group()
    diff_group.add_argument("--diff-file")
    diff_group.add_argument("--diff")
    parser.add_argument("--output")
    parser.add_argument("--json", action="store_true", dest="json_out")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--soft-fail", action="store_true")
    args = parser.parse_args()

    if args.diff:
        diff_text = args.diff
    elif args.diff_file:
        diff_text = Path(args.diff_file).read_text()
    else:
        if sys.stdin.isatty():
            print("ERROR: provide --diff-file or pipe a diff to stdin", file=sys.stderr)
            return 2
        diff_text = sys.stdin.read()

    if not diff_text.strip():
        print("[konjo-review] Empty diff — nothing to review.", file=sys.stderr)
        return 0

    if args.dry_run:
        print(f"[konjo-review] DRY RUN — model: {CRITIC_MODEL}, chars: {len(diff_text)}", file=sys.stderr)
        return 0

    try:
        anthropic = _load_anthropic()
        result = _call_api(diff_text, anthropic)
    except (ImportError, ValueError, RuntimeError) as exc:
        print(f"[konjo-review] ERROR: {exc}", file=sys.stderr)
        print("[konjo-review] Soft-failing.", file=sys.stderr)
        return 0

    verdict = result.get("verdict", "UNKNOWN")
    has_blockers = verdict == "BLOCKER" or bool(result.get("blockers"))

    if args.json_out:
        print(json.dumps(result, indent=2))
    else:
        report = _render_human(result)
        if args.output:
            Path(args.output).write_text(report)
        else:
            print(report)

    if has_blockers and not args.soft_fail:
        print(f"\n[konjo-review] VERDICT: {verdict} — merge blocked.", file=sys.stderr)
        return 1

    print(f"[konjo-review] VERDICT: {verdict}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
