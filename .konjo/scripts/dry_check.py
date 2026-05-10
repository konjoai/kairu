#!/usr/bin/env python3
"""
Konjo DRY Checker — cross-language duplicate block detector.

Algorithm (O(N) per file, stdlib-only):
  1. Tokenize each source file into normalized line sequences
  2. Build rolling fingerprints using polynomial hash (Rabin-Karp style)
  3. Cluster hash collisions by exact match on normalized content
  4. Report clusters with >= 2 members where similarity >= threshold

Language support: .rs .py .mojo .ts .js (and .tsx .jsx)
Exit codes: 0=clean, 1=violations found, 2=config error
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterator


def _strip_rust_comments(text: str) -> str:
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return text


def _strip_c_comments(text: str) -> str:
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return text


def _strip_python_comments(text: str) -> str:
    text = re.sub(r'""".*?"""', "", text, flags=re.DOTALL)
    text = re.sub(r"'''.*?'''", "", text, flags=re.DOTALL)
    text = re.sub(r"#[^\n]*", "", text)
    return text


_STRIPPERS: dict[str, object] = {
    ".rs": _strip_rust_comments,
    ".py": _strip_python_comments,
    ".mojo": _strip_python_comments,
    ".ts": _strip_c_comments,
    ".tsx": _strip_c_comments,
    ".js": _strip_c_comments,
    ".jsx": _strip_c_comments,
}


def _normalize_lines(path: Path) -> list[str]:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return []
    stripper = _STRIPPERS.get(path.suffix, lambda t: t)
    text = stripper(text)
    lines = []
    for line in text.splitlines():
        normalized = " ".join(line.split())
        if normalized:
            lines.append(normalized)
    return lines


def _window_fingerprints(lines: list[str], window: int) -> Iterator[tuple[str, int]]:
    if len(lines) < window:
        return
    for i in range(len(lines) - window + 1):
        block = "\n".join(lines[i : i + window])
        digest = hashlib.sha256(block.encode()).hexdigest()
        yield digest, i


def _similarity(a: list[str], b: list[str]) -> float:
    return SequenceMatcher(None, a, b).ratio()


SUPPORTED_EXTENSIONS = {".rs", ".py", ".mojo", ".ts", ".tsx", ".js", ".jsx"}
SKIP_DIRS = {"target", ".git", "__pycache__", "node_modules", ".venv", "venv", "dist", "build"}


def _iter_sources(root: Path, extensions: set[str]) -> Iterator[Path]:
    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix in extensions and path.is_file():
            yield path


def _staged_files(root: Path, extensions: set[str]) -> list[Path]:
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=root, capture_output=True, text=True, check=False
    )
    paths = []
    for line in result.stdout.splitlines():
        p = root / line
        if p.suffix in extensions and p.is_file():
            paths.append(p)
    return paths


def _changed_files(root: Path, extensions: set[str]) -> list[Path]:
    result = subprocess.run(
        ["git", "diff", "--name-only", "origin/main...HEAD"],
        cwd=root, capture_output=True, text=True, check=False
    )
    paths = []
    for line in result.stdout.splitlines():
        p = root / line
        if p.suffix in extensions and p.is_file():
            paths.append(p)
    return paths


def find_duplicates(
    files: list[Path],
    all_files: list[Path],
    threshold: float,
    min_lines: int,
) -> list[dict]:
    index: dict[str, list[tuple[Path, int, list[str]]]] = defaultdict(list)
    file_lines: dict[Path, list[str]] = {}

    for path in all_files:
        lines = _normalize_lines(path)
        file_lines[path] = lines
        for digest, start in _window_fingerprints(lines, min_lines):
            index[digest].append((path, start, lines))

    violations = []
    seen_pairs: set[frozenset] = set()

    for target_path in files:
        if target_path not in file_lines:
            lines = _normalize_lines(target_path)
            file_lines[target_path] = lines
        else:
            lines = file_lines[target_path]

        for digest, start in _window_fingerprints(lines, min_lines):
            matches = index.get(digest, [])
            target_block = lines[start : start + min_lines]

            for other_path, other_start, other_lines in matches:
                if other_path == target_path and other_start == start:
                    continue
                pair_key = frozenset([
                    f"{target_path}:{start}",
                    f"{other_path}:{other_start}",
                ])
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                other_block = other_lines[other_start : other_start + min_lines]
                sim = _similarity(target_block, other_block)
                if sim >= threshold:
                    violations.append({
                        "file_a": str(target_path),
                        "line_a": start + 1,
                        "file_b": str(other_path),
                        "line_b": other_start + 1,
                        "similarity": round(sim, 3),
                        "lines": min_lines,
                        "sample": "\n".join(target_block[:5]) + ("..." if len(target_block) > 5 else ""),
                    })

    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description="Konjo DRY Checker")
    parser.add_argument("--root", default=None)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--min-lines", type=int, default=10)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--staged-only", action="store_true")
    mode.add_argument("--changed-only", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_out")
    parser.add_argument("--report", help="Write JSON report to file")
    parser.add_argument("--warn-only", action="store_true")
    parser.add_argument("--extensions", default=",".join(sorted(SUPPORTED_EXTENSIONS)))
    args = parser.parse_args()

    if args.root:
        root = Path(args.root)
    else:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=False
        )
        root = Path(result.stdout.strip()) if result.returncode == 0 else Path(".")

    extensions = {e.strip() for e in args.extensions.split(",") if e.strip()}
    all_files = list(_iter_sources(root, extensions))

    if args.staged_only:
        scan_targets = _staged_files(root, extensions)
        if not scan_targets:
            if not args.json_out:
                print("[dry-check] No staged source files to check.")
            return 0
    elif args.changed_only:
        scan_targets = _changed_files(root, extensions)
        if not scan_targets:
            if not args.json_out:
                print("[dry-check] No changed source files to check.")
            return 0
    else:
        scan_targets = all_files

    violations = find_duplicates(all_files, scan_targets, args.threshold, args.min_lines)

    report = {
        "duplicates": violations,
        "count": len(violations),
        "threshold": args.threshold,
        "min_lines": args.min_lines,
        "scanned": len(scan_targets),
    }

    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))

    if args.json_out:
        print(json.dumps(report, indent=2))
    else:
        if violations:
            print(f"[dry-check] ❌ {len(violations)} DRY violation(s) found:\n")
            for v in violations:
                print(
                    f"  {v['file_a']}:{v['line_a']} ↔ "
                    f"{v['file_b']}:{v['line_b']} "
                    f"({v['similarity']*100:.0f}% similar, {v['lines']}+ lines)"
                )
                print(f"    Sample: {v['sample'][:80]}")
            print("\nAbstract duplicate logic into a shared function or module.")
        else:
            print(f"[dry-check] ✅ No DRY violations ({len(scan_targets)} files scanned).")

    if violations and not args.warn_only:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
