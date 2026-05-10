"""Tests for kairu.rubrics — the eight named prism rubrics."""
from __future__ import annotations

import re

import pytest

from kairu.evaluation import CRITERIA, RUBRICS, evaluate
from kairu.rubrics import RUBRIC_DEFS, rubric_color, rubric_criteria, rubric_names


EXPECTED_NAMES = (
    "helpfulness", "accuracy", "safety", "coherence",
    "conciseness", "creativity", "groundedness", "tone",
)
HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")


def test_all_eight_rubrics_present() -> None:
    assert set(EXPECTED_NAMES) <= set(RUBRIC_DEFS)
    assert set(EXPECTED_NAMES) <= set(RUBRICS)


def test_rubric_names_returns_canonical_order() -> None:
    names = rubric_names()
    for n in EXPECTED_NAMES:
        assert n in names
    # Canonical order must place helpfulness first, tone last.
    assert names[0] == "helpfulness"
    assert names[-1] == "tone"


def test_each_rubric_has_color_in_hex() -> None:
    for name in EXPECTED_NAMES:
        c = rubric_color(name)
        assert HEX_RE.match(c), f"{name} → {c} is not #RRGGBB"


def test_colors_unique_across_rubrics() -> None:
    colors = [RUBRIC_DEFS[n]["color"] for n in EXPECTED_NAMES]
    assert len(set(colors)) == len(colors)


def test_each_rubric_weights_reference_real_criteria() -> None:
    for name in EXPECTED_NAMES:
        for criterion in rubric_criteria(name):
            assert criterion in CRITERIA, f"{name} references unknown criterion {criterion}"


def test_each_rubric_weights_are_positive() -> None:
    for name in EXPECTED_NAMES:
        weights = RUBRIC_DEFS[name]["weights"]
        assert all(w > 0 for w in weights.values()), f"{name} has non-positive weight"


def test_each_rubric_runs_end_to_end() -> None:
    prompt = "Explain how a binary search tree maintains O(log n) lookup."
    response = (
        "A binary search tree keeps O(log n) lookup by storing values so that the "
        "left subtree of any node holds smaller keys and the right subtree holds "
        "larger ones. Each comparison halves the remaining search space, giving "
        "logarithmic depth on a balanced tree."
    )
    for name in EXPECTED_NAMES:
        ev = evaluate(prompt, response, rubric=name)
        assert ev.rubric == name
        assert 0.0 <= ev.aggregate <= 1.0


def test_unknown_rubric_color_raises() -> None:
    with pytest.raises(KeyError):
        rubric_color("does_not_exist")


def test_unknown_rubric_criteria_raises() -> None:
    with pytest.raises(KeyError):
        rubric_criteria("does_not_exist")


def test_safety_rubric_safety_weight_dominates() -> None:
    weights = RUBRIC_DEFS["safety"]["weights"]
    other_max = max(w for k, w in weights.items() if k != "safety")
    assert weights["safety"] > 2 * other_max


def test_creativity_rubric_downweights_relevance() -> None:
    weights = RUBRIC_DEFS["creativity"]["weights"]
    # Relevance < 1 — creative answers diverge from prompt vocabulary.
    assert weights["relevance"] < 1.0


def test_groundedness_emphasises_completeness() -> None:
    weights = RUBRIC_DEFS["groundedness"]["weights"]
    assert weights["completeness"] >= max(weights.values())


def test_descriptions_non_empty() -> None:
    for name in EXPECTED_NAMES:
        assert len(RUBRIC_DEFS[name]["description"]) > 10
