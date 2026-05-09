"""Tests for kairu.evaluation — rubric scorers, comparison, batching."""
from __future__ import annotations

import json

import pytest

from kairu.evaluation import (
    CRITERIA,
    RUBRICS,
    TIE_EPSILON,
    Comparison,
    Evaluation,
    compare,
    evaluate,
    evaluate_batch,
    score_coherence,
    score_completeness,
    score_conciseness,
    score_fluency,
    score_relevance,
    score_safety,
    score_specificity,
    to_csv,
)


# --------------------------- per-criterion scorers ---------------------------


def test_relevance_overlap_increases_score() -> None:
    low, _ = score_relevance("python programming", "javascript framework")
    high, _ = score_relevance("python programming", "python programming language is fun")
    assert high > low


def test_relevance_returns_zero_for_empty_response() -> None:
    s, _ = score_relevance("hello world", "")
    assert s == 0.0


def test_coherence_penalises_repetition() -> None:
    repeat = "the cat sat on the mat the cat sat on the mat the cat sat on the mat"
    diverse = "the cat sat on the mat while a brown dog watched silently from outside"
    s_rep, _ = score_coherence("", repeat)
    s_div, _ = score_coherence("", diverse)
    assert s_div > s_rep


def test_coherence_short_response_falls_back_gracefully() -> None:
    s, detail = score_coherence("", "yes")
    assert 0.0 <= s <= 1.0
    assert detail["basis"] in {0.0, 1.0, 2.0, 3.0}


def test_conciseness_peaks_near_target_ratio() -> None:
    prompt = "Explain quantum entanglement"  # 3 tokens
    short, _ = score_conciseness(prompt, "yes")              # 1 token
    target_response = " ".join(["word"] * 12)                # 12 tokens ≈ 4×
    okay, _ = score_conciseness(prompt, target_response)
    runaway, _ = score_conciseness(prompt, " ".join(["w"] * 500))
    assert okay > short
    assert okay > runaway


def test_safety_flags_pii() -> None:
    clean, _ = score_safety("", "the answer is 42")
    leaky, hits = score_safety("", "Contact me at john.doe@example.com or 555-123-4567")
    assert clean == 1.0
    assert leaky < clean
    assert hits.get("email", 0) >= 1


def test_safety_multiple_categories_compound() -> None:
    s, hits = score_safety("", "SSN 123-45-6789, email a@b.co, key sk_AAAAAAAAAAAAAAAAAAAA")
    assert s < 1.0
    assert hits["categories_hit"] >= 3


def test_fluency_zero_on_empty() -> None:
    s, _ = score_fluency("", "")
    assert s == 0.0


def test_fluency_rewards_real_sentences() -> None:
    blob, _ = score_fluency("", "x x x x x x x x x x x x x x x x x x x x")
    real, _ = score_fluency(
        "",
        "The capital of France is Paris. It sits along the Seine river. Paris has many famous landmarks.",
    )
    assert real > blob


def test_specificity_rewards_proper_nouns_and_numbers() -> None:
    vague, _ = score_specificity("", "it depends on what you want and sometimes things vary a lot")
    specific, _ = score_specificity(
        "", "Python 3.11 was released in October 2022 by the PSF foundation."
    )
    assert specific > vague


def test_completeness_addresses_all_anchors() -> None:
    full, _ = score_completeness(
        "Compare python and rust",
        "Python is dynamic; Rust is statically compiled.",
    )
    partial, _ = score_completeness(
        "Compare python and rust",
        "Python is a great language for beginners.",
    )
    assert full > partial


# ------------------------------- evaluate() ---------------------------------


def test_evaluate_default_rubric_returns_all_default_criteria() -> None:
    ev = evaluate("hello world", "hello world is a classic phrase")
    names = {s.name for s in ev.scores}
    assert names == set(RUBRICS["default"].criteria)
    assert 0.0 <= ev.aggregate <= 1.0


def test_evaluate_custom_criteria_overrides_rubric() -> None:
    ev = evaluate("p", "r", criteria=["relevance", "fluency"])
    names = [s.name for s in ev.scores]
    assert names == ["relevance", "fluency"]


def test_evaluate_unknown_rubric_raises() -> None:
    with pytest.raises(ValueError):
        evaluate("p", "r", rubric="does_not_exist")


def test_evaluate_unknown_criterion_raises() -> None:
    with pytest.raises(ValueError):
        evaluate("p", "r", criteria=["invented_metric"])


def test_evaluate_negative_weight_rejected() -> None:
    with pytest.raises(ValueError):
        evaluate("p", "r", weights={"relevance": -1.0})


def test_evaluate_non_string_input_rejected() -> None:
    with pytest.raises(TypeError):
        evaluate(123, "ok")  # type: ignore[arg-type]


def test_evaluate_aggregate_is_weighted_mean() -> None:
    ev = evaluate(
        "Compare python and rust",
        "Python is dynamic; Rust is statically compiled.",
        rubric="helpfulness",
    )
    total_w = sum(s.weight for s in ev.scores)
    weighted = sum(s.score * s.weight for s in ev.scores)
    assert ev.aggregate == pytest.approx(weighted / total_w, abs=1e-9)


def test_evaluate_to_dict_round_trip_is_json_safe() -> None:
    ev = evaluate("hi", "hello there")
    payload = json.dumps(ev.to_dict())
    assert "aggregate" in payload
    assert "scores" in payload


# -------------------------------- compare() ---------------------------------


def test_compare_picks_clear_winner() -> None:
    c = compare(
        "Explain how photosynthesis works in detail",
        "Photosynthesis is the process by which plants convert sunlight into chemical energy stored in glucose, using chlorophyll in their leaves.",
        "idk",
    )
    assert c.winner == "a"
    assert c.margin > 0.05


def test_compare_tie_within_epsilon() -> None:
    c = compare("hello", "hello world", "hello world")
    assert c.winner == "tie"
    assert c.margin < TIE_EPSILON + 1e-9


def test_compare_per_criterion_winner_consistent_with_delta() -> None:
    c = compare("p", "long detailed answer with many words", "no")
    for cc in c.per_criterion:
        if cc.delta > TIE_EPSILON:
            assert cc.winner == "a"
        elif cc.delta < -TIE_EPSILON:
            assert cc.winner == "b"
        else:
            assert cc.winner == "tie"


def test_compare_labels_propagate() -> None:
    c = compare("p", "r1", "r2", label_a="gpt-3.5", label_b="claude-haiku")
    assert c.label_a == "gpt-3.5"
    assert c.label_b == "claude-haiku"


def test_compare_unknown_rubric_raises() -> None:
    with pytest.raises(ValueError):
        compare("p", "a", "b", rubric="nope")


# ------------------------------- batch & csv --------------------------------


def test_batch_evaluates_each_item() -> None:
    items = [
        {"id": "x", "prompt": "q1", "response": "a1"},
        {"id": "y", "prompt": "q2", "response": "a2 with more words"},
    ]
    rows = evaluate_batch(items)
    assert len(rows) == 2
    assert rows[0]["id"] == "x"
    assert "aggregate" in rows[0]
    assert any(k.startswith("score_") for k in rows[0])


def test_batch_missing_keys_raises() -> None:
    with pytest.raises(ValueError):
        evaluate_batch([{"prompt": "only prompt"}])


def test_batch_assigns_default_id_when_missing() -> None:
    rows = evaluate_batch([{"prompt": "q", "response": "a"}])
    assert rows[0]["id"] == "0"


def test_to_csv_emits_header_plus_rows() -> None:
    rows = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    out = to_csv(rows)
    lines = out.strip().split("\n")
    assert lines[0] == '"a","b"'
    assert lines[1] == '"1","2"'
    assert lines[2] == '"3","4"'


def test_to_csv_empty_returns_empty_string() -> None:
    assert to_csv([]) == ""


# --------------------------- registry sanity --------------------------------


def test_all_default_rubric_criteria_exist_in_registry() -> None:
    for rubric in RUBRICS.values():
        for c in rubric.criteria:
            assert c in CRITERIA


def test_rubrics_have_unique_names() -> None:
    names = [r.name for r in RUBRICS.values()]
    assert len(names) == len(set(names))


def test_evaluate_all_criteria_bounded_to_unit_interval() -> None:
    ev = evaluate(
        "The quick brown fox jumps over the lazy dog",
        "Foxes and dogs are common in fables. Aesop wrote many such tales in 600 BC.",
        criteria=list(CRITERIA),
    )
    for s in ev.scores:
        assert 0.0 <= s.score <= 1.0
