"""Tests for kairu.squish_eval — quantization-tier quality evaluation."""
from __future__ import annotations

import pytest

from kairu.squish_eval import (
    DegradationReport,
    QuantTier,
    RubricScore,
    SquishEvaluator,
    quality_degradation_report,
    recommended_quant_tier,
)


# ── SquishEvaluator unit tests ───────────────────────────────────────────


def test_score_identical_output_against_reference_is_high():
    ev = SquishEvaluator()
    ref = "The quick brown fox jumps over the lazy dog."
    s = ev.score(ref, reference=ref)
    # Exact match → correctness, faithfulness, completeness all 1.0; safety 1.0.
    assert s.correctness == pytest.approx(1.0)
    assert s.faithfulness == pytest.approx(1.0)
    assert s.completeness == pytest.approx(1.0)
    assert s.safety == pytest.approx(1.0)
    assert 0.0 <= s.fluency <= 1.0
    assert s.aggregate >= 0.85
    assert isinstance(s, RubricScore)


def test_score_empty_output_with_reference_is_low():
    ev = SquishEvaluator()
    s = ev.score("", reference="something meaningful here")
    assert s.correctness == 0.0
    assert s.completeness == 0.0
    assert s.fluency == 0.0  # empty raw → fluency 0
    assert s.aggregate < 0.5


def test_score_no_reference_is_neutral():
    """Without a reference, faithfulness/completeness/correctness fall back to 1.0."""
    ev = SquishEvaluator()
    s = ev.score("Hello world.")
    assert s.correctness == 1.0
    assert s.faithfulness == 1.0
    assert s.completeness == 1.0
    # Safety neutral, fluency depends on heuristic.
    assert s.safety == 1.0


def test_safety_penalty_for_banned_substring():
    ev = SquishEvaluator(banned=["forbidden"])
    s = ev.score("This contains forbidden content.")
    assert s.safety == pytest.approx(0.5)
    s2 = ev.score("This contains forbidden forbidden content.")
    # Single substring match counted once per banned token, not per occurrence.
    assert s2.safety == pytest.approx(0.5)


def test_score_batch_length_mismatch_raises():
    ev = SquishEvaluator()
    with pytest.raises(ValueError):
        ev.score_batch(["a", "b"], references=["only one"])


def test_score_rejects_non_string_output():
    ev = SquishEvaluator()
    with pytest.raises(TypeError):
        ev.score(123)  # type: ignore[arg-type]


def test_repetition_window_validation():
    with pytest.raises(ValueError):
        SquishEvaluator(repetition_window=0)


# ── quality_degradation_report ───────────────────────────────────────────


def test_degradation_report_identical_outputs_show_zero_delta():
    """When every tier returns the same outputs as the baseline, all deltas are 0."""
    baseline = ["Paris is the capital of France.", "2 + 2 = 4."]
    tiers = {
        "int8": list(baseline),
        "int4": list(baseline),
        "int2": list(baseline),
    }
    report = quality_degradation_report(baseline, tiers)
    assert isinstance(report, DegradationReport)
    assert len(report.tiers) == 3
    for t in report.tiers:
        assert t.aggregate_delta == pytest.approx(0.0)
        assert t.retention_pct == pytest.approx(100.0)
        for v in t.per_criterion.values():
            assert v == pytest.approx(0.0, abs=1e-9)


def test_degradation_report_quantization_introduces_loss():
    """Progressively garbled outputs should monotonically lose retention."""
    baseline = [
        "The capital of France is Paris.",
        "Photosynthesis converts light energy into chemical energy.",
    ]
    tiers = {
        "int8": [
            "The capital of France is Paris.",
            "Photosynthesis converts light energy into chemical energy.",
        ],
        "int4": [
            "The capital of France is Lyon.",  # wrong city
            "Photosynthesis is a plant thing with light.",  # vague
        ],
        "int2": [
            "France big city.",  # incoherent
            "Plants light food.",
        ],
    }
    report = quality_degradation_report(baseline, tiers)
    by_tier = {t.tier: t for t in report.tiers}
    assert by_tier["int8"].retention_pct >= by_tier["int4"].retention_pct
    assert by_tier["int4"].retention_pct >= by_tier["int2"].retention_pct
    assert by_tier["int2"].aggregate_delta < 0.0
    # Tiers ordered by descending bit-width.
    assert [t.tier for t in report.tiers] == ["int8", "int4", "int2"]


def test_degradation_report_length_mismatch_raises():
    with pytest.raises(ValueError, match="expected 2"):
        quality_degradation_report(
            baseline_outputs=["a", "b"],
            quantized_outputs={"int4": ["only-one"]},
        )


def test_degradation_report_empty_baseline_raises():
    with pytest.raises(ValueError, match="non-empty"):
        quality_degradation_report(baseline_outputs=[], quantized_outputs={"int4": []})


def test_degradation_report_with_explicit_references():
    """When references are given, baseline retention need not be 100%."""
    refs = ["Paris is the capital of France."]
    baseline = ["Lyon is the capital of France."]  # baseline already wrong
    tiers = {"int4": ["Berlin is the capital of France."]}  # also wrong, differently
    report = quality_degradation_report(baseline, tiers, references=refs)
    # Both baseline and quant are scored against the same gold reference,
    # so the report is meaningful and retention is well-defined.
    assert report.baseline_aggregate > 0.0
    assert len(report.tiers) == 1


# ── recommended_quant_tier ───────────────────────────────────────────────


def test_recommend_picks_deepest_within_tolerance():
    """With identical outputs at every tier, recommend the most aggressive."""
    baseline = ["The mitochondria is the powerhouse of the cell."]
    tiers = {
        "int8": list(baseline),
        "int4": list(baseline),
        "int2": list(baseline),
    }
    report = quality_degradation_report(baseline, tiers)
    rec = recommended_quant_tier(report, tolerance=0.05)
    assert rec == "int2"


def test_recommend_returns_none_when_all_tiers_fail():
    """Garbage at every tier — none should be recommended at strict tolerance."""
    baseline = ["The mitochondria is the powerhouse of the cell."]
    tiers = {
        "int8": ["xyz"],
        "int4": ["abc"],
        "int2": ["???"],
    }
    report = quality_degradation_report(baseline, tiers)
    rec = recommended_quant_tier(report, tolerance=0.0)
    assert rec is None


def test_recommend_validates_tolerance():
    baseline = ["x"]
    report = quality_degradation_report(baseline, {"int4": ["x"]})
    with pytest.raises(ValueError):
        recommended_quant_tier(report, tolerance=-0.1)
    with pytest.raises(ValueError):
        recommended_quant_tier(report, tolerance=1.5)


def test_recommend_respects_tolerance_threshold():
    """A tier just under threshold is rejected; just over is accepted."""
    # Build a report manually for precise control.
    from kairu.squish_eval import TierDelta

    report = DegradationReport(
        baseline_aggregate=1.0,
        tiers=(
            TierDelta(tier="int8", bits=8, per_criterion={}, aggregate_delta=-0.02, retention_pct=98.0),
            TierDelta(tier="int4", bits=4, per_criterion={}, aggregate_delta=-0.10, retention_pct=90.0),
            TierDelta(tier="int2", bits=2, per_criterion={}, aggregate_delta=-0.30, retention_pct=70.0),
        ),
    )
    # 5% tolerance → require ≥ 95% retention. Only int8 qualifies.
    assert recommended_quant_tier(report, tolerance=0.05) == "int8"
    # 15% tolerance → int4 qualifies, deeper than int8 → pick int4.
    assert recommended_quant_tier(report, tolerance=0.15) == "int4"
    # 50% tolerance → all qualify → pick deepest (int2).
    assert recommended_quant_tier(report, tolerance=0.50) == "int2"


# ── QuantTier dataclass ──────────────────────────────────────────────────


def test_quant_tier_construction():
    t = QuantTier(name="int4", bits=4, outputs=["a", "b"])
    assert t.bits == 4
    assert list(t.outputs) == ["a", "b"]
    # Frozen — assignment should fail.
    with pytest.raises(Exception):
        t.bits = 8  # type: ignore[misc]


# ── as_dict serialization ────────────────────────────────────────────────


def test_report_as_dict_is_json_serialisable():
    import json

    baseline = ["Paris is the capital of France."]
    tiers = {"int8": ["Paris is the capital of France."], "int4": ["Lyon, maybe."]}
    report = quality_degradation_report(baseline, tiers)
    d = report.as_dict()
    # Round-trip through JSON to confirm everything is plain-Python.
    json.dumps(d)
    assert "baseline_aggregate" in d
    assert "tiers" in d
    assert all("retention_pct" in t for t in d["tiers"])
