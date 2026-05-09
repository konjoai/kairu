"""Squish integration — evaluate quality degradation across quantization tiers.

This module gives kairu the ability to compare a baseline model's outputs
(typically FP16) against the same model quantized to lower precision tiers
(``int8``, ``int4``, ``int2``) and quantify the quality loss.

Design
------
The evaluator scores each output string against a 5-criterion rubric:

    correctness   — substring/keyword overlap with the reference (or with the
                    baseline output, if no explicit reference is supplied)
    fluency       — surface-form heuristic: punctuation balance, repetition
                    penalty, average token length sanity check
    faithfulness  — fraction of reference content tokens that appear in the
                    output (recall against the reference)
    completeness  — length-ratio against the reference, capped at 1.0
    safety        — penalty for emitting any token from a small banned list
                    (overrideable via ``SquishEvaluator(banned=…)``)

Every score is in ``[0.0, 1.0]``; the aggregate score is the unweighted mean.

The implementation is deterministic, pure-Python, and has no ML dependency.
It exists to give kairu a stable yardstick when an upstream quantizer
(e.g. ``squish``) emits side-by-side outputs for comparison — *not* to
replace task-specific eval suites.

Public API
----------
``SquishEvaluator``                 — scores one output, or many in parallel
``QuantTier``                       — frozen dataclass: name + bits + outputs
``RubricScore``                     — frozen dataclass: per-criterion + aggregate
``DegradationReport``               — frozen dataclass: per-tier deltas + retention
``quality_degradation_report(...)`` — module-level convenience wrapper
``recommended_quant_tier(...)``     — pick the lowest-bit tier within tolerance
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

# Tier ordering: highest precision (most bits) first. Used for "highest
# compression that still fits tolerance" — we walk from least-compressed
# to most-compressed and accept the deepest tier still within budget.
_TIER_BITS: dict[str, int] = {
    "fp16": 16,
    "int8": 8,
    "int4": 4,
    "int2": 2,
}

# Default banned tokens — surface-form check only, deliberately small.
# Callers with real safety needs should pass their own list or, better,
# a real safety classifier from outside this module.
_DEFAULT_BANNED: frozenset[str] = frozenset({
    "<|endoftext|>",  # leaked control token
    "[REDACTED]",     # accidental redaction marker
})

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric token split. Pure stdlib, deterministic."""
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class RubricScore:
    """Per-criterion score for a single output. All fields in [0, 1]."""

    correctness: float
    fluency: float
    faithfulness: float
    completeness: float
    safety: float
    aggregate: float

    def as_dict(self) -> dict[str, float]:
        return {
            "correctness": self.correctness,
            "fluency": self.fluency,
            "faithfulness": self.faithfulness,
            "completeness": self.completeness,
            "safety": self.safety,
            "aggregate": self.aggregate,
        }


@dataclass(frozen=True)
class QuantTier:
    """One quantization tier and its outputs aligned with the baseline list."""

    name: str
    bits: int
    outputs: Sequence[str]


@dataclass(frozen=True)
class TierDelta:
    """Per-criterion score delta (quantized minus baseline) for one tier."""

    tier: str
    bits: int
    per_criterion: Mapping[str, float]
    aggregate_delta: float
    retention_pct: float  # 100 * (quant_aggregate / baseline_aggregate)


@dataclass(frozen=True)
class DegradationReport:
    """Full quality-degradation report across all tiers."""

    baseline_aggregate: float
    tiers: tuple[TierDelta, ...]

    def as_dict(self) -> dict:
        return {
            "baseline_aggregate": self.baseline_aggregate,
            "tiers": [
                {
                    "tier": t.tier,
                    "bits": t.bits,
                    "per_criterion": dict(t.per_criterion),
                    "aggregate_delta": t.aggregate_delta,
                    "retention_pct": t.retention_pct,
                }
                for t in self.tiers
            ],
        }


class SquishEvaluator:
    """Score outputs against a 5-criterion rubric.

    Parameters
    ----------
    banned:
        Iterable of substrings that trigger a safety penalty when present in
        the output. Defaults to a tiny built-in list.
    repetition_window:
        N for the n-gram repetition penalty in the fluency score (default 3).
    """

    def __init__(
        self,
        banned: Optional[Sequence[str]] = None,
        repetition_window: int = 3,
    ) -> None:
        if repetition_window < 1:
            raise ValueError("repetition_window must be >= 1")
        self._banned = frozenset(banned) if banned is not None else _DEFAULT_BANNED
        self._win = repetition_window

    # ── single-output scoring ────────────────────────────────────────────

    def score(self, output: str, reference: Optional[str] = None) -> RubricScore:
        """Score one output. ``reference`` is the gold/baseline text.

        When ``reference`` is None or empty, faithfulness and completeness
        cannot be computed — they fall back to neutral 1.0 (no penalty)
        because there is no ground truth to disagree with.
        """
        if not isinstance(output, str):
            raise TypeError("output must be a string")
        if reference is not None and not isinstance(reference, str):
            raise TypeError("reference must be a string or None")

        out_toks = _tokenize(output)
        ref_toks = _tokenize(reference) if reference else []

        c = self._correctness(out_toks, ref_toks)
        fl = self._fluency(output, out_toks)
        fa = self._faithfulness(out_toks, ref_toks)
        cp = self._completeness(out_toks, ref_toks)
        sa = self._safety(output)

        agg = (c + fl + fa + cp + sa) / 5.0
        return RubricScore(c, fl, fa, cp, sa, agg)

    def score_batch(
        self,
        outputs: Sequence[str],
        references: Optional[Sequence[str]] = None,
    ) -> list[RubricScore]:
        """Score a list of outputs. ``references`` (if given) must match length."""
        if references is not None and len(references) != len(outputs):
            raise ValueError(
                f"references length {len(references)} != outputs length {len(outputs)}"
            )
        refs = references if references is not None else [None] * len(outputs)
        return [self.score(o, r) for o, r in zip(outputs, refs)]

    # ── tier-level scoring ───────────────────────────────────────────────

    def score_tier(
        self,
        tier: QuantTier,
        references: Optional[Sequence[str]] = None,
    ) -> list[RubricScore]:
        """Score every output in a tier against the matching reference."""
        return self.score_batch(list(tier.outputs), references)

    # ── individual criteria ──────────────────────────────────────────────

    @staticmethod
    def _correctness(out_toks: list[str], ref_toks: list[str]) -> float:
        """Token-overlap precision: |out ∩ ref| / |out|.

        With no reference we cannot judge correctness — return 1.0 (neutral).
        With an empty output but a non-empty reference, precision is undefined
        but the output is plainly wrong → 0.0.
        """
        if not ref_toks:
            return 1.0
        if not out_toks:
            return 0.0
        ref_set = set(ref_toks)
        hits = sum(1 for t in out_toks if t in ref_set)
        return hits / len(out_toks)

    def _fluency(self, raw: str, toks: list[str]) -> float:
        """Surface heuristic: balanced quotes/parens, low n-gram repetition,
        plausible average token length. Each subscore in [0, 1]; mean returned.
        """
        if not raw or not toks:
            return 0.0

        # Balanced delimiters.
        pairs = (("(", ")"), ("[", "]"), ("{", "}"), ('"', '"'))
        balance_violations = 0
        for o, c in pairs:
            if o == c:  # quotes — must be even count
                if raw.count(o) % 2 != 0:
                    balance_violations += 1
            else:
                if raw.count(o) != raw.count(c):
                    balance_violations += 1
        balance = max(0.0, 1.0 - balance_violations / len(pairs))

        # Repetition penalty over n-grams (n = self._win).
        n = self._win
        if len(toks) >= n:
            grams = [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]
            unique = len(set(grams))
            repetition = unique / len(grams)
        else:
            repetition = 1.0

        # Average token length plausibility — penalise extremes.
        avg_len = sum(len(t) for t in toks) / len(toks)
        # Plausible English average is ~4.5; penalise distances > 3.
        length_score = max(0.0, 1.0 - abs(avg_len - 4.5) / 6.0)

        return (balance + repetition + length_score) / 3.0

    @staticmethod
    def _faithfulness(out_toks: list[str], ref_toks: list[str]) -> float:
        """Recall against the reference: |out ∩ ref| / |ref|."""
        if not ref_toks:
            return 1.0
        ref_set = set(ref_toks)
        out_set = set(out_toks)
        return len(ref_set & out_set) / len(ref_set)

    @staticmethod
    def _completeness(out_toks: list[str], ref_toks: list[str]) -> float:
        """Length ratio capped at 1.0. No reference → neutral 1.0."""
        if not ref_toks:
            return 1.0
        if not out_toks:
            return 0.0
        return min(1.0, len(out_toks) / len(ref_toks))

    def _safety(self, raw: str) -> float:
        """1.0 by default; subtract 0.5 per banned-substring hit, floor at 0."""
        if not raw:
            return 1.0
        hits = sum(1 for token in self._banned if token in raw)
        return max(0.0, 1.0 - 0.5 * hits)


# ──────────────────────────────────────────────────────────────────────────
# Module-level convenience API
# ──────────────────────────────────────────────────────────────────────────


def _aggregate_mean(scores: Sequence[RubricScore]) -> float:
    if not scores:
        return 0.0
    return sum(s.aggregate for s in scores) / len(scores)


def _per_criterion_mean(scores: Sequence[RubricScore]) -> dict[str, float]:
    if not scores:
        return {k: 0.0 for k in ("correctness", "fluency", "faithfulness", "completeness", "safety")}
    n = len(scores)
    return {
        "correctness": sum(s.correctness for s in scores) / n,
        "fluency": sum(s.fluency for s in scores) / n,
        "faithfulness": sum(s.faithfulness for s in scores) / n,
        "completeness": sum(s.completeness for s in scores) / n,
        "safety": sum(s.safety for s in scores) / n,
    }


def quality_degradation_report(
    baseline_outputs: Sequence[str],
    quantized_outputs: Mapping[str, Sequence[str]],
    references: Optional[Sequence[str]] = None,
    evaluator: Optional[SquishEvaluator] = None,
) -> DegradationReport:
    """Per-criterion delta + aggregate retention for each quantization tier.

    Parameters
    ----------
    baseline_outputs:
        Outputs from the reference model (typically FP16). When ``references``
        is omitted these double as the gold standard for every quantized tier.
    quantized_outputs:
        ``{tier_name: [outputs...]}`` — each list must match ``baseline_outputs``
        in length.
    references:
        Optional gold references. When provided, both the baseline and every
        quantized tier are scored against these. When omitted, the baseline
        outputs themselves are used as the reference (so baseline retention is
        always 100%).
    evaluator:
        Optional pre-configured ``SquishEvaluator``. A default instance is
        created if not supplied.

    Returns
    -------
    DegradationReport with one ``TierDelta`` per quantized tier, ordered by
    descending bit-width (least → most aggressive compression).
    """
    if not baseline_outputs:
        raise ValueError("baseline_outputs must be non-empty")
    n = len(baseline_outputs)
    for tier_name, outs in quantized_outputs.items():
        if len(outs) != n:
            raise ValueError(
                f"tier {tier_name!r} has {len(outs)} outputs; expected {n}"
            )
    if references is not None and len(references) != n:
        raise ValueError(
            f"references length {len(references)} != baseline length {n}"
        )

    ev = evaluator or SquishEvaluator()
    refs_for_baseline = references  # may be None — neutral scoring
    refs_for_quant = references if references is not None else list(baseline_outputs)

    baseline_scores = ev.score_batch(list(baseline_outputs), refs_for_baseline)
    baseline_per_crit = _per_criterion_mean(baseline_scores)
    baseline_agg = _aggregate_mean(baseline_scores)

    deltas: list[TierDelta] = []
    # Sort by bits descending so least-aggressive compression is reported first.
    ordered = sorted(
        quantized_outputs.items(),
        key=lambda kv: -_TIER_BITS.get(kv[0], 0),
    )
    for tier_name, outs in ordered:
        bits = _TIER_BITS.get(tier_name, 0)
        scores = ev.score_batch(list(outs), refs_for_quant)
        per_crit = _per_criterion_mean(scores)
        agg = _aggregate_mean(scores)
        delta = {k: per_crit[k] - baseline_per_crit[k] for k in per_crit}
        retention = (agg / baseline_agg * 100.0) if baseline_agg > 0 else 0.0
        deltas.append(
            TierDelta(
                tier=tier_name,
                bits=bits,
                per_criterion=delta,
                aggregate_delta=agg - baseline_agg,
                retention_pct=retention,
            )
        )

    return DegradationReport(baseline_aggregate=baseline_agg, tiers=tuple(deltas))


def recommended_quant_tier(
    rubric: DegradationReport,
    tolerance: float = 0.05,
) -> Optional[str]:
    """Return the most aggressive tier whose quality loss is within tolerance.

    "Within tolerance" means ``retention_pct >= (1 - tolerance) * 100``.
    Walks tiers in order of *decreasing* bit-width, so the deepest acceptable
    compression is returned. If the report has no tiers, or no tier meets the
    bar, returns ``None``.
    """
    if not 0.0 <= tolerance <= 1.0:
        raise ValueError("tolerance must be in [0, 1]")
    threshold = (1.0 - tolerance) * 100.0
    # tiers are stored in descending bit-width order; iterate that way and
    # remember the deepest one that still meets the bar.
    accepted: Optional[str] = None
    for t in rubric.tiers:
        if t.retention_pct + 1e-9 >= threshold:
            accepted = t.tier
        else:
            # Once we drop below threshold at a given depth, deeper tiers
            # only degrade further — but we still let the loop finish in
            # case the report is malformed (out of order). Cheap.
            continue
    return accepted


__all__ = [
    "QuantTier",
    "RubricScore",
    "TierDelta",
    "DegradationReport",
    "SquishEvaluator",
    "quality_degradation_report",
    "recommended_quant_tier",
]
