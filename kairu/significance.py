"""Statistical significance for A/B response comparisons.

Given two responses to the same prompt and a rubric, the **paired sample**
is the vector of per-criterion score differences ``(a_i - b_i)``. A paired
t-test on those differences asks: *is the mean difference significantly
different from zero?*

We additionally report:

* **Cohen's d (paired)** — the effect size, ``mean_diff / stdev_diff``.
* **95 % confidence interval** for the mean difference using the t critical
  value at ``df = n - 1``.
* **Winner** with the explicit rule: report a winner only when
  ``p_value < alpha`` (default 0.05) AND ``|d| >= small_effect_threshold``;
  otherwise the result is ``"tie"``. Statistical significance without a
  meaningful effect size is not a winner.

Pure-stdlib implementation. The Student's t CDF is computed via Simpson's
rule numerical integration of the PDF — accurate to ~5 decimal places for
``n`` in the relevant range (2 … several hundred), no scipy required.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

DEFAULT_ALPHA: float = 0.05
SMALL_EFFECT_THRESHOLD: float = 0.20    # |Cohen's d| below this → no winner
SIMPSON_STEPS: int = 2000               # even number; one-sided tail integration


# ─────────────────────────────────────────────────────────────────────────
# Student's t distribution — PDF and CDF without scipy
# ─────────────────────────────────────────────────────────────────────────

def _lgamma(x: float) -> float:
    """Wrapper for math.lgamma — kept named so the formula reads cleanly."""
    return math.lgamma(x)


def _student_t_pdf(t: float, df: float) -> float:
    """Student's t PDF at ``t`` with ``df`` degrees of freedom.

        f(t; df) = Γ((df+1)/2) / (√(π·df) · Γ(df/2)) · (1 + t²/df)^(-(df+1)/2)
    """
    log_coeff = _lgamma((df + 1) / 2) - _lgamma(df / 2) - 0.5 * math.log(df * math.pi)
    log_kernel = -((df + 1) / 2) * math.log1p(t * t / df)
    return math.exp(log_coeff + log_kernel)


def _student_t_cdf(t: float, df: float) -> float:
    """Student's t CDF at ``t`` via Simpson's rule.

    The integral is symmetric around 0, so we compute the half-area from 0
    to ``|t|`` and add 0.5. For large ``|t|`` the integral converges fast;
    SIMPSON_STEPS=2000 over [0, |t|] is overkill but cheap (<1 ms).
    """
    if df <= 0:
        raise ValueError("df must be > 0")
    if t == 0.0:
        return 0.5
    a = 0.0
    b = abs(t)
    n = SIMPSON_STEPS                              # already even
    h = (b - a) / n
    s = _student_t_pdf(a, df) + _student_t_pdf(b, df)
    # Simpson's 1/3 — coefficients 4 on odd indices, 2 on even.
    for i in range(1, n):
        x = a + i * h
        s += (4 if i % 2 == 1 else 2) * _student_t_pdf(x, df)
    half_area = (h / 3) * s
    cdf_abs = 0.5 + half_area
    return cdf_abs if t > 0 else 1.0 - cdf_abs


def _student_t_critical(df: float, alpha: float) -> float:
    """Find ``t*`` such that ``P(|T| <= t*) = 1 - alpha``. Bisection on the
    survival function — monotone, well behaved."""
    target = 1.0 - alpha / 2.0
    lo, hi = 0.0, 100.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if _student_t_cdf(mid, df) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# ─────────────────────────────────────────────────────────────────────────
# Paired t-test + Cohen's d
# ─────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SignificanceResult:
    """One-shot comparison outcome.

    Attributes
    ----------
    n
        Number of paired observations (criteria in the rubric).
    mean_diff
        Mean of ``(a_i - b_i)``. Positive ⇒ A's scores tend to be higher.
    stdev_diff
        Sample stddev of the differences (``ddof = 1``); 0.0 when ``n < 2``.
    t_stat
        Paired t statistic. ``nan`` when ``stdev_diff == 0`` (no variance).
    df
        Degrees of freedom (``n - 1``).
    p_value
        Two-sided p-value from the Student's t CDF. Bounded to ``[0, 1]``.
    effect_size
        Cohen's d for paired samples = ``mean_diff / stdev_diff``.
    effect_label
        Human-readable bucket: negligible / small / medium / large.
    confidence_interval
        95 % CI on the mean difference, ``(lo, hi)``.
    winner
        ``"a"``, ``"b"``, or ``"tie"`` per the rule documented in the
        module docstring.
    """

    n: int
    mean_diff: float
    stdev_diff: float
    t_stat: float
    df: float
    p_value: float
    effect_size: float
    effect_label: str
    confidence_interval: Tuple[float, float]
    winner: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "n": self.n,
            "mean_diff": self.mean_diff,
            "stdev_diff": self.stdev_diff,
            "t_stat": self.t_stat,
            "df": self.df,
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "effect_label": self.effect_label,
            "confidence_interval": list(self.confidence_interval),
            "winner": self.winner,
        }


def _effect_label(d: float) -> str:
    a = abs(d)
    if a < SMALL_EFFECT_THRESHOLD:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    return "large"


def paired_t_test(
    a_scores: Sequence[float],
    b_scores: Sequence[float],
    alpha: float = DEFAULT_ALPHA,
) -> SignificanceResult:
    """Paired t-test on per-criterion differences ``a - b``.

    Parameters
    ----------
    a_scores, b_scores
        Equal-length sequences of per-criterion scores in ``[0, 1]``.
    alpha
        Significance threshold for the winner rule. Default 0.05.

    Raises
    ------
    ValueError
        If lengths differ or are < 2.
    """
    if len(a_scores) != len(b_scores):
        raise ValueError(
            f"a and b must be the same length (got {len(a_scores)} vs {len(b_scores)})"
        )
    if len(a_scores) < 2:
        raise ValueError("paired t-test requires at least 2 paired observations")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    diffs: List[float] = [float(a) - float(b) for a, b in zip(a_scores, b_scores)]
    n = len(diffs)
    df = n - 1
    mean_diff = statistics.fmean(diffs)
    stdev_diff = statistics.stdev(diffs)               # ddof = 1

    # Floating-point tolerance: when every paired diff is identical the
    # sample stddev is mathematically 0 but `statistics.stdev` can return
    # a value on the order of 1e-17 due to accumulated rounding. Anything
    # below this floor is indistinguishable from "no variance" for the
    # purpose of computing a t-statistic.
    if stdev_diff < 1e-12:
        stdev_diff = 0.0
    if stdev_diff == 0.0:
        # Degenerate case: every paired difference is identical (often 0).
        t_stat = float("nan") if mean_diff == 0.0 else float("inf") * (1.0 if mean_diff > 0 else -1.0)
        p_value = 1.0 if mean_diff == 0.0 else 0.0
        effect = 0.0 if mean_diff == 0.0 else float("inf") * (1.0 if mean_diff > 0 else -1.0)
        t_crit = _student_t_critical(df, alpha)
        ci_half = t_crit * 0.0                          # SE = 0
        ci = (mean_diff - ci_half, mean_diff + ci_half)
    else:
        se = stdev_diff / math.sqrt(n)
        t_stat = mean_diff / se
        # Two-sided p-value:  2 * P(T >= |t|) = 2 * (1 - CDF(|t|)).
        p_value = max(0.0, min(1.0, 2.0 * (1.0 - _student_t_cdf(abs(t_stat), df))))
        effect = mean_diff / stdev_diff
        t_crit = _student_t_critical(df, alpha)
        ci_half = t_crit * se
        ci = (mean_diff - ci_half, mean_diff + ci_half)

    significant = p_value < alpha
    if significant and abs(effect) >= SMALL_EFFECT_THRESHOLD:
        winner = "a" if mean_diff > 0 else "b"
    else:
        winner = "tie"

    return SignificanceResult(
        n=n,
        mean_diff=mean_diff,
        stdev_diff=stdev_diff,
        t_stat=t_stat,
        df=df,
        p_value=p_value,
        effect_size=effect,
        effect_label=_effect_label(effect),
        confidence_interval=ci,
        winner=winner,
    )


def per_criterion_diffs(
    a_scores: Mapping[str, float],
    b_scores: Mapping[str, float],
) -> Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[str, ...]]:
    """Align two ``{criterion: score}`` dicts into (a, b, names) tuples.

    Used by the HTTP layer to bridge :class:`kairu.Evaluation.scores`
    (criterion-keyed) into the positional arrays the t-test expects.
    """
    shared = [k for k in a_scores if k in b_scores]
    if not shared:
        raise ValueError("no overlapping criteria between A and B")
    return (
        tuple(float(a_scores[k]) for k in shared),
        tuple(float(b_scores[k]) for k in shared),
        tuple(shared),
    )


__all__ = [
    "DEFAULT_ALPHA",
    "SMALL_EFFECT_THRESHOLD",
    "SignificanceResult",
    "paired_t_test",
    "per_criterion_diffs",
]
