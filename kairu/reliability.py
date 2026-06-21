"""Psychometric reliability metrics for judge ensembles.

A judge ensemble produces a *judges × criteria* score matrix. Two questions
decide whether that matrix is trustworthy, and kairu's ensemble output
answered neither before this module:

1. **Internal consistency** — do the criteria measure one coherent construct,
   or are they pulling in different directions? *Cronbach's alpha* over the
   criteria (treated as test items, judges as observations) answers this.
2. **Inter-rater agreement** — do the judges actually agree, or is the median
   hiding a coin-flip? Two complementary answers:
   * *ICC(2,1)* — two-way-random, single-rater, absolute-agreement intraclass
     correlation on the continuous [0, 1] scores.
   * *Fleiss' kappa* — chance-corrected agreement on the pass/fail
     binarisation at a threshold, the categorical view auditors expect.

All three are classical, pure-stdlib computations (Autorubric,
arXiv:2603.00077). Each needs at least two judges and two criteria; when the
matrix is too small a metric returns ``None`` rather than a fabricated number.

Interpretation bands follow the standard references — Cronbach (consistency),
Cicchetti (ICC), and Landis & Koch (kappa).

No ML deps — pure stdlib + :mod:`kairu.ensemble`.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from kairu.ensemble import EnsembleResult

# Pass/fail cut applied before Fleiss' kappa — scores at or above this count
# as "pass". 0.5 is the natural midpoint of the [0, 1] scoring scale.
DEFAULT_PASS_THRESHOLD: float = 0.5


# ─────────────────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReliabilityReport:
    """Inter-rater reliability of one judge × criteria score matrix.

    Attributes
    ----------
    cronbach_alpha:
        Internal consistency across criteria; ``None`` when fewer than two
        criteria or two judges, or when no variance makes it undefined.
    cronbach_label:
        Band for ``cronbach_alpha`` (e.g. ``"good"``) or ``"undefined"``.
    icc:
        ICC(2,1) absolute-agreement inter-judge correlation; ``None`` when
        undefined.
    icc_label:
        Band for ``icc`` or ``"undefined"``.
    fleiss_kappa:
        Chance-corrected agreement on pass/fail binarisation; ``None`` when
        undefined.
    fleiss_label:
        Band for ``fleiss_kappa`` or ``"undefined"``.
    pass_threshold:
        Cut used for the kappa binarisation.
    n_judges, n_criteria:
        Matrix dimensions the metrics were computed from.
    """

    cronbach_alpha: Optional[float]
    cronbach_label: str
    icc: Optional[float]
    icc_label: str
    fleiss_kappa: Optional[float]
    fleiss_label: str
    pass_threshold: float
    n_judges: int
    n_criteria: int

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable view of the report."""
        return {
            "cronbach_alpha": self.cronbach_alpha,
            "cronbach_label": self.cronbach_label,
            "icc": self.icc,
            "icc_label": self.icc_label,
            "fleiss_kappa": self.fleiss_kappa,
            "fleiss_label": self.fleiss_label,
            "pass_threshold": self.pass_threshold,
            "n_judges": self.n_judges,
            "n_criteria": self.n_criteria,
        }


# ─────────────────────────────────────────────────────────────────────────
# Interpretation bands
# ─────────────────────────────────────────────────────────────────────────


def _band(value: Optional[float], cuts: Sequence[tuple[float, str]]) -> str:
    """Map a value to the first band whose upper cut it falls under."""
    if value is None:
        return "undefined"
    for upper, label in cuts:
        if value < upper:
            return label
    return cuts[-1][1]


_CRONBACH_CUTS = (
    (0.5, "unacceptable"),
    (0.6, "poor"),
    (0.7, "questionable"),
    (0.8, "acceptable"),
    (0.9, "good"),
    (float("inf"), "excellent"),
)
_ICC_CUTS = (
    (0.5, "poor"),
    (0.75, "moderate"),
    (0.9, "good"),
    (float("inf"), "excellent"),
)
_KAPPA_CUTS = (
    (0.0, "poor"),
    (0.2, "slight"),
    (0.4, "fair"),
    (0.6, "moderate"),
    (0.8, "substantial"),
    (float("inf"), "almost_perfect"),
)


# ─────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────


def cronbach_alpha(matrix: Sequence[Sequence[float]]) -> Optional[float]:
    """Cronbach's alpha over a judges × criteria matrix.

    Criteria are the test items, judges the observations. Returns ``None``
    when there are fewer than two judges or two criteria, or when the total
    score has zero variance (alpha undefined).

    Parameters
    ----------
    matrix:
        Rows are judges, columns are criteria. Rows must be equal length.
    """
    n_judges = len(matrix)
    if n_judges < 2:
        return None
    k = len(matrix[0])
    if k < 2:
        return None

    column_var = sum(statistics.variance([row[j] for row in matrix]) for j in range(k))
    totals = [sum(row) for row in matrix]
    total_var = statistics.variance(totals)
    if total_var == 0:
        return None
    return (k / (k - 1)) * (1.0 - column_var / total_var)


def _mean_squares(matrix: Sequence[Sequence[float]]) -> tuple[float, float, float]:
    """Two-way ANOVA mean squares (rows, columns, residual) for a grid."""
    n = len(matrix)  # subjects (criteria)
    k = len(matrix[0])  # raters (judges)
    flat = [v for row in matrix for v in row]
    grand = statistics.fmean(flat)

    row_means = [statistics.fmean(row) for row in matrix]
    col_means = [statistics.fmean([row[j] for row in matrix]) for j in range(k)]

    ss_row = k * sum((m - grand) ** 2 for m in row_means)
    ss_col = n * sum((m - grand) ** 2 for m in col_means)
    ss_total = sum((v - grand) ** 2 for v in flat)
    ss_error = ss_total - ss_row - ss_col

    ms_row = ss_row / (n - 1)
    ms_col = ss_col / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    return ms_row, ms_col, ms_error


def intraclass_correlation(matrix: Sequence[Sequence[float]]) -> Optional[float]:
    """ICC(2,1): two-way-random, single-rater, absolute agreement.

    Inter-judge agreement on the continuous scores. The matrix is oriented
    criteria × judges (subjects × raters). Returns ``None`` when undefined
    (fewer than two of either dimension, or a zero denominator).

    Parameters
    ----------
    matrix:
        Rows are criteria (subjects), columns are judges (raters).
    """
    n = len(matrix)
    if n < 2:
        return None
    k = len(matrix[0])
    if k < 2:
        return None

    ms_row, ms_col, ms_error = _mean_squares(matrix)
    denom = ms_row + (k - 1) * ms_error + k * (ms_col - ms_error) / n
    if denom == 0:
        return None
    return (ms_row - ms_error) / denom


def fleiss_kappa(
    matrix: Sequence[Sequence[float]],
    threshold: float = DEFAULT_PASS_THRESHOLD,
) -> Optional[float]:
    """Fleiss' kappa on the pass/fail binarisation of a criteria × judges grid.

    Each score is mapped to ``1`` (pass) when ``>= threshold`` else ``0``.
    Returns ``None`` when there are fewer than two criteria or two judges, or
    when every judgment is identical (kappa undefined: ``1 - P_e == 0``).

    Parameters
    ----------
    matrix:
        Rows are criteria (subjects), columns are judges (raters).
    threshold:
        Pass/fail cut applied to each score.
    """
    n = len(matrix)
    if n < 2:
        return None
    k = len(matrix[0])
    if k < 2:
        return None

    # n_pass[i] = number of judges that passed criterion i.
    pass_counts = [sum(1 for v in row if v >= threshold) for row in matrix]
    p_bar = statistics.fmean(
        (p * p + (k - p) * (k - p) - k) / (k * (k - 1)) for p in pass_counts
    )
    prop_pass = sum(pass_counts) / (n * k)
    p_e = prop_pass**2 + (1.0 - prop_pass) ** 2
    if p_e >= 1.0:
        return None
    return (p_bar - p_e) / (1.0 - p_e)


# ─────────────────────────────────────────────────────────────────────────
# Public entry points
# ─────────────────────────────────────────────────────────────────────────


def compute_reliability(
    matrix: Sequence[Sequence[float]],
    *,
    pass_threshold: float = DEFAULT_PASS_THRESHOLD,
) -> ReliabilityReport:
    """Compute all three reliability metrics from a judges × criteria matrix.

    Cronbach's alpha reads the matrix as judges × criteria; ICC and Fleiss'
    kappa need it transposed to criteria × judges, which this function handles
    internally.

    Parameters
    ----------
    matrix:
        Rows are judges, columns are criteria. Equal-length rows.
    pass_threshold:
        Cut for the kappa binarisation.

    Returns
    -------
    ReliabilityReport
    """
    n_judges = len(matrix)
    n_criteria = len(matrix[0]) if matrix else 0

    alpha = cronbach_alpha(matrix)
    transposed: List[List[float]] = (
        [[row[j] for row in matrix] for j in range(n_criteria)] if matrix else []
    )
    icc = intraclass_correlation(transposed)
    kappa = fleiss_kappa(transposed, pass_threshold)

    return ReliabilityReport(
        cronbach_alpha=alpha,
        cronbach_label=_band(alpha, _CRONBACH_CUTS),
        icc=icc,
        icc_label=_band(icc, _ICC_CUTS),
        fleiss_kappa=kappa,
        fleiss_label=_band(kappa, _KAPPA_CUTS),
        pass_threshold=pass_threshold,
        n_judges=n_judges,
        n_criteria=n_criteria,
    )


def reliability_from_ensemble(
    result: EnsembleResult,
    *,
    pass_threshold: float = DEFAULT_PASS_THRESHOLD,
) -> ReliabilityReport:
    """Build a :class:`ReliabilityReport` from an :class:`EnsembleResult`.

    Reads each judge's per-criterion scores into a judges × criteria matrix,
    using the criteria of the first judge as the canonical column order so a
    judge that dropped a criterion does not silently misalign the columns.

    Parameters
    ----------
    result:
        The ensemble verdict whose per-judge scores are assessed.
    pass_threshold:
        Cut for the kappa binarisation.

    Returns
    -------
    ReliabilityReport
    """
    judges = result.judges
    if not judges:
        return compute_reliability([], pass_threshold=pass_threshold)
    # Keep only criteria every judge scored, so the matrix stays rectangular
    # even if one judge dropped a criterion.
    criteria = [c for c in judges[0].scores if all(c in j.scores for j in judges)]
    matrix = [[j.scores[c] for c in criteria] for j in judges]
    return compute_reliability(matrix, pass_threshold=pass_threshold)


__all__ = [
    "DEFAULT_PASS_THRESHOLD",
    "ReliabilityReport",
    "cronbach_alpha",
    "intraclass_correlation",
    "fleiss_kappa",
    "compute_reliability",
    "reliability_from_ensemble",
]
