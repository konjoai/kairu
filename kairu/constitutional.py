"""Policy-document-driven rubric generation — pure stdlib, deterministic.

Given plain-text policy documents, this module extracts obligation and
prohibition clauses using keyword triggers, derives criterion names, and
produces a scored :class:`GeneratedRubric` that integrates with
:data:`kairu.evaluation.RUBRICS`.

Constitutional scoring is entirely separate from the main evaluation scorers:
positive clauses use token-overlap F1; negative clauses measure absence of
forbidden terms.  Both are bounded to [0.0, 1.0] and depend only on stdlib.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from kairu.evaluation import RUBRIC_REGISTRY, RUBRIC_VERSION, RUBRICS, Rubric


__all__ = [
    "ConstitutionalClause",
    "ClauseScore",
    "ConstitutionalEvaluation",
    "GeneratedRubric",
    "extract_clauses",
    "score_response",
    "generate_rubric",
]

# ─────────────────────────────────────────────────────────────────────────
# Trigger phrases — negative checked first (longer match wins)
# ─────────────────────────────────────────────────────────────────────────

_NEGATIVE_TRIGGERS: Tuple[str, ...] = (
    "must not",
    "shall not",
    "prohibited",
    "forbidden",
    "not allowed",
    "not permitted",
    "may not",
)

_POSITIVE_TRIGGERS: Tuple[str, ...] = (
    "must",
    "shall",
    "required",
    "mandatory",
    "obligated",
)

# ─────────────────────────────────────────────────────────────────────────
# Sentence-splitting pattern
# ─────────────────────────────────────────────────────────────────────────

_SENT_SPLIT_RE = re.compile(r"[.!?]")

_MIN_SENT_LEN = 10
_MAX_SENT_LEN = 500

# ─────────────────────────────────────────────────────────────────────────
# Stopwords (shared with evaluation.py pattern — no coupling needed)
# ─────────────────────────────────────────────────────────────────────────

_STOPWORDS: frozenset[str] = frozenset(
    """a an and are as at be been being but by could did do does for from had
    has have he her him his i if in into is it its me my no not of on or our
    she so than that the their them these they this those to too us was we
    were what when where which who whom why will with would you your""".split()
)

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")


# ─────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ConstitutionalClause:
    """One obligation or prohibition extracted from a policy document.

    Attributes
    ----------
    text:
        Original sentence from the policy.
    polarity:
        ``"positive"`` for obligation triggers (must/shall/required);
        ``"negative"`` for prohibition triggers (must not/prohibited/…).
    trigger:
        The exact trigger phrase that matched.
    criterion_name:
        Snake-case name derived from the clause content.
    weight:
        Scoring weight — defaults to 1.0.
    """

    text: str
    polarity: str
    trigger: str
    criterion_name: str
    weight: float = 1.0


@dataclass(frozen=True)
class ClauseScore:
    """Score for one clause against a response.

    Attributes
    ----------
    criterion_name:
        Matches :attr:`ConstitutionalClause.criterion_name`.
    polarity:
        ``"positive"`` or ``"negative"``.
    score:
        Value in [0.0, 1.0].
    detail:
        Diagnostic breakdown of how the score was computed.
    """

    criterion_name: str
    polarity: str
    score: float
    detail: Dict[str, float]


@dataclass(frozen=True)
class ConstitutionalEvaluation:
    """Aggregate result of scoring a response against all policy clauses.

    Attributes
    ----------
    goal:
        The response text that was evaluated.
    n_clauses:
        Total number of clauses evaluated.
    n_positive:
        Count of positive (obligation) clauses.
    n_negative:
        Count of negative (prohibition) clauses.
    per_clause:
        One :class:`ClauseScore` per clause, in extraction order.
    aggregate:
        Unweighted mean of per-clause scores.
    """

    goal: str
    n_clauses: int
    n_positive: int
    n_negative: int
    per_clause: Tuple[ClauseScore, ...]
    aggregate: float


@dataclass(frozen=True)
class GeneratedRubric:
    """A rubric derived from policy text, ready for use with the registry.

    Attributes
    ----------
    name:
        Stable rubric identifier.
    clauses:
        All extracted :class:`ConstitutionalClause` objects.
    n_clauses:
        Total clause count.
    n_positive:
        Obligation clause count.
    n_negative:
        Prohibition clause count.
    criteria:
        Tuple of criterion names (one per clause).
    weights:
        Mapping of criterion name → weight (all 1.0 by default).
    """

    name: str
    clauses: Tuple[ConstitutionalClause, ...]
    n_clauses: int
    n_positive: int
    n_negative: int
    criteria: Tuple[str, ...]
    weights: Dict[str, float]


# ─────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────


def _split_sentences(text: str) -> List[str]:
    """Split *text* on sentence-ending punctuation and filter by length."""
    raw = _SENT_SPLIT_RE.split(text)
    result: List[str] = []
    for sent in raw:
        sent = sent.strip()
        if _MIN_SENT_LEN <= len(sent) <= _MAX_SENT_LEN:
            result.append(sent)
    return result


def _content_tokens(text: str) -> List[str]:
    """Lowercase word tokens with stopwords removed."""
    return [
        m.group(0).lower()
        for m in _WORD_RE.finditer(text)
        if m.group(0).lower() not in _STOPWORDS
    ]


def _detect_trigger(sentence_lower: str) -> Tuple[str, str] | None:
    """Return (trigger, polarity) for the first matching trigger, or None.

    Negative triggers are checked before positive so that "must not"
    is not also claimed by the "must" positive trigger.
    """
    for trigger in _NEGATIVE_TRIGGERS:
        if trigger in sentence_lower:
            return trigger, "negative"
    for trigger in _POSITIVE_TRIGGERS:
        if trigger in sentence_lower:
            return trigger, "positive"
    return None


def _make_criterion_name(sentence: str) -> str:
    """Derive a snake_case criterion name from the first 3 content tokens."""
    tokens = _content_tokens(sentence)
    chosen = tokens[:3] if len(tokens) >= 3 else tokens
    if not chosen:
        return "clause"
    return "_".join(chosen)


def _deduplicate_names(raw_names: List[str]) -> List[str]:
    """Append ``_2``, ``_3``, … to duplicate criterion names."""
    seen: Dict[str, int] = {}
    result: List[str] = []
    for name in raw_names:
        count = seen.get(name, 0)
        if count == 0:
            result.append(name)
        else:
            result.append(f"{name}_{count + 1}")
        seen[name] = count + 1
    return result


# ─────────────────────────────────────────────────────────────────────────
# Scoring helpers
# ─────────────────────────────────────────────────────────────────────────


def _score_positive(
    clause_terms: List[str], response_terms: List[str]
) -> Tuple[float, Dict[str, float]]:
    """F1 of content-token overlap for a positive (obligation) clause."""
    c_set = set(clause_terms)
    r_set = set(response_terms)
    overlap = len(c_set & r_set)
    precision = overlap / len(r_set) if r_set else 0.0
    recall = overlap / len(c_set) if c_set else 0.0
    denom = precision + recall
    f1 = (2.0 * precision * recall / denom) if denom > 0.0 else 0.0
    detail: Dict[str, float] = {
        "overlap": float(overlap),
        "precision": precision,
        "recall": recall,
        "clause_terms": float(len(c_set)),
    }
    return f1, detail


def _score_negative(
    clause_terms: List[str], response_terms: List[str]
) -> Tuple[float, Dict[str, float]]:
    """Absence score for a negative (prohibition) clause."""
    c_set = set(clause_terms)
    r_set = set(response_terms)
    forbidden_found = float(len(c_set & r_set))
    total_forbidden = float(max(1, len(c_set)))
    score = 1.0 - min(1.0, forbidden_found / total_forbidden)
    detail: Dict[str, float] = {
        "forbidden_terms_found": forbidden_found,
        "total_forbidden_terms": total_forbidden,
    }
    return score, detail


# ─────────────────────────────────────────────────────────────────────────
# Public functions
# ─────────────────────────────────────────────────────────────────────────


def extract_clauses(text: str, max_clauses: int = 20) -> List[ConstitutionalClause]:
    """Extract obligation and prohibition clauses from policy text.

    Parameters
    ----------
    text:
        Plain policy text — no PDF parsing; raw string only.
    max_clauses:
        Hard cap on returned clauses. Clauses beyond this limit are silently
        dropped. Defaults to 20.

    Returns
    -------
    List[ConstitutionalClause]
        Clauses in document order, each with a deduplicated criterion name.
    """
    sentences = _split_sentences(text)
    raw_names: List[str] = []
    candidate_triggers: List[Tuple[str, str, str]] = []  # (sentence, trigger, polarity)

    for sentence in sentences:
        if len(candidate_triggers) >= max_clauses:
            break
        result = _detect_trigger(sentence.lower())
        if result is None:
            continue
        trigger, polarity = result
        raw_name = _make_criterion_name(sentence)
        raw_names.append(raw_name)
        candidate_triggers.append((sentence, trigger, polarity))

    deduped = _deduplicate_names(raw_names)
    clauses: List[ConstitutionalClause] = []
    for (sentence, trigger, polarity), name in zip(candidate_triggers, deduped):
        clauses.append(
            ConstitutionalClause(
                text=sentence,
                polarity=polarity,
                trigger=trigger,
                criterion_name=name,
                weight=1.0,
            )
        )
    return clauses


def score_response(
    response: str,
    clauses: Sequence[ConstitutionalClause],
) -> ConstitutionalEvaluation:
    """Score a response against a set of extracted policy clauses.

    Parameters
    ----------
    response:
        The model output to evaluate.
    clauses:
        Non-empty sequence of :class:`ConstitutionalClause` objects (from
        :func:`extract_clauses` or built manually).

    Returns
    -------
    ConstitutionalEvaluation
        Per-clause scores and an unweighted aggregate.

    Raises
    ------
    ValueError
        If *clauses* is empty.
    """
    if not clauses:
        raise ValueError("clauses must be non-empty — nothing to score against")

    response_terms = _content_tokens(response)
    scored: List[ClauseScore] = []

    for clause in clauses:
        clause_terms = _content_tokens(clause.text)
        if clause.polarity == "positive":
            raw_score, detail = _score_positive(clause_terms, response_terms)
        else:
            raw_score, detail = _score_negative(clause_terms, response_terms)
        score = max(0.0, min(1.0, raw_score))
        scored.append(
            ClauseScore(
                criterion_name=clause.criterion_name,
                polarity=clause.polarity,
                score=score,
                detail=detail,
            )
        )

    aggregate = sum(cs.score for cs in scored) / len(scored)
    n_positive = sum(1 for c in clauses if c.polarity == "positive")
    n_negative = sum(1 for c in clauses if c.polarity == "negative")

    return ConstitutionalEvaluation(
        goal=response,
        n_clauses=len(clauses),
        n_positive=n_positive,
        n_negative=n_negative,
        per_clause=tuple(scored),
        aggregate=aggregate,
    )


def generate_rubric(text: str, name: str, max_clauses: int = 20) -> GeneratedRubric:
    """Extract clauses from policy text and return a registered rubric.

    The resulting rubric is inserted into :data:`kairu.evaluation.RUBRICS`
    and :data:`kairu.evaluation.RUBRIC_REGISTRY` so it is discoverable by
    name from the rest of the evaluation pipeline.  Constitutional rubrics
    carry their criterion names and equal weights (1.0 each); they are
    scored via :func:`score_response`, not through ``evaluate()``.

    Parameters
    ----------
    text:
        Plain policy text.
    name:
        Stable identifier for the rubric (must be non-empty).
    max_clauses:
        Maximum number of clauses to extract. Defaults to 20.

    Returns
    -------
    GeneratedRubric
        Frozen result with clause details, criterion names, and weights.

    Raises
    ------
    ValueError
        If *name* is empty or *text* yields no clauses.
    """
    if not name:
        raise ValueError("name must be non-empty")

    clauses = extract_clauses(text, max_clauses=max_clauses)

    criteria = tuple(c.criterion_name for c in clauses)
    weights = {c.criterion_name: c.weight for c in clauses}

    n_positive = sum(1 for c in clauses if c.polarity == "positive")
    n_negative = sum(1 for c in clauses if c.polarity == "negative")

    description = (
        f"Constitutional rubric '{name}' — {len(clauses)} clause(s) from policy text."
    )
    rubric = Rubric(
        name=name,
        description=description,
        criteria=criteria,
        weights=weights,
        version=RUBRIC_VERSION,
    )
    versions = RUBRIC_REGISTRY.setdefault(name, {})
    versions[RUBRIC_VERSION] = rubric
    RUBRICS[name] = rubric

    return GeneratedRubric(
        name=name,
        clauses=tuple(clauses),
        n_clauses=len(clauses),
        n_positive=n_positive,
        n_negative=n_negative,
        criteria=criteria,
        weights=weights,
    )
