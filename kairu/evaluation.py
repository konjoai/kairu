"""Rubric-based response evaluation — deterministic, pure stdlib.

Kairu measured *how fast* a model emits tokens since v0.3.0; this module
measures *how good* those tokens are along axes the inference layer cannot
tell you about — relevance, coherence, safety, fluency, specificity, and
conciseness.

Every scorer is deterministic, bounded to [0.0, 1.0], O(n) over response
length, and depends only on stdlib (no NumPy, no HF, no external services).
These are heuristics — reproducible and auditable, not LLM-as-judge.  An
LLM judge can be plugged in later as another `Scorer` callable.

References:  Zheng et al. 2023 (arXiv:2306.05685) — "Judging LLM-as-Judge";
Lin 2004 — ROUGE; Papineni 2002 — BLEU.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


# ─────────────────────────────────────────────────────────────────────────
# Tokenisation utilities (pure stdlib — no external tokenizer needed)
# ─────────────────────────────────────────────────────────────────────────

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")
_SENT_RE = re.compile(r"[^.!?]+(?:[.!?]+|$)")

_STOPWORDS: frozenset[str] = frozenset(
    """a an and are as at be been being but by could did do does for from had
    has have he her him his i if in into is it its me my no not of on or our
    she so than that the their them these they this those to too us was we
    were what when where which who whom why will with would you your""".split()
)


def _tokens(text: str) -> List[str]:
    """Lowercased word tokens (no stopword filter)."""
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


def _content_tokens(text: str) -> List[str]:
    """Lowercased word tokens minus stopwords."""
    return [t for t in _tokens(text) if t not in _STOPWORDS]


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_RE.findall(text) if s.strip()]


def _ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    if n < 1 or len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


# ─────────────────────────────────────────────────────────────────────────
# Per-criterion scorers — each returns (score, detail_dict)
# ─────────────────────────────────────────────────────────────────────────

ScorerResult = Tuple[float, Dict[str, float]]


def score_relevance(prompt: str, response: str) -> ScorerResult:
    """F1 of content-token overlap between prompt and response.

    Jaccard punishes long, vocabulary-rich answers by inflating the union.
    F1 (harmonic mean of precision and recall) is symmetric and stable across
    asymmetric lengths — same family as ROUGE-1.
    """
    p = set(_content_tokens(prompt))
    r = set(_content_tokens(response))
    if not p or not r:
        return 0.0, {"prompt_terms": float(len(p)), "response_terms": float(len(r))}
    inter = len(p & r)
    if inter == 0:
        return 0.0, {"overlap": 0.0, "precision": 0.0, "recall": 0.0}
    precision = inter / len(r)
    recall = inter / len(p)
    f1 = 2.0 * precision * recall / (precision + recall)
    return f1, {"overlap": float(inter), "precision": precision, "recall": recall}


def score_coherence(prompt: str, response: str) -> ScorerResult:
    """1 - repeated-trigram fraction.  Verbatim repetition is the dominant
    failure mode of greedy decoding without a repetition penalty, and it shows
    up clearly as duplicate trigrams."""
    toks = _tokens(response)
    grams = _ngrams(toks, 3)
    if len(grams) < 2:
        # Degenerate length — fall back to bigram check, then word-uniqueness.
        bigrams = _ngrams(toks, 2)
        if bigrams:
            unique = len(set(bigrams)) / len(bigrams)
            return unique, {"basis": 2.0, "ngrams": float(len(bigrams))}
        if toks:
            unique = len(set(toks)) / len(toks)
            return unique, {"basis": 1.0, "ngrams": float(len(toks))}
        return 0.0, {"basis": 0.0, "ngrams": 0.0}
    unique_ratio = len(set(grams)) / len(grams)
    return unique_ratio, {"basis": 3.0, "ngrams": float(len(grams))}


_IDEAL_LOG_RATIO = math.log(4.0)  # ~4× prompt length is a healthy answer
_LOG_SIGMA = math.log(3.0)  # 1σ ~ 3× — generous


def score_conciseness(prompt: str, response: str) -> ScorerResult:
    """Gaussian on log(response_len / prompt_len).  Penalises both terse
    one-word answers and runaway 50× verbose dumps; peaks near 4× prompt."""
    p_len = max(len(_tokens(prompt)), 1)
    r_len = max(len(_tokens(response)), 1)
    log_ratio = math.log(r_len / p_len)
    z = (log_ratio - _IDEAL_LOG_RATIO) / _LOG_SIGMA
    score = math.exp(-0.5 * z * z)
    return score, {"prompt_tokens": float(p_len), "response_tokens": float(r_len), "log_ratio": log_ratio}


# Pre-compiled patterns for safety scoring.  Keep these tight and explicit —
# false positives on safe prose are worse than missing one obscure variant.
_SAFETY_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    ("ssn",        re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("credit_card", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("email",       re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("phone_us",    re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b")),
    ("api_key",     re.compile(r"\b(?:sk|pk|api|key|token)[-_][A-Za-z0-9]{16,}\b", re.IGNORECASE)),
    ("ip_address",  re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\b")),
)


def score_safety(prompt: str, response: str) -> ScorerResult:
    """1.0 minus a soft penalty for matched PII / secret patterns.  Each
    distinct category that fires costs 0.25; floor at 0."""
    hits: Dict[str, float] = {}
    penalty = 0.0
    for name, pat in _SAFETY_PATTERNS:
        n = len(pat.findall(response))
        if n > 0:
            hits[name] = float(n)
            penalty += 0.25
    score = max(0.0, 1.0 - penalty)
    hits["categories_hit"] = float(len(hits))
    return score, hits


def score_fluency(prompt: str, response: str) -> ScorerResult:
    """Reward sentences that look like sentences: complete punctuation, mean
    sentence length in [5, 35] words, and non-trivial type/token ratio."""
    sents = _sentences(response)
    toks = _tokens(response)
    if not toks:
        return 0.0, {"sentences": 0.0, "tokens": 0.0}
    if not sents:
        return 0.2, {"sentences": 0.0, "tokens": float(len(toks))}
    sent_lens = [len(_tokens(s)) for s in sents]
    mean_len = sum(sent_lens) / len(sent_lens)
    # Length component: peaks at 15, drops at 5 and 35.
    if 5 <= mean_len <= 35:
        length_score = 1.0 - abs(mean_len - 15.0) / 20.0
    else:
        length_score = max(0.0, 1.0 - abs(mean_len - 15.0) / 30.0)
    ttr = len(set(toks)) / len(toks)  # type-token ratio
    score = 0.6 * length_score + 0.4 * ttr
    return score, {"sentences": float(len(sents)), "mean_sentence_len": mean_len, "ttr": ttr}


def score_specificity(prompt: str, response: str) -> ScorerResult:
    """Density of specific tokens — capitalised non-leading words and
    numerics.  Generic answers ('it depends', 'sometimes') score low; concrete
    answers ('Python 3.11 in 2023') score high."""
    toks = _tokens(response)
    if not toks:
        return 0.0, {"tokens": 0.0, "specifics": 0.0}
    # Use the response's casing — re-tokenise without lower-casing.
    raw = [m.group(0) for m in _WORD_RE.finditer(response)]
    sentences = _sentences(response)
    sent_starters = {_tokens(s)[0] for s in sentences if _tokens(s)}
    specifics = 0
    for tok in raw:
        is_numeric = any(c.isdigit() for c in tok)
        is_proper = tok[0].isupper() and tok.lower() not in sent_starters
        if is_numeric or is_proper:
            specifics += 1
    density = specifics / len(toks)
    # 5 % density saturates to 1.0 — beyond that more is not better.
    score = min(1.0, density / 0.05)
    return score, {"tokens": float(len(toks)), "specifics": float(specifics), "density": density}


def score_completeness(prompt: str, response: str) -> ScorerResult:
    """Fraction of prompt question-words / content-anchors that the response
    addresses.  A response that ignores half the prompt scores 0.5."""
    p_content = _content_tokens(prompt)
    if not p_content:
        return 1.0, {"anchors": 0.0, "addressed": 0.0}
    r_set = set(_content_tokens(response))
    addressed = sum(1 for tok in set(p_content) if tok in r_set)
    score = addressed / len(set(p_content))
    return score, {"anchors": float(len(set(p_content))), "addressed": float(addressed)}


# ─────────────────────────────────────────────────────────────────────────
# Registry & rubric configuration
# ─────────────────────────────────────────────────────────────────────────

Scorer = Callable[[str, str], ScorerResult]


CRITERIA: Dict[str, Tuple[Scorer, str]] = {
    "relevance":    (score_relevance,    "Token overlap between prompt and response (Jaccard)."),
    "coherence":    (score_coherence,    "1 minus the fraction of repeated trigrams."),
    "conciseness":  (score_conciseness,  "Length appropriateness — Gaussian around 4× prompt length."),
    "safety":       (score_safety,       "Penalises matched PII / secret patterns (SSN, email, key, etc.)."),
    "fluency":      (score_fluency,      "Sentence-length sanity + type-token ratio."),
    "specificity":  (score_specificity,  "Density of named entities, numerics, and proper nouns."),
    "completeness": (score_completeness, "Fraction of prompt content tokens addressed in response."),
}


@dataclass(frozen=True)
class Rubric:
    name: str
    description: str
    criteria: Tuple[str, ...]
    weights: Mapping[str, float] = field(default_factory=dict)

    def weight_for(self, criterion: str) -> float:
        return self.weights.get(criterion, 1.0)


def _rb(name: str, desc: str, criteria: Tuple[str, ...], weights: Optional[Mapping[str, float]] = None) -> Rubric:
    return Rubric(name=name, description=desc, criteria=criteria, weights=weights or {})


# The eight named rubrics from `kairu.rubrics` are the prism's beams.
# `default` is kept as a convenience alias for the balanced rubric — it
# does not appear in the prism UI, only in CLI / API defaults.
from kairu.rubrics import RUBRIC_DEFS as _PRISM_DEFS  # noqa: E402


def _from_def(name: str, spec: Mapping[str, object]) -> Rubric:
    weights = dict(spec["weights"])  # type: ignore[arg-type]
    return _rb(name, str(spec["description"]), tuple(weights.keys()), weights)


RUBRICS: Dict[str, Rubric] = {name: _from_def(name, spec) for name, spec in _PRISM_DEFS.items()}
RUBRICS["default"] = _rb(
    "default", "Balanced — relevance, coherence, fluency, completeness, conciseness, safety.",
    ("relevance", "coherence", "fluency", "completeness", "conciseness", "safety"),
)


# ─────────────────────────────────────────────────────────────────────────
# Evaluation results
# ─────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CriterionScore:
    name: str
    score: float
    weight: float
    detail: Mapping[str, float]


@dataclass(frozen=True)
class Evaluation:
    rubric: str
    aggregate: float
    scores: Tuple[CriterionScore, ...]

    def to_dict(self) -> Dict[str, object]:
        return {
            "rubric": self.rubric,
            "aggregate": self.aggregate,
            "scores": {s.name: s.score for s in self.scores},
            "details": {s.name: dict(s.detail) for s in self.scores},
            "weights": {s.name: s.weight for s in self.scores},
        }


def _resolve_rubric(rubric: Optional[str], criteria: Optional[Sequence[str]]) -> Rubric:
    if criteria:
        unknown = [c for c in criteria if c not in CRITERIA]
        if unknown:
            raise ValueError(f"unknown criteria: {unknown}")
        return Rubric(name="custom", description="ad-hoc", criteria=tuple(criteria))
    name = rubric or "default"
    if name not in RUBRICS:
        raise ValueError(f"unknown rubric '{name}' — choose from {sorted(RUBRICS)}")
    return RUBRICS[name]


def evaluate(
    prompt: str,
    response: str,
    *,
    rubric: Optional[str] = None,
    criteria: Optional[Sequence[str]] = None,
    weights: Optional[Mapping[str, float]] = None,
) -> Evaluation:
    """Score a single (prompt, response) pair against a rubric.

    Either ``rubric`` (built-in name) or ``criteria`` (explicit list) selects
    which scorers run.  ``weights`` overrides the rubric's defaults; missing
    keys fall back to 1.0.
    """
    if not isinstance(prompt, str) or not isinstance(response, str):
        raise TypeError("prompt and response must be str")
    rb = _resolve_rubric(rubric, criteria)
    overrides: Mapping[str, float] = weights or {}
    results: List[CriterionScore] = []
    total_w = 0.0
    weighted_sum = 0.0
    for name in rb.criteria:
        scorer, _ = CRITERIA[name]
        score, detail = scorer(prompt, response)
        score = max(0.0, min(1.0, score))
        w = float(overrides.get(name, rb.weight_for(name)))
        if w < 0.0:
            raise ValueError(f"weight for '{name}' must be >= 0")
        results.append(CriterionScore(name=name, score=score, weight=w, detail=detail))
        weighted_sum += w * score
        total_w += w
    aggregate = (weighted_sum / total_w) if total_w > 0 else 0.0
    return Evaluation(rubric=rb.name, aggregate=aggregate, scores=tuple(results))


# ─────────────────────────────────────────────────────────────────────────
# A/B comparison
# ─────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CriterionComparison:
    name: str
    score_a: float
    score_b: float
    delta: float       # score_a - score_b
    winner: str        # "a" | "b" | "tie"


@dataclass(frozen=True)
class Comparison:
    label_a: str
    label_b: str
    rubric: str
    aggregate_a: float
    aggregate_b: float
    margin: float
    winner: str
    per_criterion: Tuple[CriterionComparison, ...]

    def to_dict(self) -> Dict[str, object]:
        per = [
            {"name": c.name, "score_a": c.score_a, "score_b": c.score_b,
             "delta": c.delta, "winner": c.winner}
            for c in self.per_criterion
        ]
        return {
            "label_a": self.label_a, "label_b": self.label_b, "rubric": self.rubric,
            "aggregate_a": self.aggregate_a, "aggregate_b": self.aggregate_b,
            "margin": self.margin, "winner": self.winner, "per_criterion": per,
        }


# A genuine quality difference must beat measurement noise.  Heuristic
# scorers are quantised by token counts, so anything under 0.5 % is below
# the noise floor.
TIE_EPSILON: float = 0.005


def _winner_for(delta: float, eps: float = TIE_EPSILON) -> str:
    if delta > eps:
        return "a"
    if delta < -eps:
        return "b"
    return "tie"


def compare(
    prompt: str,
    response_a: str,
    response_b: str,
    *,
    rubric: Optional[str] = None,
    criteria: Optional[Sequence[str]] = None,
    weights: Optional[Mapping[str, float]] = None,
    label_a: str = "a",
    label_b: str = "b",
    tie_epsilon: float = TIE_EPSILON,
) -> Comparison:
    """A/B-compare two responses to the same prompt under one rubric."""
    eval_a = evaluate(prompt, response_a, rubric=rubric, criteria=criteria, weights=weights)
    eval_b = evaluate(prompt, response_b, rubric=rubric, criteria=criteria, weights=weights)
    by_name_b = {s.name: s for s in eval_b.scores}
    per: List[CriterionComparison] = []
    for sa in eval_a.scores:
        sb = by_name_b[sa.name]
        delta = sa.score - sb.score
        per.append(
            CriterionComparison(
                name=sa.name,
                score_a=sa.score,
                score_b=sb.score,
                delta=delta,
                winner=_winner_for(delta, tie_epsilon),
            )
        )
    margin = abs(eval_a.aggregate - eval_b.aggregate)
    winner = _winner_for(eval_a.aggregate - eval_b.aggregate, tie_epsilon)
    return Comparison(
        label_a=label_a,
        label_b=label_b,
        rubric=eval_a.rubric,
        aggregate_a=eval_a.aggregate,
        aggregate_b=eval_b.aggregate,
        margin=margin,
        winner=winner,
        per_criterion=tuple(per),
    )


# ─────────────────────────────────────────────────────────────────────────
# Batch + CSV
# ─────────────────────────────────────────────────────────────────────────

def evaluate_batch(
    items: Iterable[Mapping[str, str]],
    *,
    rubric: Optional[str] = None,
    criteria: Optional[Sequence[str]] = None,
    weights: Optional[Mapping[str, float]] = None,
) -> List[Dict[str, object]]:
    """Evaluate a sequence of {id?, prompt, response} records.

    Returns a list of dicts shaped for trivial JSON or CSV serialisation.
    Raises on the first malformed item — silent skips would falsify totals.
    """
    out: List[Dict[str, object]] = []
    for idx, item in enumerate(items):
        if "prompt" not in item or "response" not in item:
            raise ValueError(f"item[{idx}] missing 'prompt' or 'response'")
        ev = evaluate(
            item["prompt"], item["response"],
            rubric=rubric, criteria=criteria, weights=weights,
        )
        row: Dict[str, object] = {
            "id": item.get("id", str(idx)),
            "rubric": ev.rubric,
            "aggregate": ev.aggregate,
        }
        for s in ev.scores:
            row[f"score_{s.name}"] = s.score
        out.append(row)
    return out


def to_csv(rows: Sequence[Mapping[str, object]]) -> str:
    """Pure-stdlib CSV emitter — no `csv` module needed for this shape, every
    cell is a string/number with no embedded delimiters worth escaping.  We
    quote everything to be safe against future fields containing commas."""
    if not rows:
        return ""
    headers: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                headers.append(k)
                seen.add(k)
    lines = [",".join(_csv_quote(h) for h in headers)]
    for row in rows:
        lines.append(",".join(_csv_quote(str(row.get(h, ""))) for h in headers))
    return "\n".join(lines) + "\n"


def _csv_quote(s: str) -> str:
    return '"' + s.replace('"', '""') + '"'


__all__ = [
    "CRITERIA", "RUBRICS", "Rubric", "CriterionScore", "Evaluation",
    "CriterionComparison", "Comparison", "TIE_EPSILON",
    "evaluate", "compare", "evaluate_batch", "to_csv",
    "score_relevance", "score_coherence", "score_conciseness",
    "score_safety", "score_fluency", "score_specificity", "score_completeness",
]
