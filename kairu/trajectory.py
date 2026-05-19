"""Agentic trajectory scoring — pure stdlib, deterministic, bounded [0, 1].

Scores a sequence of (step, tool_call, observation, response) records on four
quality dimensions:

- **tool_selection** — relevance of the tool call to the stated goal.
- **error_recovery** — quality of recovery when an error observation is seen.
- **goal_progress** — token-overlap of the response with the goal description.
- **efficiency** — non-repetition / non-stagnation across steps.

Each per-step score is bundled into a :class:`StepScore`; the aggregate over
all steps is returned as a :class:`TrajectoryEvaluation`.

All result types are frozen dataclasses.  No external dependencies are used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

__all__ = [
    "TrajectoryStep",
    "StepScore",
    "TrajectoryEvaluation",
    "evaluate_trajectory",
]

# ---------------------------------------------------------------------------
# Tokenisation (pure stdlib — mirrors evaluation._content_tokens pattern)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")

_STOPWORDS: frozenset[str] = frozenset(
    (
        "a an and are as at be been by do for from has have he her him his i "
        "in is it its me my no not of on or our she so than that the their them "
        "they this to too us was we were what when where who will with you your"
    ).split()
)

_ERROR_SIGNALS: frozenset[str] = frozenset(
    [
        "error",
        "fail",
        "exception",
        "not found",
        "invalid",
        "timeout",
        "traceback",
        "refused",
    ]
)

_RECOVERY_SIGNALS: frozenset[str] = frozenset(
    [
        "retry",
        "alternative",
        "instead",
        "try",
        "fallback",
        "adjust",
        "reconsider",
        "different",
        "another",
        "fix",
    ]
)


def _content_tokens(text: str) -> list[str]:
    """Return lowercased word tokens with stopwords removed."""
    return [
        m.group(0).lower()
        for m in _WORD_RE.finditer(text)
        if m.group(0).lower() not in _STOPWORDS
    ]


def _has_error_signal(observation: str) -> bool:
    """Return True if *observation* contains at least one error-signal phrase."""
    obs_lower = observation.lower()
    return any(signal in obs_lower for signal in _ERROR_SIGNALS)


def _count_recovery_signals(response: str) -> int:
    """Count how many distinct recovery-signal words appear in *response*."""
    resp_lower = response.lower()
    return sum(1 for word in _RECOVERY_SIGNALS if word in resp_lower)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrajectoryStep:
    """A single step in an agentic trajectory.

    Parameters
    ----------
    step:
        0-indexed step number.
    tool_call:
        The action/tool the agent invoked (``None`` = no tool used).
    observation:
        Environment feedback / tool result (``None`` = no feedback yet).
    response:
        Agent's reasoning/response text at this step.
    """

    step: int
    tool_call: Optional[str]
    observation: Optional[str]
    response: str


@dataclass(frozen=True)
class StepScore:
    """Quality scores for a single :class:`TrajectoryStep`.

    All dimension scores are bounded to ``[0, 1]``.

    Parameters
    ----------
    step:
        0-indexed step number (mirrors :attr:`TrajectoryStep.step`).
    tool_selection:
        Relevance of the tool call to the goal.
    error_recovery:
        Recovery quality after error observations.
    goal_progress:
        Token overlap of the response with the goal.
    efficiency:
        Non-repetition / non-stagnation relative to the previous step.
    score:
        Unweighted mean of the four dimension scores.
    """

    step: int
    tool_selection: float
    error_recovery: float
    goal_progress: float
    efficiency: float
    score: float


@dataclass(frozen=True)
class TrajectoryEvaluation:
    """Aggregate evaluation of an agentic trajectory.

    Parameters
    ----------
    goal:
        The objective the agent was trying to accomplish.
    n_steps:
        Total number of steps evaluated.
    steps:
        Per-step :class:`StepScore` records.
    tool_selection:
        Mean per-step tool_selection score.
    error_recovery:
        Mean per-step error_recovery score.
    goal_completion:
        Final-step ``goal_progress`` — proxy for whether the goal was met.
    efficiency:
        Mean per-step (or optimal-steps-scaled) efficiency score.
    aggregate:
        Unweighted mean of the four dimension means.
    """

    goal: str
    n_steps: int
    steps: Tuple[StepScore, ...]
    tool_selection: float
    error_recovery: float
    goal_completion: float
    efficiency: float
    aggregate: float


# ---------------------------------------------------------------------------
# Per-step dimension scorers
# ---------------------------------------------------------------------------


def _score_tool_selection(goal_toks: set[str], tool_call: Optional[str]) -> float:
    """Return the tool-selection score for a single step.

    If ``tool_call`` is ``None``, no tool was needed so the score is 1.0.
    Otherwise returns the content-token recall of goal terms in the tool call.
    """
    if tool_call is None:
        return 1.0
    if not goal_toks:
        return 1.0
    tc_toks = set(_content_tokens(tool_call))
    return len(goal_toks & tc_toks) / len(goal_toks)


def _score_error_recovery(observation: Optional[str], response: str) -> float:
    """Return the error-recovery score for a single step.

    Returns 1.0 when there is no error signal in ``observation``.
    When an error is observed, score = min(1.0, recovery_signals / 2).
    """
    if observation is None or not _has_error_signal(observation):
        return 1.0
    recovery_count = _count_recovery_signals(response)
    return min(1.0, recovery_count / 2.0)


def _score_goal_progress(goal_toks: set[str], response: str) -> float:
    """Return content-token recall of goal terms found in *response*."""
    if not goal_toks:
        return 1.0
    resp_toks = set(_content_tokens(response))
    return len(goal_toks & resp_toks) / len(goal_toks)


def _score_efficiency(
    step_index: int,
    tool_call: Optional[str],
    goal_progress: float,
    prev_tool_call: Optional[str],
) -> float:
    """Return the efficiency score for step *step_index*.

    Step 0 always gets 1.0.  Subsequent steps lose 0.5 for repeating the
    previous tool call and 0.5 for making no goal progress (stagnation).
    Result is clamped to [0, 1].
    """
    if step_index == 0:
        return 1.0
    score = 1.0
    if tool_call is not None and tool_call == prev_tool_call:
        score -= 0.5
    if goal_progress == 0.0:
        score -= 0.5
    return max(0.0, min(1.0, score))


def _score_step(
    step: TrajectoryStep,
    goal_toks: set[str],
    prev_tool_call: Optional[str],
) -> StepScore:
    """Compute all four dimension scores for a single :class:`TrajectoryStep`."""
    ts = _score_tool_selection(goal_toks, step.tool_call)
    er = _score_error_recovery(step.observation, step.response)
    gp = _score_goal_progress(goal_toks, step.response)
    ef = _score_efficiency(step.step, step.tool_call, gp, prev_tool_call)
    mean_score = (ts + er + gp + ef) / 4.0
    return StepScore(
        step=step.step,
        tool_selection=ts,
        error_recovery=er,
        goal_progress=gp,
        efficiency=ef,
        score=mean_score,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_trajectory(
    goal: str,
    steps: Sequence[TrajectoryStep],
    *,
    optimal_steps: Optional[int] = None,
) -> TrajectoryEvaluation:
    """Evaluate an agentic trajectory on four quality dimensions.

    Parameters
    ----------
    goal:
        The objective the agent is trying to accomplish.
    steps:
        Sequence of :class:`TrajectoryStep` records in execution order.
    optimal_steps:
        If provided, an efficiency bonus/penalty scale is applied:
        ``actual_steps / optimal_steps`` ratio modulates the mean efficiency
        score.  When ``None``, per-step efficiency only.

    Returns
    -------
    TrajectoryEvaluation
        Per-step breakdown and aggregate dimension means.

    Raises
    ------
    ValueError
        If ``steps`` is empty or ``optimal_steps`` is not None and <= 0.
    """
    if not steps:
        raise ValueError("steps must be non-empty")
    if optimal_steps is not None and optimal_steps <= 0:
        raise ValueError("optimal_steps must be > 0")

    goal_toks: set[str] = set(_content_tokens(goal))

    scored: list[StepScore] = []
    prev_tool_call: Optional[str] = None
    for raw_step in steps:
        ss = _score_step(raw_step, goal_toks, prev_tool_call)
        scored.append(ss)
        prev_tool_call = raw_step.tool_call

    n = len(scored)
    mean_ts = sum(s.tool_selection for s in scored) / n
    mean_er = sum(s.error_recovery for s in scored) / n
    mean_gp = sum(s.goal_progress for s in scored) / n
    mean_ef = sum(s.efficiency for s in scored) / n

    goal_completion = scored[-1].goal_progress

    if optimal_steps is not None:
        step_ratio = len(steps) / optimal_steps
        mean_ef = max(0.0, min(1.0, mean_ef / step_ratio))

    aggregate = (mean_ts + mean_er + mean_ef + mean_gp) / 4.0

    return TrajectoryEvaluation(
        goal=goal,
        n_steps=n,
        steps=tuple(scored),
        tool_selection=mean_ts,
        error_recovery=mean_er,
        goal_completion=goal_completion,
        efficiency=mean_ef,
        aggregate=aggregate,
    )
