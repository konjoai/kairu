"""Multi-model tournament — round-robin pairwise compare + Elo ranking.

Run every (model_a, model_b) pair across every prompt, score each pair
via the judge ensemble, and roll the results up into:

  * an N × N **win matrix** (entries are win counts, diagonal is zero),
  * Elo-style ratings (start 1500, K = 32),
  * per-criterion dominance (which model wins each criterion most often),
  * an overall ranking by Elo, with ties broken by total wins.

Each "model" in the request is a row of pre-generated responses — one
per prompt. We never call an LLM here; the caller has already produced
each model's responses (typically from an offline benchmark batch).
That keeps the tournament endpoint pure-CPU, deterministic, and fast.

Wire shape
----------
::

    {
      "prompts": ["What is 2+2?", "Capital of France?", ...],
      "models": [
        {"name": "gpt-4o",    "responses": ["Four.",  "Paris.", ...]},
        {"name": "claude",    "responses": ["Four",   "Paris is the capital.", ...]},
        ...
      ],
      "judges": [{"name": "j1"}, {"name": "j2"}]
    }

Elo arithmetic
--------------
Per match between rated R_a, R_b: expected score for A is
``E_a = 1 / (1 + 10**((R_b - R_a) / 400))``. After a match scored s_a
(1 for win, 0.5 for tie, 0 for loss), the update is
``R_a' = R_a + K * (s_a - E_a)`` with ``K = 32``. Standard chess Elo.
"""
from __future__ import annotations

import statistics
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from kairu.ensemble import (
    DEFAULT_DISAGREEMENT_THRESHOLD,
    EnsembleComparison,
    JudgeConfig,
    ensemble_compare,
)


DEFAULT_ELO_K: float = 32.0
DEFAULT_ELO_START: float = 1500.0


@dataclass(frozen=True)
class TournamentMatch:
    """One pairwise comparison result."""

    prompt_idx: int
    model_a: str
    model_b: str
    winner: str          # "a" | "b" | "tie"
    median_diff: float
    disagreement_flag: bool
    per_criterion: Dict[str, str]  # criterion → "a" | "b" | "tie"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelRanking:
    """One row of the final leaderboard."""

    model: str
    elo: float
    wins: int
    losses: int
    ties: int
    rank: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TournamentResult:
    """Aggregate of all matches in a tournament."""

    tournament_id: str
    created_utc: float
    models: Tuple[str, ...]
    n_prompts: int
    n_matches: int
    matches: Tuple[TournamentMatch, ...]
    win_matrix: Dict[str, Dict[str, int]]
    elo: Dict[str, float]
    rankings: Tuple[ModelRanking, ...]
    per_criterion_dominance: Dict[str, Dict[str, int]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tournament_id": self.tournament_id,
            "created_utc": self.created_utc,
            "models": list(self.models),
            "n_prompts": self.n_prompts,
            "n_matches": self.n_matches,
            "matches": [m.to_dict() for m in self.matches],
            "win_matrix": {k: dict(v) for k, v in self.win_matrix.items()},
            "elo": dict(self.elo),
            "rankings": [r.to_dict() for r in self.rankings],
            "per_criterion_dominance": {
                k: dict(v) for k, v in self.per_criterion_dominance.items()
            },
        }


# ─────────────────────────────────────────────────────────────────────────
# Elo update
# ─────────────────────────────────────────────────────────────────────────


def _expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _update_elo(
    rating_a: float, rating_b: float, score_a: float, k: float = DEFAULT_ELO_K,
) -> Tuple[float, float]:
    e_a = _expected_score(rating_a, rating_b)
    e_b = 1.0 - e_a
    delta_a = k * (score_a - e_a)
    delta_b = k * ((1.0 - score_a) - e_b)
    return rating_a + delta_a, rating_b + delta_b


# ─────────────────────────────────────────────────────────────────────────
# Tournament runner
# ─────────────────────────────────────────────────────────────────────────


def run_tournament(
    models: Sequence[Mapping[str, Any]],
    prompts: Sequence[str],
    judges: Sequence[JudgeConfig],
    *,
    elo_start: float = DEFAULT_ELO_START,
    elo_k: float = DEFAULT_ELO_K,
    disagreement_threshold: float = DEFAULT_DISAGREEMENT_THRESHOLD,
) -> TournamentResult:
    """Run every pair of models across every prompt."""
    if len(models) < 2:
        raise ValueError("tournament requires at least 2 models")
    if len(prompts) < 1:
        raise ValueError("tournament requires at least 1 prompt")
    if len(judges) < 1:
        raise ValueError("tournament requires at least 1 judge")

    names: List[str] = []
    response_table: List[List[str]] = []
    for i, m in enumerate(models):
        name = m.get("name") if isinstance(m, Mapping) else None
        if not isinstance(name, str) or not name:
            raise TypeError(f"model {i}: missing string 'name'")
        responses = m.get("responses") if isinstance(m, Mapping) else None
        if not isinstance(responses, (list, tuple)):
            raise TypeError(f"model {i}: 'responses' must be a list")
        if len(responses) != len(prompts):
            raise ValueError(
                f"model '{name}': {len(responses)} responses for {len(prompts)} prompts"
            )
        for j, r in enumerate(responses):
            if not isinstance(r, str):
                raise TypeError(f"model '{name}': response[{j}] must be a string")
        names.append(name)
        response_table.append(list(responses))
    if len(set(names)) != len(names):
        raise ValueError("model names must be unique")

    # Per-model bookkeeping.
    elo: Dict[str, float] = {n: float(elo_start) for n in names}
    wins  = {n: 0 for n in names}
    losses = {n: 0 for n in names}
    ties   = {n: 0 for n in names}
    win_matrix: Dict[str, Dict[str, int]] = {
        a: {b: 0 for b in names if b != a} for a in names
    }
    per_criterion_dominance: Dict[str, Dict[str, int]] = {}

    matches: List[TournamentMatch] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name_a, name_b = names[i], names[j]
            for p_idx, prompt in enumerate(prompts):
                resp_a = response_table[i][p_idx]
                resp_b = response_table[j][p_idx]
                cmp: EnsembleComparison = ensemble_compare(
                    prompt, resp_a, resp_b, judges,
                    disagreement_threshold=disagreement_threshold,
                )
                if cmp.winner == "a":
                    score_a = 1.0
                    wins[name_a] += 1
                    losses[name_b] += 1
                    win_matrix[name_a][name_b] += 1
                elif cmp.winner == "b":
                    score_a = 0.0
                    wins[name_b] += 1
                    losses[name_a] += 1
                    win_matrix[name_b][name_a] += 1
                else:
                    score_a = 0.5
                    ties[name_a] += 1
                    ties[name_b] += 1

                elo[name_a], elo[name_b] = _update_elo(
                    elo[name_a], elo[name_b], score_a, k=elo_k,
                )

                per_crit: Dict[str, str] = {}
                for crit, body in cmp.per_criterion.items():
                    w = body["winner"]
                    per_crit[crit] = w
                    bucket = per_criterion_dominance.setdefault(crit, {n: 0 for n in names})
                    if w == "a":
                        bucket[name_a] = bucket.get(name_a, 0) + 1
                    elif w == "b":
                        bucket[name_b] = bucket.get(name_b, 0) + 1

                matches.append(TournamentMatch(
                    prompt_idx=p_idx,
                    model_a=name_a, model_b=name_b,
                    winner=cmp.winner, median_diff=cmp.median_diff,
                    disagreement_flag=cmp.disagreement_flag,
                    per_criterion=per_crit,
                ))

    # Ranking: by Elo desc, tie-break by total wins desc, then name asc.
    ranked_names = sorted(
        names, key=lambda n: (-elo[n], -wins[n], n),
    )
    rankings = tuple(
        ModelRanking(
            model=n, elo=round(elo[n], 2),
            wins=wins[n], losses=losses[n], ties=ties[n], rank=idx + 1,
        )
        for idx, n in enumerate(ranked_names)
    )

    return TournamentResult(
        tournament_id=uuid.uuid4().hex,
        created_utc=time.time(),
        models=tuple(names),
        n_prompts=len(prompts),
        n_matches=len(matches),
        matches=tuple(matches),
        win_matrix=win_matrix,
        elo={k: round(v, 2) for k, v in elo.items()},
        rankings=rankings,
        per_criterion_dominance=per_criterion_dominance,
    )


# ─────────────────────────────────────────────────────────────────────────
# In-memory tournament store
# ─────────────────────────────────────────────────────────────────────────


class TournamentStore:
    """Process-local tournament store. Newest first on `list`."""

    def __init__(self) -> None:
        self._tournaments: Dict[str, TournamentResult] = {}

    def save(self, result: TournamentResult) -> str:
        self._tournaments[result.tournament_id] = result
        return result.tournament_id

    def get(self, tournament_id: str) -> TournamentResult:
        if tournament_id not in self._tournaments:
            raise KeyError(tournament_id)
        return self._tournaments[tournament_id]

    def list(self) -> List[TournamentResult]:
        return sorted(
            self._tournaments.values(),
            key=lambda t: t.created_utc, reverse=True,
        )

    def __len__(self) -> int:
        return len(self._tournaments)


__all__ = [
    "DEFAULT_ELO_K",
    "DEFAULT_ELO_START",
    "TournamentMatch",
    "ModelRanking",
    "TournamentResult",
    "TournamentStore",
    "run_tournament",
]
