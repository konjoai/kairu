"""Tests for kairu.tournament — round-robin pairwise compare + Elo."""
from __future__ import annotations

import json

import pytest

from kairu.ensemble import JudgeConfig
from kairu.tournament import (
    DEFAULT_ELO_START,
    ModelRanking,
    TournamentResult,
    TournamentStore,
    run_tournament,
)


PROMPTS = [
    "What is the capital of France?",
    "Briefly explain photosynthesis.",
    "Define recursion.",
]

# Three models with clear quality stratification: GOOD > MID > BAD.
GOOD = [
    "Paris is the capital of France.",
    "Photosynthesis converts light into chemical energy stored as glucose; oak leaves absorb sunlight via chlorophyll.",
    "Recursion is when a function calls itself with a smaller input until it hits a base case.",
]
MID = [
    "Paris.",
    "Plants use sunlight to make food.",
    "Recursion calls itself.",
]
BAD = [
    "idk",
    "plants",
    "loop thing",
]

JUDGES = [JudgeConfig(name="j1"), JudgeConfig(name="j2"), JudgeConfig(name="j3")]


def _models():
    return [
        {"name": "good", "responses": GOOD},
        {"name": "mid",  "responses": MID},
        {"name": "bad",  "responses": BAD},
    ]


# ── Validation ─────────────────────────────────────────────────────────


def test_requires_at_least_two_models():
    with pytest.raises(ValueError):
        run_tournament([{"name": "solo", "responses": ["x"]}], ["p"], JUDGES)


def test_requires_at_least_one_prompt():
    with pytest.raises(ValueError):
        run_tournament(_models(), [], JUDGES)


def test_requires_at_least_one_judge():
    with pytest.raises(ValueError):
        run_tournament(_models(), PROMPTS, [])


def test_response_count_must_match_prompt_count():
    with pytest.raises(ValueError):
        run_tournament(
            [{"name": "a", "responses": ["x"]}, {"name": "b", "responses": ["y", "z"]}],
            ["p1", "p2"], JUDGES,
        )


def test_model_names_must_be_unique():
    with pytest.raises(ValueError):
        run_tournament(
            [{"name": "dup", "responses": GOOD}, {"name": "dup", "responses": MID}],
            PROMPTS, JUDGES,
        )


def test_rejects_bad_model_payload_types():
    with pytest.raises(TypeError):
        run_tournament([{"responses": ["x"]}, {"name": "b", "responses": ["y"]}],
                       ["p"], JUDGES)
    with pytest.raises(TypeError):
        run_tournament([{"name": "a", "responses": "not-a-list"},
                        {"name": "b", "responses": ["y"]}], ["p"], JUDGES)


# ── Result shape + Elo ────────────────────────────────────────────────


def test_result_shape_is_complete():
    result = run_tournament(_models(), PROMPTS, JUDGES)
    assert isinstance(result, TournamentResult)
    assert result.n_prompts == 3
    # 3 models choose-2 × 3 prompts = 9 matches.
    assert result.n_matches == 9
    assert set(result.models) == {"good", "mid", "bad"}
    # Win matrix is square (minus diagonal) and only contains other models.
    for a in result.models:
        assert a not in result.win_matrix[a]
        for b in result.models:
            if a == b:
                continue
            assert b in result.win_matrix[a]


def test_better_model_wins_more():
    result = run_tournament(_models(), PROMPTS, JUDGES)
    # Total wins by model.
    wins = {r.model: r.wins for r in result.rankings}
    assert wins["good"] >= wins["mid"] >= wins["bad"]


def test_rankings_are_sorted_by_elo_desc():
    result = run_tournament(_models(), PROMPTS, JUDGES)
    assert [r.rank for r in result.rankings] == sorted([r.rank for r in result.rankings])
    elos = [r.elo for r in result.rankings]
    assert elos == sorted(elos, reverse=True)


def test_top_ranked_model_is_the_strongest():
    result = run_tournament(_models(), PROMPTS, JUDGES)
    assert result.rankings[0].model == "good"


def test_elo_starts_at_default_and_diverges():
    result = run_tournament(_models(), PROMPTS, JUDGES, elo_start=1500.0)
    elos = list(result.elo.values())
    # Some model gained; some lost.
    assert max(elos) > 1500.0
    assert min(elos) < 1500.0
    # Total Elo conservation: sum should stay close to N × 1500 modulo float drift.
    assert abs(sum(elos) - 1500.0 * len(result.models)) < 1e-6


def test_per_criterion_dominance_is_populated():
    result = run_tournament(_models(), PROMPTS, JUDGES)
    assert result.per_criterion_dominance  # at least one criterion
    # Each bucket includes every model name as a key.
    for crit, bucket in result.per_criterion_dominance.items():
        assert set(bucket.keys()).issubset(set(result.models))


def test_identical_models_tie_throughout():
    twins = [
        {"name": "alpha", "responses": GOOD},
        {"name": "bravo", "responses": list(GOOD)},
    ]
    result = run_tournament(twins, PROMPTS, JUDGES)
    # All matches are ties → both models keep Elo = 1500.
    for r in result.rankings:
        assert r.wins == 0
        assert r.losses == 0
        assert r.ties == 3
        assert r.elo == pytest.approx(1500.0, abs=1e-6)


def test_to_dict_is_json_serialisable():
    result = run_tournament(_models(), PROMPTS, JUDGES)
    json.dumps(result.to_dict())  # no raise
    d = result.to_dict()
    assert "rankings" in d and len(d["rankings"]) == 3


# ── TournamentStore ───────────────────────────────────────────────────


def test_store_save_get_list():
    store = TournamentStore()
    result = run_tournament(_models(), PROMPTS, JUDGES)
    tid = store.save(result)
    assert tid == result.tournament_id
    assert store.get(tid).tournament_id == tid
    assert tid in [t.tournament_id for t in store.list()]


def test_store_unknown_raises_keyerror():
    store = TournamentStore()
    with pytest.raises(KeyError):
        store.get("no-such-id")
