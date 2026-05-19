"""Tests for kairu.constitutional — policy-document rubric generation."""

from __future__ import annotations

import pytest

from kairu.constitutional import (
    ConstitutionalClause,
    GeneratedRubric,
    extract_clauses,
    generate_rubric,
    score_response,
)


# ── extract_clauses ──────────────────────────────────────────────────────


def test_extract_clauses_finds_positive_trigger():
    """A sentence with 'must' is extracted as a positive clause."""
    clauses = extract_clauses("The system must encrypt all data.")
    assert len(clauses) == 1
    assert clauses[0].polarity == "positive"
    assert clauses[0].trigger == "must"


def test_extract_clauses_finds_negative_trigger():
    """A sentence with 'must not' is extracted as a negative clause."""
    clauses = extract_clauses("Users must not share passwords.")
    assert len(clauses) == 1
    assert clauses[0].polarity == "negative"
    assert clauses[0].trigger == "must not"


def test_extract_clauses_negative_beats_positive_on_overlap():
    """'must not' is not double-counted as a positive 'must' clause."""
    clauses = extract_clauses(
        "Administrators must not access user data without consent."
    )
    assert len(clauses) == 1
    assert clauses[0].polarity == "negative"
    assert clauses[0].trigger == "must not"


def test_criterion_names_are_snake_case():
    """Criterion names contain only lowercase words joined by underscores."""
    clauses = extract_clauses(
        "The service must log all API requests for auditing purposes."
    )
    assert len(clauses) == 1
    name = clauses[0].criterion_name
    assert "_" in name or name.isalpha()
    assert name == name.lower()
    assert " " not in name


def test_criterion_names_are_deduplicated():
    """Two clauses that produce the same raw name get a numeric suffix."""
    policy = "Users must register before login. Users must verify email before login."
    clauses = extract_clauses(policy)
    assert len(clauses) == 2
    names = [c.criterion_name for c in clauses]
    assert len(set(names)) == 2, f"Expected unique names, got: {names}"


def test_max_clauses_cap_is_respected():
    """extract_clauses never returns more than max_clauses entries."""
    sentences = " ".join(
        f"Rule {i}: the user must comply with policy {i}." for i in range(30)
    )
    clauses = extract_clauses(sentences, max_clauses=5)
    assert len(clauses) <= 5


def test_empty_text_returns_no_clauses():
    """Empty string yields an empty list."""
    assert extract_clauses("") == []


def test_text_with_no_triggers_returns_no_clauses():
    """Text without any trigger word yields an empty list."""
    clauses = extract_clauses(
        "The sky is blue. Water is wet. Grass is green and pleasant."
    )
    assert clauses == []


# ── score_response ───────────────────────────────────────────────────────


def test_score_response_positive_clause_compliant_scores_higher():
    """A response that echoes the obligation terms scores higher than one that does not."""
    clause = ConstitutionalClause(
        text="The system must encrypt all data.",
        polarity="positive",
        trigger="must",
        criterion_name="system_encrypt_data",
    )
    compliant = "All data is encrypted using AES-256 encryption."
    non_compliant = "The weather today is quite pleasant outside."
    ev_good = score_response(compliant, [clause])
    ev_bad = score_response(non_compliant, [clause])
    assert ev_good.aggregate > ev_bad.aggregate


def test_score_response_negative_clause_forbidden_terms_lower_score():
    """A response containing the forbidden terms scores lower than one that avoids them."""
    clause = ConstitutionalClause(
        text="Users must not share passwords.",
        polarity="negative",
        trigger="must not",
        criterion_name="users_share_passwords",
    )
    clean = "Please keep your credentials secure at all times."
    violating = "You can share passwords with trusted colleagues."
    ev_clean = score_response(clean, [clause])
    ev_violating = score_response(violating, [clause])
    assert ev_clean.aggregate > ev_violating.aggregate


def test_score_response_empty_clauses_raises():
    """Passing an empty clause list raises ValueError."""
    with pytest.raises(ValueError, match="non-empty"):
        score_response("some response", [])


def test_clause_score_is_bounded():
    """ClauseScore.score is always in [0.0, 1.0] for arbitrary inputs."""
    clauses = [
        ConstitutionalClause(
            text="Systems must log every action taken.",
            polarity="positive",
            trigger="must",
            criterion_name="systems_log_action",
        ),
        ConstitutionalClause(
            text="Users must not delete audit records.",
            polarity="negative",
            trigger="must not",
            criterion_name="users_delete_audit",
        ),
    ]
    for response_text in [
        "",
        "log action taken systems audit records delete",
        "completely unrelated text about weather and sunshine",
        "must must must must must shall required mandatory obligated",
    ]:
        ev = score_response(response_text, clauses)
        for cs in ev.per_clause:
            assert 0.0 <= cs.score <= 1.0, (
                f"score {cs.score} out of bounds for response: {response_text!r}"
            )


# ── generate_rubric ──────────────────────────────────────────────────────


def test_generate_rubric_returns_correct_counts():
    """n_positive and n_negative sum to n_clauses."""
    policy = (
        "The platform must store data securely. "
        "Users must not upload malware. "
        "All requests shall be authenticated. "
        "Sharing credentials is prohibited."
    )
    rubric = generate_rubric(policy, name="test_counts_rubric")
    assert isinstance(rubric, GeneratedRubric)
    assert rubric.n_positive + rubric.n_negative == rubric.n_clauses
    assert rubric.n_positive >= 1
    assert rubric.n_negative >= 1


def test_generate_rubric_registers_in_rubrics():
    """After generate_rubric, the rubric name is present in kairu.evaluation.RUBRICS."""
    from kairu.evaluation import RUBRICS

    policy = "The system must validate all inputs before processing."
    rubric_name = "test_registration_rubric"
    generate_rubric(policy, name=rubric_name)
    assert rubric_name in RUBRICS
    assert RUBRICS[rubric_name].name == rubric_name


def test_constitutional_evaluation_aggregate_is_unweighted_mean():
    """ConstitutionalEvaluation.aggregate equals the unweighted mean of per-clause scores."""
    clauses = [
        ConstitutionalClause(
            text="The service must provide uptime guarantees.",
            polarity="positive",
            trigger="must",
            criterion_name="service_provide_uptime",
        ),
        ConstitutionalClause(
            text="Operators must not disable monitoring systems.",
            polarity="negative",
            trigger="must not",
            criterion_name="operators_disable_monitoring",
        ),
        ConstitutionalClause(
            text="All deployments shall be reviewed before release.",
            polarity="positive",
            trigger="shall",
            criterion_name="deployments_reviewed_release",
        ),
    ]
    response = "The service ensures uptime and monitoring is always active."
    ev = score_response(response, clauses)
    expected = sum(cs.score for cs in ev.per_clause) / len(ev.per_clause)
    assert ev.aggregate == pytest.approx(expected)
