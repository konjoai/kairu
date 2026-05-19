"""Tests for kairu.trajectory — agentic trajectory scoring."""

from __future__ import annotations

import pytest

from kairu.trajectory import (
    TrajectoryEvaluation,
    TrajectoryStep,
    evaluate_trajectory,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _step(
    step: int,
    response: str,
    tool_call: str | None = None,
    observation: str | None = None,
) -> TrajectoryStep:
    return TrajectoryStep(
        step=step, tool_call=tool_call, observation=observation, response=response
    )


# ---------------------------------------------------------------------------
# 1. Empty steps raises ValueError
# ---------------------------------------------------------------------------


def test_empty_steps_raises():
    with pytest.raises(ValueError, match="non-empty"):
        evaluate_trajectory("find the file", [])


# ---------------------------------------------------------------------------
# 2. optimal_steps=0 raises ValueError
# ---------------------------------------------------------------------------


def test_optimal_steps_zero_raises():
    s = _step(0, "I searched for the file")
    with pytest.raises(ValueError, match="optimal_steps"):
        evaluate_trajectory("find the file", [s], optimal_steps=0)


# ---------------------------------------------------------------------------
# 3. Single step, no tool call, response addresses goal → high aggregate
# ---------------------------------------------------------------------------


def test_single_step_no_tool_addresses_goal_high_aggregate():
    goal = "summarize the document"
    s = _step(0, "I summarized the document carefully")
    ev = evaluate_trajectory(goal, [s])
    assert ev.aggregate > 0.6
    assert isinstance(ev, TrajectoryEvaluation)


# ---------------------------------------------------------------------------
# 4. tool_selection=1.0 when tool_call is None
# ---------------------------------------------------------------------------


def test_tool_selection_is_1_when_no_tool_call():
    s = _step(0, "thinking through the problem")
    ev = evaluate_trajectory("solve the problem", [s])
    assert ev.steps[0].tool_selection == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 5. tool_selection scores higher when tool_call mentions goal keywords
# ---------------------------------------------------------------------------


def test_tool_selection_higher_with_goal_keyword_in_tool_call():
    goal = "search database records"
    s_relevant = _step(0, "querying now", tool_call="search database query")
    s_irrelevant = _step(0, "querying now", tool_call="open browser window")
    ev_rel = evaluate_trajectory(goal, [s_relevant])
    ev_irr = evaluate_trajectory(goal, [s_irrelevant])
    assert ev_rel.steps[0].tool_selection > ev_irr.steps[0].tool_selection


# ---------------------------------------------------------------------------
# 6. tool_selection=0.0 when tool_call has no goal keyword overlap
# ---------------------------------------------------------------------------


def test_tool_selection_zero_with_no_overlap():
    goal = "search database records"
    # tool_call contains none of the goal content tokens
    s = _step(0, "response", tool_call="xyz abc completely unrelated")
    ev = evaluate_trajectory(goal, [s])
    assert ev.steps[0].tool_selection == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 7. error_recovery=1.0 when observation has no error signals
# ---------------------------------------------------------------------------


def test_error_recovery_is_1_with_no_error_observation():
    s = _step(0, "done", observation="Operation completed successfully.")
    ev = evaluate_trajectory("run the task", [s])
    assert ev.steps[0].error_recovery == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 8. error_recovery > 0 when observation has error and response has recovery signal
# ---------------------------------------------------------------------------


def test_error_recovery_positive_with_error_and_recovery():
    s = _step(
        0,
        response="I will retry with a different approach instead.",
        observation="error: connection timeout",
    )
    ev = evaluate_trajectory("connect to server", [s])
    assert ev.steps[0].error_recovery > 0.0


# ---------------------------------------------------------------------------
# 9. error_recovery=0.0 when observation has error and response has no recovery signals
# ---------------------------------------------------------------------------


def test_error_recovery_zero_with_error_and_no_recovery():
    s = _step(
        0,
        response="Hmm.",
        observation="traceback: NullPointerException at line 42",
    )
    ev = evaluate_trajectory("run program", [s])
    assert ev.steps[0].error_recovery == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 10. goal_progress is higher when response repeats goal terms
# ---------------------------------------------------------------------------


def test_goal_progress_higher_when_response_mentions_goal_terms():
    goal = "search external database records"
    s_match = _step(0, "I searched external database records successfully")
    s_miss = _step(0, "All done now")
    ev_match = evaluate_trajectory(goal, [s_match])
    ev_miss = evaluate_trajectory(goal, [s_miss])
    assert ev_match.steps[0].goal_progress > ev_miss.steps[0].goal_progress


# ---------------------------------------------------------------------------
# 11. efficiency=1.0 for step 0
# ---------------------------------------------------------------------------


def test_efficiency_is_1_for_first_step():
    s = _step(0, "starting the task", tool_call="search tool")
    ev = evaluate_trajectory("find the file", [s])
    assert ev.steps[0].efficiency == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 12. efficiency penalized when tool_call repeats previous step
# ---------------------------------------------------------------------------


def test_efficiency_penalized_for_repeated_tool_call():
    goal = "find the answer"
    s0 = _step(0, "searching for answer", tool_call="search engine")
    s1 = _step(1, "searching for answer again", tool_call="search engine")
    ev = evaluate_trajectory(goal, [s0, s1])
    # Step 1 repeats tool_call AND makes some goal progress — deduction of 0.5
    # but goal_progress may be > 0, so only 0.5 deduction from repetition.
    assert ev.steps[1].efficiency < 1.0


# ---------------------------------------------------------------------------
# 13. efficiency penalized when response has zero goal overlap
# ---------------------------------------------------------------------------


def test_efficiency_penalized_for_zero_goal_progress():
    goal = "summarize quarterly financials"
    s0 = _step(0, "summarize quarterly financials report")
    s1 = _step(1, "xyz irrelevant tangent")  # no goal token overlap → stagnation
    ev = evaluate_trajectory(goal, [s0, s1])
    assert ev.steps[1].efficiency < 1.0


# ---------------------------------------------------------------------------
# 14. optimal_steps scales down efficiency when too many steps taken
# ---------------------------------------------------------------------------


def test_optimal_steps_scales_down_efficiency_for_excess_steps():
    goal = "find configuration file"
    steps = [_step(i, "finding configuration file") for i in range(10)]
    ev_no_opt = evaluate_trajectory(goal, steps)
    # optimal=2 but 10 steps taken — efficiency should be penalised
    ev_with_opt = evaluate_trajectory(goal, steps, optimal_steps=2)
    assert ev_with_opt.efficiency < ev_no_opt.efficiency


# ---------------------------------------------------------------------------
# 15. TrajectoryEvaluation.aggregate = unweighted mean of four dimension means
# ---------------------------------------------------------------------------


def test_aggregate_is_unweighted_mean_of_four_dimension_means():
    goal = "write unit tests"
    steps = [
        _step(0, "writing unit tests for the module"),
        _step(1, "writing more unit tests", observation="error: import failed"),
    ]
    ev = evaluate_trajectory(goal, steps)
    # goal_completion is not used in aggregate — aggregate uses mean_gp internally.
    # Recompute using what the spec says: mean of four dimension MEANS.
    # aggregate = (mean_ts + mean_er + mean_gp + mean_ef) / 4
    # goal_completion = last step's goal_progress, not the mean.
    mean_gp = sum(s.goal_progress for s in ev.steps) / ev.n_steps
    recomputed = (ev.tool_selection + ev.error_recovery + mean_gp + ev.efficiency) / 4.0
    assert ev.aggregate == pytest.approx(recomputed, abs=1e-9)
