"""Tests for kairu.early_exit — static + CALM-style adaptive early exit."""

from __future__ import annotations

import numpy as np
import pytest

from kairu.base import ModelInterface
from kairu.early_exit import EarlyExitDecoder


class FixedLogitsModel(ModelInterface):
    """Returns the same logits at every step — makes exit decisions deterministic."""

    def __init__(self, logits) -> None:
        self._logits = np.asarray(logits, dtype=np.float32)

    @property
    def vocab_size(self) -> int:
        return int(self._logits.size)

    def next_token_logits(self, token_ids: list[int]) -> np.ndarray:
        return self._logits

    def max_seq_len(self) -> int:
        return 4096


# --------------------------------------------------------------------------- #
# Construction validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kwargs",
    [
        {"confidence_threshold": 0.0},
        {"confidence_threshold": 1.5},
        {"min_confidence": -0.1},
        {"min_confidence": 1.1},
        {"entropy_floor": -1.0},
        {"adapt_decay": -0.5},
    ],
)
def test_rejects_invalid_params(kwargs):
    with pytest.raises(ValueError):
        EarlyExitDecoder(FixedLogitsModel([1.0, 1.0]), **kwargs)


def test_min_confidence_clamped_to_base():
    dec = EarlyExitDecoder(
        FixedLogitsModel([1.0, 1.0]),
        confidence_threshold=0.6,
        adaptive=True,
        min_confidence=0.9,  # above the base → must clamp down to 0.6
    )
    assert dec.min_confidence == pytest.approx(0.6)
    # span collapses to zero → schedule is constant at the base.
    assert dec.effective_confidence(0) == pytest.approx(0.6)
    assert dec.effective_confidence(50) == pytest.approx(0.6)


# --------------------------------------------------------------------------- #
# Effective threshold schedule
# --------------------------------------------------------------------------- #


def test_static_threshold_is_constant():
    dec = EarlyExitDecoder(FixedLogitsModel([1.0, 1.0]), confidence_threshold=0.8)
    assert dec.effective_confidence(0) == 0.8
    assert dec.effective_confidence(100) == 0.8


def test_adaptive_threshold_decays_monotonically_toward_floor():
    dec = EarlyExitDecoder(
        FixedLogitsModel([1.0, 1.0]),
        confidence_threshold=0.9,
        adaptive=True,
        min_confidence=0.5,
        adapt_decay=0.5,
    )
    assert dec.effective_confidence(0) == pytest.approx(0.9)  # base at t=0
    assert dec.effective_confidence(0) > dec.effective_confidence(1)
    assert dec.effective_confidence(1) > dec.effective_confidence(5)
    # As t → ∞ the schedule approaches the floor.
    assert dec.effective_confidence(1000) == pytest.approx(0.5, abs=1e-3)


# --------------------------------------------------------------------------- #
# Generation / exit reasons
# --------------------------------------------------------------------------- #


def test_confidence_exit_on_peaked_distribution():
    dec = EarlyExitDecoder(
        FixedLogitsModel([10.0, 0.0, 0.0, 0.0]), confidence_threshold=0.9
    )
    gen, stats = dec.generate([1], max_new_tokens=5)
    assert stats["exit_reason"] == "confidence"
    assert stats["early_exit"] is True
    assert stats["tokens_generated"] == 1
    assert gen[0] == 0  # emits the argmax token before halting


def test_entropy_exit_when_confidence_bar_unreached():
    # top prob 0.85 < 0.99 (no confidence exit) but entropy ≈ 0.59 < floor 1.0.
    soft = FixedLogitsModel(np.log([0.85, 0.05, 0.05, 0.05]))
    dec = EarlyExitDecoder(soft, confidence_threshold=0.99, entropy_floor=1.0)
    _, stats = dec.generate([1], max_new_tokens=5)
    assert stats["exit_reason"] == "entropy"
    assert stats["early_exit"] is True


def test_runs_to_max_tokens_when_uncertain():
    uniform = FixedLogitsModel([1.0, 1.0, 1.0, 1.0])  # top 0.25, entropy ln4 ≈ 1.39
    dec = EarlyExitDecoder(uniform, confidence_threshold=0.9, entropy_floor=0.5)
    gen, stats = dec.generate([1], max_new_tokens=4)
    assert stats["exit_reason"] == "max_tokens"
    assert stats["early_exit"] is False
    assert stats["tokens_generated"] == 4
    assert len(gen) == 4


def test_stats_carry_adaptive_metadata():
    dec = EarlyExitDecoder(FixedLogitsModel([1.0, 1.0, 1.0, 1.0]), entropy_floor=0.5)
    _, stats = dec.generate([1], max_new_tokens=2)
    assert set(stats) >= {
        "tokens_generated",
        "exit_reason",
        "max_new_tokens",
        "early_exit",
        "adaptive",
        "final_confidence_threshold",
    }
    assert stats["adaptive"] is False
    assert stats["final_confidence_threshold"] == pytest.approx(0.9)


def test_adaptive_exits_where_static_would_not():
    """A mid-confidence (0.7) model never clears a static 0.9 bar, but the
    adaptive schedule relaxes below 0.7 and exits at step 2."""
    model = FixedLogitsModel(np.log([0.7, 0.1, 0.1, 0.1]))

    static = EarlyExitDecoder(model, confidence_threshold=0.9, entropy_floor=0.0)
    _, s_stats = static.generate([1], max_new_tokens=5)
    assert s_stats["exit_reason"] == "max_tokens"

    adaptive = EarlyExitDecoder(
        model,
        confidence_threshold=0.9,
        entropy_floor=0.0,
        adaptive=True,
        min_confidence=0.5,
        adapt_decay=0.5,
    )
    gen, a_stats = adaptive.generate([1], max_new_tokens=5)
    assert a_stats["adaptive"] is True
    assert a_stats["exit_reason"] == "confidence"
    # threshold(2) = 0.5 + 0.4·e^-1 ≈ 0.647 ≤ 0.7 → exits at step index 2.
    assert a_stats["tokens_generated"] == 3
    assert a_stats["final_confidence_threshold"] == pytest.approx(0.6472, abs=1e-3)
    assert gen[-1] == 0  # final token is the argmax at the exit step
