"""Tests for kairu.gamma_scheduler — AIMD adaptive γ for speculative decoding."""
from __future__ import annotations

import pytest

from kairu.gamma_scheduler import DynamicGammaScheduler
from kairu.mock_model import MockModel
from kairu.speculative import SpeculativeDecoder


def test_initial_state():
    s = DynamicGammaScheduler(initial=4)
    assert s.gamma == 4
    assert s.rolling_rate() == 0.0
    assert s.stats()["adjustments"] == 0


def test_high_acceptance_grows_gamma():
    s = DynamicGammaScheduler(initial=2, max_gamma=8, high_threshold=0.7, window=2)
    s.update(2, 2)  # rolling 1.0 ≥ 0.7 → 2 → 3
    assert s.gamma == 3
    s.update(2, 2)  # rolling 1.0 → 3 → 4
    assert s.gamma == 4


def test_gamma_clamped_at_max():
    s = DynamicGammaScheduler(initial=7, max_gamma=8, high_threshold=0.5, window=1)
    for _ in range(5):
        s.update(1, 1)
    assert s.gamma == 8  # clamped


def test_low_acceptance_shrinks_gamma():
    s = DynamicGammaScheduler(initial=8, min_gamma=1, low_threshold=0.4, decrease_factor=0.5, window=1)
    s.update(0, 4)  # rate 0.0 ≤ 0.4 → 8 * 0.5 = 4
    assert s.gamma == 4
    s.update(0, 4)
    assert s.gamma == 2
    s.update(0, 4)
    assert s.gamma == 1
    s.update(0, 4)
    assert s.gamma == 1  # floor


def test_mid_acceptance_does_not_change():
    s = DynamicGammaScheduler(initial=4, high_threshold=0.8, low_threshold=0.4, window=1)
    s.update(2, 4)  # rate 0.5 — between thresholds
    assert s.gamma == 4
    assert s.stats()["adjustments"] == 0


def test_rolling_window_smooths():
    """A burst of high acceptance must not move γ if the rolling mean stays mid-band."""
    s = DynamicGammaScheduler(initial=4, high_threshold=0.8, low_threshold=0.4, window=4)
    s.update(2, 4)  # 0.5
    s.update(2, 4)  # 0.5
    s.update(4, 4)  # rolling mean (0.5+0.5+1.0)/3 ≈ 0.67 — between thresholds
    s.update(2, 4)  # mean (0.5+0.5+1.0+0.5)/4 = 0.625 — still mid
    assert s.gamma == 4


def test_rejects_bad_config():
    with pytest.raises(ValueError):
        DynamicGammaScheduler(initial=10, max_gamma=8)
    with pytest.raises(ValueError):
        DynamicGammaScheduler(initial=4, low_threshold=0.9, high_threshold=0.5)
    with pytest.raises(ValueError):
        DynamicGammaScheduler(initial=4, decrease_factor=2.0)
    with pytest.raises(ValueError):
        DynamicGammaScheduler(initial=4, increase=0)
    with pytest.raises(ValueError):
        DynamicGammaScheduler(initial=4, window=0)


def test_update_validates_args():
    s = DynamicGammaScheduler(initial=2)
    with pytest.raises(ValueError):
        s.update(accepted=-1, attempted=4)
    with pytest.raises(ValueError):
        s.update(accepted=5, attempted=4)
    with pytest.raises(ValueError):
        s.update(accepted=0, attempted=0)


def test_speculative_decoder_records_scheduler_stats():
    s = DynamicGammaScheduler(initial=3, window=4)
    dec = SpeculativeDecoder(MockModel(), MockModel(), gamma=3, scheduler=s)
    _, stats = dec.generate([1, 2], max_new_tokens=12)
    assert "final_gamma" in stats
    assert "gamma_adjustments" in stats
    assert s.stats()["window_size"] >= 1


def test_scheduler_adjustments_counter_monotonic():
    s = DynamicGammaScheduler(initial=2, max_gamma=4, high_threshold=0.5, window=1)
    s.update(2, 2)
    a1 = s.stats()["adjustments"]
    s.update(2, 2)
    a2 = s.stats()["adjustments"]
    assert a2 >= a1
