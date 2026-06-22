"""Tests for kairu.auto_profile — strategy recommendation."""

from __future__ import annotations

from kairu.auto_profile import AutoProfile, DecoderProfile
from kairu.layered import MockLayeredModel
from kairu.mock_model import MockModel


def test_recommend_returns_profile():
    p = AutoProfile.recommend(MockModel())
    assert isinstance(p, DecoderProfile)
    assert p.strategy in {"vanilla", "early_exit", "layered_early_exit", "speculative"}
    assert p.use_cache is True
    assert p.cache_capacity > 0


def test_draft_hint_picks_vanilla():
    p = AutoProfile.recommend(MockModel(), name_hint="tinyllama-draft")
    assert p.strategy == "vanilla"
    assert "draft" in p.rationale.lower()


def test_layered_model_picks_layered_early_exit():
    p = AutoProfile.recommend(MockLayeredModel(num_layers=12))
    assert p.strategy == "layered_early_exit"


def test_speculative_when_draft_supplied_and_large_vocab():
    class Big(MockModel):
        @property
        def vocab_size(self) -> int:
            return 50_000

    p = AutoProfile.recommend(Big(), name_hint="llama-3-8b", has_draft=True)
    assert p.strategy == "speculative"
    assert p.gamma >= 4


def test_speculative_huge_vocab_bumps_gamma():
    class Huge(MockModel):
        @property
        def vocab_size(self) -> int:
            return 150_000

    p = AutoProfile.recommend(Huge(), name_hint="qwen-2-72b", has_draft=True)
    assert p.gamma == 6


def test_mid_size_model_picks_early_exit():
    p = AutoProfile.recommend(MockModel())  # vocab 1000 — below 5k floor
    assert p.strategy == "vanilla"

    class Mid(MockModel):
        @property
        def vocab_size(self) -> int:
            return 32_000

    p2 = AutoProfile.recommend(Mid())
    assert p2.strategy == "early_exit"


def test_draft_hint_overrides_size():
    """A small name override should win even if vocab is large + has_draft."""

    class Big(MockModel):
        @property
        def vocab_size(self) -> int:
            return 100_000

    p = AutoProfile.recommend(Big(), name_hint="opt-125m-draft", has_draft=True)
    assert p.strategy == "vanilla"


def test_profile_is_immutable():
    p = AutoProfile.recommend(MockModel())
    try:
        p.strategy = "speculative"  # type: ignore[misc]
        assert False, "should have raised"
    except Exception:
        pass


def test_recommendation_is_deterministic():
    a = AutoProfile.recommend(MockModel(), name_hint="qwen", has_draft=False)
    b = AutoProfile.recommend(MockModel(), name_hint="qwen", has_draft=False)
    assert a == b


def test_early_exit_gated_for_encoder_architecture():
    """A mid-size model is normally early_exit, but an encoder-style name
    (no left-to-right confidence trajectory) must fall back to vanilla."""

    class Mid(MockModel):
        @property
        def vocab_size(self) -> int:
            return 32_000

    p = AutoProfile.recommend(Mid(), name_hint="bert-large")
    assert p.strategy == "vanilla"
    assert "encoder" in p.rationale.lower()


def test_layered_gated_when_too_shallow():
    """A layered model below the minimum depth is too shallow to exit."""
    p = AutoProfile.recommend(MockLayeredModel(num_layers=4))
    assert p.strategy == "vanilla"


def test_layered_deep_enough_still_early_exits():
    p = AutoProfile.recommend(MockLayeredModel(num_layers=12))
    assert p.strategy == "layered_early_exit"


def _big_with_draft(**kwargs):
    class Big(MockModel):
        @property
        def vocab_size(self) -> int:
            return 50_000

    return AutoProfile.recommend(
        Big(), name_hint="llama-3-8b", has_draft=True, **kwargs
    )


def test_speculative_has_no_warnings_by_default():
    p = _big_with_draft()
    assert p.strategy == "speculative"
    assert p.warnings == ()


def test_four_bit_draft_emits_warning():
    p = _big_with_draft(quant="int4")
    assert p.strategy == "speculative"
    assert any("4-bit" in w for w in p.warnings)


def test_tree_draft_emits_warning():
    p = _big_with_draft(draft_kind="tree")
    assert any("tree" in w.lower() for w in p.warnings)


def test_both_caveats_stack():
    p = _big_with_draft(quant="NF4", draft_kind="Tree")  # case-insensitive
    assert len(p.warnings) == 2


def test_non_speculative_profile_has_no_warnings():
    # quant/draft_kind only matter on the speculative path.
    p = AutoProfile.recommend(MockModel(), quant="int4", draft_kind="tree")
    assert p.warnings == ()
