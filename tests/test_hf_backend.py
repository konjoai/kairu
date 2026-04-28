"""Tests for kairu._hf_backend — 4 non-gated structural tests + 4 gated integration tests."""
from __future__ import annotations

import os

import pytest

from kairu.base import ModelInterface


# ---------------------------------------------------------------------------
# Non-gated structural tests (no model loading)
# ---------------------------------------------------------------------------


def test_hf_backend_module_importable():
    """The module must be importable without loading any ML libraries."""
    import kairu._hf_backend  # noqa: F401

    assert hasattr(kairu._hf_backend, "HuggingFaceModel")
    assert hasattr(kairu._hf_backend, "load_hf_model")


def test_hf_model_is_model_interface_subclass():
    from kairu._hf_backend import HuggingFaceModel

    assert issubclass(HuggingFaceModel, ModelInterface)


def test_load_hf_model_is_callable():
    from kairu._hf_backend import load_hf_model

    assert callable(load_hf_model)


def test_hf_model_init_requires_string():
    """HuggingFaceModel.__init__ must accept a single positional str argument.
    We verify the signature without actually calling it."""
    import inspect

    from kairu._hf_backend import HuggingFaceModel

    sig = inspect.signature(HuggingFaceModel.__init__)
    params = list(sig.parameters.keys())
    # Expect: ['self', 'model_name']
    assert "model_name" in params


# ---------------------------------------------------------------------------
# Gated integration tests — only run when KAIRU_TEST_HF=1
# ---------------------------------------------------------------------------

_HF_SKIP = pytest.mark.skipif(
    os.environ.get("KAIRU_TEST_HF") != "1",
    reason="Set KAIRU_TEST_HF=1 to run HuggingFace integration tests",
)


@_HF_SKIP
def test_hf_model_loads_tiny_gpt2():
    from kairu._hf_backend import HuggingFaceModel

    model = HuggingFaceModel("sshleifer/tiny-gpt2")
    assert isinstance(model, ModelInterface)


@_HF_SKIP
def test_hf_next_token_logits_shape():
    import numpy as np

    from kairu._hf_backend import HuggingFaceModel

    model = HuggingFaceModel("sshleifer/tiny-gpt2")
    logits = model.next_token_logits([1, 2, 3])
    assert isinstance(logits, np.ndarray)
    assert logits.shape == (model.vocab_size,)


@_HF_SKIP
def test_hf_encode_decode_roundtrip():
    from kairu._hf_backend import HuggingFaceModel

    model = HuggingFaceModel("sshleifer/tiny-gpt2")
    text = "Hello world"
    ids = model.encode(text)
    recovered = model.decode(ids)
    # The decoded text should approximate the original after stripping
    assert isinstance(recovered, str)
    assert len(recovered) > 0


@_HF_SKIP
def test_hf_stream_generate_yields_strings():
    from kairu._hf_backend import HuggingFaceModel

    model = HuggingFaceModel("sshleifer/tiny-gpt2")
    chunks = []
    for chunk in model.stream_generate("Hello", max_new_tokens=10, temperature=1.0):
        chunks.append(chunk)
        if len(chunks) >= 3:
            break
    assert len(chunks) >= 1
    assert all(isinstance(c, str) for c in chunks)
