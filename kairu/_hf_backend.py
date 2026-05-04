"""HuggingFace backend — requires kairu[hf]: pip install kairu[hf]."""
from __future__ import annotations

import numpy as np

from kairu.base import ModelInterface


class HuggingFaceModel(ModelInterface):
    """
    Wraps a causal LM from HuggingFace transformers.

    Usage:
        model = HuggingFaceModel("sshleifer/tiny-gpt2")
        logits = model.next_token_logits([1, 2, 3])
    """

    def __init__(self, model_name: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        self._model.eval()
        self._torch = torch

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size

    def max_seq_len(self) -> int:
        cfg = self._model.config
        return getattr(cfg, "max_position_embeddings", 1024)

    def next_token_logits(self, token_ids: list[int]) -> np.ndarray:
        ids = self._torch.tensor([token_ids], dtype=self._torch.long)
        with self._torch.no_grad():
            out = self._model(ids)
        return out.logits[0, -1].float().detach().numpy()

    def encode(self, text: str) -> list[int]:
        """Tokenize text to token IDs."""
        return self._tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back to text."""
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)

    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
    ):
        """
        Stream tokens one by one using HF TextIteratorStreamer.
        Yields decoded text chunks (NOT token IDs) as they are generated.
        """
        import threading

        from transformers import TextIteratorStreamer

        inputs = self._tokenizer(prompt, return_tensors="pt")
        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generate_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
        )

        thread = threading.Thread(target=self._model.generate, kwargs=generate_kwargs)
        thread.start()

        for text_chunk in streamer:
            yield text_chunk

        thread.join()


def load_hf_model(model_name: str) -> HuggingFaceModel:
    return HuggingFaceModel(model_name)


class HuggingFaceKVCachedModel(HuggingFaceModel):
    """HF model with persistent ``past_key_values`` reuse across calls.

    Standard HF causal-LM forward recomputes the full attention over the
    whole prefix every call. ``past_key_values`` lets the model do a single
    *new-tokens-only* pass and append to a cached K/V tensor, reducing the
    per-token cost from O(n²) to O(n) once a prefix is established.

    We expose this as a drop-in :class:`HuggingFaceModel` subclass so it
    works behind :class:`kairu.SpeculativeDecoder`,
    :class:`kairu.LayerwiseEarlyExitDecoder`, and the streaming server
    without any changes to those callers.

    Cache invalidation rule
    -----------------------
    The cache is keyed by the longest common prefix between the current
    request and the prior one. If the new prefix diverges (e.g. a different
    prompt arrives), we drop the cache and recompute from scratch — same
    semantics as ``transformers``' built-in generation cache.

    Memory bound
    ------------
    K/V tensors grow linearly with sequence length; the cap is the model's
    own ``max_position_embeddings``. We respect that hard limit.
    """

    def __init__(self, model_name: str) -> None:
        super().__init__(model_name)
        self._cached_prefix: list[int] = []
        self._past = None  # transformers `Cache` or legacy tuple
        self._kv_hits = 0
        self._kv_misses = 0

    def _common_prefix_len(self, token_ids: list[int]) -> int:
        cap = min(len(self._cached_prefix), len(token_ids))
        i = 0
        while i < cap and self._cached_prefix[i] == token_ids[i]:
            i += 1
        return i

    def reset_cache(self) -> None:
        self._cached_prefix = []
        self._past = None

    @property
    def kv_cache_stats(self) -> dict:
        total = self._kv_hits + self._kv_misses
        return {
            "kv_hits": self._kv_hits,
            "kv_misses": self._kv_misses,
            "kv_hit_rate": (self._kv_hits / total) if total > 0 else 0.0,
            "cached_prefix_len": len(self._cached_prefix),
        }

    def next_token_logits(self, token_ids: list[int]) -> np.ndarray:
        torch = self._torch
        if not token_ids:
            raise ValueError("token_ids must be non-empty")

        # Hard cap by model's max position embedding count.
        max_len = self.max_seq_len()
        if len(token_ids) > max_len:
            token_ids = token_ids[-max_len:]
            self.reset_cache()

        common = self._common_prefix_len(token_ids)
        if self._past is None or common != len(self._cached_prefix):
            # Divergence (or first call): drop cache.
            self.reset_cache()
            common = 0

        suffix = token_ids[common:]
        if not suffix:
            # Whole prefix was cached — feed only the last token to get its logits.
            suffix = [token_ids[-1]]
            common = len(token_ids) - 1
            # Have to recompute past for the [0, common) prefix without the last token.
            if common != len(self._cached_prefix):
                self.reset_cache()
                suffix = token_ids
                common = 0

        if common > 0:
            self._kv_hits += 1
        else:
            self._kv_misses += 1

        ids = torch.tensor([suffix], dtype=torch.long)
        with torch.no_grad():
            out = self._model(ids, past_key_values=self._past, use_cache=True)
        self._past = out.past_key_values
        self._cached_prefix = list(token_ids)
        return out.logits[0, -1].float().detach().numpy()


__all__ = [
    "HuggingFaceModel",
    "HuggingFaceKVCachedModel",
    "load_hf_model",
]
