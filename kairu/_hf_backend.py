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
