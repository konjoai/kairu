"""
HuggingFace backend — only imported when torch + transformers are available.

Install the optional dependency group to enable this backend:
    pip install kairu[hf]
"""
import numpy as np

from kairu.base import ModelInterface


class HuggingFaceModel(ModelInterface):
    """
    Wraps a causal LM from the HuggingFace Hub as a ModelInterface.

    The model is loaded in float16 and kept in eval mode.
    next_token_logits performs a single forward pass and returns the
    last-position logits as a float32 NumPy array.
    """

    def __init__(self, model_name: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16
        )
        self._model.eval()

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size

    def max_seq_len(self) -> int:
        return getattr(self._model.config, "max_position_embeddings", 2048)

    def next_token_logits(self, token_ids: list[int]) -> np.ndarray:
        import torch

        ids = torch.tensor([token_ids], dtype=torch.long)
        with torch.no_grad():
            out = self._model(ids)
        return out.logits[0, -1].float().numpy()


def load_hf_model(name: str) -> HuggingFaceModel:
    """Load a HuggingFace causal LM by name or path."""
    return HuggingFaceModel(name)
