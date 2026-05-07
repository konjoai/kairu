"""Token watermarking for LLM output integrity.

Implements the Kirchenbauer et al. (2023) green/red list scheme:

    Kirchenbauer J., Geiping J., Wen Y., Kaddour J., Black A., Goldstein T.
    "A Watermark for Large Language Models" — https://arxiv.org/abs/2301.10226

Concept
-------
At each generation step the vocabulary V (size |V|) is pseudo-randomly
partitioned into a *green* list G (size ⌊|V|/2⌋) and a *red* list R = V \ G
using the previous token (or a window of context tokens) as a seed.  The
processor adds a fixed scalar bias δ to the logits of every green token before
softmax, nudging the model to prefer green tokens.

Detection is statistical: over a sequence of T tokens, the fraction of green
tokens follows a known null distribution (H₀: no watermark → Binomial(T, 0.5))
which is well-approximated by N(0.5, 0.25/T) for large T.  A z-score

    z = (|{green tokens}| / T - 0.5) / sqrt(0.25 / T)

exceeds a threshold (e.g. z > 4) with negligible false-positive probability.

Constraints (from CLAUDE.md)
-----------------------------
* No HF / torch imports — uses only NumPy and stdlib (hashlib).
* Zero overhead when no watermark is requested (no object created).
* No silent failures — every ValueError is surfaced immediately.
"""
from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Green-list seeding
# ---------------------------------------------------------------------------

def _seed_from_token(token_id: int) -> int:
    """Map a single previous token ID to a 64-bit integer seed.

    Uses SHA-256 over a packed 4-byte token id so collisions are negligible
    even for small vocabularies.
    """
    raw = hashlib.sha256(struct.pack("<I", token_id & 0xFFFF_FFFF)).digest()
    # Take the first 8 bytes as a little-endian uint64
    return struct.unpack("<Q", raw[:8])[0]


def _seed_from_context(context_ids: Sequence[int], window: int) -> int:
    """Hash the last *window* token IDs to a 64-bit seed.

    Provides richer context sensitivity at slightly higher cost.
    """
    tail = list(context_ids[-window:]) if context_ids else [0]
    packed = struct.pack(f"<{len(tail)}I", *[t & 0xFFFF_FFFF for t in tail])
    raw = hashlib.sha256(packed).digest()
    return struct.unpack("<Q", raw[:8])[0]


def _make_green_list(vocab_size: int, seed: int, green_fraction: float) -> np.ndarray:
    """Return a boolean mask of length *vocab_size* with True = green token.

    The mask is reproducible for the same (vocab_size, seed, green_fraction)
    triple.

    Args:
        vocab_size:      Total vocabulary size.
        seed:            64-bit integer seed.
        green_fraction:  Fraction of vocab in the green list (default 0.5).

    Returns:
        Boolean ndarray of shape (vocab_size,).
    """
    if vocab_size < 2:
        raise ValueError("vocab_size must be >= 2")
    if not (0.0 < green_fraction < 1.0):
        raise ValueError("green_fraction must be in (0, 1)")
    n_green = max(1, int(math.floor(vocab_size * green_fraction)))
    rng = np.random.default_rng(seed % (2**63))  # default_rng accepts uint63
    perm = rng.permutation(vocab_size)
    mask = np.zeros(vocab_size, dtype=bool)
    mask[perm[:n_green]] = True
    return mask


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

class WatermarkLogitsProcessor:
    """Bias logits toward hash-seeded green tokens at each decoding step.

    This is a *logits processor* (not a sampler) — it modifies the raw logit
    array before softmax, so it is agnostic to temperature and top-k/top-p.

    Args:
        vocab_size:      Vocabulary size — must match the model's vocab.
        delta:           Additive bias applied to green-list logits (default 2.0).
                         Larger δ → stronger watermark signal but harder for
                         the model to use red tokens when necessary.
        green_fraction:  Fraction of vocab placed in the green list each step.
                         Default 0.5 (half green, half red).
        context_window:  Number of preceding tokens used as the hash seed.
                         1 (default) → hash only the immediately preceding
                         token (fastest, Kirchenbauer original scheme).
                         >1 → hash the last *context_window* tokens (richer
                         context sensitivity).
        seeding_scheme:  'single' (default, hash prev token) or 'context'
                         (hash last *context_window* tokens).

    Example::

        from kairu.mock_model import MockModel
        from kairu.watermark import WatermarkLogitsProcessor
        import numpy as np

        proc = WatermarkLogitsProcessor(vocab_size=1000, delta=2.0)
        model = MockModel()
        logits = model.next_token_logits([1, 2, 3])
        biased = proc.process(logits, context_ids=[1, 2, 3])
    """

    VALID_SCHEMES = frozenset({"single", "context"})

    def __init__(
        self,
        vocab_size: int,
        delta: float = 2.0,
        green_fraction: float = 0.5,
        context_window: int = 1,
        seeding_scheme: str = "single",
    ) -> None:
        if vocab_size < 2:
            raise ValueError("vocab_size must be >= 2")
        if delta <= 0.0:
            raise ValueError("delta must be > 0")
        if not (0.0 < green_fraction < 1.0):
            raise ValueError("green_fraction must be in (0, 1)")
        if context_window < 1:
            raise ValueError("context_window must be >= 1")
        if seeding_scheme not in self.VALID_SCHEMES:
            raise ValueError(
                f"seeding_scheme must be one of {sorted(self.VALID_SCHEMES)}"
            )

        self.vocab_size = vocab_size
        self.delta = delta
        self.green_fraction = green_fraction
        self.context_window = context_window
        self.seeding_scheme = seeding_scheme

    def _compute_seed(self, context_ids: Sequence[int]) -> int:
        """Derive the integer seed from the current context."""
        if not context_ids:
            # No preceding token: use a fixed seed (handles first token edge case).
            return 0
        if self.seeding_scheme == "single":
            return _seed_from_token(context_ids[-1])
        # "context" scheme
        return _seed_from_context(context_ids, self.context_window)

    def green_list(self, context_ids: Sequence[int]) -> np.ndarray:
        """Return the boolean green-list mask for the given context.

        Deterministic: same context always yields the same mask.
        """
        seed = self._compute_seed(context_ids)
        return _make_green_list(self.vocab_size, seed, self.green_fraction)

    def process(
        self,
        logits: np.ndarray,
        context_ids: Sequence[int],
    ) -> np.ndarray:
        """Return a *copy* of *logits* with δ added to all green-list entries.

        Args:
            logits:      Raw logit array, shape (vocab_size,), float32 or float64.
            context_ids: Sequence of preceding token IDs (prompt + generated so far).

        Returns:
            Biased logit array of the same dtype, shape (vocab_size,).

        Raises:
            ValueError: if logits.shape[0] != self.vocab_size.
        """
        if logits.ndim != 1 or logits.shape[0] != self.vocab_size:
            raise ValueError(
                f"logits shape {logits.shape} incompatible with "
                f"vocab_size={self.vocab_size}"
            )
        green_mask = self.green_list(context_ids)
        biased = logits.copy()
        biased[green_mask] += self.delta
        return biased


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WatermarkResult:
    """Detection outcome for a single token sequence."""

    num_tokens: int
    """Total tokens analysed (T)."""

    num_green: int
    """Number of tokens that fall in their context-specific green list."""

    green_fraction: float
    """num_green / num_tokens."""

    z_score: float
    """Standard-normal test statistic (higher → more likely watermarked)."""

    p_value: float
    """One-sided p-value under H₀: not watermarked (P(Z ≥ z_score))."""

    decision: bool
    """True if z_score >= threshold (watermark detected)."""

    threshold: float
    """z-score threshold used for the decision."""


class WatermarkDetector:
    """Detect Kirchenbauer-style watermarks in a token sequence.

    The detector reconstructs the same green/red list the processor would have
    used at each step (using the identical seeding scheme and parameters) and
    counts how many generated tokens landed in the green list.  A z-score is
    computed against the null hypothesis (no watermark → ~50 % green).

    Args:
        vocab_size:      Must match the processor's vocab_size.
        green_fraction:  Must match the processor's green_fraction.
        context_window:  Must match the processor's context_window.
        seeding_scheme:  Must match the processor's seeding_scheme.
        z_threshold:     z-score above which the watermark is declared
                         present (default 4.0 → ~3×10⁻⁵ false-positive rate).

    Example::

        detector = WatermarkDetector(vocab_size=1000)
        result = detector.detect(token_ids=[1, 2, 3, ...], prefix_ids=[0])
        print(result.decision, result.z_score)
    """

    def __init__(
        self,
        vocab_size: int,
        green_fraction: float = 0.5,
        context_window: int = 1,
        seeding_scheme: str = "single",
        z_threshold: float = 4.0,
    ) -> None:
        if z_threshold <= 0.0:
            raise ValueError("z_threshold must be > 0")
        # Reuse the processor's seeding infrastructure.
        self._processor = WatermarkLogitsProcessor(
            vocab_size=vocab_size,
            delta=1.0,  # delta irrelevant for detection
            green_fraction=green_fraction,
            context_window=context_window,
            seeding_scheme=seeding_scheme,
        )
        self.z_threshold = z_threshold

    def detect(
        self,
        token_ids: Sequence[int],
        prefix_ids: Optional[Sequence[int]] = None,
    ) -> WatermarkResult:
        """Analyse *token_ids* and return a :class:`WatermarkResult`.

        Args:
            token_ids:  The generated token sequence to test (not including
                        the prompt).
            prefix_ids: The prompt token IDs that preceded the sequence.  If
                        None, the context for the first generated token is
                        assumed empty (seed = 0).

        Returns:
            :class:`WatermarkResult`.

        Raises:
            ValueError: if token_ids is empty.
        """
        if len(token_ids) == 0:
            raise ValueError("token_ids must be non-empty")

        prefix: list[int] = list(prefix_ids) if prefix_ids else []
        context: list[int] = list(prefix)
        n_green = 0

        for tok in token_ids:
            green_mask = self._processor.green_list(context)
            if green_mask[tok]:
                n_green += 1
            context.append(tok)

        T = len(token_ids)
        green_frac = n_green / T
        # Null distribution: Binomial(T, 0.5) ≈ N(T*0.5, T*0.25)
        # z = (n_green - T*0.5) / sqrt(T * 0.25)
        expected = T * 0.5
        std = math.sqrt(T * 0.25)
        z = (n_green - expected) / std

        # One-sided p-value P(Z >= z) using a rational approximation of erfc.
        p_value = _norm_sf(z)

        return WatermarkResult(
            num_tokens=T,
            num_green=n_green,
            green_fraction=green_frac,
            z_score=z,
            p_value=p_value,
            decision=z >= self.z_threshold,
            threshold=self.z_threshold,
        )


# ---------------------------------------------------------------------------
# Normal survival function (no scipy)
# ---------------------------------------------------------------------------

def _norm_sf(z: float) -> float:
    """Compute P(Z >= z) for Z ~ N(0,1) using the complementary error function.

    Uses Python's stdlib math.erfc which is accurate to machine precision.
    For z → ±∞ the result is clamped to [0, 1].
    """
    # P(Z >= z) = 0.5 * erfc(z / sqrt(2))
    val = 0.5 * math.erfc(z / math.sqrt(2.0))
    return max(0.0, min(1.0, val))


__all__ = [
    "WatermarkLogitsProcessor",
    "WatermarkDetector",
    "WatermarkResult",
]
