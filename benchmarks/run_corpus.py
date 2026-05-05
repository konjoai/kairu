#!/usr/bin/env python3
"""Real-workload benchmark harness for kairu inference.

Runs a fixed 100-prompt corpus against one or more model backends and
publishes results JSON to ``benchmarks/results/``.

Design
------
* Corpus is embedded (no external file required) — 100 short prompts
  representative of real LLM workloads: instruction-following, Q&A,
  summarisation, coding, and free-form generation.
* When ``--model mock`` is passed (the default) the harness runs fully
  offline using :class:`kairu.MockModel` — zero ML dependencies.
* When ``--model tiny-gpt2`` or ``--model gpt2`` is passed it defers to
  :class:`kairu._hf_backend.HuggingFaceModel` (requires ``kairu[hf]``
  and a live internet connection on first run to pull weights from HF Hub).
* Results are saved as timestamped JSON under ``benchmarks/results/`` and
  NEVER overwrite an existing file (:meth:`BenchmarkResult.save`).
* Summary statistics (p50/p95/p99/mean/tok·s⁻¹) are printed to stdout in
  a human-readable table.

Usage::

    # Offline smoke-test (MockModel, no ML deps)
    python benchmarks/run_corpus.py --model mock --tokens 50 --warmup 3 --runs 10

    # Real GPT-2 (requires: pip install 'kairu[hf]')
    python benchmarks/run_corpus.py --model gpt2 --tokens 100 --warmup 5 --runs 20

    # tiny-gpt2 (the fastest real model, ~117 M params)
    python benchmarks/run_corpus.py --model sshleifer/tiny-gpt2 --tokens 64 --runs 20

    # Custom output directory
    python benchmarks/run_corpus.py --model mock --output /tmp/bench_results
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# Ensure the repo root is on sys.path so the script is importable from any CWD.
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kairu.bench import BenchmarkResult, BenchmarkRunner, _collect_hardware, _percentile
from kairu.base import ModelInterface
from kairu.mock_model import MockModel
from kairu.streaming import StreamingDecoder


# ---------------------------------------------------------------------------
# Fixed 100-prompt corpus
# ---------------------------------------------------------------------------

CORPUS: List[str] = [
    # Instruction following (20)
    "Explain the difference between supervised and unsupervised learning.",
    "Write a haiku about artificial intelligence.",
    "Summarise the French Revolution in three sentences.",
    "Translate 'Good morning, how are you?' into Spanish.",
    "List five benefits of daily exercise.",
    "Describe how photosynthesis works.",
    "What are the main causes of climate change?",
    "Explain Newton's three laws of motion.",
    "Write a one-sentence definition of quantum entanglement.",
    "What is the Pythagorean theorem?",
    "Describe the water cycle.",
    "What is the difference between RAM and ROM?",
    "Explain what a neural network is.",
    "What are the main programming paradigms?",
    "Write a brief biography of Alan Turing.",
    "Explain the concept of recursion in programming.",
    "What is the Big Bang theory?",
    "Describe what DNA is and its role in genetics.",
    "Explain the difference between HTTP and HTTPS.",
    "What is machine learning?",
    # Question & Answer (20)
    "What is the capital of France?",
    "Who wrote 'Pride and Prejudice'?",
    "What year did World War II end?",
    "How many planets are in the solar system?",
    "What is the speed of light?",
    "Who invented the telephone?",
    "What is the largest ocean on Earth?",
    "What language is most widely spoken in the world?",
    "What is the chemical formula for water?",
    "Who painted the Mona Lisa?",
    "What is the boiling point of water in Celsius?",
    "How many bones are in the human body?",
    "What is the tallest mountain in the world?",
    "What currency does Japan use?",
    "What is the powerhouse of the cell?",
    "Who wrote the Iliad?",
    "What is the largest country by area?",
    "How many continents are there?",
    "What is the atomic number of carbon?",
    "What is the speed of sound in air at sea level?",
    # Coding tasks (20)
    "Write a Python function to reverse a string.",
    "Write a Python one-liner to flatten a list of lists.",
    "What does the Python 'yield' keyword do?",
    "Write a SQL query to select all rows from a table called 'users'.",
    "Explain the difference between '==' and 'is' in Python.",
    "Write a bash command to count lines in a file.",
    "What is a REST API?",
    "Write a Python function to compute the Fibonacci sequence.",
    "What is the difference between a list and a tuple in Python?",
    "Explain what a hash map is.",
    "Write a Python decorator that logs function calls.",
    "What does O(n log n) mean in complexity analysis?",
    "Explain the difference between TCP and UDP.",
    "Write a Python context manager for timing code.",
    "What is a closure in programming?",
    "Explain what Git branching is.",
    "What is a Docker container?",
    "Write a Python generator that yields prime numbers.",
    "Explain dependency injection.",
    "What is the difference between a process and a thread?",
    # Summarisation / comprehension (20)
    "In one sentence, what is the theory of relativity?",
    "What are the key ideas in Adam Smith's 'The Wealth of Nations'?",
    "Briefly describe the plot of Romeo and Juliet.",
    "What is the main idea behind Agile software development?",
    "Summarise the concept of supply and demand.",
    "What is blockchain technology?",
    "Describe the purpose of the United Nations.",
    "What is the scientific method?",
    "What are the main themes of George Orwell's '1984'?",
    "What is DevOps?",
    "Briefly explain what the Turing Test is.",
    "What is cloud computing?",
    "Summarise the life of Albert Einstein.",
    "What are microservices in software architecture?",
    "What is the open-source software movement?",
    "Explain what cryptocurrency is.",
    "Briefly describe the Renaissance period.",
    "What is functional programming?",
    "What are the main types of machine learning?",
    "Explain what containerisation is in software development.",
    # Free-form / creative (20)
    "Write a short poem about the ocean.",
    "Describe a futuristic city in two sentences.",
    "What would a world without the internet look like?",
    "Write a motivational quote about perseverance.",
    "Describe the smell of rain.",
    "What is the most important invention of the 20th century?",
    "Write a short riddle.",
    "Imagine you are an astronaut. Describe what you see.",
    "What does 'the ends justify the means' mean?",
    "Write a short story opening set in ancient Rome.",
    "What would you tell your younger self?",
    "Describe the feeling of learning something new.",
    "What makes a great leader?",
    "Describe a perfect day.",
    "What is the meaning of life according to philosophy?",
    "Write a fortune cookie message.",
    "What would you do if you had one hour left?",
    "Describe the experience of reading a great book.",
    "What is the most beautiful thing about science?",
    "Imagine meeting an alien. What is the first thing you ask?",
]

assert len(CORPUS) == 100, f"Corpus must have exactly 100 prompts, got {len(CORPUS)}"


# ---------------------------------------------------------------------------
# Per-prompt latency measurement
# ---------------------------------------------------------------------------

def _run_prompt(
    decoder: StreamingDecoder,
    prompt_token_ids: List[int],
    num_tokens: int,
) -> float:
    """Time a single generation run.  Returns elapsed seconds."""
    t0 = time.perf_counter()
    for _ in decoder.stream(prompt_token_ids, max_new_tokens=num_tokens):
        pass
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Corpus harness
# ---------------------------------------------------------------------------

class CorpusBenchmarkRunner:
    """Run the full 100-prompt corpus and emit a :class:`BenchmarkResult`.

    Args:
        model:      Any :class:`ModelInterface`.  Pass :class:`MockModel` for
                    offline execution.
        name:       Label embedded in the output filename and JSON.
        tokenizer:  Optional callable ``str → List[int]``.  Defaults to a
                    whitespace tokeniser (safe for MockModel; for HF models
                    pass the model's tokeniser).
    """

    def __init__(
        self,
        model: ModelInterface,
        name: str = "corpus",
        tokenizer=None,
    ) -> None:
        self._model = model
        self.name = name
        self._tok = tokenizer or self._default_tokenizer

    @staticmethod
    def _default_tokenizer(text: str) -> List[int]:
        """Whitespace tokeniser — maps each word to a stable hash mod vocab."""
        vocab = 1000
        return [(hash(w) & 0x7FFFFFFF) % vocab for w in text.split()] or [0]

    def run(
        self,
        num_tokens: int = 64,
        warmup: int = 5,
        prompts: Optional[List[str]] = None,
    ) -> BenchmarkResult:
        """Execute the corpus benchmark.

        Args:
            num_tokens: Tokens to generate per prompt.
            warmup:     Warmup runs (over the first N prompts) to discard.
            prompts:    Override the default 100-prompt corpus.

        Returns:
            :class:`BenchmarkResult` with per-prompt latency statistics.
        """
        corpus = prompts if prompts is not None else CORPUS
        decoder = StreamingDecoder(self._model, temperature=1.0)

        # Warmup phase — run over a slice of the corpus, discard timings.
        warmup_slice = corpus[:warmup] if warmup > 0 else []
        for prompt in warmup_slice:
            ids = self._tok(prompt)
            _run_prompt(decoder, ids, num_tokens)

        # Measured phase.
        latencies: List[float] = []
        for prompt in corpus:
            ids = self._tok(prompt)
            elapsed = _run_prompt(decoder, ids, num_tokens)
            latencies.append(elapsed)

        # Statistics (pure stdlib — no scipy).
        sorted_lat = sorted(latencies)
        p50 = _percentile(sorted_lat, 50.0)
        p95 = _percentile(sorted_lat, 95.0)
        p99 = _percentile(sorted_lat, 99.0)
        mean = statistics.mean(latencies) if latencies else 0.0
        stddev = statistics.stdev(latencies) if len(latencies) >= 2 else 0.0
        tok_s_mean = (num_tokens / mean) if mean > 0 else 0.0

        return BenchmarkResult(
            name=self.name,
            model_name=type(self._model).__name__,
            num_tokens=num_tokens,
            num_runs=len(corpus),
            warmup=warmup,
            latencies_s=latencies,
            p50=p50,
            p95=p95,
            p99=p99,
            mean=mean,
            stddev=stddev,
            tokens_per_s_mean=tok_s_mean,
            hardware=_collect_hardware(),
            timestamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_model(model_arg: str) -> ModelInterface:
    if model_arg == "mock":
        return MockModel()
    try:
        from kairu._hf_backend import HuggingFaceModel  # type: ignore[import]
    except ImportError as exc:
        print(
            f"ERROR: HuggingFace backend not available — "
            f"install with: pip install 'kairu[hf]'\n  ({exc})",
            file=sys.stderr,
        )
        sys.exit(1)
    return HuggingFaceModel(model_arg)  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python benchmarks/run_corpus.py",
        description=(
            "Kairu 100-prompt corpus benchmark — measures p50/p95/p99 "
            "token-generation latency across a representative prompt set."
        ),
    )
    p.add_argument(
        "--model", default="mock",
        help="'mock' (offline) or a HuggingFace model id e.g. 'gpt2' (requires kairu[hf]).",
    )
    p.add_argument("--tokens", type=int, default=64, metavar="N",
                   help="Tokens to generate per prompt (default: 64).")
    p.add_argument("--warmup", type=int, default=5, metavar="N",
                   help="Warmup prompts to discard (default: 5).")
    p.add_argument("--runs", type=int, default=100, metavar="N",
                   help="Number of corpus prompts to benchmark (default: 100, full corpus).")
    p.add_argument("--name", default="corpus",
                   help="Label embedded in result filename and JSON.")
    p.add_argument("--output", default=None,
                   help="Override save directory (default: benchmarks/results).")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    model = _load_model(args.model)
    # Subset the corpus if fewer runs requested.
    corpus = CORPUS[: args.runs] if args.runs < len(CORPUS) else CORPUS

    runner = CorpusBenchmarkRunner(model, name=args.name)

    print(
        f"\nKairu Corpus Benchmark — model={args.model!r}  "
        f"tokens={args.tokens}  prompts={len(corpus)}  warmup={args.warmup}\n"
        f"{'─' * 60}"
    )

    result = runner.run(num_tokens=args.tokens, warmup=args.warmup, prompts=corpus)

    col_w = 22
    print(f"{'Metric':<{col_w}}  {'Value':>15}")
    print(f"{'─' * col_w}  {'─' * 15}")
    print(f"{'p50 latency (s)':<{col_w}}  {result.p50:>15.6f}")
    print(f"{'p95 latency (s)':<{col_w}}  {result.p95:>15.6f}")
    print(f"{'p99 latency (s)':<{col_w}}  {result.p99:>15.6f}")
    print(f"{'mean latency (s)':<{col_w}}  {result.mean:>15.6f}")
    print(f"{'stddev (s)':<{col_w}}  {result.stddev:>15.6f}")
    print(f"{'tokens/s (mean)':<{col_w}}  {result.tokens_per_s_mean:>15.2f}")
    print(f"{'─' * col_w}  {'─' * 15}")

    save_kwargs: dict = {}
    if args.output:
        save_kwargs["base_dir"] = args.output

    saved_path = result.save(**save_kwargs)
    print(f"\nResult saved → {saved_path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
