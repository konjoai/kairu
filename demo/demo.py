"""Kairu — Demo Day script.

Exercises the live library against deterministic mock models so the demo runs
in <2s without a single byte of model weights. Run from the repo root::

    python -m demo.demo

Sections
--------
1. AutoProfile.recommend() over three model archetypes
2. LogitsCache LRU eviction trace
3. DynamicGammaScheduler over a 20-step accept/reject sequence
4. SpeculativeDecoder driven by mock draft + target — accepted tokens & speedup
5. LayerwiseEarlyExitDecoder — per-token exit layer and compute saved
"""
from __future__ import annotations

import time

import numpy as np
from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from kairu import (
    AutoProfile,
    DynamicGammaScheduler,
    LayerwiseEarlyExitDecoder,
    LogitsCache,
    MockLayeredModel,
)
from kairu.mock_model import MockModel
from kairu.speculative import SpeculativeDecoder


console = Console()


# ────────────────────────────────────────────────────────────── helpers

CYAN = "bold cyan"
GOLD = "bold yellow"
GREEN = "bold green"
RED = "bold red"
DIM = "dim"


def banner() -> None:
    title = Text("流  K A I R U  —  D E M O   D A Y", justify="center")
    title.stylize("bold magenta")
    sub = Text("Adaptive inference optimization · live mock models · zero weights",
               justify="center", style="cyan")
    console.print(Panel(Group(title, sub), border_style="magenta", padding=(1, 2)))


def section(title: str, n: int) -> None:
    console.print()
    console.print(Rule(f"[bold cyan]§{n} · {title}", style="cyan"))


# ────────────────────────────────────────────────────────────── 1. AutoProfile

def demo_auto_profile() -> None:
    section("AutoProfile — strategy recommendation", 1)

    class BigVocab(MockModel):
        @property
        def vocab_size(self) -> int:
            return 128_000

    cases = [
        (MockModel(), "tinyllama-125m", False, "tiny / vanilla"),
        (MockLayeredModel(num_layers=24), "kairu-mid", False, "layered architecture"),
        (BigVocab(), "llama-3-70b", True, "frontier + draft model"),
    ]

    table = Table(box=box.ROUNDED, show_lines=True, header_style="bold cyan")
    table.add_column("Scenario", style="bold")
    table.add_column("Strategy", style=GREEN)
    table.add_column("γ", justify="right", style=GOLD)
    table.add_column("Threshold", justify="right")
    table.add_column("Cache", justify="right")
    table.add_column("Rationale", style=DIM)

    for model, hint, has_draft, label in cases:
        p = AutoProfile.recommend(model, name_hint=hint, has_draft=has_draft)
        table.add_row(
            f"{label}\n[dim]{hint}[/dim]",
            p.strategy,
            str(p.gamma),
            f"{p.early_exit_threshold:.2f}",
            str(p.cache_capacity),
            p.rationale,
        )
    console.print(table)


# ────────────────────────────────────────────────────────────── 2. LogitsCache

def demo_logits_cache() -> None:
    section("LogitsCache — LRU eviction trace", 2)
    cap = 6
    cache = LogitsCache(capacity=cap)

    log = Table(box=box.SIMPLE, header_style="bold cyan")
    log.add_column("step", justify="right", style=DIM)
    log.add_column("op", style="bold")
    log.add_column("key")
    log.add_column("size", justify="right")
    log.add_column("hits", justify="right", style=GREEN)
    log.add_column("misses", justify="right", style=RED)
    log.add_column("evict", justify="right", style=GOLD)

    rng = np.random.default_rng(7)
    plan = [("put", k) for k in range(10)] + [
        ("get", 0), ("get", 1), ("get", 8), ("get", 9), ("get", 99),
        ("put", 10), ("put", 11),
    ]
    for i, (op, k) in enumerate(plan, 1):
        if op == "put":
            cache.put((k,), rng.standard_normal(4).astype(np.float32))
        else:
            cache.get((k,))
        s = cache.stats()
        log.add_row(
            str(i), op, f"({k},)", f"{s['size']}/{cap}",
            str(s["hits"]), str(s["misses"]), str(s["evictions"]),
        )

    console.print(log)
    s = cache.stats()
    console.print(Panel(
        f"[bold]hit_rate[/] = [green]{s['hit_rate']:.1%}[/]   "
        f"[bold]evictions[/] = [yellow]{s['evictions']}[/]   "
        f"[bold]final size[/] = [cyan]{s['size']}/{cap}[/]\n"
        f"[dim]Insert 10 keys into a 6-slot cache → 4 evictions. "
        f"Touching key 0 right after insertion 9 promotes it to MRU; "
        f"subsequent puts evict the next-LRU instead.[/]",
        title="Cache summary", border_style="cyan",
    ))


# ────────────────────────────────────────────────────────────── 3. DynamicGammaScheduler

def demo_gamma_scheduler() -> None:
    section("DynamicGammaScheduler — AIMD over γ", 3)
    sched = DynamicGammaScheduler(
        initial=2, min_gamma=1, max_gamma=8,
        high_threshold=0.7, low_threshold=0.3,
        increase=1, decrease_factor=0.5, window=4,
    )

    # 20-step trace: 7 high, 7 low, 6 mid — exercise grow/shrink/hold
    regimes = ["HIGH"] * 7 + ["LOW"] * 7 + ["MID"] * 6

    table = Table(box=box.SIMPLE, header_style="bold cyan")
    table.add_column("step", justify="right", style=DIM)
    table.add_column("regime")
    table.add_column("γ used", justify="right")
    table.add_column("accepted", justify="right")
    table.add_column("rate", justify="right")
    table.add_column("rolling", justify="right", style=GOLD)
    table.add_column("γ next", justify="right", style=GREEN)
    table.add_column("bar")

    for i, regime in enumerate(regimes, 1):
        gamma_used = sched.gamma
        if regime == "HIGH":
            accepted = gamma_used
        elif regime == "LOW":
            accepted = 0
        else:
            accepted = max(1, gamma_used // 2)
        new_gamma = sched.update(accepted, gamma_used)
        rate = accepted / gamma_used
        rolling = sched.rolling_rate()
        regime_color = {"HIGH": GREEN, "LOW": RED, "MID": "yellow"}[regime]
        bar = "█" * gamma_used + "░" * (sched._max - gamma_used)
        arrow = "↑" if new_gamma > gamma_used else ("↓" if new_gamma < gamma_used else "·")
        table.add_row(
            str(i),
            f"[{regime_color}]{regime}[/]",
            str(gamma_used),
            str(accepted),
            f"{rate:.2f}",
            f"{rolling:.2f}",
            f"{new_gamma} {arrow}",
            f"[cyan]{bar}[/]",
        )

    console.print(table)
    s = sched.stats()
    console.print(Panel(
        f"[bold]final γ[/] = [green]{s['gamma']}[/]   "
        f"[bold]adjustments[/] = [yellow]{s['adjustments']}[/]   "
        f"[bold]rolling acceptance[/] = [cyan]{s['rolling_acceptance']:.2f}[/]\n"
        f"[dim]Pattern: γ grows from 2 → 8 (clamped) under HIGH, then halves "
        f"each LOW round, then holds in the MID band. Same control law as TCP.[/]",
        title="Scheduler summary", border_style="cyan",
    ))


# ────────────────────────────────────────────────────────────── 4. SpeculativeDecoder

def demo_speculative() -> None:
    section("SpeculativeDecoder — accepted tokens + speedup", 4)
    target = MockModel()
    draft = MockModel()
    gamma = 4
    decoder = SpeculativeDecoder(target, draft, gamma=gamma, temperature=1.0)

    prompt = [10, 20, 30]
    table = Table(box=box.SIMPLE, header_style="bold cyan")
    table.add_column("call", justify="right", style=DIM)
    table.add_column("max_new")
    table.add_column("generated", style="cyan")
    table.add_column("accepted", style=GREEN, justify="right")
    table.add_column("rejected", style=RED, justify="right")
    table.add_column("acc rate", justify="right", style=GOLD)
    table.add_column("E[speedup]", justify="right", style="bold magenta")

    total_accepted = total_attempted = 0
    for call in range(1, 6):
        t0 = time.perf_counter()
        toks, stats = decoder.generate(prompt, max_new_tokens=8)
        dt = (time.perf_counter() - t0) * 1000
        rho = stats["acceptance_rate"]
        # Theoretical speedup from Leviathan et al. 2023
        speedup = (1 - rho ** (gamma + 1)) / (1 - rho) if rho < 1 else float(gamma + 1)
        total_accepted += stats["accepted_tokens"]
        total_attempted += stats["accepted_tokens"] + stats["rejected_tokens"]
        preview = " ".join(str(t) for t in toks[:6]) + (" …" if len(toks) > 6 else "")
        table.add_row(
            str(call), "8", preview,
            str(stats["accepted_tokens"]),
            str(stats["rejected_tokens"]),
            f"{rho:.2f}",
            f"{speedup:.2f}×",
        )

    console.print(table)
    overall_rho = total_accepted / total_attempted if total_attempted else 0.0
    overall_speedup = (
        (1 - overall_rho ** (gamma + 1)) / (1 - overall_rho)
        if overall_rho < 1 else float(gamma + 1)
    )
    console.print(Panel(
        f"[bold]overall acceptance[/] = [yellow]{overall_rho:.2f}[/]   "
        f"[bold]theoretical speedup[/] = [magenta]{overall_speedup:.2f}×[/] "
        f"[dim](γ={gamma}, formula: E[T] = (1-ρ^(γ+1))/(1-ρ))[/]",
        title="Speculative summary", border_style="cyan",
    ))


# ────────────────────────────────────────────────────────────── 5. LayerwiseEarlyExitDecoder

def demo_layerwise_exit() -> None:
    section("LayerwiseEarlyExitDecoder — per-token exit layer", 5)
    L = 12
    model = MockLayeredModel(num_layers=L)
    decoder = LayerwiseEarlyExitDecoder(model, confidence_threshold=0.4, min_layer=1, temperature=0.1)
    _, stats = decoder.generate([1, 2, 3], max_new_tokens=10)

    table = Table(box=box.SIMPLE, header_style="bold cyan")
    table.add_column("token", justify="right", style=DIM)
    table.add_column("exit layer", justify="right", style=GREEN)
    table.add_column("layers used", justify="right")
    table.add_column("layers skipped", justify="right", style=GOLD)
    table.add_column("layer bar")

    for i, layer in enumerate(stats["exit_layers"], 1):
        used = layer
        skipped = L - layer
        bar = "[green]" + "█" * used + "[/]" + "[dim]" + "░" * skipped + "[/]"
        table.add_row(str(i), str(layer), str(used), str(skipped), bar)

    console.print(table)
    saved = stats["compute_saved"]
    console.print(Panel(
        f"[bold]mean exit layer[/] = [cyan]{stats['mean_exit_layer']:.2f}[/] / {L}   "
        f"[bold]compute saved[/] = [green]{saved:.1%}[/]\n"
        f"[dim]compute_saved = 1 − mean_exit_layer / num_layers   "
        f"(uniform per-layer cost assumption holds for transformer decoder stacks)[/]",
        title="Layerwise summary", border_style="cyan",
    ))


# ────────────────────────────────────────────────────────────── main

def main() -> None:
    banner()
    demo_auto_profile()
    demo_logits_cache()
    demo_gamma_scheduler()
    demo_speculative()
    demo_layerwise_exit()
    console.print()
    console.print(Align.center(
        Panel(
            "[bold magenta]ቆንጆ · 根性 · 康宙 · कोहजो · ᨀᨚᨐᨚ · конйо · 건조 · কুঞ্জ[/]\n"
            "[cyan]Build, ship, repeat.[/]",
            border_style="magenta", padding=(1, 4),
        )
    ))


if __name__ == "__main__":
    main()
