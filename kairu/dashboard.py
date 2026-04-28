"""
Live Rich dashboard for monitoring inference metrics in real time.
Uses Rich tables and Live display so metrics refresh while tokens stream.
"""
from __future__ import annotations

from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from kairu.metrics import GenerationMetrics

console = Console()


def render_metrics_table(metrics: GenerationMetrics) -> Table:
    """Build a Rich Table snapshot from a GenerationMetrics instance."""
    table = Table(show_header=True, header_style="bold cyan", box=None)
    table.add_column("Metric", style="dim", width=25)
    table.add_column("Value", justify="right")

    table.add_row("Prompt tokens", str(metrics.prompt_tokens))
    table.add_row("Generated tokens", str(metrics.generated_tokens))
    table.add_row("Total time (ms)", f"{metrics.total_time_ms:.1f}")
    table.add_row("Throughput (tok/s)", f"{metrics.tokens_per_second:.1f}")
    table.add_row("Mean latency (ms/tok)", f"{metrics.mean_latency_ms:.2f}")
    table.add_row("Acceptance rate", f"{metrics.acceptance_rate:.1%}")
    return table


class KairuDashboard:
    """
    Context manager that renders a live-updating metrics panel.

    Example::

        metrics = GenerationMetrics(prompt_tokens=16)
        with KairuDashboard() as dash:
            dash.attach(metrics)
            for tok in model.stream():
                metrics.record_token()
                dash.update()
        metrics.finish()
    """

    def __init__(self, refresh_per_second: int = 4):
        self._refresh = refresh_per_second
        self._metrics: GenerationMetrics | None = None
        self._live: Live | None = None

    def attach(self, metrics: GenerationMetrics) -> None:
        """Bind a GenerationMetrics instance to this dashboard."""
        self._metrics = metrics

    def update(self) -> None:
        """Push the latest metric snapshot to the live display."""
        if self._live is not None and self._metrics is not None:
            self._live.update(
                Panel(
                    render_metrics_table(self._metrics),
                    title="[bold green]Kairu[/bold green]",
                )
            )

    def __enter__(self) -> "KairuDashboard":
        self._live = Live(
            Panel(Text("Initializing..."), title="[bold green]Kairu[/bold green]"),
            refresh_per_second=self._refresh,
            console=console,
        )
        self._live.__enter__()
        return self

    def __exit__(self, *args) -> None:
        if self._live is not None:
            self.update()
            self._live.__exit__(*args)
