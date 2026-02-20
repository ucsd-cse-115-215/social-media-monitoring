"""Rich console output for matched posts."""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from monitor.filters.base import Post
from monitor.pipeline import Pipeline

console = Console()


def display_post(post: Post) -> None:
    """Display a matched post with its LLM analysis."""
    analysis = post.metadata.get("llm_analysis", {})

    # Build the panel content
    body = Text(post.text.strip())

    subtitle_parts = []
    if analysis.get("explanation"):
        subtitle_parts.append(analysis["explanation"])
    confidence = analysis.get("confidence")
    if confidence is not None:
        subtitle_parts.append(f"confidence: {confidence:.0%}")

    subtitle = " | ".join(subtitle_parts) if subtitle_parts else None

    panel = Panel(
        body,
        title="[bold magenta]poem found[/]",
        subtitle=f"[dim]{subtitle}[/]" if subtitle else None,
        border_style="magenta",
        padding=(1, 2),
    )
    console.print(panel)

    if post.url:
        console.print(f"  [dim link={post.url}]{post.url}[/]")
    console.print()


def display_stats(pipeline: Pipeline) -> None:
    """Display pipeline filtering statistics."""
    summary = pipeline.get_summary()

    table = Table(title="Pipeline Stats", border_style="dim")
    table.add_column("Stage", style="cyan")
    table.add_column("Passed", justify="right", style="green")
    table.add_column("Rejected", justify="right", style="red")
    table.add_column("Pass Rate", justify="right")

    for stage_name, stage_stats in summary["stages"].items():
        table.add_row(
            stage_name,
            str(stage_stats["passed"]),
            str(stage_stats["rejected"]),
            stage_stats["pass_rate"],
        )

    console.print()
    console.print(table)
    console.print(
        f"[dim]Total: {summary['total_seen']} posts in {summary['elapsed_seconds']}s "
        f"({summary['posts_per_second']} posts/sec)[/]"
    )
