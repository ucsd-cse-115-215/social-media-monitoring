"""Rich console output for matched posts."""

import asyncio

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from monitor.filters.base import Post
from monitor.pipeline import Pipeline

console = Console()

# Live status line for rejected-post counter
_live: Live | None = None


def start_status() -> None:
    """Start the live status line."""
    global _live
    _live = Live("", console=console, refresh_per_second=4)
    _live.start()


def stop_status() -> None:
    """Stop the live status line."""
    global _live
    if _live:
        _live.stop()
        _live = None


def _update_status(pipeline: Pipeline, queue: asyncio.Queue | None = None) -> None:
    if _live is None:
        return
    s = pipeline.get_summary()
    parts = [f"[dim]{s['total_seen']} seen"]
    for name, st in s["stages"].items():
        if st["rejected"]:
            parts.append(f"{name}: -{st['rejected']}")
    parts.append(f"({s['posts_per_second']}/s)")
    if queue is not None:
        parts.append(f"queue: {queue.qsize()}")
    _live.update(" | ".join(parts) + "[/]")


def display_post(post: Post, pipeline: Pipeline | None = None, queue: asyncio.Queue | None = None) -> None:
    """Display a matched post with its LLM analysis."""
    # Temporarily stop live display so the panel prints cleanly
    if _live:
        _live.stop()

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

    # Restart live display
    if _live:
        _live.start(refresh=True)


def display_rejected(pipeline: Pipeline, queue: asyncio.Queue | None = None) -> None:
    """Update the live status line with current pipeline stats."""
    _update_status(pipeline, queue)


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
