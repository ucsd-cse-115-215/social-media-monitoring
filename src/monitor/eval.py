"""Evaluate the LLM filter against the gold set.

Usage:
    python -m monitor.eval                        # defaults
    python -m monitor.eval --model gpt-4o         # try a different model
    python -m monitor.eval --threshold 0.5        # lower confidence threshold
"""

import asyncio
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import openai
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from monitor.filters.base import Post
from monitor.filters.llm import LLMFilter

console = Console()

DEFAULT_GOLD_PATH = Path("data/gold.jsonl")


def load_gold(gold_path: Path) -> list[dict]:
    entries = []
    with open(gold_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


async def evaluate(
    gold_path: Path,
    model: str,
    threshold: float,
) -> None:
    load_dotenv()
    client = openai.AsyncOpenAI()
    llm = LLMFilter(client, model=model, confidence_threshold=threshold)

    entries = load_gold(gold_path)
    if not entries:
        console.print("[red]Gold set is empty.[/]")
        return

    console.print(
        f"[bold]Evaluating LLM filter[/]: model={model}, threshold={threshold}"
    )
    console.print(f"Gold set: {len(entries)} entries from {gold_path}\n")

    tp = fp = tn = fn = 0
    false_positives: list[tuple[dict, dict]] = []
    false_negatives: list[tuple[dict, dict]] = []

    for i, entry in enumerate(entries, 1):
        post = Post(
            text=entry["text"],
            author_did="",
            created_at=datetime.now(timezone.utc),
            url=entry.get("url"),
        )

        predicted = await llm.matches(post)
        actual = entry["is_poetry"]
        analysis = post.metadata.get("llm_analysis", {})

        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
            false_positives.append((entry, analysis))
        elif not predicted and actual:
            fn += 1
            false_negatives.append((entry, analysis))
        else:
            tn += 1

        # Progress
        total = tp + fp + tn + fn
        console.print(f"  [{total}/{len(entries)}] evaluated...", end="\r")

    console.print()

    # --- Metrics ---
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    metrics = Table(title="Results", border_style="bold")
    metrics.add_column("Metric", style="cyan")
    metrics.add_column("Value", justify="right")
    metrics.add_row("True Positives", str(tp))
    metrics.add_row("False Positives", str(fp))
    metrics.add_row("True Negatives", str(tn))
    metrics.add_row("False Negatives", str(fn))
    metrics.add_row("Precision", f"{precision:.1%}")
    metrics.add_row("Recall", f"{recall:.1%}")
    metrics.add_row("F1", f"{f1:.1%}")
    console.print(metrics)

    # --- Disagreements ---
    if false_positives:
        console.print(f"\n[bold red]False Positives[/] ({len(false_positives)}):")
        console.print("[dim]LLM said poetry, human said not[/]\n")
        for entry, analysis in false_positives:
            _show_disagreement(entry, analysis)

    if false_negatives:
        console.print(f"\n[bold yellow]False Negatives[/] ({len(false_negatives)}):")
        console.print("[dim]LLM said not poetry, human said poetry[/]\n")
        for entry, analysis in false_negatives:
            _show_disagreement(entry, analysis)


def _show_disagreement(entry: dict, analysis: dict) -> None:
    explanation = analysis.get("explanation", "no explanation")
    confidence = analysis.get("confidence")
    subtitle = f"{explanation}"
    if confidence is not None:
        subtitle += f" | confidence: {confidence:.0%}"

    panel = Panel(
        Text(entry["text"].strip()),
        subtitle=f"[dim]{subtitle}[/]",
        border_style="dim",
        padding=(0, 2),
    )
    console.print(panel)
    if entry.get("url"):
        console.print(f"  [dim]{entry['url']}[/]")
    console.print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM filter against gold set")
    parser.add_argument(
        "--gold",
        type=Path,
        default=DEFAULT_GOLD_PATH,
        help=f"Path to gold set (default: {DEFAULT_GOLD_PATH})",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to evaluate (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Confidence threshold (default: 0.7)",
    )
    args = parser.parse_args()

    if not args.gold.exists():
        console.print(f"[red]Gold set not found: {args.gold}[/]")
        console.print("Run the monitor with --log-dir, then label with: python -m monitor.label")
        raise SystemExit(1)

    asyncio.run(evaluate(args.gold, args.model, args.threshold))


if __name__ == "__main__":
    main()
