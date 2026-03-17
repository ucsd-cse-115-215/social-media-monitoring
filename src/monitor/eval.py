"""Evaluate the LLM filter against the gold set.

Usage:
    python -m monitor.eval                        # defaults
    python -m monitor.eval --model gpt-4o         # try a different model
    python -m monitor.eval --threshold 0.5        # lower confidence threshold

LLM predictions are cached in data/eval_{model}.jsonl. If the file exists,
cached results are reused and only new gold-set entries are sent to the LLM.
Delete the file to force a full re-run.
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
COSTS_PATH = Path("data/model_costs.json")


def _results_path(model: str) -> Path:
    """Cache file for a given model's predictions."""
    return Path(f"data/eval_{model}.jsonl")


def load_costs() -> dict[str, dict]:
    if COSTS_PATH.exists():
        with open(COSTS_PATH) as f:
            data = json.load(f)
        data.pop("_comment", None)
        return data
    return {}


def load_jsonl(path: Path) -> list[dict]:
    entries = []
    if not path.exists():
        return entries
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_results(path: Path) -> dict[str, dict]:
    """Load cached results keyed by post text."""
    return {e["text"]: e for e in load_jsonl(path)}


async def run_llm_predictions(
    gold: list[dict],
    cached: dict[str, dict],
    model: str,
    results_path: Path,
) -> dict[str, dict]:
    """Call the LLM for any gold entries not already cached. Returns full results dict."""
    missing = [e for e in gold if e["text"] not in cached]
    if not missing:
        console.print(f"[dim]Using cached predictions from {results_path}[/]")
        return cached

    console.print(
        f"Running LLM on {len(missing)} new entries "
        f"({len(cached)} cached)..."
    )

    load_dotenv()
    client = openai.AsyncOpenAI()
    # Use threshold=0 so we always get the full analysis; threshold is applied later
    llm = LLMFilter(client, model=model, confidence_threshold=0)

    results_path.parent.mkdir(parents=True, exist_ok=True)

    for i, entry in enumerate(missing, 1):
        post = Post(
            text=entry["text"],
            author_did="",
            created_at=datetime.now(timezone.utc),
            url=entry.get("url"),
        )
        await llm.matches(post)

        result = {
            "text": entry["text"],
            "llm_analysis": post.metadata.get("llm_analysis", {}),
            "usage": post.metadata.get("usage", {}),
        }

        # Append to cache file incrementally
        with open(results_path, "a") as f:
            f.write(json.dumps(result) + "\n")

        cached[entry["text"]] = result
        console.print(f"  [{i}/{len(missing)}] evaluated...", end="\r")

    console.print()
    return cached


def compute_metrics(
    gold: list[dict],
    results: dict[str, dict],
    model: str,
    threshold: float,
) -> None:
    """Compute and display accuracy metrics and cost from cached results."""
    tp = fp = tn = fn = 0
    total_input_tokens = 0
    total_output_tokens = 0
    false_positives: list[tuple[dict, dict]] = []
    false_negatives: list[tuple[dict, dict]] = []

    for entry in gold:
        result = results.get(entry["text"])
        if result is None:
            continue

        analysis = result.get("llm_analysis", {})
        usage = result.get("usage", {})
        total_input_tokens += usage.get("input_tokens", 0)
        total_output_tokens += usage.get("output_tokens", 0)

        predicted = (
            analysis.get("is_poetry", False)
            and analysis.get("confidence", 0) >= threshold
        )
        actual = entry["is_poetry"]

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

    # --- Cost ---
    costs = load_costs()
    model_cost = costs.get(model)

    cost_table = Table(title="Cost", border_style="bold")
    cost_table.add_column("Metric", style="cyan")
    cost_table.add_column("Value", justify="right")
    cost_table.add_row("Input tokens", f"{total_input_tokens:,}")
    cost_table.add_row("Output tokens", f"{total_output_tokens:,}")

    if model_cost:
        input_cost = total_input_tokens / 1_000_000 * model_cost["input"]
        output_cost = total_output_tokens / 1_000_000 * model_cost["output"]
        total_cost = input_cost + output_cost
        n = max(tp + fp + tn + fn, 1)
        cost_table.add_row("Input cost", f"${input_cost:.4f}")
        cost_table.add_row("Output cost", f"${output_cost:.4f}")
        cost_table.add_row("Total cost", f"${total_cost:.4f}")
        cost_table.add_row("Cost per post", f"${total_cost / n:.6f}")
    else:
        cost_table.add_row("Cost", f"[yellow]no pricing for {model} in {COSTS_PATH}[/]")

    console.print(cost_table)

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


async def evaluate(gold_path: Path, model: str, threshold: float) -> None:
    gold = load_jsonl(gold_path)
    if not gold:
        console.print("[red]Gold set is empty.[/]")
        return

    console.print(
        f"[bold]Evaluating LLM filter[/]: model={model}, threshold={threshold}"
    )
    console.print(f"Gold set: {len(gold)} entries from {gold_path}\n")

    rpath = _results_path(model)
    cached = load_results(rpath)
    results = await run_llm_predictions(gold, cached, model, rpath)
    compute_metrics(gold, results, model, threshold)


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
