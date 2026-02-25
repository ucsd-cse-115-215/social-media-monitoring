"""Interactive labeling tool for building the gold evaluation set.

Usage:
    python -m monitor.label logs/llm.jsonl

Reads posts from an LLM filter log, presents them for human labeling,
and appends labeled entries to data/gold.jsonl.

By default, shows all LLM-positive posts and a random sample of negatives
(controlled by --negative-sample-rate).
"""

import argparse
import json
import random
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

DEFAULT_GOLD_PATH = Path("data/gold.jsonl")


def load_log(log_path: Path) -> list[dict]:
    entries = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_existing_texts(gold_path: Path) -> set[str]:
    """Load texts already in the gold set to avoid re-labeling."""
    texts = set()
    if gold_path.exists():
        with open(gold_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    texts.add(entry["text"])
    return texts


def select_entries(entries: list[dict], negative_sample_rate: float) -> list[dict]:
    """Select all positives and a sample of negatives."""
    positives = [e for e in entries if e.get("passed")]
    negatives = [e for e in entries if not e.get("passed")]

    n_negatives = max(1, int(len(negatives) * negative_sample_rate))
    sampled_negatives = random.sample(negatives, min(n_negatives, len(negatives)))

    # Shuffle so the labeler doesn't see all positives first
    selected = positives + sampled_negatives
    random.shuffle(selected)
    return selected


def prompt_label() -> str:
    """Prompt the user for a label. Returns 'p', 'n', 's', or 'q'."""
    console.print(
        "  [green][p][/]oetry  [red][n][/]ot poetry  [dim][s][/]kip  [dim][q][/]uit"
    )
    while True:
        try:
            choice = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "q"
        if choice in ("p", "n", "s", "q"):
            return choice
        console.print("  [dim]press p, n, s, or q[/]")


def run_labeler(
    log_path: Path,
    gold_path: Path,
    negative_sample_rate: float,
) -> None:
    entries = load_log(log_path)
    existing = load_existing_texts(gold_path)

    # Filter out already-labeled posts
    entries = [e for e in entries if e["text"] not in existing]
    if not entries:
        console.print("[yellow]No new posts to label (all already in gold set).[/]")
        return

    selected = select_entries(entries, negative_sample_rate)

    console.print(
        f"[bold]Labeling session[/]: {len(selected)} posts "
        f"(from {len(entries)} new entries in log)"
    )
    console.print()

    gold_path.parent.mkdir(parents=True, exist_ok=True)
    labeled = 0

    for i, entry in enumerate(selected, 1):
        console.print(f"[dim]— {i}/{len(selected)} —[/]")

        panel = Panel(
            Text(entry["text"].strip()),
            border_style="cyan",
            padding=(1, 2),
        )
        console.print(panel)

        choice = prompt_label()

        if choice == "q":
            break
        elif choice == "s":
            continue
        else:
            is_poetry = choice == "p"
            gold_entry = {
                "text": entry["text"],
                "url": entry.get("url"),
                "is_poetry": is_poetry,
            }
            with open(gold_path, "a") as f:
                f.write(json.dumps(gold_entry) + "\n")
            labeled += 1

            label_str = "[green]poetry[/]" if is_poetry else "[red]not poetry[/]"
            console.print(f"  → {label_str}")

        console.print()

    console.print(f"[bold]Done.[/] Labeled {labeled} posts → {gold_path}")


def main():
    parser = argparse.ArgumentParser(description="Label posts for the gold eval set")
    parser.add_argument("log_file", type=Path, help="Path to LLM filter log (e.g. logs/llm.jsonl)")
    parser.add_argument(
        "--gold",
        type=Path,
        default=DEFAULT_GOLD_PATH,
        help=f"Path to gold set output (default: {DEFAULT_GOLD_PATH})",
    )
    parser.add_argument(
        "--negative-sample-rate",
        type=float,
        default=0.2,
        help="Fraction of LLM-negative posts to include for labeling (default: 0.2)",
    )
    args = parser.parse_args()

    if not args.log_file.exists():
        console.print(f"[red]Log file not found: {args.log_file}[/]")
        raise SystemExit(1)

    run_labeler(args.log_file, args.gold, args.negative_sample_rate)


if __name__ == "__main__":
    main()
