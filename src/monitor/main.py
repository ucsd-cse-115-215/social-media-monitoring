"""CLI entry point for the social media monitor."""

import argparse
import asyncio
import logging
import signal
from pathlib import Path

import openai
from dotenv import load_dotenv

from monitor.filters import BasicFilter, LLMFilter, ProfanityFilter, StructuralFilter
from monitor.output import console, display_post, display_rejected, display_stats, start_status, stop_status
from monitor.pipeline import Pipeline
from monitor.source import stream_posts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Social Media Monitor — Poetry Detector")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory for filter JSONL logs (one file per stage)",
    )
    parser.add_argument(
        "--log-stages",
        nargs="*",
        default=None,
        help="Which stages to log (e.g. --log-stages structural llm). Default: all stages",
    )
    return parser.parse_args()


async def run(args: argparse.Namespace) -> None:
    load_dotenv()

    client = openai.AsyncOpenAI()  # reads OPENAI_API_KEY from env

    filters = [
        BasicFilter(min_length=20),
        StructuralFilter(min_lines=3),
        ProfanityFilter(),
        LLMFilter(client),
    ]

    # Set up per-filter logging
    if args.log_dir:
        args.log_dir.mkdir(parents=True, exist_ok=True)
        log_stages = set(args.log_stages) if args.log_stages else None
        for f in filters:
            if log_stages is None or f.name in log_stages:
                f.log_path = args.log_dir / f"{f.name}.jsonl"
        enabled = [f.name for f in filters if f.log_path]
        console.print(f"[dim]Logging enabled for: {', '.join(enabled)} → {args.log_dir}/[/]")

    pipeline = Pipeline(filters)

    console.print("[bold]Social Media Monitor — Poetry Detector[/]")
    console.print("Connecting to Bluesky firehose...\n")

    # Print stats on Ctrl+C before exiting
    loop = asyncio.get_event_loop()
    stop = asyncio.Event()

    def handle_signal():
        console.print("\n[yellow]Shutting down...[/]")
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_signal)

    start_status()
    try:
        async for post in stream_posts():
            if stop.is_set():
                break
            if await pipeline.process(post):
                display_post(post, pipeline)
            else:
                display_rejected(pipeline)
    finally:
        stop_status()
        display_stats(pipeline)


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
