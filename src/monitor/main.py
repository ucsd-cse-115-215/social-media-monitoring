"""CLI entry point for the social media monitor."""

import asyncio
import logging
import signal

import openai
from dotenv import load_dotenv

from monitor.filters import BasicFilter, LLMFilter, StructuralFilter
from monitor.output import console, display_post, display_stats
from monitor.pipeline import Pipeline
from monitor.source import stream_posts


async def run() -> None:
    load_dotenv()

    client = openai.AsyncOpenAI()  # reads OPENAI_API_KEY from env

    pipeline = Pipeline(
        [
            BasicFilter(min_length=20),
            StructuralFilter(min_lines=3),
            LLMFilter(client),
        ]
    )

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

    try:
        async for post in stream_posts():
            if stop.is_set():
                break
            if await pipeline.process(post):
                display_post(post)
    finally:
        display_stats(pipeline)


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(run())


if __name__ == "__main__":
    main()
