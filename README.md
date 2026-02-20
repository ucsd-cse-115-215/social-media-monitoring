# Social Media Monitor — Poetry Detector

A multi-stage filtering pipeline that monitors the Bluesky firehose in real time
and detects poetry using a combination of cheap heuristics and LLM classification.

This is the example project for CSE 215: Software Engineering for LLM-powered Systems.

## Architecture

```
Bluesky firehose (~1000 posts/sec)
        │
   ┌────▼─────┐
   │  Stage 1  │  BasicFilter: language, length, non-empty
   │  (free)   │  Drops ~40% of posts
   └────┬──────┘
        │
   ┌────▼─────┐
   │  Stage 2  │  StructuralFilter: line breaks, short lines, not a list
   │  (free)   │  Drops ~99% of remaining posts
   └────┬──────┘
        │
   ┌────▼─────┐
   │  Stage 3  │  LLMFilter: GPT-4o-mini judges if it's actually poetry
   │  ($$$)    │  Only sees ~0.1% of original volume
   └────┬──────┘
        │
   ┌────▼─────┐
   │  Output   │  Rich console display with poem + LLM explanation
   └──────────┘
```

The key idea: each stage is cheaper and faster than the next. By the time we
reach the expensive LLM call, we've already filtered out 99.9%+ of posts.

## Setup

```bash
# Clone and enter the repo
cd social-media-monitoring

# Create a virtual environment and install
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Add your OpenAI API key
# Edit .env and set OPENAI_API_KEY=sk-...
```

## Usage

```bash
# Run the monitor
python -m monitor.main

# Run tests
python -m pytest tests/ -v
```

Press Ctrl+C to stop — the app will print pipeline stats before exiting.

## Project Structure

```
src/monitor/
├── main.py              # CLI entry point, wires everything together
├── source.py            # Bluesky Jetstream WebSocket client
├── pipeline.py          # Multi-stage filter chain with stats
├── output.py            # Rich console display
└── filters/
    ├── base.py          # Post dataclass + Filter ABC
    ├── basic.py         # Stage 1: language, length, non-empty
    ├── structural.py    # Stage 2: poetry shape heuristics
    └── llm.py           # Stage 3: OpenAI classification
```

## Customization

To build your own monitor (e.g. detecting misinformation, job postings, haiku, etc.):

1. **Swap the filters** — keep `BasicFilter`, replace `StructuralFilter` with
   your own heuristics, and update the LLM prompt in `llm.py`.
2. **Tune the pipeline** — adjust thresholds, add/remove stages, reorder filters.
3. **Change the output** — swap `ConsoleOutput` for Slack webhooks, email, a database, etc.

## Future Directions

- Embedding-based pre-filter (semantic similarity to known poetry)
- Prompt optimization with DSPy
- Declarative pipeline with LOTUS/Palimpzest
- Web dashboard with Streamlit
- Cost tracking and rate limiting
