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

### Logging

Enable per-stage logging to capture posts for debugging and labeling:

```bash
# Log all stages
python -m monitor.main --log-dir logs/

# Log only specific stages
python -m monitor.main --log-dir logs/ --log-stages structural llm
```

Each stage writes to its own JSONL file (e.g. `logs/llm.jsonl`).

### Building a Gold Set

To tune the LLM filter, you need labeled data. The workflow:

1. **Collect candidates** — run the monitor with LLM logging for a while:
   ```bash
   python -m monitor.main --log-dir logs/ --log-stages llm
   ```
2. **Label them** — the labeling tool shows all LLM-positive posts and a random
   sample of negatives, and asks you to classify each one:
   ```bash
   python -m monitor.label logs/llm.jsonl
   ```
   Press `p` (poetry), `n` (not poetry), `s` (skip), or `q` (quit).
   Labels are appended to `data/gold.jsonl`. Already-labeled posts are skipped.

   To include more negatives for labeling (default is 20%):
   ```bash
   python -m monitor.label logs/llm.jsonl --negative-sample-rate 0.5
   ```

### Evaluating the LLM Filter

Run the gold set through the LLM filter and see precision/recall/F1:

```bash
python -m monitor.eval
```

Compare different models or thresholds:

```bash
python -m monitor.eval --model gpt-4o --threshold 0.5
python -m monitor.eval --model gpt-4o-mini --threshold 0.9
```

The eval shows every disagreement between the LLM and human labels,
along with the LLM's confidence and reasoning — useful for improving the prompt.

## Project Structure

```
src/monitor/
├── main.py              # CLI entry point, wires everything together
├── source.py            # Bluesky Jetstream WebSocket client
├── pipeline.py          # Multi-stage filter chain with stats
├── output.py            # Rich console display
├── label.py             # Interactive labeling tool
├── eval.py              # LLM filter evaluation
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
