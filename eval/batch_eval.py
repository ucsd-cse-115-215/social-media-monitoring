"""Submit and retrieve OpenAI Batch API evaluations.

Usage:
    python scripts/batch_eval.py submit --model gpt-5-mini gpt-5.4
    python scripts/batch_eval.py status
    python scripts/batch_eval.py download
"""

import argparse
import json
import tempfile
from pathlib import Path

import openai
from dotenv import load_dotenv

load_dotenv()

GOLD_PATH = Path("data/gold.jsonl")
BATCH_META_PATH = Path("data/batch_jobs.json")

SYSTEM_PROMPT = """\
You are a poetry detector. Given a social media post, determine whether it
contains an original poem (not just a quote or song lyrics).

Respond with JSON:
{
  "is_poetry": true/false,
  "confidence": 0.0-1.0,
  "explanation": "brief reason"
}
"""

# Chain-of-thought variant: justification comes before the boolean decision,
# forcing the model to reason before committing to an answer.
COT_PROMPT = """\
You are a poetry detector. Given a social media post, determine whether it
contains an original poem (not just a quote or song lyrics).

Think step-by-step, then respond with JSON:
{
  "justification": "your reasoning about whether this is an original poem",
  "is_poetry": true/false,
  "confidence": 0.0-1.0
}
"""

# Reasoning-effort variant: no explanation needed — the model's internal
# reasoning (via reasoning_effort) replaces the explanation field.
REASONING_PROMPT = """\
You are a poetry detector. Given a social media post, determine whether it
contains an original poem (not just a quote or song lyrics).

Respond with JSON:
{
  "is_poetry": true/false,
  "confidence": 0.0-1.0
}
"""


def load_gold() -> list[dict]:
    entries = []
    with open(GOLD_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_batch_meta() -> dict:
    if BATCH_META_PATH.exists():
        with open(BATCH_META_PATH) as f:
            return json.load(f)
    return {}


def save_batch_meta(meta: dict):
    with open(BATCH_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)


def submit(models: list[str], tag: str | None = None,
           reasoning_effort: str | None = None, cot: bool = False):
    client = openai.OpenAI()
    gold = load_gold()
    meta = load_batch_meta()

    # Pick the right prompt variant
    if reasoning_effort:
        prompt = REASONING_PROMPT
    elif cot:
        prompt = COT_PROMPT
    else:
        prompt = SYSTEM_PROMPT

    for model in models:
        label = tag or model
        results_path = Path(f"data/eval_{label}.jsonl")
        if results_path.exists():
            # Load existing cached texts to skip them
            cached_texts = set()
            with open(results_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        cached_texts.add(json.loads(line)["text"])
            entries = [e for e in gold if e["text"] not in cached_texts]
            if not entries:
                print(f"  {label}: all {len(gold)} entries already cached, skipping")
                continue
            print(f"  {label}: {len(cached_texts)} cached, submitting {len(entries)} new entries")
        else:
            entries = gold

        # Build batch request JSONL
        requests = []
        for i, entry in enumerate(entries):
            body = {
                "model": model,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": entry["text"]},
                ],
                "response_format": {"type": "json_object"},
            }
            if reasoning_effort:
                body["reasoning_effort"] = reasoning_effort
            requests.append({
                "custom_id": f"gold-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            })

        # Write to temp file and upload
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as tmp:
            for req in requests:
                tmp.write(json.dumps(req) + "\n")
            tmp_path = tmp.name

        print(f"Uploading batch file for {label} ({len(requests)} requests)...")
        with open(tmp_path, "rb") as f:
            batch_file = client.files.create(file=f, purpose="batch")

        print(f"Submitting batch for {label}...")
        batch = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"model": model, "description": f"eval {label} on gold set"},
        )

        meta[label] = {
            "batch_id": batch.id,
            "input_file_id": batch_file.id,
            "status": batch.status,
            "n_requests": len(requests),
            # Store the gold texts in submission order so we can map results back
            "texts": [e["text"] for e in entries],
        }
        print(f"  Batch {batch.id} submitted ({batch.status})")

    save_batch_meta(meta)
    print(f"\nBatch metadata saved to {BATCH_META_PATH}")


def status():
    client = openai.OpenAI()
    meta = load_batch_meta()
    if not meta:
        print("No batch jobs found.")
        return

    for model, info in meta.items():
        batch = client.batches.retrieve(info["batch_id"])
        info["status"] = batch.status
        counts = batch.request_counts
        print(
            f"  {model}: {batch.status} "
            f"(completed={counts.completed}/{counts.total}, failed={counts.failed})"
        )
        if batch.output_file_id:
            info["output_file_id"] = batch.output_file_id

    save_batch_meta(meta)


def download():
    client = openai.OpenAI()
    meta = load_batch_meta()
    if not meta:
        print("No batch jobs found.")
        return

    for label, info in meta.items():
        batch = client.batches.retrieve(info["batch_id"])
        info["status"] = batch.status

        if batch.status != "completed":
            print(f"  {label}: not ready yet ({batch.status})")
            continue

        if not batch.output_file_id:
            print(f"  {label}: completed but no output file?")
            continue

        print(f"  {label}: downloading results...")
        content = client.files.content(batch.output_file_id).text
        texts = info["texts"]

        results_path = Path(f"data/eval_{label}.jsonl")
        with open(results_path, "a") as out:
            for line in content.strip().split("\n"):
                resp = json.loads(line)
                idx = int(resp["custom_id"].split("-")[1])
                text = texts[idx]

                body = resp["response"]["body"]
                choice = body["choices"][0]
                usage = body["usage"]
                llm_analysis = json.loads(choice["message"]["content"])

                result = {
                    "text": text,
                    "llm_analysis": llm_analysis,
                    "usage": {
                        "input_tokens": usage["prompt_tokens"],
                        "output_tokens": usage["completion_tokens"],
                    },
                }
                out.write(json.dumps(result) + "\n")

        print(f"    Saved {len(texts)} results to {results_path}")

    save_batch_meta(meta)


def main():
    parser = argparse.ArgumentParser(description="OpenAI Batch API eval")
    sub = parser.add_subparsers(dest="command", required=True)

    p_submit = sub.add_parser("submit", help="Submit batch jobs")
    p_submit.add_argument("--model", nargs="+", required=True, help="Models to evaluate")
    p_submit.add_argument("--tag", help="Override output filename (e.g. eval_{tag}.jsonl)")
    p_submit.add_argument("--reasoning-effort", choices=["low", "medium", "high"],
                          help="Set reasoning effort (uses simplified prompt without explanation)")
    p_submit.add_argument("--cot", action="store_true",
                          help="Use chain-of-thought prompt (justification before decision)")

    sub.add_parser("status", help="Check batch job status")
    sub.add_parser("download", help="Download completed results")

    args = parser.parse_args()

    if args.command == "submit":
        submit(args.model, tag=args.tag,
               reasoning_effort=args.reasoning_effort, cot=args.cot)
    elif args.command == "status":
        status()
    elif args.command == "download":
        download()


if __name__ == "__main__":
    main()
