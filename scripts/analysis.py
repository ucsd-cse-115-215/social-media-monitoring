# %% [markdown]
# # LLM Filter Evaluation
#
# Compare models and sweep thresholds using cached eval results.
# Run `python -m monitor.eval --model <model>` first to generate result files.

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# %%
# --- Load data ---

GOLD_PATH = Path("../data/gold.jsonl")
COSTS_PATH = Path("../data/model_costs.json")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


gold = load_jsonl(GOLD_PATH)
gold_by_text = {e["text"]: e["is_poetry"] for e in gold}

with open(COSTS_PATH) as f:
    costs_data = json.load(f)
    costs_data.pop("_comment", None)

# Discover all eval result files
result_files = sorted(Path("../data").glob("eval_*.jsonl"))
models = {}
for rf in result_files:
    model_name = rf.stem.removeprefix("eval_")
    results = load_jsonl(rf)
    models[model_name] = {r["text"]: r for r in results}

print(f"Gold set: {len(gold)} entries")
print(f"Models: {', '.join(models.keys())}")

# %%
# --- Metrics at a given threshold ---


def compute_metrics(results_by_text, threshold):
    tp = fp = tn = fn = 0
    for text, label in gold_by_text.items():
        r = results_by_text.get(text)
        if r is None:
            continue
        analysis = r.get("llm_analysis", {})
        predicted = (
            analysis.get("is_poetry", False)
            and analysis.get("confidence", 0) >= threshold
        )
        if predicted and label:
            tp += 1
        elif predicted and not label:
            fp += 1
        elif not predicted and label:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


def total_cost(model_name, results_by_text):
    pricing = costs_data.get(model_name)
    if not pricing:
        return None
    input_tok = sum(r.get("usage", {}).get("input_tokens", 0) for r in results_by_text.values())
    output_tok = sum(r.get("usage", {}).get("output_tokens", 0) for r in results_by_text.values())
    return (input_tok / 1e6 * pricing["input"]) + (output_tok / 1e6 * pricing["output"])


# Print a quick comparison at the default threshold
print(f"{'Model':<20} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Cost':>10}")
print("-" * 52)
for name, results in models.items():
    m = compute_metrics(results, threshold=0.7)
    c = total_cost(name, results)
    cost_str = f"${c:.4f}" if c is not None else "N/A"
    print(f"{name:<20} {m['precision']:>6.1%} {m['recall']:>6.1%} {m['f1']:>6.1%} {cost_str:>10}")

# %%
# --- Threshold sweep (table) ---

thresholds = np.arange(0.0, 1.01, 0.05)

# Precompute sweep for all models
sweep = {}
for name, results in models.items():
    sweep[name] = [compute_metrics(results, t) for t in thresholds]

# Print text table
def fmt(v):
    return f"{v:.1%}" if v is not None else "  N/A"

header = f"{'Thresh':>6}"
for name in models:
    header += f"  | {'Prec':>6} {'Rec':>6} {'F1':>6}"
print(header)
print("-" * len(header))
for i, t in enumerate(thresholds):
    row = f"{t:>6.2f}"
    for name in models:
        m = sweep[name][i]
        row += f"  | {fmt(m['precision']):>6} {fmt(m['recall']):>6} {fmt(m['f1']):>6}"
    print(row)

# %%
# --- Threshold sweep (plots) ---

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for name in models:
    # Filter out None values for plotting
    ts = [thresholds[i] for i in range(len(thresholds))]
    precisions = [sweep[name][i]["precision"] for i in range(len(thresholds))]
    recalls = [sweep[name][i]["recall"] for i in range(len(thresholds))]
    f1s = [sweep[name][i]["f1"] for i in range(len(thresholds))]

    # Only plot non-None points
    def plot_filtered(ax, xs, ys, **kwargs):
        valid = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if valid:
            ax.plot([v[0] for v in valid], [v[1] for v in valid], **kwargs)

    plot_filtered(axes[0], ts, precisions, label=name)
    plot_filtered(axes[1], ts, recalls, label=name)
    plot_filtered(axes[2], ts, f1s, label=name)

for ax, title in zip(axes, ["Precision", "Recall", "F1"]):
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

fig.tight_layout()
plt.show()

# %%
# --- Precision-Recall curve ---

fig, ax = plt.subplots(figsize=(6, 5))

for name, results in models.items():
    points = []
    for t in thresholds:
        m = compute_metrics(results, t)
        if m["precision"] is not None and m["recall"] is not None:
            points.append((m["recall"], m["precision"]))
    if points:
        ax.plot([p[0] for p in points], [p[1] for p in points],
                "o-", label=name, markersize=3)

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve")
ax.legend()
ax.set_xlim(0, 1.05)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()

# %%
# --- Cost vs F1 at optimal threshold ---

fig, ax = plt.subplots(figsize=(6, 4))

for name, results in models.items():
    f1_values = [compute_metrics(results, t)["f1"] for t in thresholds]
    best_f1 = max((v for v in f1_values if v is not None), default=0)
    c = total_cost(name, results)
    if c is not None:
        ax.scatter(c, best_f1, s=100, zorder=5)
        ax.annotate(name, (c, best_f1), textcoords="offset points",
                     xytext=(8, 4), fontsize=10)

ax.set_xlabel("Cost (USD, on eval set)")
ax.set_ylabel("Best F1")
ax.set_title("Cost vs Best F1")
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()

# %%
# --- Disagreements between models ---
#
# Posts where the two models give different predictions at threshold=0.7.

if len(models) >= 2:
    model_names = list(models.keys())
    a_name, b_name = model_names[0], model_names[1]
    a_results, b_results = models[a_name], models[b_name]

    print(f"Disagreements: {a_name} vs {b_name} (threshold=0.7)\n")
    for text, label in gold_by_text.items():
        a = a_results.get(text, {}).get("llm_analysis", {})
        b = b_results.get(text, {}).get("llm_analysis", {})
        a_pred = a.get("is_poetry", False) and a.get("confidence", 0) >= 0.7
        b_pred = b.get("is_poetry", False) and b.get("confidence", 0) >= 0.7
        if a_pred != b_pred:
            label_str = "POETRY" if label else "not poetry"
            a_str = f"yes ({a.get('confidence', '?')})" if a_pred else f"no ({a.get('confidence', '?')})"
            b_str = f"yes ({b.get('confidence', '?')})" if b_pred else f"no ({b.get('confidence', '?')})"
            print(f"[{label_str}] {a_name}={a_str}, {b_name}={b_str}")
            print(f"  {text[:80]}...")
            print()

# %%
