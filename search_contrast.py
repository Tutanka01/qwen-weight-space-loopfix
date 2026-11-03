"""Search #1 (failed): ablate the restart-vs-stop contrast direction from the top-scoring ffn_down
columns of the layers that write it (blog-like). Hurts healthy </think> predictions.
Usage: uv run search_contrast.py   (writes results/search.json)
"""

import json
import sys

import torch

from edit_lab import Lab

TRACES = ["results/base_default.jsonl", "results/base_qwen_reco.jsonl"]
LAYERS = [14, 15, 16, 17, 18, 19, 21, 22]

lab = Lab(TRACES, TRACES)
base = lab.evaluate()
print("base", base, flush=True)
scores = lab.neuron_scores(LAYERS)
torch.save(scores, "results/acts/neuron_scores.pt")
for l in LAYERS:
    s = scores[l]
    top = s.topk(5)
    print(f"layer {l}: top neuron scores {[round(v, 3) for v in top.values.tolist()]} (sum top64={s.topk(64).values.sum():.2f})")


def top(l, k):
    return scores[l].topk(k).indices.tolist()


candidates = {
    "L15+L19 top64 (blog-like)": {15: (top(15, 64), 15), 19: (top(19, 64), 19)},
    "L15+L18+L19 top64": {l: (top(l, 64), l) for l in (15, 18, 19)},
    "L15+L18+L19 top128": {l: (top(l, 128), l) for l in (15, 18, 19)},
    "L14-L19 top64": {l: (top(l, 64), l) for l in range(14, 20)},
    "L15-L22 top64": {l: (top(l, 64), l) for l in (15, 16, 17, 18, 19, 21, 22)},
    "L15+L18+L19 all columns": {l: (list(range(3584)), l) for l in (15, 18, 19)},
}
results = {}
for name, edit in candidates.items():
    n = lab.apply(edit)
    m = lab.evaluate()
    results[name] = {"weights_changed": n, **m}
    print(f"{name:<28} changed={n:>9}  {m}", flush=True)
lab.reset()
json.dump({"base": base, **results}, open("results/search.json", "w"), indent=1)
