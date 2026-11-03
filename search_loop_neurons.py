"""Search #2 (the one that works): edit only LOOP-SPECIFIC neurons, so healthy behaviour is untouched
by construction. Compares two edit forms on held-out traces:
  proj : remove the restart direction from the neuron's output column (blog-style)
  zero : zero the neuron's output column
Usage: uv run search_loop_neurons.py   (writes results/search2.json)
"""

import json

import torch

from edit_lab import DEV, Lab
from neuron_selection import as_edit, loop_specific

TRACES = ["results/base_default.jsonl", "results/base_qwen_reco.jsonl"]


def apply_zero(lab, edit):
    lab.reset()
    n = 0
    for l, (neurons, _) in edit.items():
        W = lab.model.model.layers[l].mlp.down_proj.weight.data
        W[:, torch.as_tensor(neurons, device=DEV)] = 0
        n += W.shape[0] * len(neurons)
    return n


def main():
    lab = Lab(TRACES, TRACES)
    results = {"base": lab.evaluate()}
    print("base", results["base"], flush=True)
    for quiet in (0.15, 0.3):
        cands = loop_specific(quiet)
        print(f"\nquiet<{quiet}: {len(cands)} candidates; top:", [(round(s, 3), f"L{l}n{j}") for s, l, j in cands[:8]])
        for n in (8, 32, 128):
            edit = as_edit(cands, n)
            layers = {l: len(v[0]) for l, v in edit.items()}
            for mode, fn in (("proj", lab.apply), ("zero", lambda e: apply_zero(lab, e))):
                changed = fn(edit)
                m = lab.evaluate()
                name = f"quiet<{quiet} top{n} {mode}"
                results[name] = {"weights_changed": changed, "layers": layers, **m}
                print(f"{name:<24} changed={changed:>7} layers={layers}  {m}", flush=True)
    lab.reset()
    json.dump(results, open("results/search2.json", "w"), indent=1)


if __name__ == "__main__":
    main()
