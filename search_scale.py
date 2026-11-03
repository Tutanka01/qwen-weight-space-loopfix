"""Search #3: scale up the best family (loop-specific neurons, quiet<0.3, direction projection)
and write the edit specs used to build the GGUFs (edits/loopfix_q03_top{N}.json).
Usage: uv run search_scale.py   (writes results/search3.json)
"""

import json
from pathlib import Path

from edit_lab import Lab
from neuron_selection import as_edit, loop_specific

TRACES = ["results/base_default.jsonl", "results/base_qwen_reco.jsonl"]


def main():
    lab = Lab(TRACES, TRACES)
    print("base", lab.evaluate(), flush=True)
    cands = loop_specific(0.3)
    Path("edits").mkdir(exist_ok=True)
    res = {}
    for n in (128, 256, 512):
        edit = as_edit(cands, n)
        spec = {"mode": "proj", "layers": {str(l): v[0] for l, v in sorted(edit.items())}}
        Path(f"edits/loopfix_q03_top{n}.json").write_text(json.dumps(spec))
        changed = lab.apply(edit)
        m = lab.evaluate()
        res[n] = {"weights_changed": changed, **m}
        print(f"quiet<0.3 top{n} proj changed={changed} layers={ {l: len(v[0]) for l, v in edit.items()} } {m}", flush=True)
    lab.reset()
    json.dump(res, open("results/search3.json", "w"), indent=1)


if __name__ == "__main__":
    main()
