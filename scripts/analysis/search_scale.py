"""Search #3: scale up the best family (loop-specific neurons, quiet<0.3, direction projection)
and write the edit specs used to build the GGUFs (edits/loopfix_q03_top{N}.json).
Usage: uv run scripts/analysis/search_scale.py   (writes results/search/scale.json)
"""

import json
from pathlib import Path

from loopfix.paths import ACTS, EDITS, SEARCH, TRACES
from loopfix.lab import Lab
from loopfix.selection import as_edit, loop_specific



def main():
    lab = Lab(TRACES, TRACES)
    print("base", lab.evaluate(), flush=True)
    cands = loop_specific(0.3)
    EDITS.mkdir(exist_ok=True)
    res = {}
    for n in (128, 256, 512):
        edit = as_edit(cands, n)
        spec = {"mode": "proj", "layers": {str(l): v[0] for l, v in sorted(edit.items())}}
        (EDITS / f"loopfix_q03_top{n}.json").write_text(json.dumps(spec))
        changed = lab.apply(edit)
        m = lab.evaluate()
        res[n] = {"weights_changed": changed, **m}
        print(f"quiet<0.3 top{n} proj changed={changed} layers={ {l: len(v[0]) for l, v in edit.items()} } {m}", flush=True)
    lab.reset()
    json.dump(res, open(SEARCH / "scale.json", "w"), indent=1)


if __name__ == "__main__":
    main()
