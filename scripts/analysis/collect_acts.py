"""Teacher-force LM Studio traces through the model and record residual-stream and MLP-output
activations at three kinds of positions (the activation at t-1 is what predicts token t):

  restart_loop : just before a "Wait"/"Okay"/"Actually"/... restart, in traces that hit the token limit
  restart_ok   : same restarts, but in traces that eventually finished
  stop         : just before "</think>" in traces that finished
  neutral      : random positions inside reasoning of finished traces (for side-effect checks)

Only discovery prompts (odd ids) are used, the even ids are kept for evaluation.
Usage: uv run scripts/analysis/collect_acts.py [traces.jsonl ...]   (default: results/baseline/*.jsonl)
"""

import json
import random
import sys

import numpy as np
import torch

from loopfix.gguf_model import load
from loopfix.paths import ACTS as OUT
from loopfix.paths import TRACES
from loopfix.traces import RESTART, build_text, is_discovery, load_rows

MAX_PER_TRACE = 25


def main():
    rows = load_rows(sys.argv[1:] or TRACES)
    rows = [r for r in rows if is_discovery(r["id"])]
    model, tok = load(device="mps")
    n_layers = model.config.num_hidden_layers
    rng = random.Random(0)

    mlp_out = {}
    hooks = [layer.mlp.register_forward_hook(lambda m, i, o, l=l: mlp_out.__setitem__(l, o[0]))
             for l, layer in enumerate(model.model.layers)]

    store = {k: {"resid": [], "mlp": []} for k in ("restart_loop", "restart_ok", "stop", "neutral")}
    meta = []
    for n, r in enumerate(rows):
        prompt, text = build_text(tok, r)
        enc = tok(text, return_offsets_mapping=True, return_tensors="pt")
        ids, offsets = enc.input_ids, enc.offset_mapping[0].tolist()
        start_char = len(prompt)
        char_to_tok = lambda c: next(i for i, (a, b) in enumerate(offsets) if b > c)

        picks = []  # (kind, token index whose *next* token is the event)
        restarts = [m.start(1) for m in RESTART.finditer(text, start_char)]
        restarts = rng.sample(restarts, min(MAX_PER_TRACE, len(restarts)))
        kind = "restart_loop" if r["truncated"] else "restart_ok"
        picks += [(kind, char_to_tok(c) - 1) for c in restarts]
        if not r["truncated"]:
            picks.append(("stop", len(offsets) - 2))  # position predicting "</think>"
            first = char_to_tok(start_char)
            picks += [("neutral", rng.randrange(first, len(offsets) - 2)) for _ in range(10)]

        with torch.no_grad():  # body only: full logits for 8k tokens x 248k vocab would be ~8 GB
            out = model.model(ids.to("mps"), output_hidden_states=True, use_cache=False)
        hs = out.hidden_states  # embeddings + one per layer
        for kind, t in picks:
            store[kind]["resid"].append(torch.stack([hs[l + 1][0, t] for l in range(n_layers)]).half().cpu().numpy())
            store[kind]["mlp"].append(torch.stack([mlp_out[l][t] for l in range(n_layers)]).half().cpu().numpy())
            meta.append({"kind": kind, "id": r["id"], "config": r["config"], "pos": t,
                         "next": tok.decode(ids[0, t + 1: t + 3])})
        print(f"[{n + 1}/{len(rows)}] {r['config']} {r['id']} tokens={ids.shape[1]} picks={len(picks)}", flush=True)

    for h in hooks:
        h.remove()
    OUT.mkdir(parents=True, exist_ok=True)
    for kind, d in store.items():
        np.savez(OUT / f"{kind}.npz", resid=np.array(d["resid"]), mlp=np.array(d["mlp"]))
        print(kind, len(d["resid"]))
    (OUT / "meta.jsonl").write_text("\n".join(json.dumps(m, ensure_ascii=False) for m in meta))


if __name__ == "__main__":
    main()
