"""Neuron-level search + held-out proxy evaluation of ffn_down edits (no gradients).

Edit (same form as the blog's Experiment 8): for the top-k neurons j of layer l that push the
residual stream along the 'restart' direction d_l, remove that component from their output column:
    W_down[:, j] <- W_down[:, j] - (d_l . W_down[:, j]) d_l

Proxy metrics on HELD-OUT traces (even prompt ids), teacher-forced:
  restart_gap : mean log P(</think>) - log P(actual restart token) at restart positions   (want up)
  stop_lp     : mean log P(</think>) where the model really closed its thinking            (want ~same)
  neutral_kl  : mean KL(base || edited) of next-token distributions at random positions     (want ~0)
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from collect_acts import RESTART, build_text, is_discovery
from qwen_gguf import load

A = Path("results/acts")
DEV = "mps"


def load_rows(paths):
    return [json.loads(l) for p in paths for l in Path(p).read_text().splitlines() if l.strip()]


def positions(tok, r, rng, max_restarts=25, n_neutral=10):
    prompt, text = build_text(tok, r)
    enc = tok(text, return_offsets_mapping=True, return_tensors="pt")
    offsets = enc.offset_mapping[0].tolist()
    c2t = lambda c: next(i for i, (a, b) in enumerate(offsets) if b > c)
    rs = [m.start(1) for m in RESTART.finditer(text, len(prompt))]
    rs = rng.sample(rs, min(max_restarts, len(rs)))
    pos = {"restart": [c2t(c) - 1 for c in rs], "stop": [], "neutral": []}
    if not r["truncated"]:
        pos["stop"] = [len(offsets) - 2]
    first = c2t(len(prompt))
    pos["neutral"] = [rng.randrange(first, len(offsets) - 2) for _ in range(n_neutral)]
    return enc.input_ids, pos


class Lab:
    def __init__(self, train_paths, test_paths):
        self.model, self.tok = load(device=DEV)
        self.think_end = self.tok.convert_tokens_to_ids("</think>")
        self.orig = {l: layer.mlp.down_proj.weight.detach().clone() for l, layer in enumerate(self.model.model.layers)}
        rng = random.Random(1)
        test_rows = [r for r in load_rows(test_paths) if not is_discovery(r["id"])]
        self.test = [(r, *positions(self.tok, r, rng)) for r in test_rows]
        self.train_rows = [r for r in load_rows(train_paths) if is_discovery(r["id"])]
        self.dirs = {int(k): torch.tensor(v, device=DEV) for k, v in np.load(A / "dirs.npz").items()}
        self.base_logp = None

    # ---------- neuron scoring on discovery traces ----------
    def neuron_scores(self, layers):
        """score[l][j] = mean_restart(a_j * w_j.d_l) - mean_stop(a_j * w_j.d_l)"""
        acts = {}
        hooks = [self.model.model.layers[l].mlp.down_proj.register_forward_hook(
            lambda m, i, o, l=l: acts.__setitem__(l, i[0][0])) for l in layers]
        rng = random.Random(0)
        sums = {k: {l: torch.zeros(self.model.config.intermediate_size, device=DEV) for l in layers} for k in ("restart", "stop")}
        counts = {"restart": 0, "stop": 0}
        for r in self.train_rows:
            ids, pos = positions(self.tok, r, rng)
            with torch.no_grad():
                self.model.model(ids.to(DEV), use_cache=False)
            for kind in ("restart", "stop"):
                if r["truncated"] is False and kind == "restart":
                    continue  # restarts only from looping traces
                p = pos[kind]
                if not p:
                    continue
                for l in layers:
                    proj = self.orig[l].T @ self.dirs[l]  # w_j . d_l for every neuron j
                    sums[kind][l] += (acts[l][p] * proj).sum(0)
                counts[kind] += len(p)
        for h in hooks:
            h.remove()
        return {l: (sums["restart"][l] / counts["restart"] - sums["stop"][l] / counts["stop"]).cpu() for l in layers}

    # ---------- edits ----------
    def reset(self):
        for l, w in self.orig.items():
            self.model.model.layers[l].mlp.down_proj.weight.data.copy_(w)

    def apply(self, edit):
        """edit: {layer: (neuron_indices, direction_layer)} ; projects d out of the selected columns."""
        self.reset()
        changed = 0
        for l, (neurons, dl) in edit.items():
            W = self.model.model.layers[l].mlp.down_proj.weight.data  # [hidden, inter]
            d = self.dirs[dl]
            idx = torch.as_tensor(neurons, device=DEV)
            cols = W[:, idx]
            W[:, idx] = cols - torch.outer(d, d @ cols)
            changed += cols.numel()
        return changed

    # ---------- held-out proxy ----------
    def _logp_at(self, ids, pos_list):
        with torch.no_grad():
            h = self.model.model(ids.to(DEV), use_cache=False).last_hidden_state[0, pos_list]
            return F.log_softmax(self.model.lm_head(h).float(), -1)

    def evaluate(self):
        gaps, stops, kls = [], [], []
        cur_logp = []
        for i, (r, ids, pos) in enumerate(self.test):
            allpos = pos["restart"] + pos["stop"] + pos["neutral"]
            if not allpos:
                cur_logp.append(None)
                continue
            lp = self._logp_at(ids, allpos)
            cur_logp.append(lp[len(pos["restart"]) + len(pos["stop"]):].cpu())
            nr = len(pos["restart"])
            if nr and r["truncated"]:
                nxt = ids[0, torch.tensor(pos["restart"]) + 1].to(DEV)
                gaps += (lp[:nr, self.think_end] - lp[torch.arange(nr), nxt]).tolist()
            if pos["stop"]:
                stops.append(lp[nr, self.think_end].item())
            if self.base_logp is not None and self.base_logp[i] is not None:
                b = self.base_logp[i]
                kls += (b.exp() * (b - cur_logp[-1])).sum(-1).tolist()
        if self.base_logp is None:
            self.base_logp = cur_logp
        return {"restart_gap": float(np.mean(gaps)), "stop_lp": float(np.mean(stops)),
                "neutral_kl": float(np.mean(kls)) if kls else 0.0}


if __name__ == "__main__":
    lab = Lab(sys.argv[1:], sys.argv[1:])
    print("base", lab.evaluate(), flush=True)
