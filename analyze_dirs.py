"""Find the 'keep second-guessing' direction and which MLP layers write it.

For each layer: d_l = mean(resid | restart in looping trace) - mean(resid | position before </think>),
evaluated with a prompt-level split so separation is not just memorised.
"""

import json
from pathlib import Path

import numpy as np

A = Path("results/acts")
load = lambda k: np.load(A / f"{k}.npz")
meta = [json.loads(l) for l in (A / "meta.jsonl").read_text().splitlines()]
ids = {k: [m["id"] for m in meta if m["kind"] == k] for k in ("restart_loop", "restart_ok", "stop", "neutral")}
R, S, O, N = (load(k) for k in ("restart_loop", "stop", "restart_ok", "neutral"))


def auc(pos, neg):
    s = np.concatenate([pos, neg])
    ranks = s.argsort().argsort() + 1
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def split(kind):  # half of the discovery prompts to fit the direction, the other half to score it
    a = np.array([int(i[1:]) % 4 == 1 for i in ids[kind]])
    return a, ~a


rl_fit, rl_val = split("restart_loop")
st_fit, st_val = split("stop")
print("layer  |d|/|resid|  AUC(restart_loop vs stop, held-out prompts)  AUC(restart_ok vs stop)  mlp_push")
dirs = {}
for l in range(R["resid"].shape[1]):
    r, s, o = (x["resid"][:, l].astype(np.float32) for x in (R, S, O))
    d = r[rl_fit].mean(0) - s[st_fit].mean(0)
    rel = np.linalg.norm(d) / np.linalg.norm(r, axis=1).mean()
    d /= np.linalg.norm(d)
    dirs[l] = d
    a1 = auc(r[rl_val] @ d, s[st_val] @ d)
    a2 = auc(o @ d, s @ d)
    # how much this layer's MLP output pushes along its own-layer direction (restart minus stop)
    m_r, m_s = R["mlp"][:, l].astype(np.float32), S["mlp"][:, l].astype(np.float32)
    push = (m_r @ d).mean() - (m_s @ d).mean()
    print(f"{l:>5}  {rel:>10.3f}  {a1:>10.3f}  {a2:>10.3f}  {push:>9.3f}")
np.savez(A / "dirs.npz", **{str(l): d for l, d in dirs.items()})

# Which MLP layers write the late-layer direction? project every MLP output onto d at a late layer.
for target in (19, 22):
    d = dirs[target]
    print(f"\nMLP contributions to layer-{target} direction (mean restart_loop - mean stop):")
    contrib = [(R["mlp"][:, l].astype(np.float32) @ d).mean() - (S["mlp"][:, l].astype(np.float32) @ d).mean()
               for l in range(target + 1)]
    for l in np.argsort(contrib)[::-1][:8]:
        print(f"  layer {l:>2}: {contrib[l]:+.3f}")
