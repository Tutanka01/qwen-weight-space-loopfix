"""Per-neuron mean activation at restart_loop / restart_ok / stop / neutral positions (discovery traces)."""
import random, torch
from edit_lab import Lab, positions, DEV
TR = ["results/base_default.jsonl", "results/base_qwen_reco.jsonl"]
LAYERS = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
lab = Lab(TR, [])
acts = {}
hooks = [lab.model.model.layers[l].mlp.down_proj.register_forward_hook(lambda m, i, o, l=l: acts.__setitem__(l, i[0][0])) for l in LAYERS]
H = lab.model.config.intermediate_size
kinds = ("restart_loop", "restart_ok", "stop", "neutral")
S = {k: {l: torch.zeros(H, device=DEV) for l in LAYERS} for k in kinds}
A = {k: {l: torch.zeros(H, device=DEV) for l in LAYERS} for k in kinds}
C = {k: 0 for k in kinds}
rng = random.Random(0)
for r in lab.train_rows:
    ids, pos = positions(lab.tok, r, rng)
    with torch.no_grad(): lab.model.model(ids.to(DEV), use_cache=False)
    for k, p in (("restart_loop" if r["truncated"] else "restart_ok", pos["restart"]), ("stop", pos["stop"]),
                 ("neutral", pos["neutral"] if not r["truncated"] else [])):
        if not p: continue
        for l in LAYERS:
            S[k][l] += acts[l][p].sum(0); A[k][l] += acts[l][p].abs().sum(0)
        C[k] += len(p)
for h in hooks: h.remove()
out = {k: {l: (S[k][l] / C[k]).cpu() for l in LAYERS} for k in kinds}
out_abs = {k: {l: (A[k][l] / C[k]).cpu() for l in LAYERS} for k in kinds}
proj = {l: (lab.orig[l].T @ lab.dirs[l]).cpu() for l in LAYERS}
torch.save({"mean": out, "absmean": out_abs, "proj": proj, "counts": C}, "results/acts/neuron_stats.pt")
print(C)
for l in LAYERS:
    contrib = out["restart_loop"][l] * proj[l]
    for j in contrib.topk(3).indices.tolist():
        print(f"L{l:>2} n{j:>4} w.d={proj[l][j]:+.3f} | mean act: loop={out['restart_loop'][l][j]:+.3f} ok={out['restart_ok'][l][j]:+.3f} "
              f"stop={out['stop'][l][j]:+.3f} neutral={out['neutral'][l][j]:+.3f} | |act| neutral={out_abs['neutral'][l][j]:.3f}")
