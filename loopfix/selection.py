"""Selection of loop-specific MLP neurons (shared by the search scripts).

Uses the per-neuron statistics written by scripts/analysis/neuron_stats.py.
"""

from functools import cache

import torch

from loopfix.paths import ACTS

STATS = ACTS / "neuron_stats.pt"


@cache
def stats():
    return torch.load(STATS)


def loop_specific(max_quiet: float):
    """Neurons ranked by how much MORE they push along the restart direction d_l in looping traces than
    in healthy restarts, keeping only neurons that stay quiet (|mean act| < max_quiet) on healthy
    restarts, before </think> and at neutral positions. Returns [(score, layer, neuron), ...]."""
    st = stats()
    mean, absm, proj = st["mean"], st["absmean"], st["proj"]
    cands = []
    for l in sorted(proj):
        extra = (mean["restart_loop"][l] - mean["restart_ok"][l]) * proj[l]
        quiet = ((mean["stop"][l].abs() < max_quiet) & (absm["neutral"][l] < max_quiet)
                 & (mean["restart_ok"][l].abs() < max_quiet))
        for j in torch.nonzero(quiet).flatten().tolist():
            if extra[j] > 0:
                cands.append((extra[j].item(), l, j))
    return sorted(cands, reverse=True)


def as_edit(cands, n):
    """Top-n candidates -> {layer: ([neurons], direction_layer)} as expected by loopfix.lab.Lab.apply."""
    edit = {}
    for _, l, j in cands[:n]:
        edit.setdefault(l, ([], l))[0].append(j)
    return edit
