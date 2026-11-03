"""Paired comparison of held-out runs (first file = baseline).

Usage: uv run scripts/eval/compare.py --seed 0 results/heldout/base.jsonl results/heldout/loopfix128.jsonl ...
"""

import argparse
import json
import random
from pathlib import Path
from statistics import mean, median


SEED = None


def load(p):
    rows = (json.loads(l) for l in Path(p).read_text().splitlines() if l.strip())
    return {(r["id"], r["seed"]): r for r in rows if SEED is None or r["seed"] == SEED}


def boot_ci(diffs, n=5000, seed=0):
    """95% CI of the mean paired difference, resampling prompts (both seeds of a prompt together)."""
    rng = random.Random(seed)
    by_prompt = {}
    for (pid, _), d in diffs.items():
        by_prompt.setdefault(pid, []).append(d)
    groups = list(by_prompt.values())
    stats = sorted(mean(x for g in rng.choices(groups, k=len(groups)) for x in g) for _ in range(n))
    return stats[int(0.025 * n)], stats[int(0.975 * n)]


def row(name, rs):
    rs = list(rs)
    fr = [r["answer_fr_score"] for r in rs if r["answer_fr_score"] is not None]
    return (f"{name:<16} n={len(rs):>3}  truncated={100 * mean(r['truncated'] for r in rs):5.1f}%  "
            f"answered={100 * mean(r['has_answer'] for r in rs):5.1f}%  median_tok={int(median(r['completion_tokens'] for r in rs)):>5}  "
            f"mean_tok={int(mean(r['completion_tokens'] for r in rs)):>5}  answer_FR={mean(fr):.2f}")


def main():
    global SEED
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, help="only use generations with this seed")
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()
    SEED = args.seed
    base_path, *others = args.files
    base = load(base_path)
    print(row(Path(base_path).stem, base.values()))
    for p in others:
        o = load(p)
        keys = sorted(base.keys() & o.keys())
        name = Path(p).stem
        print(row(name, (o[k] for k in keys)))
        for metric in ("truncated", "has_answer"):
            d = {k: float(o[k][metric]) - float(base[k][metric]) for k in keys}
            lo, hi = boot_ci(d)
            print(f"    Δ {metric:<10} vs base: {100 * mean(d.values()):+5.1f} pts  95% CI [{100 * lo:+.1f}, {100 * hi:+.1f}]")
        for cat in ("persona", "plain"):
            ks = [k for k in keys if base[k]["cat"] == cat]
            if not ks:
                continue
            print(f"    {cat:<8} n={len(ks):>3}  truncated: base {100 * mean(base[k]['truncated'] for k in ks):5.1f}%  ->  {100 * mean(o[k]['truncated'] for k in ks):5.1f}%")


if __name__ == "__main__":
    main()
