"""Measure thinking-loop rate of a model served by LM Studio (OpenAI-compatible API).

Usage: uv run measure.py --config default --out results/base_default.jsonl
"""

import argparse
import json
import re
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

API = "http://localhost:1234/v1/chat/completions"

SAMPLING = {
    # Whatever LM Studio uses when the client sends nothing (what the user sees in the UI).
    "default": {},
    "greedy": {"temperature": 0.0},
    # Sampling commonly recommended for Qwen thinking mode (anti-repetition).
    "qwen_reco": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5},
}

FR_WORDS = set("le la les un une des et est je tu il nous vous de du pas que qui pour avec sur dans ce ça c'est mais".split())
EN_WORDS = set("the a an and is i you he we of not that who for with on in this it but are".split())


def repetition_stats(text: str) -> dict:
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 15]
    if not lines:
        return {"dup_line_ratio": 0.0, "max_line_repeat": 0}
    counts = Counter(lines)
    dup = sum(c - 1 for c in counts.values())
    return {"dup_line_ratio": round(dup / len(lines), 3), "max_line_repeat": max(counts.values())}


def lang_score(text: str) -> float | None:
    """Share of French function words among FR+EN function words (1.0 = French)."""
    words = re.findall(r"[a-zà-ÿ']+", text.lower())
    fr = sum(w in FR_WORDS for w in words)
    en = sum(w in EN_WORDS for w in words)
    return round(fr / (fr + en), 3) if fr + en else None


def run_one(model: str, prompt: dict, sampling: dict, max_tokens: int, seed: int) -> dict:
    messages = ([{"role": "system", "content": prompt["system"]}] if prompt["system"] else []) + [
        {"role": "user", "content": prompt["user"]}
    ]
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "seed": seed, **sampling}
    req = urllib.request.Request(API, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        resp = json.load(r)
    choice = resp["choices"][0]
    msg = choice["message"]
    reasoning = msg.get("reasoning_content") or ""
    content = msg.get("content") or ""
    return {
        "id": prompt["id"],
        "cat": prompt["cat"],
        "seed": seed,
        "finish": choice["finish_reason"],
        "truncated": choice["finish_reason"] == "length",
        "completion_tokens": resp["usage"]["completion_tokens"],
        "reasoning_tokens": resp["usage"].get("completion_tokens_details", {}).get("reasoning_tokens"),
        "has_answer": bool(content.strip()),
        **repetition_stats(reasoning),
        "answer_fr_score": lang_score(content),
        "seconds": round(time.time() - t0, 1),
        "system": prompt["system"],
        "user": prompt["user"],
        "reasoning": reasoning,
        "content": content,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-0.8b")
    ap.add_argument("--config", choices=SAMPLING, default="default")
    ap.add_argument("--prompts", default="data/prompts_fr.jsonl")
    ap.add_argument("--out", required=True)
    # LM Studio shares its context across parallel slots: workers x (max_tokens + prompt) must fit
    # (model loaded with 66816 ctx / 4 slots).
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--seeds", type=int, default=1, help="samples per prompt")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    prompts = [json.loads(l) for l in Path(args.prompts).read_text().splitlines() if l.strip()]
    out = Path(args.out)
    done = set()
    if out.exists():
        done = {(r["id"], r["seed"]) for r in map(json.loads, out.read_text().splitlines())}
    jobs = [(p, s) for p in prompts for s in range(args.seeds) if (p["id"], s) not in done]
    print(f"{len(jobs)} generations to run ({len(done)} already done) -> {out}", flush=True)

    sampling = SAMPLING[args.config]
    with ThreadPoolExecutor(args.workers) as pool, out.open("a") as f:
        futures = [pool.submit(run_one, args.model, p, sampling, args.max_tokens, s) for p, s in jobs]
        for i, fut in enumerate(futures, 1):
            try:
                r = fut.result()
            except Exception as e:  # keep going; rerun resumes missing items
                body = e.read().decode()[:300] if hasattr(e, "read") else ""
                print(f"[{i}/{len(jobs)}] FAILED: {e} {body}", flush=True)
                continue
            r["config"] = args.config
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(jobs)}] {r['id']} s{r['seed']} finish={r['finish']} tok={r['completion_tokens']} dup={r['dup_line_ratio']}", flush=True)


if __name__ == "__main__":
    main()
