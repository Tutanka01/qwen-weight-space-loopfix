"""Helpers on LM Studio traces: discovery/test split, prompt rebuilding and key token positions."""

import json
import re
from pathlib import Path

# A line where the model restarts its reasoning ("Wait, ...", "Okay, final decision", "Actually", ...)
RESTART = re.compile(r"(?m)^[\s*\-\d.()]*(Wait|Okay|Actually|Let's|Let me|Hmm|No,|But wait|Final)")


def is_discovery(pid: str) -> bool:
    """Odd prompt ids are used for analysis; even ids are held out for evaluation."""
    return int(pid[1:]) % 2 == 1


def load_rows(paths):
    return [json.loads(l) for p in paths for l in Path(p).read_text().splitlines() if l.strip()]


def build_text(tok, r):
    """Rebuild exactly what the model saw/generated: chat prompt (thinking on) + reasoning (+ </think>)."""
    msgs = ([{"role": "system", "content": r["system"]}] if r["system"] else []) + [{"role": "user", "content": r["user"]}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    text = prompt + r["reasoning"]
    if not r["truncated"]:
        text = text.rstrip() + "\n</think>"
    return prompt, text


def positions(tok, r, rng, max_restarts=25, n_neutral=10):
    """Token ids of a trace + positions whose NEXT token is a restart, </think>, or random (neutral)."""
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
