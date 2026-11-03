"""Summarize measurement files: uv run scripts/eval/summarize.py results/baseline/*.jsonl"""

import json
import sys
from pathlib import Path
from statistics import mean, median


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "n": n,
        "truncated_%": round(100 * sum(r["truncated"] for r in rows) / n, 1),
        "answered_%": round(100 * sum(r["has_answer"] for r in rows) / n, 1),
        "repetitive_%": round(100 * sum(r["dup_line_ratio"] >= 0.2 for r in rows) / n, 1),
        "median_tokens": int(median(r["completion_tokens"] for r in rows)),
        "mean_tokens": int(mean(r["completion_tokens"] for r in rows)),
    }


def main():
    print(f"{'file':<28}{'subset':<9}{'n':>4}{'trunc%':>8}{'answ%':>7}{'repet%':>8}{'med_tok':>9}{'mean_tok':>9}")
    for path in sys.argv[1:]:
        rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
        for subset in ("all", "persona", "plain"):
            sub = rows if subset == "all" else [r for r in rows if r["cat"] == subset]
            if not sub:
                continue
            s = summarize(sub)
            print(f"{Path(path).stem:<28}{subset:<9}{s['n']:>4}{s['truncated_%']:>8}{s['answered_%']:>7}"
                  f"{s['repetitive_%']:>8}{s['median_tokens']:>9}{s['mean_tokens']:>9}")


if __name__ == "__main__":
    main()
