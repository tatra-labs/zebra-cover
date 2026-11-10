"""Command line: ``python -m zebracover <command>``.

    generate --size 4x4 --seed 3          print a fresh puzzle (add --json for the record with gold calls)
    gold --size 5x5 --seed 0              encode a puzzle from its gold calls, solve it, print the grid
    encode reply.txt --size 4x4 --seed 3  check a model's reply (tool calls) against that puzzle
    solve-openai --base-url URL --model M [--sizes 3x3 4x4] [--per-size 5]
                                          run the loop against any OpenAI-compatible endpoint (costs money
                                          on hosted APIs; nothing in this repository has run it)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from . import OpenAIBackend, calls_to_text, check, generate, grade, solve_nec, solve_plain


def size(text: str) -> tuple[int, int]:
    n, m = text.lower().split("x")
    return int(n), int(m)


def gold_text(p) -> str:
    init = {"name": "init_zebra", "arguments": {
        "number_of_places": p.houses, "number_of_clues": len(p.calls),
        "properties": {g: [f"{g}_{v}" for v in vs] for g, vs in p.groups.items()}}}
    return calls_to_text([init] + p.calls)


def show_grid(p, grid) -> str:
    groups = list(p.groups)
    rows = [["House"] + groups]
    for h in sorted(grid, key=int):
        row = grid[h]
        rows.append([str(h)] + [str(row.get(g, "?")).split("_", 1)[-1] for g in groups])
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in rows)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="zebracover")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--size", type=size, default=(4, 4))
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--json", action="store_true")
    o = sub.add_parser("gold")
    o.add_argument("--size", type=size, default=(4, 4))
    o.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("encode")
    e.add_argument("reply")
    e.add_argument("--size", type=size, default=(4, 4))
    e.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("solve-openai")
    s.add_argument("--base-url", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    s.add_argument("--sizes", type=size, nargs="+", default=[(3, 3), (4, 4), (5, 5)])
    s.add_argument("--per-size", type=int, default=5)
    s.add_argument("--plain", action="store_true", help="also run the plain baseline")
    a = ap.parse_args(argv)

    if a.cmd == "generate":
        p = generate(*a.size, a.seed)
        print(json.dumps(p.to_json(), indent=1) if a.json else p.text())
    elif a.cmd == "gold":
        p = generate(*a.size, a.seed)
        start = time.perf_counter()
        verdict, _, grid = check(gold_text(p))
        ms = (time.perf_counter() - start) * 1000
        print(p.text(), f"\n\nsolver: {verdict} in {ms:.1f} ms\n")
        print(show_grid(p, {str(h): r for h, r in grid.items()}))
    elif a.cmd == "encode":
        p = generate(*a.size, a.seed)
        verdict, feedback, grid = check(open(a.reply, encoding="utf-8").read())
        print(f"verdict: {verdict}")
        if grid:
            solved, acc = grade(p, {str(h): r for h, r in grid.items()})
            print(show_grid(p, {str(h): r for h, r in grid.items()}), f"\ncorrect: {solved} (cells {acc:.2f})")
        else:
            print(feedback)
    else:
        backend = OpenAIBackend(a.base_url, a.model, a.api_key)
        for n, m in a.sizes:
            for seed in range(a.per_size):
                p = generate(n, m, seed)
                for att in ([solve_plain(p, backend)] if a.plain else []) + [solve_nec(p, backend)]:
                    print(json.dumps({k: v for k, v in att.to_json().items() if k != "rounds"}
                                     | {"verdicts": [r["verdict"] for r in att.rounds]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
