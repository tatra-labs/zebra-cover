"""Run a local model on the benchmark set, plainly and through the exact-cover loop. Resumable.

    python experiments/run_llm.py --model Qwen/Qwen3-0.6B
    python experiments/run_llm.py --model Qwen/Qwen3-1.7B --methods nec+feedback --threads 16

``nec`` (one shot, no feedback) is the first round of ``nec+feedback``, so only the latter is run and the
report derives both from it. Every reply is stored, so verdicts can be re-checked without the model.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zebracover import TransformersBackend, generate, solve_nec, solve_plain  # noqa: E402

SIZES = [(2, 3), (3, 3), (3, 4), (4, 4), (4, 5), (5, 5)]


def benchmark(per_size: int):
    return [generate(n, m, seed) for n, m in SIZES for seed in range(per_size)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--methods", nargs="+", default=["plain", "nec+feedback"])
    ap.add_argument("--per-size", type=int, default=10)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--max-rounds", type=int, default=4)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--reverse", action="store_true", help="largest puzzles first, to split work across processes")
    ap.add_argument("--out", default="results/runs")
    a = ap.parse_args()
    backend = TransformersBackend(a.model, thinking=a.thinking, max_new_tokens=a.max_new_tokens, threads=a.threads)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for method in a.methods:
        path = Path(a.out) / f"{backend.name}__{method}.jsonl"
        puzzles = benchmark(a.per_size)
        for puzzle in reversed(puzzles) if a.reverse else puzzles:
            done = {json.loads(line)["puzzle"] for line in path.open()} if path.exists() else set()
            if puzzle.id in done:
                continue
            start, tokens = time.perf_counter(), backend.generated_tokens
            if method == "plain":
                att = solve_plain(puzzle, backend)
            else:
                att = solve_nec(puzzle, backend, feedback=True, max_rounds=a.max_rounds)
            record = att.to_json() | {"seconds": time.perf_counter() - start,
                                      "tokens": backend.generated_tokens - tokens}
            with path.open("a") as f:
                f.write(json.dumps(record) + "\n")
            verdicts = ",".join(r["verdict"] for r in att.rounds)
            print(f"{backend.name:14s} {method:13s} {puzzle.id:14s} solved={att.solved!s:5s} "
                  f"cells={att.cell_accuracy:.2f} [{verdicts}] {record['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
