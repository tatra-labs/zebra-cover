"""How fast is the pure-Python Algorithm C on gold encodings, by puzzle size? Writes results/solver_speed.json.

    python experiments/solver_speed.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zebracover import generate  # noqa: E402

SIZES = [(2, 2), (3, 3), (4, 4), (5, 5), (5, 6), (6, 6)]


def main(seeds: int = 5):
    out = []
    for n, m in SIZES:
        times, clues = [], []
        for seed in range(seeds):
            p = generate(n, m, seed)
            z = p.zebra()
            start = time.perf_counter()
            verdict, _ = z.status()
            times.append(time.perf_counter() - start)
            assert verdict == "unique"
            clues.append(len(p.calls))
        out.append({"size": f"{n}x{m}", "clues": median(clues), "median_ms": median(times) * 1000,
                    "max_ms": max(times) * 1000})
        print(f"{n}x{m}: {median(clues):.0f} clues, median {median(times) * 1000:.1f} ms, "
              f"max {max(times) * 1000:.1f} ms")
    Path("results").mkdir(exist_ok=True)
    Path("results/solver_speed.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
