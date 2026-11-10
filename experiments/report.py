"""Summarise results/runs/*.jsonl into results/summary.{json,md} and figures/accuracy.png.

``nec`` (no feedback) is read off the first round of each ``nec+feedback`` attempt: the loop stops as soon as
the solver returns one solution, so a first-round ``unique`` verdict is exactly what a one-shot run would get.
``nec-lenient`` re-reads that same first reply with the near-miss tool-name aliases of ``diagnose.py``.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from diagnose import lenient, puzzle_for
from zebracover import grade

SIZES = ["2x3", "3x3", "3x4", "4x4", "4x5", "5x5"]
METHODS = ["plain", "nec", "nec+feedback", "nec-lenient"]
COLORS = {"plain": "#2a78d6", "nec": "#eda100", "nec+feedback": "#1baf7a", "nec-lenient": "#8a5cd1"}


def load(runs="results/runs"):
    rows, seen = defaultdict(list), set()  # (model, method) -> attempts
    for path in sorted(Path(runs).glob("*.jsonl")):
        for line in path.open(encoding="utf-8"):
            att = json.loads(line)
            if (att["model"], att["method"], att["puzzle"]) in seen:  # two processes can meet on one puzzle
                continue
            seen.add((att["model"], att["method"], att["puzzle"]))
            if att["method"] == "plain":
                rows[(att["model"], "plain")].append(att)
                continue
            rows[(att["model"], "nec+feedback")].append(att)
            first = dict(att, rounds=att["rounds"][:1])
            if att["rounds"][0]["verdict"] != "unique":
                first["solved"], first["cell_accuracy"] = False, 0.0
            rows[(att["model"], "nec")].append(first)
            verdict, grid, _ = lenient(att["rounds"][0]["reply"], None)
            solved, cells = grade(puzzle_for(att["puzzle"]), grid) if verdict == "unique" else (False, 0.0)
            rows[(att["model"], "nec-lenient")].append(dict(first, method="nec-lenient", solved=solved,
                                                            cell_accuracy=cells))
    return rows


def summarize(rows):
    out = []
    for (model, method), atts in sorted(rows.items(), key=lambda kv: (kv[0][0], METHODS.index(kv[0][1]))):
        by_size = defaultdict(list)
        for a in atts:
            by_size[a["size"]].append(a)
        first = Counter(a["rounds"][0]["verdict"] for a in atts) if method != "plain" else {}
        out.append({
            "model": model, "method": method, "puzzles": len(atts),
            "solved": mean(a["solved"] for a in atts), "cells": mean(a["cell_accuracy"] for a in atts),
            "by_size": {s: mean(a["solved"] for a in by_size[s]) for s in SIZES if by_size[s]},
            "rounds": mean(len(a["rounds"]) for a in atts),
            "tokens": mean(a.get("tokens", 0) for a in atts) if method in ("plain", "nec+feedback") else None,
            "first_round_verdicts": dict(first),
        })
    return out


def markdown(summary):
    lines = ["| model | method | puzzles solved | cell accuracy | " + " | ".join(SIZES) + " | mean rounds |",
             "|---|---|---:|---:|" + "---:|" * len(SIZES) + "---:|"]
    for r in summary:
        sizes = " | ".join(f"{r['by_size'][s]:.0%}" if s in r["by_size"] else "" for s in SIZES)
        lines.append(f"| {r['model']} | {r['method']} | {r['solved']:.1%} | {r['cells']:.1%} | {sizes} | "
                     f"{r['rounds']:.2f} |")
    lines += ["", "First-round verdicts of the exact-cover loop:", "",
              "| model | unique | under | over | count | invalid | none |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        if r["method"] == "nec+feedback":
            v = r["first_round_verdicts"]
            lines.append(f"| {r['model']} | " + " | ".join(str(v.get(k, 0)) for k in
                                                           ("unique", "under", "over", "count", "invalid", "none"))
                         + " |")
    return "\n".join(lines) + "\n"


def figure(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#888888", "xtick.color": "#555555", "ytick.color": "#555555"})
    models = sorted({r["model"] for r in summary})
    fig, axes = plt.subplots(1, len(models), figsize=(5.6 * len(models), 3.8), sharey=True, squeeze=False)
    width = 0.21
    for ax, model in zip(axes[0], models):
        for k, method in enumerate(METHODS):
            r = next((r for r in summary if r["model"] == model and r["method"] == method), None)
            if r is None:
                continue
            xs = [i + (k - 1.5) * width for i in range(len(SIZES))]
            ys = [r["by_size"].get(s, 0) * 100 for s in SIZES]
            ax.bar(xs, ys, width * 0.92, color=COLORS[method], label=f"{method} ({r['solved']:.0%} overall)")
        ax.set_xticks(range(len(SIZES)))
        ax.set_xticklabels(SIZES)
        ax.set_xlabel("puzzle size (houses x attributes)")
        ax.set_title(model, fontsize=10, color="#333333")
        ax.grid(True, axis="y", color="#e6e6e6", lw=0.6)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    axes[0][0].set_ylabel("puzzles solved (%)")
    axes[0][0].set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def main():
    rows = load()
    summary = summarize(rows)
    Path("results/summary.json").write_text(json.dumps(summary, indent=1))
    md = markdown(summary)
    Path("results/summary.md").write_text(md)
    print(md)
    Path("figures").mkdir(exist_ok=True)
    figure(summary, "figures/accuracy.png")


if __name__ == "__main__":
    main()
