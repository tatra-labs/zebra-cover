"""Where do the tool-call encodings go wrong? Reads the stored replies in results/runs, no model needed.

    python experiments/diagnose.py            # writes results/diagnosis.md

For every first-round reply it scores each clue call against the gold call for the same clue (by meaning, so
``left_of(A, B)`` and ``right_of(B, A)`` count as the same), and sorts the encoding errors into kinds.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zebracover import generate  # noqa: E402
from zebracover.llm import parse_tool_calls  # noqa: E402
from zebracover.nec import grade  # noqa: E402
from zebracover.zebra import CLUE_TOOLS, EncodingError, Zebra, clue_relation  # noqa: E402

FLIPS = {"clue_property_directly_left_of_property": "clue_property_directly_right_of_property",
         "clue_property_directly_right_of_property": "clue_property_directly_left_of_property",
         "clue_property_left_of": "clue_property_right_of", "clue_property_right_of": "clue_property_left_of"}


def meaning(call, n):
    """A canonical form of one clue call: equal forms constrain the houses identically."""
    name, a = call.get("name"), call.get("arguments") or {}
    if name not in CLUE_TOOLS:
        return None
    if name in ("clue_property_at_house", "clue_not_property_at_house"):
        try:
            return name, a.get("attribute"), int(a.get("house"))
        except (TypeError, ValueError):
            return None
    x, y = a.get("attribute_1"), a.get("attribute_2")
    return frozenset(frozenset({(x, i), (y, j)}) for i, j in clue_relation(name, n))


# Near-miss names the models actually wrote, mapped to the closest real tool by name. ``*_of_property`` without
# ``directly`` goes to the non-adjacent tool, which is the weaker reading.
ALIASES = {"clue_property_left_of_property": "clue_property_left_of",
           "clue_property_right_of_property": "clue_property_right_of",
           "clue_property_directly_left_of": "clue_property_directly_left_of_property",
           "clue_property_directly_right_of": "clue_property_directly_right_of_property",
           "clue_property_property": "clue_property_at_property", "clue_property_equal": "clue_property_at_property",
           "clue_property_same_property": "clue_property_at_property",
           "clue_property_one_between_two_properties": "clue_one_between_two_properties",
           "clue_property_not_at_house": "clue_not_property_at_house"}


def lenient(reply, prev_init):
    """Re-encode a stored reply with tool-name aliases, reusing the last round's init_zebra if this one has none.
    Returns ``(verdict, grid, init)``."""
    calls, _ = parse_tool_calls(reply)
    inits = [c for c in calls if c["name"] == "init_zebra"]
    init = inits[0] if len(inits) == 1 else prev_init
    if init is None:
        return "none", None, None
    try:
        a = init["arguments"]
        z = Zebra.init_zebra(a["number_of_places"], a["number_of_clues"], a["properties"])
        for c in calls:
            if c["name"] != "init_zebra":
                z.call(ALIASES.get(c["name"], c["name"]), c.get("arguments") or {})
    except (EncodingError, KeyError, TypeError, ValueError, AttributeError):
        return "invalid", None, init
    if len(z.clues) != z.expected_clues:
        return "count", None, init
    verdict, grid = z.status()
    return verdict, grid, init


def kind_of(error):
    if "unknown tool" in error:
        return "unknown tool name"
    if "unknown attribute" in error:
        return "attribute not declared in init_zebra"
    if "init_zebra" in error:
        return "init_zebra missing, repeated or malformed"
    if "missing" in error:
        return "argument missing"
    if "house" in error:
        return "bad house number"
    if "both attributes" in error:
        return "same attribute twice"
    return "other: " + error[:60]


def puzzle_for(pid, cache={}):
    if pid not in cache:
        n, m, seed = map(int, re.match(r"zc-(\d+)x(\d+)-(\d+)", pid).groups())
        cache[pid] = generate(n, m, seed)
    return cache[pid]


def main():
    out = ["# Where the encodings go wrong", "",
           "Produced by `python experiments/diagnose.py` from the stored replies in `results/runs/`.", ""]
    for path in sorted(Path("results/runs").glob("*__nec+feedback.jsonl")):
        seen, atts = set(), []
        for line in path.open(encoding="utf-8"):
            a = json.loads(line)
            if a["puzzle"] not in seen:
                seen.add(a["puzzle"])
                atts.append(a)
        model = atts[0]["model"]
        kinds, per_size = Counter(), defaultdict(lambda: [0, 0])
        right = right_alias = flipped = total = 0
        lenient_first = lenient_any = lenient_cells = 0
        lenient_size = Counter()
        later = later_no_init = 0
        for a in atts:
            p = puzzle_for(a["puzzle"])
            calls, errors = parse_tool_calls(a["rounds"][0]["reply"])
            clues = [c for c in calls if c["name"] != "init_zebra"]
            inits = [c for c in calls if c["name"] == "init_zebra"]
            if len(inits) != 1:
                errors.append("init_zebra count")
            declared = set()
            if len(inits) == 1:
                for vs in (inits[0].get("arguments") or {}).get("properties", {}).values():
                    declared |= set(vs) if isinstance(vs, list) else set()
            for c in clues:
                if c["name"] not in CLUE_TOOLS:
                    errors.append(f"unknown tool {c['name']!r}")
                for k, v in (c.get("arguments") or {}).items():
                    if k.startswith("attribute") and declared and v not in declared:
                        errors.append(f"unknown attribute {v!r}")
            for k in {kind_of(e) for e in errors}:
                kinds[k] += 1
            for i, gold in enumerate(p.calls):
                total += 1
                per_size[a["size"]][1] += 1
                if i < len(clues):
                    got = meaning(clues[i], p.houses)
                    ok = got is not None and got == meaning(gold, p.houses)
                    right += ok
                    per_size[a["size"]][0] += ok
                    aliased = dict(clues[i], name=ALIASES.get(clues[i].get("name"), clues[i].get("name")))
                    right_alias += meaning(aliased, p.houses) == meaning(gold, p.houses)
                    if not ok and clues[i].get("name") == FLIPS.get(gold["name"]) and \
                            clues[i].get("arguments") == gold["arguments"]:
                        flipped += 1
            init, best = None, 0.0
            for k, r in enumerate(a["rounds"]):
                verdict, grid, init = lenient(r["reply"], init)
                solved, cells = grade(p, grid) if verdict == "unique" else (False, 0.0)
                lenient_first += solved and k == 0
                lenient_size[a["size"]] += solved and k == 0
                best = max(best, cells)
                if solved:
                    break
            lenient_any += best == 1.0
            lenient_cells += best
            for r in a["rounds"][1:]:
                later += 1
                later_no_init += not any(c["name"] == "init_zebra" for c in parse_tool_calls(r["reply"])[0])
        out += [f"## {model}", "", f"{len(atts)} puzzles, first round.", ""]
        if not total or not any(a["rounds"][0]["reply"].strip() for a in atts):
            out += ["No replies to score.", ""]
            continue
        out += [f"- clue calls that mean the same as the gold call for that clue: **{right}/{total} "
                f"({right / total:.0%})**",
                f"- the same, after mapping near-miss tool names to the closest real tool: {right_alias}/{total} "
                f"({right_alias / total:.0%})",
                f"- of the wrong ones, the right relation with left and right swapped: {flipped}",
                "- by puzzle size: " + ", ".join(f"{s} {r}/{t}" for s, (r, t) in sorted(per_size.items())),
                f"- feedback-round replies with no `init_zebra` call (so the whole reply is rejected): "
                f"{later_no_init}/{later}", "",
                "Re-scoring the stored replies with a lenient reader (tool-name aliases, and a feedback round "
                "with no `init_zebra` reuses the previous round's): "
                f"first round solves **{lenient_first}/{len(atts)}** ("
                + ", ".join(f"{s} {lenient_size[s]}" for s in sorted({a['size'] for a in atts})) + "), "
                f"any round solves {lenient_any}/{len(atts)}. The first-round number is exact for a one-shot "
                "run with that reader, since the first reply does not depend on feedback. The any-round number "
                "is not a rerun: with a different verdict the model would have been shown different feedback.",
                "",
                "Puzzles whose first reply has at least one error of each kind:", "",
                "| kind | puzzles |", "|---|---:|"]
        out += [f"| {k} | {v} |" for k, v in kinds.most_common()]
        out.append("")
    Path("results/diagnosis.md").write_text("\n".join(out), encoding="utf-8", newline="\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
