import itertools
import json
import random

import pytest

from zebracover import (ExactCover, Puzzle, ScriptedBackend, Zebra, calls_to_text, check, generate, grade,
                        last_json_object, parse_tool_calls, solve_nec, solve_plain)
from zebracover.zebra import CLUE_TOOLS, EncodingError, clue_relation


# --- Algorithm C --------------------------------------------------------------------------------------------

def brute_force(primary, secondary, options):
    sols = []
    for r in range(len(options) + 1):
        for combo in itertools.combinations(range(len(options)), r):
            count, colour, ok = {p: 0 for p in primary}, {}, True
            for k in combo:
                for e in options[k]:
                    name, c = e if isinstance(e, tuple) else (e, None)
                    if name in count:
                        count[name] += 1
                    elif c is None:
                        ok &= name not in colour
                        colour[name] = object()
                    else:
                        ok &= colour.get(name, c) == c
                        colour.setdefault(name, c)
            if ok and all(v == 1 for v in count.values()):
                sols.append(list(combo))
    return sols


def test_exact_cover_matches_brute_force():
    rng = random.Random(0)
    for _ in range(300):
        primary = [f"p{i}" for i in range(rng.randint(1, 5))]
        secondary = [f"s{i}" for i in range(rng.randint(0, 3))]
        ec, options = ExactCover(primary, secondary), []
        for _ in range(rng.randint(1, 9)):
            opt = rng.sample(primary, rng.randint(1, len(primary)))
            for s in secondary:
                if rng.random() < 0.5:
                    c = rng.choice([None, 1, 2])
                    opt.append(s if c is None else (s, c))
            options.append(opt)
            ec.add(opt)
        assert sorted(ec.solve()) == sorted(brute_force(primary, secondary, options))


def test_colour_conflicts():
    # options 0 and 4 disagree on y, option 2 clashes with 3 on x; only 1 + 3 survives
    ec = ExactCover(["p", "q", "r"], ["x", "y"])
    ec.add(["p", "q", ("x", 0), ("y", "A")])
    ec.add(["p", "r", ("x", "A"), "y"])
    ec.add(["p", ("x", "B")])
    ec.add(["q", ("x", "A")])
    ec.add(["r", ("y", "B")])
    assert list(ec.solve()) == [[1, 3]]


def test_solve_limit_and_count():
    ec = ExactCover(["a"])
    for _ in range(5):
        ec.add(["a"])
    assert ec.count() == 5 and ec.count(limit=2) == 2


def test_exact_cover_rejects_bad_options():
    ec = ExactCover(["a"], ["x"])
    with pytest.raises(KeyError):
        ec.add(["b"])
    with pytest.raises(ValueError):
        ec.add(["a", "a"])
    with pytest.raises(ValueError):
        ec.add([("x", 1)])
    with pytest.raises(ValueError):
        ec.add([("a", 1)])


# --- Zebra encoding -----------------------------------------------------------------------------------------

def brute_zebra(z: Zebra):
    groups = list(z.groups)
    count = 0
    for perms in itertools.product(*(itertools.permutations(z.groups[g]) for g in groups)):
        house = {v: h + 1 for perm in perms for h, v in enumerate(perm)}
        ok = True
        for name, args in z.clues:
            if "attribute" in args:
                at = house[args["attribute"]] == args["house"]
                ok &= at if name == "clue_property_at_house" else not at
            else:
                ok &= (house[args["attribute_1"]], house[args["attribute_2"]]) in clue_relation(name, z.houses)
        count += ok
    return count


@pytest.mark.parametrize("seed", range(25))
def test_cover_counts_equal_brute_force(seed):
    rng = random.Random(seed)
    n = rng.choice([2, 3, 4])
    groups = {g: [f"{g}_{i}" for i in range(n)] for g in ("A", "B", "C")[:rng.choice([2, 3])]}
    z = Zebra(n, groups)
    attrs = [a for vs in groups.values() for a in vs]
    for _ in range(rng.randint(0, 5)):
        name = rng.choice(list(CLUE_TOOLS))
        if CLUE_TOOLS[name][1] == "house":
            z.call(name, {"attribute": rng.choice(attrs), "house": rng.randint(1, n)})
        else:
            a, b = rng.sample(attrs, 2)
            z.call(name, {"attribute_1": a, "attribute_2": b})
    assert len(z.solutions(limit=None)) == brute_zebra(z)


def test_relations():
    assert clue_relation("clue_property_directly_left_of_property", 3) == [(1, 2), (2, 3)]
    assert clue_relation("clue_two_between_two_properties", 4) == [(1, 4), (4, 1)]
    assert (2, 1) in clue_relation("clue_property_right_of", 2)


def test_encoding_errors():
    z = Zebra.init_zebra(2, 1, {"Name": ["Name_A", "Name_B"], "Color": ["Color_r", "Color_g"]})
    for name, args in [("clue_magic", {}), ("clue_property_at_house", {"attribute": "Name_Z", "house": 1}),
                       ("clue_property_at_house", {"attribute": "Name_A", "house": 3}),
                       ("clue_property_at_house", {"attribute": "Name_A"}),
                       ("clue_property_at_property", {"attribute_1": "Name_A", "attribute_2": "Name_A"})]:
        with pytest.raises(EncodingError):
            z.call(name, args)
    with pytest.raises(EncodingError):
        Zebra.init_zebra(2, 0, {"Name": ["Name_A"]})


# --- generator ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("size", [(2, 2), (3, 3), (4, 3), (4, 4), (5, 4)])
def test_generated_puzzles_are_unique_and_minimal(size):
    p = generate(*size, seed=7)
    z = p.zebra()
    verdict, grid = z.status()
    assert verdict == "unique"
    assert grade(p, {h: row for h, row in grid.items()}) == (True, 1.0)
    for k in range(len(p.calls)):  # every clue is needed
        partial = Zebra(z.houses, z.groups)
        for c in p.calls[:k] + p.calls[k + 1:]:
            partial.call(c["name"], c["arguments"])
        assert len(partial.solutions(limit=2)) == 2


def test_generation_is_deterministic_and_roundtrips():
    a, b = generate(4, 4, 3), generate(4, 4, 3)
    assert a == b and a != generate(4, 4, 4)
    assert Puzzle.from_json(json.loads(json.dumps(a.to_json()))) == a
    assert len(a.clues) == len(a.calls) and "## Clues:" in a.text()


# --- the loop -----------------------------------------------------------------------------------------------

def gold_calls(p):
    init = {"name": "init_zebra", "arguments": {
        "number_of_places": p.houses, "number_of_clues": len(p.calls),
        "properties": {g: [f"{g}_{v}" for v in vs] for g, vs in p.groups.items()}}}
    return [init] + p.calls


def test_gold_encoding_solves_every_size():
    for size in [(2, 2), (3, 4), (4, 4), (5, 5)]:
        p = generate(*size, seed=1)
        att = solve_nec(p, ScriptedBackend([calls_to_text(gold_calls(p))]))
        assert att.solved and att.cell_accuracy == 1.0 and len(att.rounds) == 1


def test_feedback_repairs_a_dropped_clue():
    p = generate(3, 3, seed=2)
    calls = gold_calls(p)
    dropped = calls[:-1]
    dropped[0] = json.loads(json.dumps(dropped[0]))
    dropped[0]["arguments"]["number_of_clues"] -= 1
    backend = ScriptedBackend([calls_to_text(dropped), calls_to_text(calls)])
    att = solve_nec(p, backend)
    assert [r["verdict"] for r in att.rounds] == ["under", "unique"] and att.solved
    assert not solve_nec(p, ScriptedBackend([calls_to_text(dropped)]), feedback=False).solved


def test_feedback_verdicts():
    p = generate(3, 3, seed=2)
    calls = gold_calls(p)
    assert check("no calls here")[0] == "none"
    assert check(calls_to_text(calls[:-1]))[0] == "count"
    bad = json.loads(json.dumps(calls))
    bad[1]["arguments"] = {k: ("Name_Nobody" if k.startswith("attribute") else v)
                           for k, v in bad[1]["arguments"].items()}
    assert check(calls_to_text(bad))[0] == "invalid"
    contradiction = calls + [{"name": "clue_property_at_house", "arguments": {
        "attribute": calls[0]["arguments"]["properties"][next(iter(p.groups))][0], "house": 1}},
        {"name": "clue_not_property_at_house", "arguments": {
            "attribute": calls[0]["arguments"]["properties"][next(iter(p.groups))][0], "house": 1}}]
    contradiction[0] = json.loads(json.dumps(contradiction[0]))
    contradiction[0]["arguments"]["number_of_clues"] += 2
    assert check(calls_to_text(contradiction))[0] == "over"


def test_parse_tool_calls_handles_noise():
    text = ('<think>maybe <tool_call>{"name": "x"}</tool_call></think> Clue 1.\n<tool_call>\n'
            '{"name": "clue_property_at_house", "arguments": "{\\"attribute\\": \\"A_x\\", \\"house\\": 1}"}\n'
            '</tool_call>\n<tool_call>{oops}</tool_call>')
    calls, errors = parse_tool_calls(text)
    assert calls == [{"name": "clue_property_at_house", "arguments": {"attribute": "A_x", "house": 1}}]
    assert len(errors) == 1


def test_plain_baseline_and_grading():
    p = generate(3, 3, seed=5)
    answer = {f"House {h}": row for h, row in p.solution.items()}
    att = solve_plain(p, ScriptedBackend([f"reasoning... {{not json}} final: {json.dumps(answer)}"]))
    assert att.solved and att.cell_accuracy == 1.0
    wrong = json.loads(json.dumps(answer))
    first = next(iter(p.groups))
    wrong["House 1"][first], wrong["House 2"][first] = wrong["House 2"][first], wrong["House 1"][first]
    solved, acc = grade(p, wrong)
    assert not solved and acc == pytest.approx(1 - 2 / 9)
    assert grade(p, None) == (False, 0.0)


def test_last_json_object():
    assert last_json_object('a {"x": 1} b {"y": {"z": 2}} c {bad') == {"y": {"z": 2}}
    assert last_json_object("none") is None


def test_untagged_calls_are_parsed_in_order():
    text = ('Clue 1: x.\n<tool_call>\n{"name": "a", "arguments": {"k": "{brace}"}}\n</tool_call>\n'
            'Clue 2: y.\n{"name": "b", "arguments": {}}\nnot a call {"x": 1}')
    calls, errors = parse_tool_calls(text)
    assert [c["name"] for c in calls] == ["a", "b"] and calls[0]["arguments"] == {"k": "{brace}"} and not errors
