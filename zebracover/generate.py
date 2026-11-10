"""Fresh Zebra puzzles with exactly one solution, phrased like ZebraLogic, each with its gold tool calls.

Draw a random solution, add true clues at random until the cover has one solution, then drop every clue
that is not needed (in random order). Uniqueness is checked by the solver, not assumed.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, asdict

from .zebra import Zebra, clue_relation

GROUPS = {
    "Name": (["Alice", "Eric", "Peter", "Arnold", "Bob", "Carol", "Daniel", "Grace"], "{v}"),
    "Color": (["red", "green", "blue", "yellow", "white", "purple", "orange", "black"],
              "the person who loves {v}"),
    "Pet": (["cat", "dog", "bird", "fish", "horse", "rabbit", "hamster", "turtle"], "the person who keeps a {v}"),
    "Drink": (["tea", "coffee", "milk", "water", "juice", "soda", "lemonade", "cocoa"],
              "the person who drinks {v}"),
    "Nationality": (["Dane", "Brit", "German", "Swede", "Norwegian", "Spaniard", "Italian", "Japanese"],
                    "the {v}"),
    "Hobby": (["painting", "gardening", "cooking", "chess", "hiking", "knitting", "photography", "fishing"],
              "the person who enjoys {v}"),
    "Food": (["pizza", "sushi", "tacos", "pasta", "curry", "salad", "soup", "stew"], "the person who eats {v}"),
    "Car": (["Ford", "Tesla", "BMW", "Toyota", "Honda", "Audi", "Kia", "Fiat"], "the person who drives a {v}"),
}
INTRO = {
    "Name": "Each person has a unique name", "Color": "Each person has a unique favorite color",
    "Pet": "Each person keeps a different pet", "Drink": "Each person has a different favorite drink",
    "Nationality": "Each person has a different nationality", "Hobby": "Each person has a different hobby",
    "Food": "Each person has a different favorite food", "Car": "Each person drives a different car",
}
ORDINAL = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]
TEMPLATES = {
    "clue_property_at_house": "{A} lives in the {H} house.",
    "clue_not_property_at_house": "{A} does not live in the {H} house.",
    "clue_property_at_property": "{A} is {B}.",
    "clue_property_next_to_property": "{A} and {B} are next to each other.",
    "clue_property_left_of": "{A} is somewhere to the left of {B}.",
    "clue_property_right_of": "{A} is somewhere to the right of {B}.",
    "clue_property_directly_left_of_property": "{A} is directly left of {B}.",
    "clue_property_directly_right_of_property": "{A} is directly right of {B}.",
    "clue_one_between_two_properties": "There is one house between {A} and {B}.",
    "clue_two_between_two_properties": "There are two houses between {A} and {B}.",
}
WEIGHTS = {"clue_property_at_house": 1, "clue_not_property_at_house": 1, "clue_property_at_property": 4,
           "clue_property_next_to_property": 2, "clue_property_left_of": 2, "clue_property_right_of": 2,
           "clue_property_directly_left_of_property": 2, "clue_property_directly_right_of_property": 2,
           "clue_one_between_two_properties": 1, "clue_two_between_two_properties": 1}


def attr(group: str, value: str) -> str:
    return f"{group}_{value}"


@dataclass
class Puzzle:
    id: str
    houses: int
    groups: dict[str, list[str]]
    clues: list[str]
    calls: list[dict]
    solution: dict[str, dict[str, str]]

    @property
    def size(self) -> str:
        return f"{self.houses}x{len(self.groups)}"

    def text(self) -> str:
        n = self.houses
        lines = [f"There are {n} houses, numbered 1 to {n} from left to right, as seen from across the street. "
                 "Each house is occupied by a different person. Each house has a unique attribute for each of "
                 "the following characteristics:"]
        for g, values in self.groups.items():
            lines.append(f" - {INTRO[g]}: " + ", ".join(f"`{v}`" for v in values))
        lines.append("")
        lines.append("## Clues:")
        lines += [f"{i}. {c}" for i, c in enumerate(self.clues, 1)]
        return "\n".join(lines)

    def zebra(self) -> Zebra:
        z = Zebra.init_zebra(self.houses, len(self.calls),
                             {g: [attr(g, v) for v in vs] for g, vs in self.groups.items()})
        for c in self.calls:
            z.call(c["name"], c["arguments"])
        return z

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "Puzzle":
        return cls(**d)


def describe(group: str, value: str, capital: bool) -> str:
    s = GROUPS[group][1].format(v=value)
    return s[0].upper() + s[1:] if capital else s


def true_clues(houses, where, rng):
    """Every clue that holds in the solution ``where`` (attribute -> (group, value, house)), shuffled."""
    out = []
    items = list(where)
    for a in items:
        g, v, h = where[a]
        out.append(("clue_property_at_house", {"attribute": a, "house": h}))
        for other in range(1, houses + 1):
            if other != h:
                out.append(("clue_not_property_at_house", {"attribute": a, "house": other}))
    for a in items:
        for b in items:
            if a == b:
                continue
            ga, _, ha = where[a]
            gb, _, hb = where[b]
            for name in WEIGHTS:
                if name in ("clue_property_at_house", "clue_not_property_at_house"):
                    continue
                if name == "clue_property_at_property" and (ga == gb or a > b):
                    continue
                if name in ("clue_property_next_to_property", "clue_one_between_two_properties",
                            "clue_two_between_two_properties") and a > b:
                    continue
                if (ha, hb) in clue_relation(name, houses):
                    out.append((name, {"attribute_1": a, "attribute_2": b}))
    weighted = [c for c in out for _ in range(WEIGHTS[c[0]])]
    rng.shuffle(weighted)
    seen, uniq = set(), []
    for name, args in weighted:
        key = (name, tuple(sorted(args.items())))
        if key not in seen:
            seen.add(key)
            uniq.append((name, args))
    return uniq


def phrase(name, args, where, rng):
    if "attribute" in args:
        g, v, _ = where[args["attribute"]]
        return TEMPLATES[name].format(A=describe(g, v, True), H=ORDINAL[args["house"] - 1])
    (g1, v1, _), (g2, v2, _) = where[args["attribute_1"]], where[args["attribute_2"]]
    symmetric = name in ("clue_property_at_property", "clue_property_next_to_property",
                         "clue_one_between_two_properties", "clue_two_between_two_properties")
    if symmetric and rng.random() < 0.5:
        (g1, v1), (g2, v2) = (g2, v2), (g1, v1)
    capital = not TEMPLATES[name].startswith("There")
    return TEMPLATES[name].format(A=describe(g1, v1, capital), B=describe(g2, v2, False))


def generate(houses: int, groups: int, seed: int) -> Puzzle:
    rng = random.Random(f"{houses}x{groups}-{seed}")
    names = rng.sample(list(GROUPS), groups)
    if "Name" not in names and rng.random() < 0.7:
        names[0] = "Name"
    chosen = {g: rng.sample(GROUPS[g][0], houses) for g in names}
    display = {g: sorted(vs) for g, vs in chosen.items()}
    where = {attr(g, v): (g, v, h) for g, vs in chosen.items() for h, v in enumerate(vs, 1)}
    groups_attr = {g: [attr(g, v) for v in vs] for g, vs in display.items()}

    def status(clues):
        z = Zebra(houses, groups_attr)
        for name, args in clues:
            z.call(name, args)
        return len(z.solutions(limit=2))

    pool = true_clues(houses, where, rng)
    clues: list = []
    for clue in pool:
        clues.append(clue)
        if status(clues) == 1:
            break
    for clue in rng.sample(clues, len(clues)):
        trial = [c for c in clues if c is not clue]
        if status(trial) == 1:
            clues = trial
    rng.shuffle(clues)
    solution = {str(h): {} for h in range(1, houses + 1)}
    for g, vs in chosen.items():
        for h, v in enumerate(vs, 1):
            solution[str(h)][g] = v
    return Puzzle(id=f"zc-{houses}x{groups}-{seed:03d}", houses=houses, groups=display,
                  clues=[phrase(n, a, where, rng) for n, a in clues],
                  calls=[{"name": n, "arguments": a} for n, a in clues], solution=solution)
