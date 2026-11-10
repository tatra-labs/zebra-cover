"""A Zebra puzzle as an exact-cover-with-colours instance, built through the tool calls an LLM would make.

Encoding (after NeuralExactCover, arXiv 2611.03641, written from its description):

* primary ``V:<attr>`` for every attribute value — each value sits in exactly one house;
* primary ``H:<house>:<group>`` — each house has exactly one value of every group;
* primary ``C:<k>`` for the k-th clue — every clue is satisfied by exactly one of its options;
* secondary ``P:<attr>``, coloured by the house the value is in.

An assignment option is ``V:a  H:h:g  P:a=h``. A clue option names the houses it needs, as colours of ``P``
items, and Algorithm C keeps only the options whose colours agree with the assignment. Because every clue
fixes the houses of the attributes it mentions, each solution of the puzzle is exactly one solution of the
cover, so counting cover solutions counts puzzle solutions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .exact_cover import ExactCover

CLUE_TOOLS = {
    "clue_property_at_house": ("attribute", "house"),
    "clue_not_property_at_house": ("attribute", "house"),
    "clue_property_at_property": ("attribute_1", "attribute_2"),
    "clue_property_next_to_property": ("attribute_1", "attribute_2"),
    "clue_property_left_of": ("attribute_1", "attribute_2"),
    "clue_property_right_of": ("attribute_1", "attribute_2"),
    "clue_property_directly_left_of_property": ("attribute_1", "attribute_2"),
    "clue_property_directly_right_of_property": ("attribute_1", "attribute_2"),
    "clue_one_between_two_properties": ("attribute_1", "attribute_2"),
    "clue_two_between_two_properties": ("attribute_1", "attribute_2"),
}


def clue_relation(name: str, n: int):
    """The (house_1, house_2) pairs a two-attribute clue allows, houses numbered 1..n."""
    rel = {
        "clue_property_at_property": lambda a, b: a == b,
        "clue_property_next_to_property": lambda a, b: abs(a - b) == 1,
        "clue_property_left_of": lambda a, b: a < b,
        "clue_property_right_of": lambda a, b: a > b,
        "clue_property_directly_left_of_property": lambda a, b: b == a + 1,
        "clue_property_directly_right_of_property": lambda a, b: a == b + 1,
        "clue_one_between_two_properties": lambda a, b: abs(a - b) == 2,
        "clue_two_between_two_properties": lambda a, b: abs(a - b) == 3,
    }[name]
    return [(a, b) for a in range(1, n + 1) for b in range(1, n + 1) if rel(a, b)]


class EncodingError(ValueError):
    """A tool call that cannot be encoded: unknown tool, unknown attribute, house out of range."""


@dataclass
class Zebra:
    houses: int
    groups: dict[str, list[str]]
    expected_clues: int | None = None
    clues: list[tuple[str, dict]] = field(default_factory=list)

    def __post_init__(self):
        self.group_of = {}
        for g, values in self.groups.items():
            if len(values) != self.houses:
                raise EncodingError(f"group {g!r} has {len(values)} values for {self.houses} houses")
            for v in values:
                if v in self.group_of:
                    raise EncodingError(f"attribute {v!r} appears in two groups")
                self.group_of[v] = g

    # ---- the tool-call surface -----------------------------------------------------------------------------
    @classmethod
    def init_zebra(cls, number_of_places: int, number_of_clues: int, properties: dict[str, list[str]]) -> "Zebra":
        return cls(int(number_of_places), {str(k): [str(v) for v in vs] for k, vs in properties.items()},
                   int(number_of_clues))

    def call(self, name: str, arguments: dict) -> None:
        """Apply one clue tool call, validating it."""
        if name not in CLUE_TOOLS:
            raise EncodingError(f"unknown tool {name!r}; the clue tools are {', '.join(CLUE_TOOLS)}")
        params = CLUE_TOOLS[name]
        missing = [p for p in params if p not in arguments]
        if missing:
            raise EncodingError(f"{name} is missing {', '.join(missing)}")
        args = {p: arguments[p] for p in params}
        for p in params:
            if p.startswith("attribute"):
                if args[p] not in self.group_of:
                    raise EncodingError(f"{name}: unknown attribute {args[p]!r}")
            else:
                try:
                    args[p] = int(args[p])
                except (TypeError, ValueError):
                    raise EncodingError(f"{name}: house must be an integer, got {args[p]!r}") from None
                if not 1 <= args[p] <= self.houses:
                    raise EncodingError(f"{name}: house {args[p]} is outside 1..{self.houses}")
        if len(params) == 2 and params[0].startswith("attribute") and args[params[0]] == args[params[1]]:
            raise EncodingError(f"{name}: both attributes are {args[params[0]]!r}")
        self.clues.append((name, args))

    # ---- solving -------------------------------------------------------------------------------------------
    def cover(self) -> tuple[ExactCover, list]:
        n = self.houses
        attrs = list(self.group_of)
        primary = [f"V:{a}" for a in attrs]
        primary += [f"H:{h}:{g}" for h in range(1, n + 1) for g in self.groups]
        primary += [f"C:{k}" for k in range(len(self.clues))]
        ec = ExactCover(primary, [f"P:{a}" for a in attrs])
        meaning = []
        for a in attrs:
            for h in range(1, n + 1):
                ec.add([f"V:{a}", f"H:{h}:{self.group_of[a]}", (f"P:{a}", h)])
                meaning.append(("place", a, h))
        for k, (name, args) in enumerate(self.clues):
            if name == "clue_property_at_house":
                ec.add([f"C:{k}", (f"P:{args['attribute']}", args["house"])])
                meaning.append(("clue", k))
            elif name == "clue_not_property_at_house":
                for h in range(1, n + 1):
                    if h != args["house"]:
                        ec.add([f"C:{k}", (f"P:{args['attribute']}", h)])
                        meaning.append(("clue", k))
            else:
                a1, a2 = args["attribute_1"], args["attribute_2"]
                same_group = self.group_of[a1] == self.group_of[a2]
                for h1, h2 in clue_relation(name, n):
                    if h1 == h2 and same_group:
                        continue  # two values of one group never share a house
                    ec.add([f"C:{k}", (f"P:{a1}", h1), (f"P:{a2}", h2)])
                    meaning.append(("clue", k))
        return ec, meaning

    def solutions(self, limit: int | None = 2) -> list[dict[int, dict[str, str]]]:
        """Up to ``limit`` solutions as ``{house: {group: value}}``."""
        ec, meaning = self.cover()
        out = []
        for sol in ec.solve(limit):
            grid = {h: {} for h in range(1, self.houses + 1)}
            for o in sol:
                if meaning[o][0] == "place":
                    _, a, h = meaning[o]
                    grid[h][self.group_of[a]] = a
            out.append(grid)
        return out

    def status(self) -> tuple[str, dict | None]:
        """``("unique", grid)``, ``("over", None)`` (no solution) or ``("under", None)`` (several)."""
        sols = self.solutions(limit=2)
        if not sols:
            return "over", None
        if len(sols) > 1:
            return "under", None
        return "unique", sols[0]
