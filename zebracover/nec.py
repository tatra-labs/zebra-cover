"""The two ways to ask a model for a Zebra solution, and the grader.

* ``solve_plain``: the model reads the puzzle and writes the grid as JSON. The neural baseline.
* ``solve_nec``: the model writes one tool call per clue, the exact-cover solver finds the solution, and
  (optionally) the solver's verdict goes back to the model as feedback: malformed or incomplete calls,
  over-constrained (no solution), under-constrained (several). After the NeuralExactCover loop of
  arXiv 2611.03641.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .generate import Puzzle
from .llm import parse_tool_calls, strip_thinking
from .zebra import CLUE_TOOLS, EncodingError, Zebra

DESCRIPTIONS = {
    "clue_property_at_house": "The attribute is in the given house (1 = leftmost).",
    "clue_not_property_at_house": "The attribute is not in the given house.",
    "clue_property_at_property": "Both attributes belong to the same house (e.g. 'Alice is the Dane').",
    "clue_property_next_to_property": "The two attributes are in adjacent houses, either order.",
    "clue_property_left_of": "attribute_1 is somewhere to the left of attribute_2 (not necessarily adjacent).",
    "clue_property_right_of": "attribute_1 is somewhere to the right of attribute_2 (not necessarily adjacent).",
    "clue_property_directly_left_of_property": "attribute_1 is immediately left of attribute_2.",
    "clue_property_directly_right_of_property": "attribute_1 is immediately right of attribute_2.",
    "clue_one_between_two_properties": "Exactly one house lies between the two attributes, either order.",
    "clue_two_between_two_properties": "Exactly two houses lie between the two attributes, either order.",
}


def tool_schemas() -> list[dict]:
    tools = [{"type": "function", "function": {
        "name": "init_zebra",
        "description": "Declare the puzzle. Call once, first. Name every attribute Property_value, "
                       "e.g. Name_Alice, Color_red; use exactly these names in every clue call.",
        "parameters": {"type": "object", "required": ["number_of_places", "number_of_clues", "properties"],
                       "properties": {
                           "number_of_places": {"type": "integer"},
                           "number_of_clues": {"type": "integer"},
                           "properties": {"type": "object", "description": "Property name -> list of attributes",
                                          "additionalProperties": {"type": "array", "items": {"type": "string"}}}}}}}]
    for name, params in CLUE_TOOLS.items():
        props = {p: {"type": "integer" if p == "house" else "string"} for p in params}
        tools.append({"type": "function", "function": {
            "name": name, "description": DESCRIPTIONS[name],
            "parameters": {"type": "object", "properties": props, "required": list(params)}}})
    return tools


NEC_SYSTEM = """You translate logic grid puzzles into tool calls for an exact solver. Do not solve the puzzle.
Write ALL calls in a single reply: init_zebra first, then exactly one clue call per numbered clue,
in order. Before each call write
one short sentence restating the clue. Attribute names are Property_value, for example Name_Alice or
Color_red, and must match the names you declared in init_zebra.

Example. Puzzle: 2 houses. Names: `Bob`, `Eric`. Colors: `red`, `blue`.
Clues: 1. Bob is directly left of the person who loves blue. 2. Eric does not live in the first house.
Declare the puzzle.
<tool_call>
{"name": "init_zebra", "arguments": {"number_of_places": 2, "number_of_clues": 2, "properties": {"Name": ["Name_Bob", "Name_Eric"], "Color": ["Color_red", "Color_blue"]}}}
</tool_call>
Clue 1: Bob is immediately left of blue.
<tool_call>
{"name": "clue_property_directly_left_of_property", "arguments": {"attribute_1": "Name_Bob", "attribute_2": "Color_blue"}}
</tool_call>
Clue 2: Eric is not in house 1.
<tool_call>
{"name": "clue_not_property_at_house", "arguments": {"attribute": "Name_Eric", "house": 1}}
</tool_call>"""


def nec_system() -> str:
    """The instructions plus the tool schemas as text. Passing the schemas through a chat template's tool slot
    makes Qwen3 stop after its first call to wait for a result; the paper's format is one reply with all calls."""
    schemas = "\n".join(json.dumps(t["function"]) for t in tool_schemas())
    return NEC_SYSTEM + "\n\nTools:\n" + schemas


FEEDBACK = {
    "none": "You made no tool calls. Call init_zebra and then one clue tool per clue.",
    "count": "init_zebra declares {declared} clues but you made {made} clue calls. Emit the complete, corrected "
             "list of calls again: init_zebra, then exactly one call per clue.",
    "invalid": "Some calls could not be encoded:\n{errors}\nEmit the complete, corrected list of calls again.",
    "over": "The solver found NO solution: at least one call says something the clue does not. Re-read every "
            "clue, fix the wrong call(s), and emit the complete, corrected list of calls again.",
    "under": "The solver found MORE THAN ONE solution: at least one call is weaker than its clue or a clue is "
             "missing. Re-read every clue, fix it, and emit the complete, corrected list of calls again.",
}


def check(text: str) -> tuple[str, str, dict | None]:
    """Encode one reply. Returns ``(verdict, feedback, grid)``; verdict is unique/over/under/count/invalid/none."""
    calls, errors = parse_tool_calls(text)
    if not calls and not errors:
        return "none", FEEDBACK["none"], None
    inits = [c for c in calls if c["name"] == "init_zebra"]
    clues = [c for c in calls if c["name"] != "init_zebra"]
    if len(inits) != 1:
        errors.append(f"expected exactly one init_zebra call, got {len(inits)}")
    zebra = None
    if len(inits) == 1:
        try:
            a = inits[0]["arguments"]
            zebra = Zebra.init_zebra(a["number_of_places"], a["number_of_clues"], a["properties"])
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            errors.append(f"init_zebra: {e}")
    if zebra is not None:
        for c in clues:
            try:
                zebra.call(c["name"], c["arguments"])
            except EncodingError as e:
                errors.append(str(e))
    if errors:
        errors = list(dict.fromkeys(errors))
        return "invalid", FEEDBACK["invalid"].format(errors="\n".join(f"- {e}" for e in errors[:12])), None
    if len(clues) != zebra.expected_clues:
        return "count", FEEDBACK["count"].format(declared=zebra.expected_clues, made=len(clues)), None
    verdict, grid = zebra.status()
    return verdict, FEEDBACK.get(verdict, ""), grid


@dataclass
class Attempt:
    method: str
    model: str
    puzzle: str
    size: str
    solved: bool = False
    cell_accuracy: float = 0.0
    rounds: list = field(default_factory=list)
    grid: dict | None = None

    def to_json(self):
        return self.__dict__


def grade(puzzle: Puzzle, grid: dict | None) -> tuple[bool, float]:
    """Cell accuracy against the gold grid. ``grid`` maps house -> {anything: value}; group labels are ignored,
    values are matched case-insensitively with any ``Property_`` prefix stripped."""
    if not grid:
        return False, 0.0
    norm = {}
    for house, row in grid.items():
        try:
            h = int(re.sub(r"\D", "", str(house)) or -1)
        except ValueError:
            continue
        values = row.values() if isinstance(row, dict) else row
        norm[h] = {str(v).split("_", 1)[-1].strip().lower() for v in values}
    total = correct = 0
    for house, row in puzzle.solution.items():
        for value in row.values():
            total += 1
            correct += value.lower() in norm.get(int(house), set())
    return correct == total, correct / total


def solve_nec(puzzle: Puzzle, backend, feedback: bool = True, max_rounds: int = 4) -> Attempt:
    att = Attempt("nec+feedback" if feedback else "nec", backend.name, puzzle.id, puzzle.size)
    messages = [{"role": "system", "content": nec_system()},
                {"role": "user", "content": puzzle.text()}]
    for _ in range(max_rounds if feedback else 1):
        reply = backend(messages, None)
        verdict, message, grid = check(reply)
        att.rounds.append({"verdict": verdict, "reply": reply})
        if verdict == "unique":
            att.grid = {str(h): row for h, row in grid.items()}
            break
        messages += [{"role": "assistant", "content": strip_thinking(reply)}, {"role": "user", "content": message}]
    att.solved, att.cell_accuracy = grade(puzzle, att.grid)
    return att


PLAIN_INSTRUCTION = """Solve the puzzle. End your answer with the solution as one JSON object of this exact form:
{template}"""


def solve_plain(puzzle: Puzzle, backend) -> Attempt:
    att = Attempt("plain", backend.name, puzzle.id, puzzle.size)
    template = {f"House {h}": {g: "___" for g in puzzle.groups} for h in range(1, puzzle.houses + 1)}
    prompt = puzzle.text() + "\n\n" + PLAIN_INSTRUCTION.format(template=json.dumps(template))
    reply = backend([{"role": "user", "content": prompt}], None)
    att.rounds.append({"verdict": "answer", "reply": reply})
    att.grid = last_json_object(strip_thinking(reply))
    att.solved, att.cell_accuracy = grade(puzzle, att.grid)
    return att


def last_json_object(text: str) -> dict | None:
    """The last parseable top-level ``{...}`` in the text."""
    depth, start, found = 0, None, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start:i + 1])
                    if isinstance(obj, dict):
                        found = obj
                except json.JSONDecodeError:
                    pass
    return found
