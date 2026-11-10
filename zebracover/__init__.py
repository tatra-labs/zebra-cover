"""zebra-cover: Zebra puzzles through an LLM front end and an exact-cover back end (after arXiv 2611.03641)."""
from .exact_cover import ExactCover
from .generate import Puzzle, generate
from .llm import OpenAIBackend, ScriptedBackend, TransformersBackend, calls_to_text, parse_tool_calls
from .nec import Attempt, check, grade, last_json_object, solve_nec, solve_plain, tool_schemas
from .zebra import CLUE_TOOLS, EncodingError, Zebra

__all__ = [
    "Attempt", "CLUE_TOOLS", "EncodingError", "ExactCover", "OpenAIBackend", "Puzzle", "ScriptedBackend",
    "TransformersBackend", "Zebra", "calls_to_text", "check", "generate", "grade", "last_json_object",
    "parse_tool_calls", "solve_nec", "solve_plain", "tool_schemas",
]
__version__ = "0.1.0"
