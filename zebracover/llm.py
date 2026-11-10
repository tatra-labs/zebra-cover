"""Chat backends. Each takes a message list and the tool schemas and returns the assistant's raw text.

Tool calls travel as text in Qwen3's native form, ``<tool_call>{"name": ..., "arguments": {...}}</tool_call>``,
whatever the backend, so one parser serves them all.
"""
from __future__ import annotations

import json
import re
import urllib.request

TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def json_objects(text: str) -> list[tuple[int, str]]:
    """Every top-level balanced ``{...}`` span, with its start offset; braces inside JSON strings are skipped."""
    spans, depth, start, in_str, escape = [], 0, 0, False, False
    for i, ch in enumerate(text):
        if in_str:
            escape, in_str = (False, in_str) if escape else (ch == "\\", ch != '"')
            continue
        if ch == '"' and depth:
            in_str = True
        elif ch == "{":
            start = i if depth == 0 else start
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                spans.append((start, text[start:i + 1]))
    return spans


def parse_tool_calls(text: str) -> tuple[list[dict], list[str]]:
    """``(calls, errors)`` in reply order.

    A call is any JSON object with a ``name``, inside ``<tool_call>`` tags or not: small models often drop the
    tags after the first few calls, and the tags carry no information the object does not. A tagged block
    that holds no parseable object is reported as an error.
    """
    text = strip_thinking(text)
    calls, errors = [], []
    for block in TOOL_CALL.finditer(text):
        try:
            json.loads(block.group(1))
        except json.JSONDecodeError as e:
            errors.append(f"unparseable tool call {block.group(1)[:80]!r}: {e.msg}")
    for _, raw in json_objects(text):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or "name" not in obj:
            continue
        args = obj.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                errors.append(f"arguments of {obj.get('name')!r} are not JSON")
                continue
        if not isinstance(obj.get("name"), str) or not isinstance(args, dict):
            errors.append(f"malformed tool call {raw[:80]!r}")
            continue
        calls.append({"name": obj["name"], "arguments": args})
    return calls, errors


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


class TransformersBackend:
    """A local Hugging Face chat model, greedy decoding (the paper's temperature 0)."""

    def __init__(self, model_id: str = "Qwen/Qwen3-1.7B", thinking: bool = False, max_new_tokens: int = 2048,
                 dtype: str = "bfloat16", threads: int | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if threads:
            torch.set_num_threads(threads)
        self.torch = torch
        self.name = model_id.split("/")[-1] + ("-think" if thinking else "")
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype))
        self.model.eval()
        self.thinking, self.max_new_tokens = thinking, max_new_tokens
        self.generated_tokens = 0

    def __call__(self, messages: list[dict], tools: list[dict] | None = None) -> str:
        text = self.tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=self.thinking)
        ids = self.tok(text, return_tensors="pt")
        with self.torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      temperature=None, top_p=None, top_k=None)
        new = out[0, ids.input_ids.shape[1]:]
        self.generated_tokens += len(new)
        return self.tok.decode(new, skip_special_tokens=True)


class OpenAIBackend:
    """Any OpenAI-compatible ``/v1/chat/completions`` endpoint (vLLM, llama.cpp, Ollama, hosted APIs).

    Shipped, not run in this repository: grading a hosted model costs money. Native ``tool_calls`` in the
    response are re-serialised to ``<tool_call>`` text so the rest of the pipeline is unchanged.
    """

    def __init__(self, base_url: str, model: str, api_key: str = "", max_tokens: int = 4096, timeout: float = 600):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.api_key, self.max_tokens, self.timeout = model, api_key, max_tokens, timeout
        self.name = model

    def __call__(self, messages: list[dict], tools: list[dict] | None = None) -> str:
        body = {"model": self.model, "messages": messages, "temperature": 0, "max_tokens": self.max_tokens}
        if tools:
            body["tools"] = tools
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            msg = json.loads(resp.read())["choices"][0]["message"]
        text = msg.get("content") or ""
        for tc in msg.get("tool_calls") or []:
            fn = tc["function"]
            args = fn["arguments"] if isinstance(fn["arguments"], str) else json.dumps(fn["arguments"])
            text += f'\n<tool_call>\n{{"name": "{fn["name"]}", "arguments": {args}}}\n</tool_call>'
        return text


class ScriptedBackend:
    """Replies from a list, in order. For tests and for replaying the gold encoding."""

    def __init__(self, replies: list[str], name: str = "scripted"):
        self.replies, self.name, self.calls = list(replies), name, 0

    def __call__(self, messages, tools=None) -> str:
        self.calls += 1
        return self.replies.pop(0) if self.replies else ""


def calls_to_text(calls: list[dict]) -> str:
    return "\n".join(f"<tool_call>\n{json.dumps(c)}\n</tool_call>" for c in calls)
