#!/usr/bin/env python3
"""The request-shape rule of MULTI_DRAFTER §2.2 and its features: which workload a request that declares none belongs to.

The features cost microseconds and need no tokenizer, because they run on the request's JSON:

  chars          the input's text, a stand-in for prompt length
  code_share     the share of the input's non-empty lines that look like code
  tool_density   function calls and their outputs among the input items
  has_tools, shell_tool, last_kind

Classes: code (tools offered and the last item a tool result, or a shell tool and a user turn), prose (no tools) and,
for either, long when the input is over LONG_CHARS. A misroute costs speed only, never output (§0).

Usage: route_rule.py BODIES.jsonl  prints id, the features and the route of each body.
"""
import json
import re
import sys

LONG_CHARS = 60000  # about 20K tokens of input, the 30K class once the agent's prompt and tools are counted
SHELL_TOOLS = {"exec_command", "shell", "apply_patch", "local_shell"}
CODE_LINE = re.compile(r"^\s*(def |class |import |from \S+ import|#include|fn |func |pub |let |const |return\b|if\b.*[:{(]$|for\b.*[:{(]$|while\b.*[:{(]$|[{}]\s*$|[-+@]{1,3}\s|//|\w+\s*=\s*\S|\w+\(.*\)\s*[;:{]?$)")


def texts(item):
    """The strings of one Responses input item."""
    if item.get("type") == "function_call":
        yield item.get("arguments", "")
    elif item.get("type") == "function_call_output":
        out = item.get("output", "")
        yield out if isinstance(out, str) else json.dumps(out)
    else:
        content = item.get("content", "")
        if isinstance(content, str):
            yield content
        else:
            for part in content:
                yield part.get("text", "")


def features(body):
    """The features of one /v1/responses request body."""
    items = body.get("input", [])
    items = [{"type": "message", "content": items}] if isinstance(items, str) else items
    text = "\n".join(t for i in items for t in texts(i))
    lines = [ln for ln in text.split("\n") if ln.strip()]
    sample = lines[-4000:]  # the end of the context is what the next tokens continue, and this bounds the cost
    tools = [t.get("name") for t in body.get("tools") or []]
    calls = sum(i.get("type") in ("function_call", "function_call_output") for i in items)
    return {"chars": len(text), "code_share": round(sum(bool(CODE_LINE.match(ln)) for ln in sample) / max(1, len(sample)), 3),
            "tool_density": round(calls / max(1, len(items)), 3), "has_tools": bool(tools),
            "shell_tool": any(t in SHELL_TOOLS for t in tools), "last_kind": items[-1].get("type", "message") if items else "none"}


def route(body):
    """The features plus the routed workload, exactly §2.2's table, and the length bin beside it."""
    f = features(body)
    if not f["has_tools"]:
        workload = "ask"
    elif f["last_kind"] == "function_call_output" or f["shell_tool"]:
        workload = "work"
    else:
        workload = "general"
    return {**f, "workload": workload, "long": f["chars"] >= LONG_CHARS}


if __name__ == "__main__":
    for row in open(sys.argv[1], encoding="utf-8"):
        rec = json.loads(row)
        print(rec["id"], json.dumps(route(rec["body"])))
