#!/usr/bin/env python3
"""Builds the workload sets of the several-drafters measurement (MULTI_DRAFTER §4.1, phase 0) next to agentlong.

The sets go to bench/agentlong/run.py and n_streams.py through --bodies and --manifest, in the shape of agentlong's
recorded requests. Every text in a new body is public: the prompts below, and this repository's own files for the long
reads. agentlong's agent prompt and tool list are copied from its first body, on this machine, and stay on it.

  code-long   agentlong's edit turns (recorded, 25-64K tokens)
  read-long   agentlong's turns without an edit, and eight new long reads over this repository's specs and sources
              (doc20 is under 30K on purpose: asklong's documents start at 16K)
  code-short  eight small bugs to fix, with the agent's prompt and tools (about 12K tokens of context)
  prose       twenty answers without tools: chat, phone-style, thinking on, not English
  read-agent  read-long without doc*: the turns that end in a tool result, so n_streams.py --warm can prime them

asklong (MULTI_DRAFTER §4.1, the phase 0 gate) is the prose set and the four long documents, doc*.

Usage: workloads.py --repo ENGINE_CHECKOUT --out DIR [--agentlong DIR] [--manifest bench/agentlong/manifest.json]
Writes DIR/all.jsonl, DIR/all.json and one pair per workload, and prints what the rule in route_rule.py routes each to.
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from route_rule import route  # noqa: E402

MODEL = "RadixArk/Qwen3.8-27B-NVFP4"
CHARS_PER_TOKEN = 3.3  # sets the size of a long read; the server's usage reports the real count
ASK_INSTRUCTIONS = "You are a helpful assistant. Answer clearly and in full, in the language of the question."

CHAT = [
    "Why does the sky turn red at sunset? Explain it to a curious teenager, then add what changes on a very smoky day.",
    "I have to tell my team that a project they worked on for six months is cancelled. How should I run that meeting?",
    "Compare renting and buying a home for someone who may move city in three years. Set out the reasoning, not a verdict.",
    "Write a short story of about three hundred words about a lighthouse keeper who receives a letter addressed to someone else.",
    "What caused the fall of the Western Roman Empire? Summarise the main explanations historians give and where they disagree.",
    "Explain how a mortgage's interest works over its term, why early payments are mostly interest, and what overpaying does.",
    "Give me a week of simple vegetarian dinners for two, with a shopping list, using each perishable ingredient at least twice.",
    "What should a first-time manager do in their first ninety days? Be concrete, and say what to avoid as well.",
]
PHONE = [
    "How long does cooked rice keep in the fridge, and how do I reheat it safely?",
    "My train gets in at 18:40 and the concert starts at 19:30, a twenty minute walk away. Do I have time to eat? What would you do?",
    "What's the difference between a cold and the flu, and when should I actually see a doctor?",
    "Quick tips for a job interview tomorrow morning; I have not prepared at all.",
]
THINKING = [
    "A bat and a ball cost 1.10 in total and the bat costs 1.00 more than the ball. Explain the answer, then write two similar puzzles and solve them.",
    "Three friends split a bill unevenly and one of them later refunds part of it. Set out how to work out who owes what, with an example.",
    "Should a small town spend its budget on a new library or on repairing roads? Argue both sides, then decide and say why.",
    "Plan the seating of eight people at a round table where two pairs must not sit together and one couple must. Explain your reasoning.",
]
OTHER_LANGUAGES = [
    "Explique-moi pourquoi les saisons existent et pourquoi elles sont inversées entre les deux hémisphères, avec un exemple concret.",
    "Wie funktioniert eine Wärmepumpe, und für welche Häuser lohnt sie sich? Erkläre es für jemanden ohne technische Vorkenntnisse.",
    "¿Cuáles fueron las principales causas de la Revolución Francesa y cómo influyeron en el resto de Europa?",
    "Quais são os melhores hábitos para aprender uma língua nova sozinho, e que erros devo evitar nos primeiros meses?",
]
LONG_DOCS = [
    ("doc20", 20000, ["specs/DREAMFERENCE_RAGON_ENGINE_OVERVIEW.md", "specs/DREAMFERENCE_RAGON_ENGINE_RESEARCH.md", "specs/DREAMFERENCE_RAGON_ENGINE_ARCHITECTURE.md", "specs/DREAMFERENCE_RAGON_ENGINE_MODEL.md"],
     "Summarise these documents for a new engineer: what the system does, how its parts fit, and the three things to read first."),
    ("doc32", 32000, ["specs/DREAMFERENCE_RAGON_ENGINE_BATCHING.md", "specs/DREAMFERENCE_RAGON_ENGINE_SPECULATION.md", "specs/DREAMFERENCE_RAGON_ENGINE_OVERVIEW.md", "specs/DREAMFERENCE_RAGON_ENGINE_ARCHITECTURE.md", "specs/DREAMFERENCE_RAGON_ENGINE_SESSIONS.md"],
     "Explain in plain language how speculative decoding works here, why output does not change, and how serving several sessions limits its speed."),
    ("doc44", 44000, ["specs/DREAMFERENCE_RAGON_ENGINE_BACKLOG.md", "specs/DREAMFERENCE_RAGON_ENGINE_SESSIONS.md"],
     "What do these documents record about holding several sessions at once? List the decisions with the reason for each."),
    ("doc56", 56000, ["specs/DREAMFERENCE_RAGON_ENGINE_BACKLOG.md", "specs/DREAMFERENCE_RAGON_ENGINE_BATCHING.md"],
     "Which items in these documents look most valuable for a user on one machine, and which look least? Explain your choice."),
]
# The token counts are the files' text alone: the agent's prompt and tools add about 12K, and the serve flags cap a request at 64K.
REPO_READS = [
    ("repo32", 20000, ["src/serve/main.cpp"],
     "Read main.cpp and explain how a request travels from the HTTP thread to the decode loop. Say which file to change for a new flag."),
    ("repo40", 28000, ["bench/agentlong/run.py", "bench/agentlong/n_streams.py", "bench/replay/replay.py", "bench/lookup_sim.py", "bench/quant/scalesearch.py", "src/serve/main.cpp"],
     "Read these files, then tell me what each bench script measures, where they overlap, and which server counters they depend on."),
    ("repo48", 36000, ["src/serve/main.cpp", "CMakeLists.txt", "README.md", "specs/DREAMFERENCE_RAGON_ENGINE_ARCHITECTURE.md", "specs/DREAMFERENCE_RAGON_ENGINE_API.md"],
     "Find where the drafter's settings are read and how they reach the engine. Then describe how you would add a per-request drafter choice."),
    ("repo56", 44000, ["specs/DREAMFERENCE_RAGON_ENGINE_API.md", "specs/DREAMFERENCE_RAGON_ENGINE_CLI.md", "src/serve/main.cpp"],
     "Compare what the API and CLI documents promise with what main.cpp does. List the differences you find, most serious first."),
]
SMALL_BUGS = [
    ("edit_py1", "stats.py", "The median of an even-length list is wrong. Fix it and add a test.",
     "def median(xs):\n    s = sorted(xs)\n    return s[len(s) // 2]\n\n\ndef mean(xs):\n    return sum(xs) / len(xs)\n\n\ndef spread(xs):\n    return max(xs) - min(xs)\n"),
    ("edit_py2", "lru.py", "get() does not refresh an entry's position, so a recently read key can be evicted. Fix it.",
     "from collections import OrderedDict\n\n\nclass LRU:\n    def __init__(self, cap):\n        self.cap = cap\n        self.d = OrderedDict()\n\n    def get(self, k):\n        return self.d.get(k)\n\n    def put(self, k, v):\n        self.d[k] = v\n        self.d.move_to_end(k)\n        if len(self.d) > self.cap:\n            self.d.popitem(last=False)\n"),
    ("edit_py3", "paths.py", "join_url doubles slashes when the base ends in one and drops the query string. Fix both and add tests.",
     "from urllib.parse import urlsplit\n\n\ndef join_url(base, path):\n    return base + '/' + path\n\n\ndef strip_query(url):\n    return urlsplit(url).path\n"),
    ("edit_py4", "retry.py", "The retry loop sleeps after the last failed attempt and swallows the final exception. Fix it.",
     "import time\n\n\ndef retry(f, n=3, delay=0.5):\n    for i in range(n):\n        try:\n            return f()\n        except Exception:\n            time.sleep(delay * 2 ** i)\n    return None\n"),
    ("edit_cpp", "ring.h", "push overwrites the oldest element when full but size() then exceeds the capacity. Fix the class.",
     "#pragma once\n#include <vector>\n\ntemplate <class T> class Ring {\n public:\n  explicit Ring(size_t cap) : buf_(cap) {}\n  void push(T v) { buf_[head_++ % buf_.size()] = v; ++size_; }\n  size_t size() const { return size_; }\n  const T& at(size_t i) const { return buf_[i]; }\n private:\n  std::vector<T> buf_;\n  size_t head_ = 0, size_ = 0;\n};\n"),
    ("edit_rs", "parse.rs", "parse_kv panics on a line without '=' and keeps whitespace around the key. Return a Result and trim.",
     "pub fn parse_kv(line: &str) -> (String, String) {\n    let i = line.find('=').unwrap();\n    (line[..i].to_string(), line[i + 1..].to_string())\n}\n"),
    ("edit_js", "debounce.js", "The debounced function loses its arguments and `this`. Fix it and add a cancel method.",
     "export function debounce(fn, ms) {\n  let t = null;\n  return function () {\n    clearTimeout(t);\n    t = setTimeout(fn, ms);\n  };\n}\n"),
    ("edit_go", "chunk.go", "Chunk drops the last partial chunk and loops forever when n is 0. Fix both and add a table test.",
     "package chunk\n\nfunc Chunk(xs []int, n int) [][]int {\n\tvar out [][]int\n\tfor i := 0; i+n <= len(xs); i += n {\n\t\tout = append(out, xs[i:i+n])\n\t}\n\treturn out\n}\n"),
]


def canonical(body):
    """The bytes run.py and n_streams.py send and check."""
    return json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def user(text):
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def prose_body(text, effort="none", max_output=700):
    return {"model": MODEL, "instructions": ASK_INSTRUCTIONS, "input": [user(text)], "stream": True, "store": False,
            "max_output_tokens": max_output, "reasoning": {"effort": effort, "summary": "auto"}}


def read_files(repo, paths, tokens):
    """The files' text, in order, cut at the number of tokens asked for; the rest of a long list is not read."""
    left, parts = int(tokens * CHARS_PER_TOKEN), []
    for path in paths:
        with open(os.path.join(repo, path), encoding="utf-8") as f:
            text = f.read()[:left].replace("<|", "<")  # the sources quote special tokens, and a literal image placeholder is refused
        parts.append((path, text))
        left -= len(text)
        if left <= 0:
            break
    return parts


def agent_body(base, items, max_output):
    body = {k: base[k] for k in ("model", "instructions", "tools", "tool_choice", "parallel_tool_calls", "store", "stream", "reasoning")}
    return {**body, "input": items, "max_output_tokens": max_output}


def tool_read(call_id, path, text):
    args = json.dumps({"cmd": f"cat {path}"})
    return [{"type": "function_call", "name": "exec_command", "call_id": call_id, "arguments": args},
            {"type": "function_call_output", "call_id": call_id, "output": f"Process exited with code 0\nOutput:\n{text}\n"}]


def build(repo, agentlong_dir, manifest_path):
    """Every body of the sets: [(id, workload, body)]."""
    with open(os.path.join(agentlong_dir, "bodies.jsonl"), encoding="utf-8") as f:
        recorded = [json.loads(line) for line in f]
    with open(manifest_path, encoding="utf-8") as f:
        turns = {t["id"]: t for t in json.load(f)["turns"]}
    out = [(r["id"], "code-long" if turns[r["id"]]["edit"] else "read-long", r["body"]) for r in recorded]
    base = recorded[0]["body"]
    for name, tokens, paths, question in LONG_DOCS:
        text = "\n\n".join(t for _, t in read_files(repo, paths, tokens))
        out.append((name, "read-long", prose_body(f"{text}\n\n---\n\n{question}")))
    for name, tokens, paths, question in REPO_READS:
        items = [user(question)]
        for i, (path, text) in enumerate(read_files(repo, paths, tokens)):
            items += tool_read(f"call_{name}_{i}", path, text)
        out.append((name, "read-long", agent_body(base, items, 700)))
    for name, path, task, source in SMALL_BUGS:
        items = [user(f"{task} The file is {path}."), *tool_read(f"call_{name}", path, source)]
        out.append((name, "code-short", agent_body(base, items, 500)))
    for i, text in enumerate(CHAT):
        out.append((f"chat{i}", "prose", prose_body(text)))
    for i, text in enumerate(PHONE):
        out.append((f"phone{i}", "prose", prose_body(text, max_output=400)))
    for i, text in enumerate(THINKING):
        out.append((f"think{i}", "prose", prose_body(text, effort="medium", max_output=1000)))
    for i, text in enumerate(OTHER_LANGUAGES):
        out.append((f"lang{i}", "prose", prose_body(text)))
    return out


def write_set(out_dir, name, rows):
    with open(os.path.join(out_dir, f"{name}.jsonl"), "w", encoding="utf-8") as f:
        for rid, _, body in rows:
            f.write(json.dumps({"id": rid, "body": body}, ensure_ascii=False) + "\n")
    turns = [{"id": rid, "workload": wl, "rec_in": sum(len(json.dumps(i)) for i in body["input"]) // 4,
              "rec_out": body["max_output_tokens"], "body_sha256": hashlib.sha256(canonical(body).encode()).hexdigest(),
              "route": (r := route(body))["workload"], "long": r["long"], "chars": r["chars"], "code_share": r["code_share"],
              "tool_density": r["tool_density"]}
             for rid, wl, body in rows]
    with open(os.path.join(out_dir, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump({"set": name, "turns": turns}, f, indent=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.join(HERE, "..", ".."))
    ap.add_argument("--out", required=True)
    ap.add_argument("--agentlong", default=os.path.expanduser("~/bench-sets/agentlong"))
    ap.add_argument("--manifest", default=os.path.join(HERE, "..", "agentlong", "manifest.json"))
    a = ap.parse_args()
    rows = build(a.repo, a.agentlong, a.manifest)
    os.makedirs(a.out, exist_ok=True)
    write_set(a.out, "all", rows)
    for wl in sorted({r[1] for r in rows}):
        write_set(a.out, wl, [r for r in rows if r[1] == wl])
    write_set(a.out, "read-agent", [r for r in rows if r[1] == "read-long" and not r[0].startswith("doc")])
    want = {"code-long": "work", "code-short": "work", "read-long": "work", "prose": "ask"}
    routes = {}
    for rid, wl, body in rows:
        got = route(body)
        routes.setdefault(wl, {}).setdefault(got["workload"], []).append(rid)
    for wl, by in sorted(routes.items()):
        print(wl, {k: len(v) for k, v in by.items()}, "expected", want[wl])


if __name__ == "__main__":
    main()
