#!/usr/bin/env python3
"""Fan-out measure: N forks of one long context decoding at once, against a running ragon-serve.

The workload of the SGLang figures in BATCHING section P4: one prefix of --ctx tokens is warmed once, then N requests
arrive together, each the prefix and its own question, greedy, --max-new tokens. Reported: the decode span (first
token of the last fork to start, to the last token) and the combined tok/s over it, the wall clock from submission
including the questions' prefill, the tokens each fork produced, and the engine's step counters. With --api-n, one
sampled request with n forks instead: no question prefills to wait for, the forks decode together from the first pass.

Token ids come from --ids, written by build_ids.py, so the server's tokenizer plays no part.
"""
import argparse
import json
import threading
import time
import urllib.request


def post(url, body, on_line=None):
    req = urllib.request.Request(url + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        if on_line is None:
            return json.load(r)
        for line in r:
            if line.startswith(b"data: ") and not line.startswith(b"data: [DONE]"):
                on_line(json.loads(line[6:]))
    return None


def get(url, path):
    with urllib.request.urlopen(url + path, timeout=30) as r:
        return json.load(r)


def fork(url, model, ids, max_new, out, i, gate):
    times, text_pieces = [], 0
    usage = {}

    def line(d):
        nonlocal text_pieces
        usage.update(d.get("usage") or {})
        if d.get("choices") and d["choices"][0].get("text"):
            text_pieces += 1
            times.append(time.monotonic())

    gate.wait()
    body = {"model": model, "prompt": ids, "max_tokens": max_new, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}}
    post(url, body, line)
    out[i] = {"times": times, "first": times[0] if times else None, "last": times[-1] if times else None, "pieces": text_pieces,
              "tokens": usage.get("completion_tokens", text_pieces),
              "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
              "prompt": usage.get("prompt_tokens")}


def api_n(a, prefix, questions, rep):
    """One request with n forks: the question is prefilled once, then every fork decodes from the first pass."""
    before = get(a.url, "/spec_stats")
    times, usage, finish = {}, {}, {}

    def line(d):
        usage.update(d.get("usage") or {})
        for c in d.get("choices") or []:
            if c.get("finish_reason"):
                finish[c.get("index", 0)] = c["finish_reason"]
            if c.get("text"):
                times.setdefault(c.get("index", 0), []).append(time.monotonic())

    body = {"model": a.model, "prompt": prefix + questions[rep % len(questions)], "max_tokens": a.max_new, "n": a.n,
            "temperature": a.temperature, "top_k": 40, "seed": 1000 + rep, "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.monotonic()
    post(a.url, body, line)
    t1 = time.monotonic()
    after = get(a.url, "/spec_stats")
    start = max(v[0] for v in times.values())
    end = max(v[-1] for v in times.values())
    inside = sum(1 for v in times.values() for x in v if x > start)
    pieces = [len(v) for v in times.values()]
    print(json.dumps({"label": a.label, "arm": "api-n", "n": a.n, "forks_seen": len(times), "forks_finished": len(finish),
                      "finish_stop": sum(1 for v in finish.values() if v == "stop"), "rep": rep, "ctx": len(prefix),
                      "wall_s": round(t1 - t0, 3), "span_s": round(end - start, 3), "tokens": usage.get("completion_tokens"),
                      "toks_min": min(pieces), "toks_max": max(pieces),
                      "span_tok_s": round(inside / (end - start), 1) if end > start else 0,
                      "wall_tok_s": round(sum(pieces) / (t1 - t0), 1),
                      "spec_delta": {k: after[k] - before[k] for k in ("batched_steps", "fanout_steps") if k in after}}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:8300")
    ap.add_argument("--model", default="RadixArk/Qwen3.8-27B-NVFP4")
    ap.add_argument("--ids", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--ctx", type=int, default=32000)
    ap.add_argument("--max-new", type=int, default=100)
    ap.add_argument("--label", default="")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--api-n", action="store_true", help="one request with n forks instead of n requests")
    ap.add_argument("--temperature", type=float, default=0.7, help="the --api-n arm's sampling")
    a = ap.parse_args()
    d = json.load(open(a.ids))
    prefix, questions = d["prefix"][: a.ctx], d["questions"]
    t = time.monotonic()
    post(a.url, {"model": a.model, "prompt": prefix, "max_tokens": 1, "temperature": 0})
    warm = time.monotonic() - t
    for rep in range(a.repeats):
        if a.api_n:
            api_n(a, prefix, questions, rep)
            continue
        before = get(a.url, "/spec_stats")
        out = [None] * a.n
        gate = threading.Barrier(a.n + 1)
        threads = [threading.Thread(target=fork, args=(a.url, a.model, prefix + questions[(i + rep * a.n) % len(questions)],
                                                       a.max_new, out, i, gate)) for i in range(a.n)]
        for th in threads:
            th.start()
        gate.wait()
        t0 = time.monotonic()
        for th in threads:
            th.join()
        t1 = time.monotonic()
        after = get(a.url, "/spec_stats")
        firsts = [o["first"] for o in out if o and o["first"]]
        lasts = [o["last"] for o in out if o and o["last"]]
        toks = [o["tokens"] for o in out if o]
        span = max(lasts) - max(firsts) if firsts else 0.0
        # Tokens inside the span, by when each arrived: the forks prefilled first have tokens before the last one starts.
        start = max(firsts) if firsts else 0.0
        inside = sum(1 for o in out if o for x in o["times"] if x > start)
        rec = {"label": a.label, "n": a.n, "rep": rep, "ctx": len(prefix), "warm_s": round(warm, 2),
               "wall_s": round(t1 - t0, 3), "span_s": round(span, 3), "tokens": sum(toks),
               "toks_min": min(toks), "toks_max": max(toks), "span_tok_s": round(inside / span, 1) if span else 0,
               "wall_tok_s": round(sum(toks) / (t1 - t0), 1), "cached_min": min((o["cached"] or 0) for o in out),
               "spec_delta": {k: after[k] - before[k] for k in after if isinstance(after[k], (int, float))
                              and k in ("batched_steps", "fanout_steps")}}
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
