#!/usr/bin/env python3
"""Session fork bench: one agent context split into several agents that continue from it, against any
OpenAI-compatible server (ragon-serve or another engine), through /v1/chat/completions the way an agent drives it.

One case (size S, forks k, rep r):
  1. build: a fresh synthetic agent history of about S tokens (a unique nonce at token 0, so nothing earlier is
     cached), plus a short user turn; the reply is kept, so the session holds history + turn + reply.
  2. fork: k requests at once, each that whole session plus its own short instruction (greedy, --max-new tokens).
     With --arm n, one request with n=k instead (sampled, the same instruction for every choice).
Recorded per fork: time to first token (first delta of any kind: content, reasoning or tool call), prompt and
cached tokens as the server reports them in usage, completion tokens, finish reason, decode tok/s. Per case: the
total tok/s over the window where every fork decodes, the wall clock, the server's counter deltas (prompt, cached,
generated tokens) and the peak of its memory gauges during the fork minus their value before it, sampled at
--sample-hz.
Retention: after the last fork count of a size and rep, the server is left idle --idle seconds (the bench checks
the server's prompt counter did not move meanwhile), then the same session forks again into 2 with new
instructions.

Each finished case is one JSON line in --out; a restart skips the cases already there.
"""
import argparse
import hashlib
import json
import os
import random
import re
import threading
import time
import urllib.request

TOPICS = ["the retry policy", "the cache eviction order", "the error paths", "the config loader", "the test gaps",
          "the locking", "the public API", "the logging", "the memory use", "the start-up sequence"]
VERBS = ["Write a detailed review of", "Explain step by step", "Propose a refactor of", "List every risk in",
         "Write unit tests for", "Document", "Find the bugs in", "Describe a migration plan for"]
WORDS = ("request session cache token buffer queue worker config retry limit handler error state index offset page "
         "block stream reader writer parse value result count timeout lock thread pool entry table record file path "
         "update commit check flush merge split scope owner model layer").split()


def chat(url, body, on_chunk=None, timeout=3600):
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if on_chunk is None:
            return json.load(r)
        for line in r:
            if line.startswith(b"data: ") and not line.startswith(b"data: [DONE]"):
                on_chunk(json.loads(line[6:]), time.monotonic())
    return None


def get_text(url, path, timeout=10):
    with urllib.request.urlopen(url + path, timeout=timeout) as r:
        return r.read().decode()


METRIC = re.compile(r'^([a-zA-Z_:]+)(\{[^}]*\})?\s+([-+0-9.eE]+|NaN)$')
COUNTERS = ["sglang:prompt_tokens_total", "sglang:cached_tokens_total", "sglang:generation_tokens_total",
            "ragon:session_evictions_total", "ragon:kv_session_evictions_total", "ragon:kv_pool_exhausted_total",
            "ragon:memory_refusals_total", "sglang:evicted_tokens_total"]
GAUGES = ["ragon:kv_pages_total", "ragon:kv_pages_free", "ragon:sessions", "ragon:sessions_resident", "ragon:batch",
          "sglang:num_used_tokens", "sglang:token_usage", "sglang:mamba_usage", "sglang:kv_available_tokens",
          "sglang:kv_evictable_tokens", "sglang:kv_cache_memory_usage_gb", "sglang:max_total_num_tokens",
          "sglang:num_running_reqs", "sglang:num_queue_reqs"]


def metrics(url):
    """Sum of each wanted series over its labels; {} when the server has no /metrics."""
    out = {}
    try:
        text = get_text(url, "/metrics")
    except Exception:
        return out
    for line in text.splitlines():
        m = METRIC.match(line.strip())
        if m and (m.group(1) in COUNTERS or m.group(1) in GAUGES) and m.group(3) != "NaN":
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(3))
    return out


def spec_stats(url):
    try:
        return json.loads(get_text(url, "/spec_stats"))
    except Exception:
        return {}


def mem_available_kb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return None


class Sampler(threading.Thread):
    """Polls /metrics and MemAvailable until stopped; keeps the max of every gauge and the min of MemAvailable."""

    def __init__(self, url, hz):
        super().__init__(daemon=True)
        self.url, self.period, self.stop_ev = url, 1.0 / hz, threading.Event()
        self.peak, self.low_mem, self.samples = {}, None, 0

    def run(self):
        while not self.stop_ev.is_set():
            m = metrics(self.url)
            for k in GAUGES:
                if k in m:
                    self.peak[k] = max(self.peak.get(k, m[k]), m[k])
            if "ragon:kv_pages_free" in m:
                self.peak["ragon:kv_pages_free_min"] = min(self.peak.get("ragon:kv_pages_free_min", m["ragon:kv_pages_free"]),
                                                           m["ragon:kv_pages_free"])
            mem = mem_available_kb()
            if mem is not None:
                self.low_mem = mem if self.low_mem is None else min(self.low_mem, mem)
            self.samples += 1
            self.stop_ev.wait(self.period)


def history(target_tokens, nonce, chars_per_token):
    """A deterministic agent-like transcript (system, then user/assistant turns of tool output and code) of about
    target_tokens tokens; the nonce opens the system message so no earlier run's cache matches it."""
    rng = random.Random(int(hashlib.sha256(nonce.encode()).hexdigest()[:16], 16))
    msgs = [{"role": "system", "content": f"[run {nonce}] You are a coding agent working in a large repository. "
                                          "Answer precisely and keep going until the task is done."}]
    budget = int(target_tokens * chars_per_token)
    used, turn = len(msgs[0]["content"]), 0
    while used < budget:
        name = f"src/{rng.choice(WORDS)}_{rng.choice(WORDS)}_{turn}.py"
        user = f"Open {name} and tell me what {rng.choice(WORDS)} does."
        lines = []
        for i in range(rng.randint(30, 60)):
            a, b, c = rng.choice(WORDS), rng.choice(WORDS), rng.choice(WORDS)
            lines.append(f"    {a}_{b} = {c}.{rng.choice(WORDS)}({a}, {rng.randint(0, 999)})  # {b} {c} {i}")
        code = f"def {rng.choice(WORDS)}_{turn}({rng.choice(WORDS)}, {rng.choice(WORDS)}):\n" + "\n".join(lines)
        assistant = f"Here is {name}:\n```python\n{code}\n```\nIt updates the {rng.choice(WORDS)} {rng.choice(WORDS)}."
        msgs += [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
        used += len(user) + len(assistant)
        turn += 1
    return msgs


def instructions(k, salt):
    rng = random.Random(salt)
    out, seen = [], set()
    while len(out) < k:
        v, t = rng.choice(VERBS), rng.choice(TOPICS)
        if (v, t) not in seen:
            seen.add((v, t))
            out.append(f"{v} {t} in the files above. Be thorough: at least 400 words, no summary at the end.")
    return out


def body_for(a, messages, max_tokens, **extra):
    b = {"model": a.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True,
         "stream_options": {"include_usage": True}}
    if a.thinking in ("on", "off"):
        b["chat_template_kwargs"] = {"enable_thinking": a.thinking == "on"}
    b.update(extra)
    return b


def has_delta(choice):
    d = choice.get("delta") or {}
    return bool(d.get("content") or d.get("reasoning_content") or d.get("tool_calls") or choice.get("text"))


def stream_one(a, messages, max_tokens, gate=None, **extra):
    """One streamed request; per choice index: chunk times, text, finish reason; plus usage and the send time."""
    rec = {"times": {}, "text": {}, "finish": {}, "usage": {}}

    def on_chunk(d, t):
        if d.get("usage"):
            rec["usage"] = d["usage"]
        for c in d.get("choices") or []:
            i = c.get("index", 0)
            if c.get("finish_reason"):
                rec["finish"][i] = c["finish_reason"]
            if has_delta(c):
                rec["times"].setdefault(i, []).append(t)
                dl = c.get("delta") or {}
                rec["text"][i] = rec["text"].get(i, "") + (dl.get("content") or dl.get("reasoning_content") or "")

    if gate is not None:
        gate.wait()
    rec["sent"] = time.monotonic()
    try:
        chat(a.url, body_for(a, messages, max_tokens, **extra), on_chunk)
    except Exception as e:  # recorded, never fatal: a refused fork is a finding
        rec["error"] = f"{type(e).__name__}: {e}"[:300]
    rec["done"] = time.monotonic()
    return rec


def fork_figures(rec, i, completion_tokens):
    t = rec["times"].get(i) or []
    f = {"ttft": round(t[0] - rec["sent"], 4) if t else None, "chunks": len(t), "finish": rec["finish"].get(i),
         "completion_tokens": completion_tokens, "first": t[0] if t else None, "last": t[-1] if t else None}
    text = rec["text"].get(i, "")
    f["text_sha"] = hashlib.sha256(text.encode()).hexdigest()[:16]
    f["text_head"] = text[:60]
    if t and len(t) > 1 and completion_tokens:
        f["decode_tps"] = round((completion_tokens - completion_tokens / len(t)) / (t[-1] - t[0]), 2)
    return f


def overlap_tps(forks):
    """Total tok/s over [last fork's first token, first fork's last token], each fork's tokens in the window
    estimated from its share of chunks there."""
    live = [f for f in forks if f.get("first") and f.get("last") and f.get("completion_tokens")]
    if len(live) < len(forks) or not live:
        return None, None
    lo, hi = max(f["first"] for f in live), min(f["last"] for f in live)
    if hi <= lo:
        return None, 0.0
    tok = 0.0
    for f in live:
        ts = f["_times"]
        tok += f["completion_tokens"] * sum(1 for x in ts if lo < x <= hi) / len(ts)
    return round(tok / (hi - lo), 2), round(hi - lo, 3)


def run_forks(a, session, k, salt, arm):
    """Fork the session k ways; returns the per-fork figures and the case aggregates."""
    before, spec0 = metrics(a.url), spec_stats(a.url)
    mem0 = mem_available_kb()
    sampler = Sampler(a.url, a.sample_hz)
    sampler.start()
    t0 = time.monotonic()
    forks = []
    if arm == "n":
        instr = instructions(1, salt)[0]
        rec = stream_one(a, session + [{"role": "user", "content": instr}], a.max_new, n=k, temperature=0.7, seed=salt)
        usage = rec["usage"]
        for i in range(k):  # usage is summed over choices: split by chunk share
            ch = len(rec["times"].get(i) or [])
            tot = sum(len(v) for v in rec["times"].values()) or 1
            f = fork_figures(rec, i, round((usage.get("completion_tokens") or 0) * ch / tot) if ch else 0)
            f.update({"prompt_tokens": usage.get("prompt_tokens"),
                      "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                      "_times": rec["times"].get(i) or [], "error": rec.get("error")})
            forks.append(f)
    else:
        gate = threading.Barrier(k)
        recs = [None] * k
        instrs = instructions(k, salt)

        def go(i):
            recs[i] = stream_one(a, session + [{"role": "user", "content": instrs[i]}], a.max_new, gate=gate)

        th = [threading.Thread(target=go, args=(i,)) for i in range(k)]
        for x in th:
            x.start()
        for x in th:
            x.join()
        for rec in recs:
            u = rec["usage"]
            f = fork_figures(rec, 0, u.get("completion_tokens"))
            f.update({"prompt_tokens": u.get("prompt_tokens"),
                      "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
                      "_times": rec["times"].get(0) or [], "error": rec.get("error"),
                      "sent_offset": round(rec["sent"] - t0, 4)})
            forks.append(f)
    wall = time.monotonic() - t0
    sampler.stop_ev.set()
    sampler.join()
    after, spec1 = metrics(a.url), spec_stats(a.url)
    tps_all, window = overlap_tps(forks)
    for f in forks:
        f.pop("_times", None)
        f.pop("first", None)
        f.pop("last", None)
    total_tok = sum(f.get("completion_tokens") or 0 for f in forks)
    agg = {"wall_s": round(wall, 3), "total_tokens": total_tok, "tps_wall": round(total_tok / wall, 2) if wall else None,
           "tps_all_running": tps_all, "all_running_s": window,
           "counters": {k_: round(after.get(k_, 0) - before.get(k_, 0), 1) for k_ in COUNTERS if k_ in after},
           "gauges_before": {k_: before[k_] for k_ in GAUGES if k_ in before},
           "gauges_peak": sampler.peak, "samples": sampler.samples,
           "mem_available_drop_mb": round((mem0 - sampler.low_mem) / 1024, 1) if mem0 and sampler.low_mem else None,
           "fanout_steps": (spec1.get("fanout_steps", 0) - spec0.get("fanout_steps", 0)) if spec0 else None,
           "batched_steps": (spec1.get("batched_steps", 0) - spec0.get("batched_steps", 0)) if spec0 else None}
    return forks, agg


def build(a, size, nonce):
    """The parent session: history + one short turn; returns the messages including the reply, and its figures."""
    msgs = history(size, nonce, a.chars_per_token)
    msgs.append({"role": "user", "content": "Summarize in one sentence what these files have in common."})
    rec = stream_one(a, msgs, a.build_tokens)
    if rec.get("error"):
        raise RuntimeError("build failed: " + rec["error"])
    reply = rec["text"].get(0, "").strip() or "They share one config loader."
    u = rec["usage"]
    t = rec["times"].get(0) or []
    fig = {"prompt_tokens": u.get("prompt_tokens"), "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
           "ttft": round(t[0] - rec["sent"], 4) if t else None, "wall_s": round(rec["done"] - rec["sent"], 3)}
    return msgs + [{"role": "assistant", "content": reply}], fig


def done_cases(path):
    seen = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["case"])
                except (ValueError, KeyError):
                    pass
    return seen


def emit(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps({k: row[k] for k in ("case", "kind") if k in row} |
                     {"ttft": [f.get("ttft") for f in row.get("forks", [])],
                      "cached": [f.get("cached_tokens") for f in row.get("forks", [])],
                      "tps_all": (row.get("agg") or {}).get("tps_all_running")}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:18000")
    p.add_argument("--model", default="RadixArk/Qwen3.8-27B-NVFP4")
    p.add_argument("--label", required=True, help="the server mode, e.g. sglang, flat, paged, paged-fanout")
    p.add_argument("--arm", choices=["requests", "n"], default="requests",
                   help="requests: k separate requests, each its own instruction (agent-style); n: one request with n=k")
    p.add_argument("--sizes", default="8000,32000,100000")
    p.add_argument("--forks", default="2,4,8")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--max-new", type=int, default=256)
    p.add_argument("--build-tokens", type=int, default=32)
    p.add_argument("--idle", type=int, default=300, help="seconds idle before the retention fork; 0 skips it")
    p.add_argument("--retention-forks", type=int, default=2)
    p.add_argument("--thinking", choices=["on", "off", "default"], default="off")
    p.add_argument("--chars-per-token", type=float, default=3.0,
                   help="history characters per token; 3.0 gives 8178/32271/101185 tokens for 8k/32k/100k on Qwen3.8")
    p.add_argument("--sample-hz", type=float, default=4.0)
    p.add_argument("--settle", type=float, default=2.0, help="seconds between build and fork, and between cases")
    p.add_argument("--fixed-nonce", action="store_true",
                   help="the same history for every server (exactness probe: compare text_sha across runs); "
                        "a second run then starts from a warm cache")
    p.add_argument("--no-warmup", dest="warmup", action="store_false")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    seen = done_cases(a.out)
    if a.warmup:  # graphs, compiles and allocators settle on a throwaway session first; not recorded
        session, _ = build(a, 2000, f"warmup/{time.time_ns()}")
        run_forks(a, session, 2, salt=1, arm=a.arm)
    sizes = [int(x) for x in a.sizes.split(",")]
    forks = [int(x) for x in a.forks.split(",")]
    for size in sizes:
        for rep in range(a.reps):
            last_session = None
            for k in forks:
                case = f"{a.label}/{a.arm}/{size}/{k}/{rep}"
                if case in seen:
                    continue
                nonce = f"exact/{size}/{k}/{rep}" if a.fixed_nonce else f"{case}/{time.time_ns()}"
                session, bfig = build(a, size, nonce)
                time.sleep(a.settle)
                fk, agg = run_forks(a, session, k, salt=hash((size, k, rep)) & 0xffff, arm=a.arm)
                emit(a.out, {"case": case, "kind": "fork", "label": a.label, "arm": a.arm, "size": size, "k": k,
                             "rep": rep, "build": bfig, "forks": fk, "agg": agg, "max_new": a.max_new,
                             "thinking": a.thinking, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
                last_session = session
                time.sleep(a.settle)
            rcase = f"{a.label}/{a.arm}/{size}/retention/{rep}"
            if a.idle > 0 and last_session is not None and rcase not in seen:
                m0 = metrics(a.url)
                time.sleep(a.idle)
                m1 = metrics(a.url)
                quiet = m1.get("sglang:prompt_tokens_total") == m0.get("sglang:prompt_tokens_total")
                fk, agg = run_forks(a, last_session, a.retention_forks, salt=0x5000 + rep, arm=a.arm)
                emit(a.out, {"case": rcase, "kind": "retention", "label": a.label, "arm": a.arm, "size": size,
                             "k": a.retention_forks, "rep": rep, "idle_s": a.idle, "quiet": quiet, "forks": fk,
                             "agg": agg, "max_new": a.max_new, "thinking": a.thinking,
                             "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})


if __name__ == "__main__":
    main()
