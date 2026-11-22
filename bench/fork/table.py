#!/usr/bin/env python3
"""Tables from run.py's JSON lines: one row per (size, forks) and server label, medians over reps with [min-max],
and a verdict per figure against --base (better / same / worse; "same" when the medians are within --same
percent or the rep ranges overlap).

Memory per fork, in MB of KV the fork made the server hold beyond what it held before the fork:
  ragon paged   (kv_pages_free before - its minimum during the fork) x --page-tokens x --kv-bytes-per-token / k
  sglang        (num_used_tokens peak - before) x --kv-bytes-per-token / k  (its recurrent state slots apart)
  ragon flat    (sessions_resident peak - before) x --flat-session-mb / k   (whole preallocated slots)
"""
import argparse
import json
import statistics
from collections import defaultdict


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def span(xs):
    xs = [x for x in xs if x is not None]
    return (min(xs), max(xs)) if xs else (None, None)


def mem_per_fork(row, a):
    g0, gp, k = row["agg"]["gauges_before"], row["agg"]["gauges_peak"], row["k"]
    if "ragon:kv_pages_free" in g0 and g0.get("ragon:kv_pages_total"):
        used = g0["ragon:kv_pages_free"] - gp.get("ragon:kv_pages_free_min", g0["ragon:kv_pages_free"])
        return used * a.page_tokens * a.kv_bytes_per_token / 2**20 / k
    if "sglang:num_used_tokens" in g0:
        return (gp.get("sglang:num_used_tokens", 0) - g0["sglang:num_used_tokens"]) * a.kv_bytes_per_token / 2**20 / k
    if "ragon:sessions_resident" in g0:
        return (gp.get("ragon:sessions_resident", 0) - g0["ragon:sessions_resident"]) * a.flat_session_mb / k
    return None


def figures(row, a):
    f = row["forks"]
    ok = [x for x in f if not x.get("error")]
    ttft = sorted(x["ttft"] for x in ok if x.get("ttft") is not None)
    prompt = sum(x.get("prompt_tokens") or 0 for x in ok)
    cached = sum(x.get("cached_tokens") or 0 for x in ok)
    return {
        "ttft_med": med(ttft), "ttft_max": ttft[-1] if ttft else None,
        "reread": (prompt - cached) / len(ok) if ok and any(x.get("cached_tokens") is not None for x in ok) else None,
        "cached_pct": 100.0 * cached / prompt if prompt else None,
        "mem_mb": mem_per_fork(row, a),
        "dec_tps": med([x.get("decode_tps") for x in ok]),
        "tps_all": row["agg"].get("tps_all_running"), "wall": row["agg"]["wall_s"],
        "errors": len(f) - len(ok), "not_length": sum(1 for x in ok if x.get("finish") != "length"),
    }


# figure, format, which direction is better
FIGS = [("ttft_med", "{:.2f}s", "low"), ("ttft_max", "{:.2f}s", "low"), ("reread", "{:.0f}", "low"),
        ("cached_pct", "{:.1f}%", "high"), ("mem_mb", "{:.0f}", "low"), ("dec_tps", "{:.1f}", "high"),
        ("tps_all", "{:.1f}", "high"), ("wall", "{:.1f}s", "low")]


def verdict(mine, base, lo_hi_mine, lo_hi_base, better, same_pct):
    if mine is None or base is None:
        return "n/a"
    if abs(mine - base) <= same_pct / 100.0 * max(abs(base), 1e-9):
        return "same"
    if None not in lo_hi_mine + lo_hi_base and lo_hi_mine[0] <= lo_hi_base[1] and lo_hi_base[0] <= lo_hi_mine[1]:
        return "same"
    return "better" if (mine < base) == (better == "low") else "worse"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("jsonl", nargs="+")
    p.add_argument("--base", default="sglang")
    p.add_argument("--same", type=float, default=5.0)
    p.add_argument("--kv-bytes-per-token", type=int, default=65536, help="Qwen3.8-27B BF16: 16 layers x 2 x 4 x 256 x 2")
    p.add_argument("--page-tokens", type=int, default=64)
    p.add_argument("--flat-session-mb", type=float, default=196608 * 65536 / 2**20 + 151)
    a = p.parse_args()

    rows = [json.loads(l) for path in a.jsonl for l in open(path) if l.strip()]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["size"], r["kind"], r["k"], f'{r["label"]}/{r["arm"]}')].append(figures(r, a))
    labels = sorted({g[3] for g in groups})
    keys = sorted({(g[0], g[1], g[2]) for g in groups}, key=lambda x: (x[0], x[1] != "fork", x[2]))
    print("| size | case | server | n | " + " | ".join(f[0] for f in FIGS) + " | errors/not-length |")
    print("|" + "---|" * (len(FIGS) + 5))
    for size, kind, k in keys:
        for lab in labels:
            base = f'{a.base}/{lab.split("/")[1]}'
            reps = groups.get((size, kind, k, lab))
            if not reps:
                continue
            cells = []
            for name, fmt, better in FIGS:
                xs = [r[name] for r in reps]
                m, (lo, hi) = med(xs), span(xs)
                if m is None:
                    cells.append("-")
                    continue
                c = fmt.format(m) + (f" [{fmt.format(lo)}-{fmt.format(hi)}]" if len(reps) > 1 else "")
                if base and lab != base and groups.get((size, kind, k, base)):
                    bx = [r[name] for r in groups[(size, kind, k, base)]]
                    c += " " + {"better": "+", "same": "=", "worse": "-", "n/a": ""}[
                        verdict(m, med(bx), (lo, hi), span(bx), better, a.same)]
                cells.append(c)
            err = sum(r["errors"] for r in reps), sum(r["not_length"] for r in reps)
            case = f"{kind} x{k}" if kind == "fork" else f"after idle x{k}"
            print(f"| {size} | {case} | {lab} | {len(reps)} | " + " | ".join(cells) + f" | {err[0]}/{err[1]} |")
    print(f"\nverdict marks against {a.base} on the same arm: + better, = same (within {a.same}% or overlapping rep range), - worse")


if __name__ == "__main__":
    main()
