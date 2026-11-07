#!/usr/bin/env python3
"""The tables of measure.sh: each arm per workload, its ratio to the baseline, and what a per-request choice would gain.

Rates are ratios of sums over the turns every compared arm decoded: tokens per step (generated tokens over verify
passes, the first token aside), accepted drafts per step (the server's counters) and decode tokens per second, with a
bootstrap 95% interval over turns for a ratio to the baseline. Greedy arms must write what the baseline wrote, so a
greedy arm's differing turns are counted.

A choice is scored on the same turns by leave-one-out: each turn gets the arm that was fastest over the other turns of
its class, so the class's own noise does not count as a gain. "oracle" is the fastest arm for each turn, the upper bound.

Usage: by_workload.py OUTDIR [BASE_LABEL]    OUTDIR as measure.sh left it; the baseline defaults to rtn
"""
import glob
import json
import os
import random
import re
import sys

GREEDY_ONLY_MODE = "greedy"
PROSE = re.compile(r"(chat|phone|think|lang)\d+$")
AGENT = re.compile(r"t\d+$")


def load(path):
    """The turns of one run.py output with their mode: ({id: record}, mode). Turns that failed or wrote one token drop out."""
    recs = [json.loads(line) for line in open(path, encoding="utf-8")]
    head = next((r for r in recs if r.get("header")), {})
    turns = {r["id"]: r for r in recs if r.get("srv") and not r.get("error") and r["srv"]["gen"] > 1 and r["srv"]["decode_s"] > 0}
    return turns, head.get("mode", GREEDY_ONLY_MODE)


def sums(rows):
    """(tokens, steps, seconds, accepted, spec steps) over records."""
    return (sum(r["srv"]["gen"] - 1 for r in rows), sum(r["srv"]["verify"] for r in rows), sum(r["srv"]["decode_s"] for r in rows),
            sum(r.get("spec", {}).get("accepted", 0) for r in rows), sum(r.get("spec", {}).get("steps", 0) for r in rows))


def ratio_ci(base, arm, ids, seed=1):
    """The tokens-per-second ratio arm/base and its bootstrap 95% interval over the turns."""
    if not ids:
        return None, None, None
    rng, out = random.Random(seed), []

    def tps(rows, idx):
        return sum(rows[i]["srv"]["gen"] - 1 for i in idx) / sum(rows[i]["srv"]["decode_s"] for i in idx)

    for _ in range(2000):
        k = [rng.choice(ids) for _ in ids]
        out.append(tps(arm, k) / tps(base, k))
    out.sort()
    return tps(arm, ids) / tps(base, ids), out[50], out[1949]


def table_arms(arms, base, classes, modes):
    """One row per arm and workload, against the baseline's turns."""
    print("| arm | workload | turns | tokens/step | accepted/step | tok/s | x base tok/s [95%] | differing |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for label, turns in arms.items():
        for wl in sorted(set(classes.values())):
            ids = sorted(i for i in turns if classes.get(i) == wl and i in base)
            if not ids:
                continue
            g, v, d, acc, st = sums([turns[i] for i in ids])
            r, lo, hi = ratio_ci(base, turns, ids)
            diff = sum(turns[i].get("text_sha256") != base[i].get("text_sha256") for i in ids) if modes[label] == "greedy" else "-"
            print(f"| {label} | {wl} | {len(ids)} | {g / v if v else 0:.2f} | {acc / st if st else 0:.2f} | {g / d:.1f} | "
                  f"{r:.3f} [{lo:.3f}-{hi:.3f}] | {diff} |")


def table_gate(base):
    """MULTI_DRAFTER's phase 0 gate on the baseline: asklong (prose and doc*) against agentlong (t*), tokens per step."""
    sets = {"asklong": [r for i, r in base.items() if i.startswith("doc") or PROSE.match(i)],
            "agentlong": [r for i, r in base.items() if AGENT.match(i)]}
    if not all(sets.values()):
        return

    def rate(rows):
        return sum(r["srv"]["gen"] - 1 for r in rows) / sum(r["srv"]["verify"] for r in rows)

    rng, out = random.Random(1), []
    for _ in range(2000):
        out.append(rate(rng.choices(sets["asklong"], k=len(sets["asklong"]))) / rate(rng.choices(sets["agentlong"], k=len(sets["agentlong"]))))
    out.sort()
    ask, agent = rate(sets["asklong"]), rate(sets["agentlong"])
    print(f"\nPhase 0 gate: asklong {ask:.2f} tokens/step over {len(sets['asklong'])} turns, agentlong {agent:.2f} over {len(sets['agentlong'])}: "
          f"{100 * (1 - ask / agent):.1f}% fewer [{100 * (1 - out[1949]):.1f}-{100 * (1 - out[50]):.1f}], the bar is 15%.")


def pooled(picks, arms):
    """Tokens per second over the turns, each decoded by the arm picks[id] names."""
    g = sum(arms[a][i]["srv"]["gen"] - 1 for i, a in picks.items())
    return g / sum(arms[a][i]["srv"]["decode_s"] for i, a in picks.items())


def choose(arms, ids, classes):
    """Leave-one-out: each turn takes the arm fastest over the other turns of its class."""
    picks = {}
    for i in ids:
        others = [j for j in ids if classes[j] == classes[i] and j != i]
        pool = others or ids
        picks[i] = max(arms, key=lambda a: sum(arms[a][j]["srv"]["gen"] - 1 for j in pool) / sum(arms[a][j]["srv"]["decode_s"] for j in pool))
    return picks


def table_choice(arms, base_label, by_name):
    """What each way of choosing gains over always using the baseline, on the greedy arms, per class and overall."""
    ids = sorted(set.intersection(*(set(t) for t in arms.values())))
    if not ids:
        return
    base = pooled(dict.fromkeys(ids, base_label), arms)
    print(f"\nChoice among {', '.join(arms)} over {len(ids)} turns; always {base_label} = {base:.2f} tok/s.\n")
    print("| choice | tok/s | x always-base |")
    print("|---|---:|---:|")
    best = max(arms, key=lambda a: pooled(dict.fromkeys(ids, a), arms))
    print(f"| best single arm ({best}) | {pooled(dict.fromkeys(ids, best), arms):.2f} | {pooled(dict.fromkeys(ids, best), arms) / base:.3f} |")
    for name, classes in by_name.items():
        p = pooled(choose(arms, ids, classes), arms)
        print(f"| by {name}, leave-one-out | {p:.2f} | {p / base:.3f} |")
    oracle = {i: max(arms, key=lambda a: (arms[a][i]["srv"]["gen"] - 1) / arms[a][i]["srv"]["decode_s"]) for i in ids}
    p = pooled(oracle, arms)
    print(f"| oracle per turn | {p:.2f} | {p / base:.3f} |")


def table_streams(out_dir):
    """The N-session arms, one row per arm and workload (from n_streams.py's summaries)."""
    rows = sorted(glob.glob(os.path.join(out_dir, "*.streams.jsonl")))
    if not rows:
        return
    print("\n| arm | workload | streams | requests (errors) | accepted/step (seq) | tok/s per request | tok/s aggregate |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for path in rows:
        s = next((r for r in map(json.loads, open(path, encoding="utf-8")) if r.get("summary")), None)
        if s:
            arm, wl = os.path.basename(path).split(".")[:2]
            print(f"| {arm} | {wl} | {s['streams']} | {s['requests']} ({s.get('errors', 0)}) | {s.get('accepted_per_seq_step')} | "
                  f"{s.get('request_decode_tps_pooled')} | {s.get('aggregate_decode_tps_any')} |")


def main():
    out_dir = sys.argv[1]
    base_label = sys.argv[2] if len(sys.argv) > 2 else "rtn"
    with open(os.path.join(out_dir, "set", "all.json"), encoding="utf-8") as f:
        manifest = {t["id"]: t for t in json.load(f)["turns"]}
    arms, modes = {}, {}
    for path in sorted(glob.glob(os.path.join(out_dir, "*.jsonl"))):
        if path.endswith(".streams.jsonl"):
            continue
        label = os.path.basename(path)[:-len(".jsonl")]
        arms[label], modes[label] = load(path)
    base = arms[base_label]
    classes = {i: t["workload"] for i, t in manifest.items()}
    table_arms(arms, base, classes, modes)
    table_gate(base)
    greedy = {k: v for k, v in arms.items() if modes[k] == "greedy" and k != "none" and k != "rtn-again"}
    rule = {i: f"{t['route']}{'-long' if t['long'] else ''}" for i, t in manifest.items()}
    table_choice(greedy, base_label, {"workload (declared, as a hint would)": classes, "request shape (route_rule.py)": rule})
    table_streams(out_dir)
    wrong = {i: (t["workload"], t["route"]) for i, t in manifest.items()
             if t["route"] != {"prose": "ask", "code-long": "work", "code-short": "work", "read-long": "work"}[t["workload"]]
             and not i.startswith("doc")}
    print(f"\nRule: {len(manifest) - len(wrong)}/{len(manifest)} routed as their workload; misrouted {sorted(wrong)}.")


if __name__ == "__main__":
    main()
