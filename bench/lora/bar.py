#!/usr/bin/env python3
"""The verdict on the adapters' bar (LORA §6), from the files bar.steps and agree.steps leave:

    bar.py RUN_DIR REF_DIR

Speed: the zero adapter against the base, decode tok/s at least 0.95x and prefill time per 1,000 new tokens at most
1.25x, with every turn's tokens the same (else the ratios compare different work). Agreement: rand-x32 teacher-forced on
the reference's greedy continuation, the agreement gate's lines (MILESTONES) with the tolerance of VISION §20.5 item 5.
The strict flips are not significantly above the reference's own decode against its own prefill (one-sided Fisher,
alpha 0.05), none concentrates late or on long prompts, the worst gap is at most 1.75 nats or the reference's own and the
top-5 misses at most its own, and the tool names on the agent prompts match at least as often as it matches itself. The
engine without the adapter on the same cases, the control, has to fail the flip line, or the gate has no power over the
adapter. Free-running identity is no line: the reference does not reproduce itself, which the
verdict shows. Prints the verdict and writes it to RUN_DIR/bar.json; exits 1 unless the bar is met.
"""
import json, math, re, subprocess, sys
from pathlib import Path

DECODE_MIN, PREFILL_MAX = 0.95, 1.25
ALPHA, GAP_MAX = 0.05, 1.75
ABSENT = -99.0  # a token outside the reference's top-20, as analyze.py scores it


def load(path):
    return [json.loads(line) for line in open(path)]


def summary(path):
    return next(r for r in load(path) if r.get("summary"))


def tokens(path):
    return [r.get("ids", r.get("cont")) for r in load(path)]


def agreement(got, want):
    """Prompts identical to the reference (up to the shorter, one side may keep the stop token) and tokens up to the first difference."""
    same = upto = total = 0
    for g, w in zip(got, want):
        n = min(len(g), len(w))
        first = next((i for i in range(n) if g[i] != w[i]), n)
        same += first == n
        upto += first
        total += n
    return {"prompts": min(len(got), len(want)), "identical": same, "tokens_to_first_difference": upto, "tokens": total}


def upper_tail(a, n1, b, n2):
    """One-sided Fisher exact test that a/n1 exceeds b/n2: the hypergeometric upper tail, the observed table included."""
    k = a + b
    return sum(math.comb(n1, x) * math.comb(n2, k - x) for x in range(a, min(k, n1) + 1)) / math.comb(n1 + n2, k)


def best_first(pos):
    return sorted(pos, key=lambda c: -c[1])


def score(choices, reference, prompts):
    """A run's picks (a token per prompt and position) against the reference's teacher-forced top-20 on the same positions."""
    median = sorted(map(len, prompts))[len(prompts) // 2]
    out = {"positions": 0, "flips": 0, "worst_gap": 0.0, "top5_misses": 0, "gaps_over": 0}
    by_third, by_length = [[0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0]]
    for picks, ref, prompt in zip(choices, reference, prompts):
        for j, (pick, pos) in enumerate(zip(picks, ref["top"])):
            ranked = best_first(pos)
            third, length = by_third[min(j // 16, 2)], by_length[len(prompt) >= median]
            out["positions"] += 1
            third[1] += 1
            length[1] += 1
            out["top5_misses"] += pick not in [c[0] for c in ranked[:5]]
            gap = ranked[0][1] - dict(map(tuple, ranked)).get(pick, ABSENT)
            if pick != ranked[0][0] and gap > 0:
                out["flips"] += 1
                third[0] += 1
                length[0] += 1
                out["worst_gap"] = max(out["worst_gap"], gap)
                out["gaps_over"] += gap > GAP_MAX
    out["late_p"] = upper_tail(by_third[2][0], by_third[2][1], by_third[0][0], by_third[0][1])
    out["long_prompt_p"] = upper_tail(by_length[1][0], by_length[1][1], by_length[0][0], by_length[0][1])
    return out


def gate(engine_path, force_path, gen_path, prompts):
    """The engine's forced picks and the reference's own decode, each scored against the reference's prefill."""
    reference = load(force_path)
    engine = score([[p["top"][0][0] for p in r["pos"]] for r in load(engine_path)], reference, prompts)
    noise = score([[best_first(p)[0][0] for p in r["top"]] for r in load(gen_path)], reference, prompts)
    p = upper_tail(engine["flips"], engine["positions"], noise["flips"], noise["positions"])
    return {
        "engine": engine, "reference_noise": noise, "flip_p": p, "flips_ok": p >= ALPHA,
        "gap_ok": engine["worst_gap"] <= max(GAP_MAX, noise["worst_gap"]),
        "top5_ok": engine["top5_misses"] <= noise["top5_misses"],
        "concentration_ok": engine["late_p"] >= ALPHA and engine["long_prompt_p"] >= ALPHA,
    }


def tool_names(text):
    return re.findall(r"<tool_call>.*?<function=([^>\n]+)>", text, re.S)


def same_tools(a, b):
    return sum(tool_names(x["text"]) == tool_names(y["text"]) for x, y in zip(a, b))


def main():
    run, ref = Path(sys.argv[1]), Path(sys.argv[2])
    base1, base2, zero = (summary(run / f"{n}.jsonl") for n in ("base", "base2", "zero"))
    decode_base = (base1["decode_tps"] + base2["decode_tps"]) / 2
    prefill_base = (base1["prefill_ms_per_1k_new"] + base2["prefill_ms_per_1k_new"]) / 2
    decode, prefill = zero["decode_tps"] / decode_base, zero["prefill_ms_per_1k_new"] / prefill_base
    compare = subprocess.run([sys.executable, str(Path(__file__).parent.parent / "agentlong/run.py"), "--compare",
                              str(run / "base.jsonl"), str(run / "zero.jsonl")], capture_output=True, text=True).stdout
    turns = next((l for l in compare.splitlines() if l.endswith("turns identical")), "no comparison")
    same, _, total = turns.split()[0].partition("/")
    zero_same = same == total
    prompts = [r["prompt"] for r in load(run / "cases-lora.jsonl")]
    lora_gate = gate(run / "eng-lora.jsonl", ref / "force-lora.jsonl", ref / "ref-lora.jsonl", prompts)
    control = gate(run / "eng-control.jsonl", ref / "force-lora.jsonl", ref / "ref-lora.jsonl", prompts)
    long_ref, long_ref2 = load(ref / "long-lora.jsonl"), load(ref / "long-lora-run2.jsonl")
    tools, tools_self = same_tools(load(run / "long-lora.jsonl"), long_ref), same_tools(long_ref2, long_ref)
    verdict = {
        "decode_ratio": round(decode, 4), "decode_ok": decode >= DECODE_MIN,
        "prefill_ratio": round(prefill, 4), "prefill_ok": prefill <= PREFILL_MAX,
        "base_drift": round(abs(base1["decode_tps"] - base2["decode_tps"]) / decode_base, 4),
        "zero_is_base": zero_same, "zero_compare": turns,
        "lora_gate": lora_gate, "control": control, "control_fails": not control["flips_ok"],
        "tool_names": {"engine": tools, "reference_itself": tools_self, "of": len(long_ref)},
        "reference_vs_itself": agreement(tokens(ref / "ref-lora.jsonl"), tokens(ref / "ref-lora-run2.jsonl")),
    }
    verdict["agrees"] = bool(lora_gate["flips_ok"] and lora_gate["gap_ok"] and lora_gate["top5_ok"]
                             and lora_gate["concentration_ok"] and verdict["control_fails"] and tools >= tools_self)
    verdict["met"] = bool(verdict["decode_ok"] and verdict["prefill_ok"] and zero_same and verdict["agrees"])
    (run / "bar.json").write_text(json.dumps(verdict, indent=1) + "\n")
    print(json.dumps(verdict, indent=1))
    print("BAR MET" if verdict["met"] else "BAR NOT MET")
    return 0 if verdict["met"] else 1


if __name__ == "__main__":
    sys.exit(main())
