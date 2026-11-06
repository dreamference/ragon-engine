#!/usr/bin/env python3
"""The verdict on the adapters' bar (LORA §6), from the files bar.steps leaves:

    bar.py RUN_DIR REF_DIR

Speed: the zero adapter against the mean of the two base runs around it, decode tok/s at least 0.95x and prefill time
per 1,000 new tokens at most 1.25x, with every turn's tokens the same (else the ratios compare different work). Correctness: rand-split's greedy
continuations against the reference's, and the base's against the reference's base as the same comparison's floor.
Prints the verdict and writes it to RUN_DIR/bar.json; exits 1 unless the bar is met.
"""
import json, subprocess, sys
from pathlib import Path

DECODE_MIN, PREFILL_MAX = 0.95, 1.25


def summary(path):
    return next(r for r in map(json.loads, open(path)) if r.get("summary"))


def tokens(path):
    return [r.get("ids", r.get("cont")) for r in map(json.loads, open(path))]


def agreement(got, want):
    """Prompts identical to the reference (up to the shorter, one side may keep the stop token) and tokens up to the first difference."""
    same = upto = total = 0
    for g, w in zip(got, want):
        n = min(len(g), len(w))
        first = next((i for i in range(n) if g[i] != w[i]), n)
        same += first == n
        upto += first
        total += n
    return {"prompts": len(want), "identical": same, "tokens_to_first_difference": upto, "tokens": total}


def main():
    run, ref = Path(sys.argv[1]), Path(sys.argv[2])
    bases, zero = [summary(run / n) for n in ("base.jsonl", "base2.jsonl")], summary(run / "zero.jsonl")
    mean = lambda key: sum(b[key] for b in bases) / len(bases)
    decode = zero["decode_tps"] / mean("decode_tps")
    prefill = zero["prefill_ms_per_1k_new"] / mean("prefill_ms_per_1k_new")
    cmp_out = subprocess.run([sys.executable, str(Path(__file__).parent.parent / "agentlong/run.py"), "--compare",
                              str(run / "base.jsonl"), str(run / "zero.jsonl")], capture_output=True, text=True).stdout
    last = cmp_out.strip().splitlines()[-1] if cmp_out.strip() else "no comparison"
    zero_same = last.split()[0].split("/")[0] == last.split()[0].split("/")[-1] and "identical" in last
    lora = agreement(tokens(run / "gen-lora.jsonl"), tokens(ref / "ref-lora.jsonl"))
    floor = agreement(tokens(run / "gen-base.jsonl"), tokens(ref / "ref-base.jsonl"))
    self_lora = agreement(tokens(ref / "ref-lora.jsonl"), tokens(ref / "ref-lora-run2.jsonl"))
    verdict = {
        "base_drift": round(bases[1]["decode_tps"] / bases[0]["decode_tps"], 4),
        "decode_ratio": round(decode, 4), "decode_ok": decode >= DECODE_MIN,
        "prefill_ratio": round(prefill, 4), "prefill_ok": prefill <= PREFILL_MAX,
        "zero_is_base": zero_same, "zero_compare": last,
        "lora_vs_reference": lora, "base_vs_reference": floor, "reference_vs_itself": self_lora,
        "correct": lora["identical"] == lora["prompts"],
    }
    verdict["met"] = bool(verdict["decode_ok"] and verdict["prefill_ok"] and zero_same and verdict["correct"])
    (run / "bar.json").write_text(json.dumps(verdict, indent=1) + "\n")
    print(json.dumps(verdict, indent=1))
    print("BAR MET" if verdict["met"] else "BAR NOT MET")
    return 0 if verdict["met"] else 1


if __name__ == "__main__":
    sys.exit(main())
