"""Per-layer K and V scales for the FP8 KV cache (LOWBIT section 8, B41) when the checkpoint carries none.

The checkpoint's config names an FP8 static KV scheme but its index holds no k_scale or v_scale tensors, so the scales
come from a calibration run: the engine, with RAGON_KV_CALIB=RAW set, records per full-attention layer the absolute
maximum of every K and V row it writes in an eager prefill or decode (bit-exact otherwise; the knob stays off), and
writes the maxima to RAW when it exits. This script runs that and writes the sidecar --kv-fp8-scales reads:

    {"source": "...", "layers": N, "k_scale": [N], "v_scale": [N], "k_amax": [N], "v_amax": [N], "calib": {...}}

with scale = amax / 448 (E4M3's largest finite value), so no stored value saturates on the calibration set.

  kv_calib.py run --run BUILD/ragon-run --model M --prompts PROMPTS.jsonl --out SIDECAR.json [--max-tokens 32]
  kv_calib.py render --template BUILD/ragon-template --model M --bodies BODIES.jsonl --out PROMPTS.jsonl
  kv_calib.py convert RAW.json SIDECAR.json [--source TEXT]

`render` turns Responses request bodies (bench/agentlong's bodies.jsonl) into ragon-run's prompt file (one JSON string
per line), the text ragon-serve would prefill. Calibrate on a set disjoint from the one the agreement gate scores.
"""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import tempfile

E4M3_MAX = 448.0


def convert(raw_path, out_path, source, extra=None):
    raw = json.load(open(raw_path))
    k, v = raw["k_amax"], raw["v_amax"]
    if len(k) != len(v) or not k:
        sys.exit("kv_calib: the raw maxima are empty or uneven")
    n = len(k)
    # The first and last full-attention layers stay BF16 (the engine ignores their scales); every other layer needs
    # a positive maximum, or the calibration did not reach it.
    for i in range(1, n - 1):
        if not (k[i] > 0 and v[i] > 0):
            sys.exit(f"kv_calib: layer slot {i} saw no K or V (k_amax {k[i]}, v_amax {v[i]})")
    side = {
        "source": source,
        "layers": n,
        "k_scale": [max(a, 1e-12) / E4M3_MAX for a in k],
        "v_scale": [max(a, 1e-12) / E4M3_MAX for a in v],
        "k_amax": k,
        "v_amax": v,
        "calib": dict(extra or {}, runs=raw.get("runs", 1)),
    }
    with open(out_path, "w") as f:
        json.dump(side, f, indent=1)
        f.write("\n")
    print(f"kv_calib: {n} full-attention layers; K amax {min(k[1:-1]):.3g}..{max(k[1:-1]):.3g}, "
          f"V amax {min(v[1:-1]):.3g}..{max(v[1:-1]):.3g} (slots 1..{n - 2}); wrote {out_path}")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--run", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--prompts", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--max-tokens", type=int, default=32)
    r.add_argument("--max-context", type=int, default=65536)
    t = sub.add_parser("render")
    t.add_argument("--template", required=True)
    t.add_argument("--model", required=True)
    t.add_argument("--bodies", required=True)
    t.add_argument("--out", required=True)
    c = sub.add_parser("convert")
    c.add_argument("raw")
    c.add_argument("out")
    c.add_argument("--source", default="calibration (bench/quant/kv_calib.py)")
    a = ap.parse_args()

    if a.cmd == "convert":
        convert(a.raw, a.out, a.source)
    elif a.cmd == "render":
        tok = os.path.join(a.model, "tokenizer.json")
        with open(a.bodies) as src:
            p = subprocess.run([a.template, "--responses", tok], stdin=src, capture_output=True, text=True, check=True)
        n = 0
        with open(a.out, "w") as f:
            for line in p.stdout.splitlines():
                if line.strip():
                    f.write(json.dumps(json.loads(line)["text"]) + "\n")
                    n += 1
        print(f"kv_calib: rendered {n} prompts to {a.out}")
    else:
        raw = tempfile.NamedTemporaryFile(prefix="kv_amax.", suffix=".json", delete=False).name
        os.unlink(raw)
        env = dict(os.environ, RAGON_KV_CALIB=raw)
        env.pop("RAGON_KV_FP8", None)
        cmd = [a.run, "--model", a.model, "--prompts-file", a.prompts, "--max-tokens", str(a.max_tokens),
               "--max-context", str(a.max_context), "--sessions", "1"]
        p = subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL)
        if p.returncode != 0 or not os.path.exists(raw):
            sys.exit(f"kv_calib: ragon-run exited {p.returncode} without the maxima")
        n = sum(1 for l in open(a.prompts) if l.strip())
        day = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
        src = (f"calibration {day}: absmax over {n} prompts (prompts sha256 {sha(a.prompts)}), "
               f"{a.max_tokens} greedy tokens each, scale = amax/448")
        convert(raw, a.out, src, {"prompts": n, "prompts_sha256": sha(a.prompts), "max_tokens": a.max_tokens})
        os.unlink(raw)


if __name__ == "__main__":
    main()
