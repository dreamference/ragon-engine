#!/usr/bin/env python3
"""Rung 2 of a drafter A/B (DRAFTER_RETRAIN §5), the analysis and the agreement client.

  rung2.py steps DUMP [--upto BYTES]       per-position acceptance from a greedy RAGON_DUMP_STEPS file: tokens per step
                                           (accepted + 1) and the chain's acceptance at each draft depth, overall and
                                           in context buckets (<8K, 8-25K, 25-40K, 40-64K, >=64K). Greedy output is the
                                           target's own continuation whatever was drafted, so two drafters' dumps of
                                           the same greedy run are paired step for step.
  rung2.py turns AGENTLONG_JSONL            tokens per step of a production-flags agentlong run (tree verification) per
                                           context bucket, from each turn's server counters (generated / verify passes).
                                           RAGON_DUMP_STEPS records chain steps only, so `steps` needs a `--tree 0` run.
  rung2.py agree URL AGR_DIR TOKENIZER OUT greedy /v1/completions over the agreement gate's prompts (cases.jsonl: the
                                           54 prompts with production's 48-token continuations; cases-long.jsonl with
                                           prod-gen400-long.jsonl: the 24 agent prompts' 400-token first responses)
                                           through a ragon-serve running the candidate drafter; prints the exact-match
                                           count and the matched-prefix fraction against production's text (the
                                           reference ids decoded with the target's tokenizer.json).
  rung2.py same A.jsonl B.jsonl            two drafters' agree outputs on the same build: the count of texts that differ.
                                           Greedy with an exact verifier, a drafter cannot change the target's output,
                                           so this must be 0; the counts against production's tokens then equal too.
"""
import json
import struct
import sys
import urllib.request

STEP_MAGIC = 0x54524545
BUCKETS = [(0, 8192, "<8K"), (8192, 25600, "8-25K"), (25600, 40960, "25-40K"), (40960, 65536, "40-64K"),
           (65536, 1 << 30, ">=64K")]


def steps(path, upto=None):
    with open(path, "rb") as f:
        while upto is None or f.tell() < upto:
            h = f.read(36)
            if len(h) < 36:
                return
            magic, pos, anchor, E, K16, K, accepted, nxt, greedy = struct.unpack("<9i", h)
            assert magic == STEP_MAGIC, f"bad step magic at {f.tell() - 36}"
            f.seek(E * K16 * 4 + E * K16 * K16 * 4 + (E + 1) * 4 + (E + 1) * K * 8, 1)
            yield pos, E, accepted


def cmd_steps(path, upto):
    rows = {b[2]: [0, 0, None] for b in BUCKETS}
    rows["all"] = [0, 0, None]
    for pos, E, acc in steps(path, upto):
        for lo, hi, name in BUCKETS:
            if lo <= pos < hi:
                break
        for key in (name, "all"):
            r = rows[key]
            r[0] += 1
            r[1] += acc + 1
            if r[2] is None:
                r[2] = [0] * E
            for d in range(min(acc, E)):
                r[2][d] += 1
    out = {}
    for key, (n, toks, depth) in rows.items():
        if n:
            out[key] = {"steps": n, "tok_per_step": round(toks / n, 3),
                        "accept_at_depth": [round(x / n, 3) for x in depth]}
    print(json.dumps(out))


def cmd_turns(path):
    rows = {b[2]: [0, 0, 0] for b in BUCKETS}  # turns, generated tokens, verify passes
    rows["all"] = [0, 0, 0]
    for line in open(path):
        d = json.loads(line)
        srv = d.get("srv")
        if not srv or not srv.get("verify"):
            continue
        name = next(n for lo, hi, n in BUCKETS if lo <= srv["prompt"] < hi)
        for key in (name, "all"):
            r = rows[key]
            r[0] += 1
            r[1] += srv["gen"]
            r[2] += srv["verify"]
    print(json.dumps({k: {"turns": n, "tokens": g, "tok_per_step": round(g / v, 3)}
                      for k, (n, g, v) in rows.items() if n}))


def post(url, body):
    r = urllib.request.Request(url.rstrip("/") + "/v1/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=3600))


def decoder(tokenizer_json):
    """ids -> text with the byte-level BPE vocab of tokenizer.json (GPT-2 byte mapping)."""
    tok = json.load(open(tokenizer_json))
    inv = {v: k for k, v in tok["model"]["vocab"].items()}
    special = {t["id"]: t["content"] for t in tok.get("added_tokens", [])}
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    back = {chr(c): b for b, c in zip(bs, cs)}

    def decode(ids):
        out = bytearray()
        for i in ids:
            if i in special:
                out += special[i].encode()
            else:
                out += bytes(back[ch] for ch in inv[i])
        return out.decode("utf-8", "replace")
    return decode


def cmd_agree(url, d, tokenizer_json, out):
    decode = decoder(tokenizer_json)
    short = [json.loads(x) for x in open(f"{d}/cases.jsonl")]
    long_p = [json.loads(x) for x in open(f"{d}/cases-long.jsonl")]
    long_ref = [json.loads(x) for x in open(f"{d}/prod-gen400-long.jsonl")]
    cases = [(c["prompt"], c["cont"], "short") for c in short] + \
            [(p["prompt"], r["cont"], "long") for p, r in zip(long_p, long_ref)]
    res = {"short": [0, 0, 0, 0], "long": [0, 0, 0, 0]}  # cases, exact, matched chars, reference chars
    with open(out, "w") as f:
        for prompt, ref, kind in cases:
            ref_text = decode([t for t in ref if t not in (248046, 248044)])
            c = post(url, {"prompt": prompt, "max_tokens": len(ref), "temperature": 0})
            text = c["choices"][0]["text"]
            m = 0
            while m < min(len(text), len(ref_text)) and text[m] == ref_text[m]:
                m += 1
            f.write(json.dumps({"kind": kind, "matched_chars": m, "ref_chars": len(ref_text), "text": text}) + "\n")
            r = res[kind]
            r[0] += 1
            r[1] += text == ref_text
            r[2] += m
            r[3] += len(ref_text)
    print(json.dumps({k: {"cases": v[0], "exact": v[1], "matched_frac": round(v[2] / v[3], 4) if v[3] else None}
                      for k, v in res.items()}))


def cmd_same(a, b):
    ta = [json.loads(x)["text"] for x in open(a)]
    tb = [json.loads(x)["text"] for x in open(b)]
    diff = [i for i, (x, y) in enumerate(zip(ta, tb)) if x != y]
    print(json.dumps({"cases": len(ta), "compared": min(len(ta), len(tb)), "differ": len(diff), "first": diff[:10]}))
    return 1 if diff or len(ta) != len(tb) else 0


if __name__ == "__main__":
    if sys.argv[1] == "steps":
        cmd_steps(sys.argv[2], int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[3] == "--upto" else None)
    elif sys.argv[1] == "turns":
        cmd_turns(sys.argv[2])
    elif sys.argv[1] == "same":
        sys.exit(cmd_same(sys.argv[2], sys.argv[3]))
    elif sys.argv[1] == "agree":
        cmd_agree(*sys.argv[2:6])
    else:
        sys.exit(__doc__)
