"""A needle set for the FP8 KV cache's quality gate (LOWBIT section 8, B41): one fact hidden at a chosen depth in
about 32K tokens of the repository's own specs, then a question about it, answered greedily by bench/agreement's
ragon-force with the knob off and on. The gate: the FP8 build finds every needle the BF16 build finds.

  kv_needle.py build --force BUILD/ragon-force --model M --haystack 'SRC/specs/*.md' --out DIR [--tokens 32000]
  kv_needle.py score DIR/cases.jsonl LABEL=GEN.jsonl [LABEL=GEN.jsonl ...]

build writes DIR/cases.jsonl ({"prompt": ids}, ragon-force gen's input) and DIR/needles.jsonl (the code each case
hides, its depth and its token count). Run `ragon-force gen M 24 < cases.jsonl > GEN.jsonl` per build, then score.
"""
import argparse
import glob
import json
import random
import re
import subprocess
import sys

CITIES = ["Valparaiso", "Tromso", "Ushuaia", "Hobart", "Windhoek", "Ulaanbaatar", "Reykjavik", "Lhasa"]
DEPTHS = [0.03, 0.15, 0.3, 0.45, 0.6, 0.75, 0.88, 0.97]


def chat(user):
    return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def tokenize(force, model, texts):
    p = subprocess.run([force, "tokenize", model], input="".join(json.dumps(t) + "\n" for t in texts),
                       capture_output=True, text=True, check=True)
    return [json.loads(l)["ids"] for l in p.stdout.splitlines() if l.strip()]


def build(a):
    files = sorted(glob.glob(a.haystack))
    if not files:
        sys.exit("kv_needle: no haystack files")
    paras = [p.strip() for f in files for p in open(f).read().split("\n\n") if p.strip()]
    # A long run of digits in the text could pass for an answer: cut every run of five or more to its first four.
    paras = [re.sub(r"(\d{4})\d+", r"\1", p) for p in paras]
    full = "\n\n".join(paras)
    ratio = len(full) / len(tokenize(a.force, a.model, [full])[0])  # characters per token
    def make(scale):
        rng = random.Random(41)
        texts, needles = [], []
        for i, depth in enumerate(DEPTHS * 2):
            city = CITIES[i % len(CITIES)] if i < len(DEPTHS) else CITIES[(i + 3) % len(CITIES)]
            code = "".join(rng.choice("0123456789") for _ in range(7))
            needle = f"The access code for the {city} archive is {code}."
            q = (f"\n\nAccording to the document above, what is the access code for the {city} archive? "
                 "Answer with the seven digits only.")
            start = rng.randrange(len(paras))
            order = paras[start:] + paras[:start]
            budget = int((a.tokens - 80) * ratio * scale[i])
            hay, n = [], 0
            for p in order * 8:
                if n + len(p) + 2 > budget:
                    break
                hay.append(p)
                n += len(p) + 2
            hay.insert(int(len(hay) * depth), needle)
            texts.append(chat("Read the document, then answer the question after it.\n\n" + "\n\n".join(hay) + q))
            needles.append({"case": i, "city": city, "code": code, "depth": depth})
        return texts, needles

    # Twice: the second pass corrects each case's budget by its own characters per token.
    texts, needles = make([1.0] * (2 * len(DEPTHS)))
    first = tokenize(a.force, a.model, texts)
    texts, needles = make([a.tokens / len(x) for x in first])
    ids = tokenize(a.force, a.model, texts)
    with open(f"{a.out}/cases.jsonl", "w") as f:
        for x in ids:
            f.write(json.dumps({"prompt": x}) + "\n")
    with open(f"{a.out}/needles.jsonl", "w") as f:
        for nd, x in zip(needles, ids):
            f.write(json.dumps(dict(nd, tokens=len(x))) + "\n")
    lens = [len(x) for x in ids]
    print(f"kv_needle: {len(ids)} cases, {min(lens)}..{max(lens)} tokens, depths {DEPTHS} twice")


def score(a):
    needles = [json.loads(l) for l in open(a.cases.replace("cases.jsonl", "needles.jsonl"))]
    hits = {}
    for spec in a.gens:
        label, path = spec.split("=", 1)
        gen = [json.loads(l) for l in open(path)]
        hits[label] = [nd["code"] in g["text"] for nd, g in zip(needles, gen)]
        if len(gen) != len(needles):
            print(f"{label}: {len(gen)} answers for {len(needles)} cases")
    for label, h in hits.items():
        miss = [f"{n['city']}@{n['depth']}" for n, x in zip(needles, h) if not x]
        print(f"{label}: {sum(h)}/{len(h)} hits; missed {miss}")
    labels = list(hits)
    if len(labels) >= 2:
        ref, new = hits[labels[0]], hits[labels[1]]
        lost = sum(1 for r, n in zip(ref, new) if r and not n)
        print(f"{labels[1]} vs {labels[0]}: {sum(r and n for r, n in zip(ref, new))} of {labels[0]}'s {sum(ref)} hits kept, "
              f"{lost} lost")
    json.dump({k: v for k, v in hits.items()}, open(a.cases.replace("cases.jsonl", "hits.json"), "w"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--force", required=True)
    b.add_argument("--model", required=True)
    b.add_argument("--haystack", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--tokens", type=int, default=32000)
    s = sub.add_parser("score")
    s.add_argument("cases")
    s.add_argument("gens", nargs="+")
    a = ap.parse_args()
    build(a) if a.cmd == "build" else score(a)


if __name__ == "__main__":
    main()
