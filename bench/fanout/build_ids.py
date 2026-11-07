#!/usr/bin/env python3
"""Writes the token ids run.py sends: a prefix cut from a source text and one question per fork, naming its functions.

Usage: build_ids.py TOKENIZER_JSON SOURCE_TEXT OUT_JSON [--ctx N] [--questions N]
"""
import argparse
import json
import re

from tokenizers import Tokenizer

TAIL = "\n\nAnswer:"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tokenizer")
    ap.add_argument("source")
    ap.add_argument("out")
    ap.add_argument("--ctx", type=int, default=32000)
    ap.add_argument("--questions", type=int, default=64)
    a = ap.parse_args()
    tok = Tokenizer.from_file(a.tokenizer)
    text = open(a.source).read()
    names = sorted(set(re.findall(r"\b(\w{6,})\(", text)))
    ids = tok.encode(text, add_special_tokens=False).ids
    assert len(ids) >= a.ctx, f"the source has {len(ids)} tokens, {a.ctx} wanted"
    qs = []
    for i in range(a.questions):
        x, y = names[i % len(names)], names[(i * 7 + 3) % len(names)]
        q = (f"\n\nQuestion {i}: in the source code above, explain what the function `{x}` does, which arguments it "
             f"takes and what it returns, which error cases it handles, and how it relates to `{y}`. Name one caller "
             f"of `{x}` if there is one. Be concrete and refer to the code, in plain prose, without repeating the "
             f"code itself.")
        qs.append(tok.encode(q + TAIL, add_special_tokens=False).ids)
    json.dump({"prefix": ids[: a.ctx], "questions": qs}, open(a.out, "w"))
    print(f"prefix {a.ctx} of {len(ids)} tokens, {len(qs)} questions of {min(map(len, qs))}-{max(map(len, qs))} tokens")


if __name__ == "__main__":
    main()
