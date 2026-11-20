"""Compares a ragon-metal-run --dump directory with a qwen_ref.py --dump directory (METAL spec, first lane).

  python3 check_dumps.py METAL_DIR REF_DIR [--tol 2e-2]

Prints the relative L2 error of every tensor both sides wrote, in forward order, and exits 1 if any is above the
tolerance. The Metal side dumps the fused q|gate projection as "qg" [T, H, 2D]; its first D columns of each head
are the reference "q". The reference logits start at the last prompt row; the Metal logits hold every row.
"""
import argparse
import os
import re
import sys

import numpy as np


def order_key(name):
    if name == "embed":
        return (-1, 0, "")
    if name in ("final_norm", "logits"):
        return (10 ** 6, 0 if name == "final_norm" else 1, "")
    m = re.match(r"h(\d+)$", name)
    if m:
        return (int(m.group(1)), 9, "")
    m = re.match(r"l(\d+)\.(.+)$", name)
    if m:
        return (int(m.group(1)), 0, m.group(2))
    return (10 ** 7, 0, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metal")
    ap.add_argument("ref")
    ap.add_argument("--tol", type=float, default=2e-2)
    a = ap.parse_args()
    names = sorted({f[:-4] for f in os.listdir(a.metal) if f.endswith(".npy")}, key=order_key)
    worst, bad = 0.0, []
    for n in names:
        refn = n[:-1] if n.endswith(".qg") else n
        rp = os.path.join(a.ref, refn + ".npy")
        if not os.path.exists(rp):
            continue
        m = np.load(os.path.join(a.metal, n + ".npy")).astype(np.float64)
        r = np.load(rp).astype(np.float64)
        if n.endswith(".qg"):
            D = m.shape[-1] // 2
            m = m[..., :D]
        if n == "logits":
            m = m[m.shape[0] - r.shape[0]:]
        if m.size != r.size:
            print(f"{n:16s} size mismatch metal {m.shape} ref {r.shape}")
            bad.append(n)
            continue
        m = m.reshape(r.shape)
        rel = float(np.linalg.norm(m - r) / max(np.linalg.norm(r), 1e-30))
        extra = ""
        if n == "logits":
            agree = int((m.argmax(-1) == r.argmax(-1)).sum())
            extra = f"  argmax agrees on {agree}/{r.shape[0]} rows"
        flag = "" if rel <= a.tol else "  ABOVE TOLERANCE"
        print(f"{n:16s} rel L2 {rel:.3e}  max abs {np.abs(m - r).max():.3e}{extra}{flag}")
        worst = max(worst, rel)
        if rel > a.tol:
            bad.append(n)
    print(f"{len(names)} tensors, worst rel L2 {worst:.3e}, {len(bad)} above {a.tol:g}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
