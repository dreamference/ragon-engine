#!/usr/bin/env python3
"""Round a retrained BF16 DFlash2 drafter to the served NVFP4 layout (DRAFTER_RETRAIN §4.3), the first step before
bench/quant/scalesearch.py:

    requant.py BF16_DIR TEMPLATE_DIR OUT_DIR

TEMPLATE_DIR is the NVFP4 drafter production loads today (its tensor names, dtypes, config and activation scales are
kept). Every projection the template stores in NVFP4 (a `.weight_scale` beside the weight) is re-rounded from the
BF16 weight as ModelOpt does: the global scale `weight_scale_2` = amax / (6 * 448), the amax taken over the
projections the server fuses (q, k, v; gate, up); each 16-value block's E4M3 scale = block amax / (6 * global),
rounded to nearest (computed in that order: amax / 6 / global rounds differently at E4M3 ties); the values to
E2M1 under it. The activation scales
(`input_scale`) stay the template's: they come from a calibration over on-policy conversations and a fine-tune
moves activations little (re-calibrate if rung 1 shows a gap). Every other tensor the template has is taken from
BF16_DIR in the template's dtype; a tensor BF16_DIR lacks keeps the template's value, and both lists are printed.

Self-check: with BF16_DIR = the stock BF16 drafter (incoai/Qwen3.8-27B-DFlash2) and TEMPLATE_DIR = the RTNcal file it
was rounded into, the output reproduces the template's bytes (the fraction is printed per tensor). Then
`scalesearch.py BF16_DIR OUT_DIR SS_DIR` gives the file to evaluate.

Runs where torch and safetensors are installed (production's SGLang image on the CPU, as scalesearch.py).
"""
import json
import os
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "quant"))
from scalesearch import E4M3, e2m1_codes, pack  # noqa: E402

FP8_MAX, FP4_MAX = 448.0, 6.0


FUSED = [("q_proj", "k_proj", "v_proj"), ("gate_proj", "up_proj")]  # served fused: one global scale per group


def group_amax(fb, base, quant):
    """amax over the projections the server fuses with `base` (ModelOpt gives a fused group one global scale)."""
    head, last = base.rsplit(".", 1)
    for grp in FUSED:
        if last in grp:
            return max(fb.get_tensor(f"{head}.{m}.weight").float().abs().max().item()
                       for m in grp if f"{head}.{m}" in quant)
    return fb.get_tensor(base + ".weight").float().abs().max().item()


def rtn(w, amax):
    """w [N][K] float32 -> (packed E2M1 bytes [N][K/2] uint8, E4M3 block scales [N][K/16], global scale)."""
    N, K = w.shape
    g = torch.tensor(amax, dtype=torch.float32) / (FP4_MAX * FP8_MAX)
    blocks = w.reshape(N, K // 16, 16)
    s = (blocks.abs().amax(-1) / (FP4_MAX * g)).clamp(max=FP8_MAX).to(torch.float8_e4m3fn)
    sf = s.float()
    sf = torch.where(sf == 0, torch.ones_like(sf), sf)  # an all-zero block: any scale gives zero codes
    q = e2m1_codes(blocks / (sf.unsqueeze(-1) * g))
    return pack(q.reshape(N, K)), s, g


def main():
    bf16_dir, tpl_dir, out_dir = sys.argv[1:4]
    os.makedirs(out_dir, exist_ok=True)
    out, kept, from_bf16 = {}, [], 0
    with safe_open(os.path.join(tpl_dir, "model.safetensors"), "pt") as ft, \
            safe_open(os.path.join(bf16_dir, "model.safetensors"), "pt") as fb:
        meta = ft.metadata()
        bnames = set(fb.keys())
        names = list(ft.keys())
        quant = {n[: -len(".weight_scale")] for n in names if n.endswith(".weight_scale")}
        for name in names:
            t = ft.get_tensor(name)
            base = name.rsplit(".", 1)[0]
            if base in quant and not name.endswith(".input_scale"):
                continue  # written below from the BF16 weight
            if base in quant or name not in bnames:
                out[name] = t
                if base not in quant:
                    kept.append(name)
                continue
            b = fb.get_tensor(name)
            assert b.shape == t.shape, f"{name}: {tuple(b.shape)} vs template {tuple(t.shape)}"
            out[name] = b.to(t.dtype)
            from_bf16 += 1
        tot = same_tot = 0
        for base in sorted(quant):
            w = fb.get_tensor(base + ".weight").float()
            q, s, g = rtn(w, group_amax(fb, base, quant))
            tq, ts = ft.get_tensor(base + ".weight"), ft.get_tensor(base + ".weight_scale")
            assert q.shape == tq.shape and s.shape == ts.shape, base
            same = (q == tq).sum().item()
            same_s = (s.view(torch.uint8) == ts.view(torch.uint8)).float().mean().item()
            g_t = ft.get_tensor(base + ".weight_scale_2").float().item()
            out[base + ".weight"] = q
            out[base + ".weight_scale"] = s
            out[base + ".weight_scale_2"] = torch.full_like(ft.get_tensor(base + ".weight_scale_2"), g.item())
            tot += q.numel()
            same_tot += same
            print(f"{base}: bytes equal to template {same / q.numel():.5f}; scales equal {same_s:.5f}; "
                  f"global {g.item():.6e} (template {g_t:.6e})", flush=True)
    print(json.dumps({"quantized": len(quant), "from_bf16": from_bf16, "kept_from_template": kept,
                      "bf16_unused": sorted(bnames - set(out)), "bytes_equal_to_template": round(same_tot / tot, 6)}))
    save_file(out, os.path.join(out_dir, "model.safetensors"), metadata=meta)
    for fn in os.listdir(tpl_dir):
        if fn != "model.safetensors" and not os.path.isdir(os.path.join(tpl_dir, fn)):
            shutil.copy(os.path.realpath(os.path.join(tpl_dir, fn)), os.path.join(out_dir, fn))
    print("written", out_dir)


if __name__ == "__main__":
    main()
