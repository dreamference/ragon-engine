#!/usr/bin/env python3
"""CPU reference forward of the Qwen3.8-27B text model from the NVFP4 checkpoint, in numpy f32.

A slow, plain reading of the model used to check the Metal backend layer by layer: every weight is
dequantized to f32 (NVFP4 = e2m1 code * e4m3 block scale * global scale; FP8 = e4m3 * per-tensor scale),
activations stay f32, no cache (the whole sequence is run at once).

  qwen_ref.py MODEL_DIR --ids 1,2,3 [--dump DIR] [--layers N] [--check-ids 4,5,6]

--dump writes the per-layer inputs and outputs as .npy (see DUMPS below); --check-ids teacher-forces a
continuation and prints, at each position, the reference argmax, the expected id and the top-1/top-2 margin.
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

PFX = "model.language_model."

# ---- number formats ------------------------------------------------------------------------------
E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32)


def e4m3_table():
    t = np.zeros(256, np.float32)
    for b in range(256):
        s = -1.0 if b & 0x80 else 1.0
        e = (b >> 3) & 0xF
        m = b & 7
        if e == 15 and m == 7:
            t[b] = np.nan
        elif e == 0:
            t[b] = s * (m / 8.0) * 2.0 ** -6
        else:
            t[b] = s * (1 + m / 8.0) * 2.0 ** (e - 7)
    return t


E4M3 = e4m3_table()


def bf16(a):
    return (a.astype(np.uint32) << 16).view(np.float32)


# ---- safetensors ---------------------------------------------------------------------------------
class Checkpoint:
    DT = {"BF16": np.uint16, "F32": np.float32, "U8": np.uint8, "F8_E4M3": np.uint8, "I64": np.int64}

    def __init__(self, d):
        self.d = d
        idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self.where = idx
        self.hdr = {}
        self.mm = {}
        for f in sorted(set(idx.values())):
            p = os.path.join(d, f)
            with open(p, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                self.hdr[f] = (json.loads(fh.read(n)), 8 + n)
            self.mm[f] = np.memmap(p, np.uint8, "r")

    def has(self, name):
        return name in self.where

    def dtype(self, name):
        f = self.where[name]
        return self.hdr[f][0][name]["dtype"]

    def raw(self, name):
        f = self.where[name]
        h, base = self.hdr[f]
        e = h[name]
        a, b = e["data_offsets"]
        buf = self.mm[f][base + a: base + b]
        return buf.view(self.DT[e["dtype"]]).reshape(e["shape"])

    def f32(self, name):
        dt = self.dtype(name)
        a = self.raw(name)
        if dt == "BF16":
            return bf16(a)
        if dt == "F32":
            return a.astype(np.float32)
        if dt == "F8_E4M3":
            return E4M3[a]
        raise ValueError(name + " " + dt)

    def scalar(self, name):
        return float(self.f32(name).reshape(-1)[0])

    def weight(self, prefix, rows=None):
        """The f32 [N, K] matrix of a linear layer, whatever its format; rows = (r0, r1) slices."""
        name = prefix + ".weight"
        dt = self.dtype(name)
        w = self.raw(name)
        if rows is not None:
            w = w[rows[0]:rows[1]]
        if dt == "U8":  # NVFP4: two e2m1 codes a byte, low nibble first; e4m3 scale per 16; global scale_2
            sc = self.raw(prefix + ".weight_scale")
            if rows is not None:
                sc = sc[rows[0]:rows[1]]
            N, Kh = w.shape
            v = np.empty((N, Kh * 2), np.float32)
            v[:, 0::2] = E2M1[w & 15]
            v[:, 1::2] = E2M1[w >> 4]
            v = v.reshape(N, -1, 16) * E4M3[sc][:, :, None]
            return v.reshape(N, Kh * 2) * np.float32(self.scalar(prefix + ".weight_scale_2"))
        if dt == "F8_E4M3":
            return E4M3[w] * np.float32(self.scalar(prefix + ".weight_scale"))
        if dt == "BF16":
            return bf16(w)
        raise ValueError(name + " " + dt)


# ---- model ---------------------------------------------------------------------------------------
def rms(x, w=None, eps=1e-6, one_plus=True):
    y = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    if w is None:
        return y
    return y * ((1.0 + w) if one_plus else w)


def silu(x):
    return x / (1.0 + np.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softplus(x):
    return np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, 20))))


class Model:
    def __init__(self, d):
        self.ck = Checkpoint(d)
        c = json.load(open(os.path.join(d, "config.json")))
        self.c = c.get("text_config", c)
        self.L = self.c["num_hidden_layers"]
        self.types = self.c["layer_types"]
        self.eps = self.c["rms_norm_eps"]
        rp = self.c.get("rope_parameters", {})
        self.theta = rp.get("rope_theta", self.c.get("rope_theta", 1e7))
        self.rot = int(self.c["head_dim"] * self.c.get("partial_rotary_factor", rp.get("partial_rotary_factor", 0.25)))

    def lin(self, x, prefix):
        return x @ self.ck.weight(prefix).T

    def attention(self, x, p, dump, tag):
        c = self.c
        T = x.shape[0]
        H, KV, D = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        qg = self.lin(x, p + "q_proj").reshape(T, H, 2 * D)
        q, gate = qg[:, :, :D], qg[:, :, D:].reshape(T, H * D)
        k = self.lin(x, p + "k_proj").reshape(T, KV, D)
        v = self.lin(x, p + "v_proj").reshape(T, KV, D)
        q = rms(q, self.ck.f32(p + "q_norm.weight"), self.eps)
        k = rms(k, self.ck.f32(p + "k_norm.weight"), self.eps)
        r = self.rot
        inv = self.theta ** (-np.arange(0, r, 2, dtype=np.float64) / r)
        ang = np.arange(T, dtype=np.float64)[:, None] * inv[None, :]
        cos = np.concatenate([np.cos(ang)] * 2, -1).astype(np.float32)[:, None, :]
        sin = np.concatenate([np.sin(ang)] * 2, -1).astype(np.float32)[:, None, :]

        def rope(t):
            a = t[..., :r]
            h = r // 2
            rh = np.concatenate([-a[..., h:], a[..., :h]], -1)
            return np.concatenate([a * cos + rh * sin, t[..., r:]], -1)

        q, k = rope(q), rope(k)
        if dump is not None:
            dump[tag + "q"] = q
            dump[tag + "k"] = k
            dump[tag + "v"] = v
        g = H // KV
        o = np.empty((T, H, D), np.float32)
        mask = np.triu(np.full((T, T), -np.inf, np.float32), 1)
        for h in range(H):
            s = (q[:, h, :] @ k[:, h // g, :].T) * np.float32(D ** -0.5) + mask
            s = np.exp(s - s.max(-1, keepdims=True))
            s /= s.sum(-1, keepdims=True)
            o[:, h, :] = s @ v[:, h // g, :]
        o = o.reshape(T, H * D)
        if dump is not None:
            dump[tag + "attn"] = o
        return self.lin(o * sigmoid(gate), p + "o_proj")

    def gdn(self, x, p, dump, tag):
        c = self.c
        T = x.shape[0]
        Hk, Hv = c["linear_num_key_heads"], c["linear_num_value_heads"]
        Dk, Dv = c["linear_key_head_dim"], c["linear_value_head_dim"]
        K = c["linear_conv_kernel_dim"]
        qkv = self.lin(x, p + "in_proj_qkv")
        z = self.lin(x, p + "in_proj_z").reshape(T, Hv, Dv)
        b = self.lin(x, p + "in_proj_b")
        a = self.lin(x, p + "in_proj_a")
        cw = self.ck.f32(p + "conv1d.weight").reshape(qkv.shape[1], K)
        xin = np.concatenate([np.zeros((K - 1, qkv.shape[1]), np.float32), qkv], 0)
        conv = np.zeros_like(qkv)
        for i in range(K):
            conv += xin[i:i + T] * cw[:, i][None, :]
        conv = silu(conv)
        kd = Hk * Dk
        q = conv[:, :kd].reshape(T, Hk, Dk)
        k = conv[:, kd:2 * kd].reshape(T, Hk, Dk)
        v = conv[:, 2 * kd:].reshape(T, Hv, Dv)
        q = q / np.sqrt(np.sum(q * q, -1, keepdims=True) + 1e-6) * np.float32(Dk ** -0.5)
        k = k / np.sqrt(np.sum(k * k, -1, keepdims=True) + 1e-6)
        rep = Hv // Hk
        q = np.repeat(q, rep, 1)
        k = np.repeat(k, rep, 1)
        A = np.exp(self.ck.f32(p + "A_log"))
        g = np.exp(-A[None, :] * softplus(a + self.ck.f32(p + "dt_bias")[None, :]))
        beta = sigmoid(b)
        S = np.zeros((Hv, Dv, Dk), np.float32)
        y = np.empty((T, Hv, Dv), np.float32)
        for t in range(T):
            S *= g[t][:, None, None]
            kv = np.einsum("hvk,hk->hv", S, k[t])
            delta = (v[t] - kv) * beta[t][:, None]
            S += delta[:, :, None] * k[t][:, None, :]
            y[t] = np.einsum("hvk,hk->hv", S, q[t])
        if dump is not None:
            dump[tag + "qkv"] = qkv
            dump[tag + "conv"] = conv
            dump[tag + "g"] = g
            dump[tag + "beta"] = beta
            dump[tag + "core"] = y.reshape(T, Hv * Dv)
            dump[tag + "state"] = S
        o = rms(y, self.ck.f32(p + "norm.weight"), self.eps, one_plus=False) * silu(z)
        return self.lin(o.reshape(T, Hv * Dv), p + "out_proj")

    def mlp(self, x, p):
        return self.lin(silu(self.lin(x, p + "gate_proj")) * self.lin(x, p + "up_proj"), p + "down_proj")

    def forward(self, ids, dump=None, layers=None, inner=(0, 3)):
        ck = self.ck
        emb = ck.raw(PFX + "embed_tokens.weight")
        h = bf16(emb[np.asarray(ids)])
        if dump is not None:
            dump["embed"] = h.copy()
        n = self.L if layers is None else layers
        for i in range(n):
            p = PFX + f"layers.{i}."
            dd = dump if (dump is not None and i in inner) else None
            tag = f"l{i}."
            xn = rms(h, ck.f32(p + "input_layernorm.weight"), self.eps)
            if dd is not None:
                dd[tag + "in_norm"] = xn
            if self.types[i] == "full_attention":
                m = self.attention(xn, p + "self_attn.", dd, tag)
            else:
                m = self.gdn(xn, p + "linear_attn.", dd, tag)
            if dd is not None:
                dd[tag + "mixer"] = m
            h = h + m
            xn = rms(h, ck.f32(p + "post_attention_layernorm.weight"), self.eps)
            if dd is not None:
                dd[tag + "post_norm"] = xn
            f = self.mlp(xn, p + "mlp.")
            if dd is not None:
                dd[tag + "mlp"] = f
            h = h + f
            if dump is not None:
                dump[f"h{i}"] = h.copy()
            print(f"layer {i} done", file=sys.stderr, flush=True)
        if layers is not None and layers < self.L:
            return None
        hn = rms(h, ck.f32(PFX + "norm.weight"), self.eps)
        if dump is not None:
            dump["final_norm"] = hn
        head = "lm_head" if ck.has("lm_head.weight") else PFX + "embed_tokens"
        V = ck.raw(head + ".weight").shape[0]
        logits = np.empty((hn.shape[0], V), np.float32)
        step = 16384
        for r in range(0, V, step):
            e = min(V, r + step)
            logits[:, r:e] = hn @ ck.weight(head, (r, e)).T
        return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--ids", required=True)
    ap.add_argument("--check-ids", default="")
    ap.add_argument("--dump", default="")
    ap.add_argument("--layers", type=int, default=None)
    a = ap.parse_args()
    prompt = [int(t) for t in a.ids.split(",")]
    cont = [int(t) for t in a.check_ids.split(",")] if a.check_ids else []
    ids = prompt + cont
    m = Model(a.model)
    dump = {} if a.dump else None
    logits = m.forward(ids, dump, a.layers)
    if a.dump:
        os.makedirs(a.dump, exist_ok=True)
        np.save(os.path.join(a.dump, "ids.npy"), np.asarray(ids, np.int32))
        for k, v in dump.items():
            np.save(os.path.join(a.dump, k + ".npy"), np.ascontiguousarray(v, np.float32))
        if logits is not None:
            np.save(os.path.join(a.dump, "logits.npy"), logits[len(prompt) - 1:])
    if logits is None:
        return
    P = len(prompt)
    agree = 0
    for j in range(len(cont) + 1):
        row = logits[P - 1 + j]
        top = np.argsort(row)[-2:][::-1]
        exp = cont[j] if j < len(cont) else -1
        ok = exp == top[0]
        agree += ok
        print(f"pos {j}: ref {top[0]} expected {exp} {'ok' if ok or exp < 0 else 'DIFF'} "
              f"margin {row[top[0]] - row[top[1]]:.3f}")
    print(f"agreement {agree}/{len(cont)}")


if __name__ == "__main__":
    main()
