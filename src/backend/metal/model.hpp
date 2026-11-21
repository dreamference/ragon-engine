// The hybrid Qwen3.8 model on the Metal backend (METAL spec, first lane): the checkpoint mapped as shipped
// (NVFP4 FFN and head, FP8 projections, BF16 embedding), unpacked in the kernels, and a forward pass over T
// tokens at a position (prefill chunks and M=1 decode share it). Objective-C++ only.
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "backend/metal/device.hpp"

namespace ragon::metal {

struct Tensor {
  View v;
  std::string dtype;
  std::vector<int64_t> shape;
  size_t bytes = 0;
};

// safetensors shards, each mapped once and wrapped in a no-copy Metal buffer (copied when the OS refuses).
class Checkpoint {
 public:
  void open(Queue& q, const std::string& dir);
  bool has(const std::string& name) const { return where_.count(name) != 0; }
  Tensor get(const std::string& name);    // a view into the mapped shard (copied if not 16-byte aligned)
  View f32(const std::string& name);       // a small tensor converted to f32 on the host
  float scalar(const std::string& name);
  size_t mapped_bytes() const { return mapped_; }

 private:
  struct Shard {
    std::string path;
    const uint8_t* base = nullptr;
    size_t size = 0, data = 0;  // data = 8 + header length
    id<MTLBuffer> buf = nil;
    NSDictionary* hdr = nil;
  };
  Queue* q_ = nullptr;
  std::vector<Shard> shards_;
  std::map<std::string, int> where_;
  size_t mapped_ = 0;
};

struct Lin {
  enum Fmt { kNvfp4, kFp8, kBf16, kQ4 } fmt = kBf16;  // kQ4: built at load from kFp8
  View w, sc;
  float scale = 1.0f;
  uint32_t N = 0, K = 0;
  // layer-major prefill: the half weight already sits at wlay_ + dq_off (dequantized once for all chunks)
  bool dq_ready = false;
  size_t dq_off = 0;
};

struct Layer {
  bool attn = false;
  View in_norm, post_norm;
  // gated delta net
  Lin qkv, z, a, b, out;
  View conv_w, A_log, dt_bias, gnorm, conv_state, S;
  // full attention
  Lin q, k, v, o;
  View q_norm, k_norm, kc, vc;
  View ks, vs;  // int8 KV option: f32 scale a position and kv head
  // MLP
  Lin gate, up, down;
};

class Model {
 public:
  using Dump = std::function<void(const std::string& name, View v, std::vector<int64_t> shape)>;

  void load(Queue& q, const std::string& dir, int max_ctx, int chunk);
  void reset();  // a new sequence: zero the recurrent and conv states (the KV cache is overwritten by position)
  // Runs T <= chunk tokens at positions pos0.. and leaves logits for the last row (all rows if all_logits) in
  // logits(). With dump set, syncs at each check point and hands the buffer to it.
  void forward(const uint32_t* ids, int T, int pos0, bool all_logits, const Dump& dump = nullptr);
  // A new sequence's whole prompt (P tokens from position 0) in chunk-sized pieces, layer by layer: each layer's
  // weights are dequantized once for all chunks. Same kernels on the same inputs as forward() chunk by chunk, so
  // the same results; RAGON_METAL_LAYER_MAJOR=0 runs the chunk loop instead. Leaves the last row's logits.
  void prefill(const uint32_t* ids, int P);
  // Greedy decode without a CPU round trip a token: greedy_begin(id0, banned, n) puts the first id on the device;
  // each greedy_step(pos) encodes one token, picks the next id on the GPU (the largest logit outside banned, ties to
  // the lower id) into the device's id slot and commits without waiting, so the next step is encoded while this one
  // runs. greedy_wait(k) blocks until step k has run; greedy_out(k) is then {id, top, second} (floats as bits).
  void greedy_begin(uint32_t id0, const std::vector<int>& banned, int steps);
  void greedy_step(int pos);
  void greedy_wait(int k);
  const uint32_t* greedy_out(int k) const { return gout_.host<uint32_t>() + size_t(k) * 3; }
  const float* logits() const { return logits_.host<float>(); }
  int vocab() const { return V_; }
  int chunk() const { return C_; }
  std::vector<int> eos_ids;
  // RAGON_METAL_PROF=1: sync after each block and add its GPU seconds to prof[0..3] = attention, GDN, MLP, head;
  // prof[4] = the GDN recurrence alone (part of prof[1]).
  double prof[5] = {0, 0, 0, 0, 0};
  double gpu_s = 0;  // GPU time of forward()'s last command buffer, summed (a decode step is one command buffer)
  // RAGON_METAL_PROF=2 also syncs after each prefill matmul pass: pm[0..4] = weight dequant, x to half, matmul,
  // store, silu_mul; pm[5] = the other work queued before each matmul (these seconds are then missing from
  // prof[0..3]). RAGON_METAL_PROF=4 also syncs around each GEMV (decode): pm[2] = the GEMVs, pm[5] = the rest.
  double pm[6] = {0, 0, 0, 0, 0, 0};

 private:
  // y[T][N] (=|+=) x[T][K] W^T: GEMV kernels for decode and short chunks, tensor-op matmul for T >= mm_min_.
  // mode 2 (prefill matmul only, see mm_fused()): y = silu(y) * (x W^T), the MLP up projection over the gate.
  // reuse_x: x is the input of the previous gemv call and unchanged since, so its half copy is kept.
  // mode 3 (prefill matmul only): as mode 2, but the result goes only to xh2_ as half, the A of the next gemv;
  // xa: the half A that a mode-3 gemv wrote (skips the x_half pass; x is then unused).
  void gemv(const Lin& l, View x, View y, int T, int ldx, int ldy, bool add, int mode = 0, bool reuse_x = false,
            View xa = View{});
  bool mm_fused(const Lin& l, int T) const { return !mm_old_ && use_mm(l, T); }
  bool use_mm(const Lin& l, int T) const;
  void attention_mm(Layer& L, int T, int pos0);
  id<MTLComputeCommandEncoder> pm_sync(id<MTLComputeCommandEncoder> e, int k);
  // next/T: when next takes the matmul path over these T tokens, the norm also writes out's half copy into xh_
  // (no x_half pass for it); out must be [T][ldo_t] with next.K == ldo_t.
  void norm(View x, View w, View z, View out, int rows, int n, int rpt, int ldt, int ldh, int ldo_t, int ldo_h,
            int mode, float scale = 1.0f, const Lin* next = nullptr, int T = 0);
  void gdn(Layer& L, int T, const Dump& dump, const std::string& tag);
  void layer(size_t i, int T, int pos0, const Dump& dump);  // one decoder layer over h_'s T rows
  void dequant(id<MTLComputeCommandEncoder> e, const Lin& l, View out);
  void attention(Layer& L, int T, int pos0, const Dump& dump, const std::string& tag);
  Lin lin(const std::string& prefix);

  Queue* q_ = nullptr;
  Checkpoint ck_;
  std::vector<Layer> layers_;
  View embed_, final_norm_;
  Lin head_;
  int H_ = 0, KV_ = 0, D_ = 0, Hk_ = 0, Hv_ = 0, Dk_ = 0, Dv_ = 0, hidden_ = 0, ffn_ = 0, V_ = 0, rot_ = 0;
  int max_ctx_ = 0, C_ = 0, nchunk_max_ = 32, mm_min_ = 64;
  bool attn_gqa_ = true;
  bool mm_old_ = false;  // RAGON_METAL_MM_OLD=1: the lane 2 prefill matmul (64x64 tiles, separate store pass)
  const void* xh_buf_ = nullptr;  // what xh_ holds: x's buffer, offset, K, T, ldx
  size_t xh_off_ = 0;
  uint32_t xh_k_ = 0, xh_t_ = 0, xh_ld_ = 0;
  bool xh_fresh_ = false;  // a norm just wrote xh_ for the next gemv (as reuse_x for that one call)
  bool xh_fuse_ = true;    // RAGON_METAL_XH_OLD=1: the separate x_half passes (lane 3)
  View xh2_;               // half silu(gate)*up, the MLP down projection's A (mode 3)
  bool kv8_ = false;  // int8 KV cache option (RAGON_METAL_KV=int8)
  uint32_t flash_dbg_ = 0;  // flash timing probes (RAGON_METAL_FLASH_DBG, wrong results): skip 1 rescale, 2 PV, 4 softmax, 16 QK; 8 QK via registers
  int attn_flash_ = 2;  // prefill attention (RAGON_METAL_ATTN_FLASH): 0 = S/P matmuls; 2 = flash2 (softmax in registers);
                        // 4, 8, 128, 192 = the first flash kernel
  View kdq_, vdq_;    // int8 KV: bfloat scratch of one layer's cache for the prefill matmuls
  void to_q4(Lin& l);  // the Mac-native 4-bit projection option (RAGON_METAL_PROJ=q4)
  int attn_nch_ = 16, attn_chunk_min_ = 64;  // fused decode attention: max key chunks, min keys a chunk
  int prof_ = 0;
  size_t wdq_cap_ = 0;
  float eps_ = 1e-6f;
  // activations [C][...] f32
  View gout_;
  std::vector<id<MTLCommandBuffer>> gcb_;  // the last command buffer of each greedy step
  std::vector<uint32_t> gban_;
  int gstep_ = 0;
  bool greedy_ = false;  // inside greedy_step: forward() ends with the pick and a commit instead of a sync
  View ids_, h_, xn_, qkv_, zb_, ab_, bb_, core_, gated_, qg_, kb_, vb_, ob_, part_, gb_, ub_, logits_, cos_, sin_;
  // prefill matmul scratch: half weights, half activations, f32 product; attention Q, scores, probabilities
  View wdq_, xh_, mmc_, qb_, S_, P_;
  // layer-major prefill: the whole prompt's hidden rows and one layer's half weights
  View hall_, wlay_;
  bool layer_major_ = true;
  int flush_ = 4;  // decode: commit the command buffer every flush_ layers so the GPU starts while the CPU encodes
  // chunked GDN prefill (RAGON_METAL_GDN_OLD=1 keeps the token loop): half q/k/v, per (sub-chunk, head) W, Uv, A2, lg
  View gq_, gk_, gv_, gW_, gU_, gA2_, glg_;
  double gtp_[3] = {0, 0, 0};  // RAGON_METAL_PROF=3: the chunked kernels apart (measurement only)
  long gtn_ = 0;
  int gdn_min_ = 64;  // chunk sizes from which the chunked recurrence runs; 0 = never
};

}  // namespace ragon::metal
