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
  const float* logits() const { return logits_.host<float>(); }
  int vocab() const { return V_; }
  int chunk() const { return C_; }
  std::vector<int> eos_ids;
  // RAGON_METAL_PROF=1: sync after each block and add its GPU seconds to prof[0..3] = attention, GDN, MLP, head.
  double prof[4] = {0, 0, 0, 0};

 private:
  // y[T][N] (=|+=) x[T][K] W^T: GEMV kernels for decode and short chunks, tensor-op matmul for T >= mm_min_.
  void gemv(const Lin& l, View x, View y, int T, int ldx, int ldy, bool add);
  bool use_mm(const Lin& l, int T) const;
  void attention_mm(Layer& L, int T, int pos0);
  void norm(View x, View w, View z, View out, int rows, int n, int rpt, int ldt, int ldh, int ldo_t, int ldo_h,
            int mode, float scale = 1.0f);
  void gdn(Layer& L, int T, const Dump& dump, const std::string& tag);
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
  bool kv8_ = false;  // int8 KV cache option (RAGON_METAL_KV=int8)
  uint32_t flash_dbg_ = 0;  // flash timing probes (RAGON_METAL_FLASH_DBG, wrong results): skip 1 rescale, 2 PV, 4 softmax, 16 QK; 8 QK via registers
  int attn_flash_ = 8;  // prefill attention (RAGON_METAL_ATTN_FLASH): 0 = S/P matmuls; 4, 8, 128, 192 = flash kernel
  View kdq_, vdq_;    // int8 KV: bfloat scratch of one layer's cache for the prefill matmuls
  void to_q4(Lin& l);  // the Mac-native 4-bit projection option (RAGON_METAL_PROJ=q4)
  int attn_nch_ = 16, attn_chunk_min_ = 64;  // fused decode attention: max key chunks, min keys a chunk
  bool prof_ = false;
  size_t wdq_cap_ = 0;
  float eps_ = 1e-6f;
  // activations [C][...] f32
  View ids_, h_, xn_, qkv_, zb_, ab_, bb_, core_, gated_, qg_, kb_, vb_, ob_, part_, gb_, ub_, logits_, cos_, sin_;
  // prefill matmul scratch: half weights, half activations, f32 product; attention Q, scores, probabilities
  View wdq_, xh_, mmc_, qb_, S_, P_;
};

}  // namespace ragon::metal
