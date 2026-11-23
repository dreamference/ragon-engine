#include "backend/metal/model.hpp"

#include <dispatch/dispatch.h>

#include "core/accept.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ragon::metal {

[[noreturn]] static void fail(const std::string& msg) {
  std::fprintf(stderr, "metal model: %s\n", msg.c_str());
  std::exit(2);
}

static int pad64(int n) { return (n + 63) / 64 * 64; }

static NSDictionary* read_json(const std::string& path) {
  NSData* d = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path.c_str()]];
  if (!d) fail("cannot read " + path);
  NSError* err = nil;
  id j = [NSJSONSerialization JSONObjectWithData:d options:0 error:&err];
  if (![j isKindOfClass:[NSDictionary class]]) fail("bad json " + path);
  return j;
}

static float bf16_to_f32(uint16_t v) {
  uint32_t u = uint32_t(v) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// ---- checkpoint ---------------------------------------------------------------------------------

void Checkpoint::open(Queue& q, const std::string& dir) {
  q_ = &q;
  const std::string ix = dir + "/model.safetensors.index.json";
  const bool single = ![[NSFileManager defaultManager] fileExistsAtPath:[NSString stringWithUTF8String:ix.c_str()]];
  NSDictionary* wm = single ? @{} : read_json(ix)[@"weight_map"];
  if (single) {  // one model.safetensors: the names come from its header
    Shard s;
    s.path = dir + "/model.safetensors";
    shards_.push_back(s);
  }
  std::map<std::string, int> file_ix;
  for (NSString* k in wm) {
    std::string f = [wm[k] UTF8String];
    if (!file_ix.count(f)) {
      file_ix[f] = int(shards_.size());
      Shard s;
      s.path = dir + "/" + f;
      shards_.push_back(s);
    }
    where_[[k UTF8String]] = file_ix[f];
  }
  const size_t page = size_t(getpagesize());
  for (Shard& s : shards_) {
    int fd = ::open(s.path.c_str(), O_RDONLY);
    if (fd < 0) fail("cannot open " + s.path);
    struct stat st;
    fstat(fd, &st);
    s.size = size_t(st.st_size);
    const size_t len = (s.size + page - 1) / page * page;
    void* p = mmap(nullptr, len, PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);
    if (p == MAP_FAILED) fail("mmap " + s.path);
    s.base = static_cast<const uint8_t*>(p);
    uint64_t n;
    std::memcpy(&n, s.base, 8);
    s.data = 8 + n;
    NSData* hd = [NSData dataWithBytesNoCopy:(void*)(s.base + 8) length:n freeWhenDone:NO];
    NSError* err = nil;
    s.hdr = [NSJSONSerialization JSONObjectWithData:hd options:0 error:&err];
    if (!s.hdr) fail("bad header " + s.path);
    if (single)
      for (NSString* k in s.hdr)
        if (![k isEqualToString:@"__metadata__"]) where_[[k UTF8String]] = 0;
    s.buf = [q.dev newBufferWithBytesNoCopy:(void*)s.base length:len options:MTLResourceStorageModeShared
                                deallocator:nil];
    if (!s.buf) {  // the OS refused to wrap the mapping: copy (slower load, same result)
      std::fprintf(stderr, "metal model: no-copy wrap refused for %s, copying\n", s.path.c_str());
      s.buf = [q.dev newBufferWithBytes:s.base length:s.size options:MTLResourceStorageModeShared];
      if (!s.buf) fail("buffer for " + s.path);
    }
    mapped_ += s.size;
  }
}

Tensor Checkpoint::get(const std::string& name) {
  auto it = where_.find(name);
  if (it == where_.end()) fail("missing tensor " + name);
  Shard& s = shards_[it->second];
  NSDictionary* e = s.hdr[[NSString stringWithUTF8String:name.c_str()]];
  Tensor t;
  t.dtype = [e[@"dtype"] UTF8String];
  for (NSNumber* d in e[@"shape"]) t.shape.push_back([d longLongValue]);
  NSArray* offs = e[@"data_offsets"];
  const size_t a = [offs[0] unsignedLongLongValue], b = [offs[1] unsignedLongLongValue];
  t.bytes = b - a;
  const size_t off = s.data + a;
  if (off % 16 == 0) {
    t.v = {s.buf, off};
  } else {
    t.v = q_->alloc(t.bytes);
    std::memcpy(t.v.host<uint8_t>(), s.base + off, t.bytes);
  }
  return t;
}

View Checkpoint::f32(const std::string& name) {
  Tensor t = get(name);
  size_t n = 1;
  for (int64_t d : t.shape) n *= size_t(d);
  View out = q_->alloc(n * 4);
  float* o = out.host<float>();
  if (t.dtype == "BF16") {
    const uint16_t* p = t.v.host<uint16_t>();
    for (size_t i = 0; i < n; ++i) o[i] = bf16_to_f32(p[i]);
  } else if (t.dtype == "F32") {
    std::memcpy(o, t.v.host<float>(), n * 4);
  } else {
    fail(name + ": unexpected dtype " + t.dtype);
  }
  return out;
}

float Checkpoint::scalar(const std::string& name) {
  Tensor t = get(name);
  if (t.dtype == "F32") return *t.v.host<float>();
  if (t.dtype == "BF16") return bf16_to_f32(*t.v.host<uint16_t>());
  fail(name + ": unexpected scalar dtype " + t.dtype);
}

// ---- model --------------------------------------------------------------------------------------

static const char* kPfx = "model.language_model.";

Lin Model::lin_of(Checkpoint& ck_, const std::string& prefix) {
  Lin l;
  Tensor w = ck_.get(prefix + ".weight");
  l.w = w.v;
  l.N = uint32_t(w.shape[0]);
  if (w.dtype == "U8") {
    l.fmt = Lin::kNvfp4;
    l.K = uint32_t(w.shape[1] * 2);
    l.sc = ck_.get(prefix + ".weight_scale").v;
    l.scale = ck_.scalar(prefix + ".weight_scale_2");
  } else if (w.dtype == "F8_E4M3") {
    l.fmt = Lin::kFp8;
    l.K = uint32_t(w.shape[1]);
    l.scale = ck_.scalar(prefix + ".weight_scale");
  } else if (w.dtype == "BF16") {
    l.fmt = Lin::kBf16;
    l.K = uint32_t(w.shape[1]);
  } else {
    fail(prefix + ": unsupported weight dtype " + w.dtype);
  }
  if (l.K % 256) fail(prefix + ": K not a multiple of 256");
  return l;
}

void Model::load(Queue& q, const std::string& dir, int max_ctx, int chunk) {
  q_ = &q;
  if (const char* kv = std::getenv("RAGON_METAL_KV")) {
    if (std::string(kv) == "int8") kv8_ = true;
    else if (std::string(kv) != "bf16") fail(std::string("RAGON_METAL_KV: unknown value ") + kv + " (bf16 or int8)");
  }
  max_ctx_ = max_ctx;
  C_ = chunk;
  NSDictionary* cfg = read_json(dir + "/config.json");
  NSDictionary* tc = cfg[@"text_config"] ? cfg[@"text_config"] : cfg;
  auto I = [&](NSString* k) { return [tc[k] intValue]; };
  hidden_ = I(@"hidden_size");
  ffn_ = I(@"intermediate_size");
  V_ = I(@"vocab_size");
  H_ = I(@"num_attention_heads");
  KV_ = I(@"num_key_value_heads");
  D_ = I(@"head_dim");
  Hk_ = I(@"linear_num_key_heads");
  Hv_ = I(@"linear_num_value_heads");
  Dk_ = I(@"linear_key_head_dim");
  Dv_ = I(@"linear_value_head_dim");
  eps_ = [tc[@"rms_norm_eps"] floatValue];
  NSDictionary* rp = tc[@"rope_parameters"] ? tc[@"rope_parameters"] : tc;
  double theta = rp[@"rope_theta"] ? [rp[@"rope_theta"] doubleValue] : 1e7;
  double prf = rp[@"partial_rotary_factor"] ? [rp[@"partial_rotary_factor"] doubleValue] : 0.25;
  rot_ = int(D_ * prf);
  if (D_ != 256 || Dk_ != 128 || hidden_ % 256) fail("kernel shape assumptions (head_dim 256, Dk 128) do not hold");
  id eos = tc[@"eos_token_id"] ? tc[@"eos_token_id"] : cfg[@"eos_token_id"];
  if ([eos isKindOfClass:[NSArray class]]) {
    for (NSNumber* e in eos) eos_ids.push_back([e intValue]);
  } else if (eos) {
    eos_ids.push_back([eos intValue]);
  }
  // generation_config.json lists every end id (end of text and end of turn); the recipe bans them all
  if (NSDictionary* gen = [[NSFileManager defaultManager]
                              fileExistsAtPath:[NSString stringWithUTF8String:(dir + "/generation_config.json").c_str()]]
                              ? read_json(dir + "/generation_config.json")
                              : nil) {
    id ge = gen[@"eos_token_id"];
    NSArray* gl = [ge isKindOfClass:[NSArray class]] ? ge : (ge ? @[ ge ] : @[]);
    for (NSNumber* x in gl)
      if (std::find(eos_ids.begin(), eos_ids.end(), [x intValue]) == eos_ids.end()) eos_ids.push_back([x intValue]);
  }
  NSArray* types = tc[@"layer_types"];
  const int L = I(@"num_hidden_layers");

  ck_.open(q, dir);
  Tensor emb = ck_.get(std::string(kPfx) + "embed_tokens.weight");
  if (emb.dtype != "BF16") fail("embedding is not BF16");
  embed_ = emb.v;
  final_norm_ = ck_.f32(std::string(kPfx) + "norm.weight");
  head_ = lin(ck_.has("lm_head.weight") ? "lm_head" : std::string(kPfx) + "embed_tokens");

  layers_.resize(L);
  for (int i = 0; i < L; ++i) {
    Layer& Ly = layers_[i];
    const std::string p = std::string(kPfx) + "layers." + std::to_string(i) + ".";
    Ly.attn = [types[i] isEqualToString:@"full_attention"];
    Ly.in_norm = ck_.f32(p + "input_layernorm.weight");
    Ly.post_norm = ck_.f32(p + "post_attention_layernorm.weight");
    if (Ly.attn) {
      const std::string a = p + "self_attn.";
      Ly.q = lin(a + "q_proj");
      Ly.k = lin(a + "k_proj");
      Ly.v = lin(a + "v_proj");
      Ly.o = lin(a + "o_proj");
      Ly.q_norm = ck_.f32(a + "q_norm.weight");
      Ly.k_norm = ck_.f32(a + "k_norm.weight");
      const size_t kvb = kv8_ ? 1 : 2;
      Ly.kc = q.alloc(size_t(pad64(max_ctx)) * KV_ * D_ * kvb);  // padded: prefill matmuls read whole 64-key tiles
      Ly.vc = q.alloc(size_t(pad64(max_ctx)) * KV_ * D_ * kvb);
      if (kv8_) {
        Ly.ks = q.alloc(size_t(pad64(max_ctx)) * KV_ * 4);
        Ly.vs = q.alloc(size_t(pad64(max_ctx)) * KV_ * 4);
      }
    } else {
      const std::string a = p + "linear_attn.";
      Ly.qkv = lin(a + "in_proj_qkv");
      Ly.z = lin(a + "in_proj_z");
      Ly.a = lin(a + "in_proj_a");
      Ly.b = lin(a + "in_proj_b");
      Ly.out = lin(a + "out_proj");
      Ly.conv_w = ck_.f32(a + "conv1d.weight");
      Ly.A_log = ck_.f32(a + "A_log");
      Ly.dt_bias = ck_.f32(a + "dt_bias");
      Ly.gnorm = ck_.f32(a + "norm.weight");
      Ly.conv_state = q.alloc(size_t(3) * Ly.qkv.N * 4);
      Ly.S = q.alloc(size_t(Hv_) * Dv_ * Dk_ * 4);
    }
    Ly.gate = lin(p + "mlp.gate_proj");
    Ly.up = lin(p + "mlp.up_proj");
    Ly.down = lin(p + "mlp.down_proj");
  }

  // Projection weights: the checkpoint's FP8 as shipped (default) or the Mac-native 4-bit option, converted here.
  if (const char* pj = std::getenv("RAGON_METAL_PROJ")) {
    if (std::string(pj) == "q4") {
      // RAGON_METAL_PROJ_KEEP: comma-separated projections that stay FP8 (qkv,z,a,b,out,q,k,v,o).
      const char* kp = std::getenv("RAGON_METAL_PROJ_KEEP");
      const std::string keep = std::string(",") + (kp ? kp : "") + ",";
      static const char* names[] = {"qkv", "z", "a", "b", "out", "q", "k", "v", "o"};
      for (Layer& Ly : layers_) {
        Lin* lins[] = {&Ly.qkv, &Ly.z, &Ly.a, &Ly.b, &Ly.out, &Ly.q, &Ly.k, &Ly.v, &Ly.o};
        for (int i = 0; i < 9; ++i)
          if (lins[i]->fmt == Lin::kFp8 && lins[i]->N && keep.find(std::string(",") + names[i] + ",") == std::string::npos)
            to_q4(*lins[i]);
      }
      q.sync();
    } else if (std::string(pj) != "fp8") {
      fail(std::string("RAGON_METAL_PROJ: unknown value ") + pj + " (fp8 or q4)");
    }
  }

  const size_t C = size_t(pad64(C_));  // prefill matmuls write whole 64-row tiles
  const size_t conv_w = size_t(2 * Hk_ * Dk_ + Hv_ * Dv_);
  ids_ = q.alloc(C * 4);
  h_ = q.alloc(C * hidden_ * 4);
  xn_ = q.alloc(C * hidden_ * 4);
  fxw_ = q.alloc(hidden_ * 4);
  fpart_ = q.alloc(hidden_ / 8 * 4);
  g8_ = !std::getenv("RAGON_METAL_G8_OFF");
  if (const char* v = std::getenv("RAGON_METAL_G8_MIN_N")) g8_min_n_ = uint32_t(std::atoi(v));
  if (const char* v = std::getenv("RAGON_METAL_MQ_NCH")) mq_nch_ = std::max(0, std::atoi(v));
  xp8_cap_ = size_t(std::max({hidden_, ffn_, H_ * D_, 8192})) * 16;
  xp8_ = q.alloc(xp8_cap_);
  qkv_ = q.alloc(C * std::max(conv_w, size_t(H_) * 2 * D_) * 4);
  zb_ = q.alloc(C * Hv_ * Dv_ * 4);
  ab_ = q.alloc(C * Hv_ * 4);
  bb_ = q.alloc(C * Hv_ * 4);
  core_ = q.alloc(C * Hv_ * Dv_ * 4);
  gated_ = q.alloc(C * std::max(Hv_ * Dv_, H_ * D_) * 4);
  qg_ = qkv_;  // attention layers reuse the GDN projection buffer
  kb_ = q.alloc(C * KV_ * D_ * 4);
  vb_ = q.alloc(C * KV_ * D_ * 4);
  ob_ = core_;
  part_ = q.alloc(C * H_ * nchunk_max_ * (2 + D_) * 4);
  gb_ = q.alloc(C * ffn_ * 4);
  ub_ = q.alloc(C * ffn_ * 4);
  logits_ = q.alloc(C * size_t(V_) * 4);
  if (const char* pr = std::getenv("RAGON_METAL_PROF")) prof_ = std::max(1, std::atoi(pr));
  if (const char* c = std::getenv("RAGON_METAL_CONC")) conc_ = std::atoi(c) != 0;
  if (const char* mm = std::getenv("RAGON_METAL_MM_MIN")) mm_min_ = std::max(1, std::atoi(mm));
  mm_old_ = std::getenv("RAGON_METAL_MM_OLD") != nullptr;
  // Prefill: the norms and the MLP up matmul write the next matmul's half A themselves; =1 keeps the x_half passes.
  xh_fuse_ = std::getenv("RAGON_METAL_XH_OLD") == nullptr;
  if (const char* v = std::getenv("RAGON_METAL_NORM_FOLD")) fold_ = std::atoi(v) != 0;
  if (const char* v = std::getenv("RAGON_METAL_LAYER_MAJOR")) layer_major_ = std::atoi(v) != 0;
  if (const char* v = std::getenv("RAGON_METAL_FLUSH")) flush_ = std::max(0, std::atoi(v));
  // Decode attention: the GQA-fused kernel by default; RAGON_METAL_ATTN_OLD=1 keeps the per-head kernel for A/B checks.
  attn_gqa_ = std::getenv("RAGON_METAL_ATTN_OLD") == nullptr;
  if (const char* v = std::getenv("RAGON_METAL_ATTN_NCH")) attn_nch_ = std::max(1, std::atoi(v));
  if (const char* v = std::getenv("RAGON_METAL_ATTN_CHUNK_MIN")) attn_chunk_min_ = std::max(4, std::atoi(v));
  // Prefill attention: flash2 by default; RAGON_METAL_ATTN_FLASH=0 keeps the S/P matmuls, 8 the first flash kernel.
  if (const char* v = std::getenv("RAGON_METAL_ATTN_FLASH")) attn_flash_ = std::atoi(v);
  if (attn_flash_ != 0 && attn_flash_ != 2 && attn_flash_ != 4 && attn_flash_ != 8 && attn_flash_ != 128 && attn_flash_ != 192)
    fail("RAGON_METAL_ATTN_FLASH must be 0, 2, 4, 8, 128 or 192");
  if (D_ != 256 || H_ != 6 * KV_) attn_flash_ = 0;
  if (const char* v = std::getenv("RAGON_METAL_FLASH_DBG")) flash_dbg_ = uint32_t(std::atoi(v));
  if (std::getenv("RAGON_METAL_GDN_OLD")) gdn_min_ = 0;
  if (const char* v = std::getenv("RAGON_METAL_GDN_MIN")) gdn_min_ = std::max(0, std::atoi(v));
  if (gdn_min_ && Dk_ == 128 && Dv_ == 128 && Hv_ % Hk_ == 0) {
    gq_ = q.alloc(C * Hk_ * Dk_ * 2);
    gk_ = q.alloc(C * Hk_ * Dk_ * 2);
    gv_ = q.alloc(C * Hv_ * Dv_ * 2);
    gW_ = q.alloc(C * Hv_ * Dk_ * 2);
    gU_ = q.alloc(C * Hv_ * Dv_ * 2);
    gA2_ = q.alloc(C * Hv_ * (Dk_ + 64) * 2);
    glg_ = q.alloc(C * Hv_ * 2 * 4);
  } else {
    gdn_min_ = 0;
  }
  std::fprintf(stderr, "metal model: GDN prefill %s\n", gdn_min_ ? ("chunked from " + std::to_string(gdn_min_) + " tokens").c_str() : "token loop");
  if (C_ >= mm_min_) {
    size_t maxk = 0, maxn = 0;
    for (const Layer& Ly : layers_)
      for (const Lin* l : {&Ly.qkv, &Ly.z, &Ly.out, &Ly.q, &Ly.k, &Ly.v, &Ly.o, &Ly.gate, &Ly.up, &Ly.down})
        if (l->fmt != Lin::kBf16 && l->N) {
          wdq_cap_ = std::max(wdq_cap_, size_t(l->N) * l->K);
          maxk = std::max(maxk, size_t(l->K));
          maxn = std::max(maxn, size_t(l->N));
        }
    const size_t kvp = size_t(pad64(max_ctx));
    wdq_ = q.alloc(wdq_cap_ * 2);
    xh_ = q.alloc(C * maxk * 2);
    if (xh_fuse_) xh2_ = q.alloc(C * size_t(ffn_) * 2);
    mmc_ = q.alloc(C * maxn * 4);
    qb_ = q.alloc(size_t(H_) * C * D_ * 2);
    if (!attn_flash_) S_ = q.alloc(size_t(H_) * C * kvp * 4);
    if (kv8_) {
      kdq_ = q.alloc(kvp * KV_ * D_ * 2);
      vdq_ = q.alloc(kvp * KV_ * D_ * 2);
    }
    if (!attn_flash_) P_ = q.alloc(size_t(H_) * C * kvp * 2);
  }
  // RoPE table in double: inv_freq_i = theta^(-2i/rot), pairs (i, i + rot/2)
  const int hr = rot_ / 2;
  cos_ = q.alloc(size_t(max_ctx) * hr * 4);
  sin_ = q.alloc(size_t(max_ctx) * hr * 4);
  float* cs = cos_.host<float>();
  float* sn = sin_.host<float>();
  for (int p = 0; p < max_ctx; ++p)
    for (int i = 0; i < hr; ++i) {
      const double ang = double(p) * std::pow(theta, -2.0 * i / rot_);
      cs[size_t(p) * hr + i] = float(std::cos(ang));
      sn[size_t(p) * hr + i] = float(std::sin(ang));
    }
}

void Model::to_q4(Lin& l) {
  if (l.K % 64) fail("q4: K not a multiple of 64");
  const uint32_t ng = l.N * (l.K / 64);
  View w = q_->alloc(size_t(l.N) * l.K / 2), sb = q_->alloc(size_t(ng) * 4);
  id<MTLComputeCommandEncoder> e = q_->encoder();
  [e setComputePipelineState:q_->pipe("q4_quant")];
  bind(e, {l.w, w, sb});
  [e setBytes:&ng length:4 atIndex:3];
  [q_->pre(e) dispatchThreads:MTLSizeMake(ng, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
  l.fmt = Lin::kQ4;
  l.w = w;
  l.sc = sb;  // l.scale (the per-tensor FP8 scale) stays
}

void Model::reset() {
  for (Layer& L : layers_) {
    if (L.attn) continue;
    std::memset(L.conv_state.host<uint8_t>(), 0, [L.conv_state.buf length]);
    std::memset(L.S.host<uint8_t>(), 0, [L.S.buf length]);
  }
}

struct GemvArgs {
  uint32_t N, K, ldx, ldy, rows;
  float scale;
  uint32_t add;
  float eps;
  uint32_t npart;
};

struct MmArgs {
  uint32_t M, N, K, lda, ldb, ldc, sa, sb, sc, bdiv;
};

bool Model::use_mm(const Lin& l, int T) const {
  return T >= mm_min_ && !wdq_.empty() && l.fmt != Lin::kBf16 && l.N % 64 == 0 && l.K % 64 == 0 &&
         size_t(l.N) * l.K <= wdq_cap_;
}

id<MTLComputeCommandEncoder> Model::pm_sync(id<MTLComputeCommandEncoder> e, int k) {
  if (prof_ < 2) return e;
  pm[k] += q_->sync();
  return q_->encoder();
}

void Model::dequant(id<MTLComputeCommandEncoder> e, const Lin& l, View out) {
  if (l.fmt == Lin::kNvfp4 || l.fmt == Lin::kQ4) {
    const uint32_t n32 = l.N * (l.K / 32);
    [e setComputePipelineState:q_->pipe(l.fmt == Lin::kQ4 ? "dq_q4" : "dq_nvfp4")];
    bind(e, {l.w, l.sc, out});
    [e setBytes:&n32 length:4 atIndex:3];
    [q_->pre(e) dispatchThreads:MTLSizeMake(n32, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  } else {
    const uint32_t n16 = l.N * (l.K / 16);
    [e setComputePipelineState:q_->pipe("dq_fp8")];
    bind(e, {l.w, out});
    [e setBytes:&n16 length:4 atIndex:2];
    [q_->pre(e) dispatchThreads:MTLSizeMake(n16, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  }
}

void Model::gemv(const Lin& l, View x, View y, int T, int ldx, int ldy, bool add, int mode, bool reuse_x,
                 View xa) {
  id<MTLComputeCommandEncoder> e = q_->encoder();
  const bool fresh = xh_fresh_;
  xh_fresh_ = false;
  if (use_mm(l, T)) {
    const uint32_t Mp = uint32_t(pad64(T));
    e = pm_sync(e, 5);
    const View wd = l.dq_ready ? wlay_.at(l.dq_off) : wdq_;
    if (!l.dq_ready) dequant(e, l, wdq_);
    e = pm_sync(e, 0);
    const bool have_x = !xa.empty() || ((reuse_x || fresh) && !mm_old_ && xh_buf_ == (__bridge const void*)x.buf &&
                                         xh_off_ == x.off && xh_k_ == l.K && xh_t_ == uint32_t(T) &&
                                         xh_ld_ == uint32_t(ldx));
    if (!have_x) {
      const uint32_t xa[4] = {l.K, uint32_t(T), uint32_t(ldx), 0};
      [e setComputePipelineState:q_->pipe("x_half")];
      bind(e, {x, xh_});
      [e setBytes:xa length:16 atIndex:2];
      [q_->pre(e) dispatchThreads:MTLSizeMake(l.K, Mp, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      xh_buf_ = (__bridge const void*)x.buf;
      xh_off_ = x.off;
      xh_k_ = l.K;
      xh_t_ = uint32_t(T);
      xh_ld_ = uint32_t(ldx);
      e = pm_sync(e, 1);
    }
    if (!mm_old_) {
      // one pass: 64x128 tiles (64x64 when N is not a multiple of 128) with the scale, add and row clip fused in
      const bool wide = l.N % 128 == 0;
      const uint32_t md = mode >= 2 ? uint32_t(mode) : add ? 1u : 0u;
      struct { uint32_t M, N, K, T, ldy, mode; float scale; } ea = {Mp, l.N, l.K, uint32_t(T), uint32_t(ldy), md, l.scale};
      [e setComputePipelineState:q_->pipe(wide ? "mm_ep_64x128" : "mm_ep_64x64")];
      bind(e, {xa.empty() ? xh_ : xa, wd, y});
      [e setBytes:&ea length:sizeof(ea) atIndex:3];
      [e setBuffer:(mode == 3 ? xh2_ : y).buf offset:(mode == 3 ? xh2_ : y).off atIndex:4];
      [q_->pre(e) dispatchThreadgroups:MTLSizeMake(l.N / (wide ? 128 : 64), Mp / 64, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      e = pm_sync(e, 2);
      return;
    }
    MmArgs m{Mp, l.N, l.K, l.K, l.K, l.N, 0, 0, 0, 1};
    [e setComputePipelineState:q_->pipe("mm_nt_h")];
    bind(e, {xh_, wd, mmc_});
    bind_args(e, 3, m);
    [q_->pre(e) dispatchThreadgroups:MTLSizeMake(l.N / 64, Mp / 64, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    e = pm_sync(e, 2);
    const uint32_t sa[4] = {l.N, uint32_t(T), uint32_t(ldy), add ? 1u : 0u};
    const float f[2] = {l.scale, 0};
    [e setComputePipelineState:q_->pipe("mm_store")];
    bind(e, {mmc_, y});
    [e setBytes:sa length:16 atIndex:2];
    [e setBytes:f length:8 atIndex:3];
    [q_->pre(e) dispatchThreads:MTLSizeMake(l.N, T, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    e = pm_sync(e, 3);
    return;
  }
  if (g8_ && T >= 2 && T <= 8 && (l.fmt == Lin::kNvfp4 || l.fmt == Lin::kQ4) && l.K % 256 == 0 && l.N >= g8_min_n_ &&
      size_t(l.K) * 16 <= xp8_cap_) {
    // the verify rows: x packed as half pairs, then the simdgroup-matrix GEMV (gemv8 in kernels.metal)
    const bool have = xp8_buf_ == (__bridge const void*)x.buf && xp8_off_ == x.off && xp8_k_ == l.K &&
                      xp8_t_ == uint32_t(T) && xp8_ld_ == uint32_t(ldx) && reuse_x;
    if (!have) {
      const uint32_t pa[4] = {l.K, uint32_t(T), uint32_t(ldx), 0};
      [e setComputePipelineState:q_->pipe("x_pack8")];
      bind(e, {x, xp8_});
      [e setBytes:pa length:16 atIndex:2];
      [q_->pre(e) dispatchThreads:MTLSizeMake(l.K, 4, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      xp8_buf_ = (__bridge const void*)x.buf;
      xp8_off_ = x.off;
      xp8_k_ = l.K;
      xp8_t_ = uint32_t(T);
      xp8_ld_ = uint32_t(ldx);
    }
    GemvArgs a{l.N, l.K, uint32_t(ldx), uint32_t(ldy), uint32_t(T), l.scale, add ? 1u : 0u, eps_, 0};
    [e setComputePipelineState:q_->pipe(l.fmt == Lin::kQ4 ? "gemv8_q4" : "gemv8_nvfp4")];
    bind(e, {xp8_, l.w, l.sc, y});
    bind_args(e, 4, a);
    [q_->pre(e) dispatchThreadgroups:MTLSizeMake((l.N + 63) / 64, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    return;
  }
  const char* k = l.fmt == Lin::kNvfp4 ? "gemv_nvfp4"
                  : l.fmt == Lin::kQ4  ? "gemv_q4"
                  : l.fmt == Lin::kFp8 ? "gemv_fp8"
                                       : "gemv_bf16";
  if (prof_ == 4) e = pm_sync(e, 5);  // decode split: the GEMV passes apart from the rest
  // decode: a GEMV of the normed row reads fxw_ and folds the norm in; a residual add writes the next fxw_ (see fold_in_)
  const bool fin = fold_in_ && T == 1 && x.buf == xn_.buf && x.off == xn_.off;
  if (fin && l.N % 8) fail("gemv: the norm fold needs N a multiple of 8 (RAGON_METAL_NORM_FOLD=0)");
  const bool fout = !fold_out_.empty() && T == 1 && add && y.buf == h_.buf && y.off == h_.off && l.N % 8 == 0;
  if (fin) x = fxw_;
  [e setComputePipelineState:q_->pipe(k, T == 1 ? 1 : 8, (fin ? 1u : 0u) | (fout ? 2u : 0u))];
  if (fin) [e setBuffer:fpart_.buf offset:fpart_.off atIndex:6];
  if (fout) {
    [e setBuffer:fold_out_.buf offset:fold_out_.off atIndex:7];
    [e setBuffer:fxw_.buf offset:fxw_.off atIndex:8];
    [e setBuffer:fpart_.buf offset:fpart_.off atIndex:9];
  }
  GemvArgs a{l.N, l.K, uint32_t(ldx), uint32_t(ldy), uint32_t(T), l.scale, add ? 1u : 0u, eps_, uint32_t(l.K / 8)};
  if (l.fmt == Lin::kNvfp4 || l.fmt == Lin::kQ4) {
    bind(e, {x, l.w, l.sc, y});
    bind_args(e, 4, a);
  } else {
    bind(e, {x, l.w, y});
    bind_args(e, 3, a);
  }
  // one dispatch: threadgroup (x, y) takes output rows 8x.. and activation rows 8y..
  [q_->pre(e) dispatchThreadgroups:MTLSizeMake((l.N + 7) / 8, (T + 7) / 8, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  if (prof_ == 4) pm_sync(e, 2);
}

struct NormArgs {
  uint32_t n, rpt, ldt, ldh, ldo_t, ldo_h;
  float eps;
  uint32_t mode;
  float scale;
  uint32_t half_out, tvalid;
};

void Model::norm(View x, View w, View z, View out, int rows, int n, int rpt, int ldt, int ldh, int ldo_t,
                 int ldo_h, int mode, float scale, const Lin* next, int T) {
  const bool ho = xh_fuse_ && next && T > 0 && mm_fused(*next, T) && int(next->K) == ldo_t && rows == T * rpt &&
                  ldo_t == n * rpt && (rpt == 1 || ldo_h == n);
  if (ho) rows = pad64(T) * rpt;  // the pad rows zero xh_ as x_half does
  id<MTLComputeCommandEncoder> e = q_->encoder();
  [e setComputePipelineState:q_->pipe("rmsnorm_rows")];
  NormArgs a{uint32_t(n),  uint32_t(rpt), uint32_t(ldt), uint32_t(ldh), uint32_t(ldo_t), uint32_t(ldo_h), eps_,
             uint32_t(mode), scale, ho ? 1u : 0u, uint32_t(T)};
  bind(e, {x, w.empty() ? x : w, z.empty() ? x : z, out});
  bind_args(e, 4, a);
  const View oh = xh_.empty() ? out : xh_;  // xh_ exists only with the prefill matmul
  [e setBuffer:oh.buf offset:oh.off atIndex:5];
  [q_->pre(e) dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  if (ho) {
    xh_buf_ = (__bridge const void*)out.buf;
    xh_off_ = out.off;
    xh_k_ = next->K;
    xh_t_ = uint32_t(T);
    xh_ld_ = uint32_t(ldo_t);
    xh_fresh_ = true;
  }
}

struct GdnArgs {
  uint32_t T, Hk, Hv, Dk, Dv, ldc, keep;
};
struct GdnChunkArgs {
  uint32_t T, Hk, Hv, ldc;
};

void Model::gdn(Layer& L, int T, const Dump& dump, const std::string& tag) {
  const int cw = int(L.qkv.N), vw = Hv_ * Dv_;
  gemv(L.qkv, xn_, qkv_, T, hidden_, cw, false);
  q_->overlap();  // z, a and b read xn_ like qkv
  gemv(L.z, xn_, zb_, T, hidden_, vw, false, 0, true);
  q_->overlap();
  gemv(L.a, xn_, ab_, T, hidden_, Hv_, false, 0, true);
  q_->overlap();
  gemv(L.b, xn_, bb_, T, hidden_, Hv_, false, 0, true);
  if (dump) {
    q_->sync();
    dump(tag + "qkv", qkv_, {T, cw});
  }
  const size_t li = size_t(&L - layers_.data());
  if (verify_) copy_rows(qkv_, vpre_[li], T, cw, cw, cw);  // the conv input, for the commit's replay
  id<MTLComputeCommandEncoder> e = q_->encoder();
  const bool chunked = !verify_ && gdn_min_ && T >= gdn_min_;  // the chunked path runs its own conv (half output)
  if (!chunked) {
    [e setComputePipelineState:q_->pipe("gdn_conv")];
    bind(e, {qkv_, L.conv_state, L.conv_w});
    uint32_t ca[4] = {uint32_t(cw), uint32_t(T), verify_ ? 1u : 0u, 0};
    [e setBytes:ca length:16 atIndex:3];
    [q_->pre(e) dispatchThreads:MTLSizeMake(cw, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  }
  if (!chunked) q_->overlap();  // the gates do not touch the conv's buffers
  [e setComputePipelineState:q_->pipe("gdn_gates")];
  bind(e, {ab_, bb_, L.A_log, L.dt_bias});
  uint32_t ga[2] = {uint32_t(Hv_), uint32_t(T)};
  [e setBytes:ga length:8 atIndex:4];
  [q_->pre(e) dispatchThreads:MTLSizeMake(Hv_ * T, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
  if (verify_) {
    copy_rows(ab_, vg_[li], T, Hv_, Hv_, Hv_);
    q_->overlap();
    copy_rows(bb_, vbeta_[li], T, Hv_, Hv_, Hv_);
    e = q_->encoder();
  }
  if (dump) {
    q_->sync();
    if (!chunked) dump(tag + "conv", qkv_, {T, cw});
    dump(tag + "g", ab_, {T, Hv_});
    dump(tag + "beta", bb_, {T, Hv_});
    e = q_->encoder();
  }
  if (prof_) {
    prof[1] += q_->sync();
    e = q_->encoder();
  }
  if (chunked) {
    // chunked recurrence: the conv in token blocks to raw half q/k/v, every (sub-chunk, head) in parallel, then the
    // sub-chunks in order per head
    const uint32_t nc = uint32_t((T + 63) / 64);
    GdnChunkArgs ga2{uint32_t(T), uint32_t(Hk_), uint32_t(Hv_), uint32_t(cw)};
    [e setComputePipelineState:q_->pipe("gdn_conv_h")];
    bind(e, {qkv_, L.conv_state, L.conv_w, gq_, gk_, gv_});
    bind_args(e, 6, ga2);
    [q_->pre(e) dispatchThreads:MTLSizeMake(cw, nc, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    if (nc > 1) {
      [e setComputePipelineState:q_->pipe("gdn_conv_state")];
      bind(e, {qkv_, L.conv_state});
      bind_args(e, 2, ga2);
      [q_->pre(e) dispatchThreads:MTLSizeMake(cw, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    if (prof_ >= 3) {
      gtp_[0] += q_->sync();
      e = q_->encoder();
    }
    [e setComputePipelineState:q_->pipe("gdn_chunk_prep")];
    bind(e, {gq_, gk_, gv_, ab_, bb_, gW_, gU_, gA2_, glg_});
    bind_args(e, 9, ga2);
    [q_->pre(e) dispatchThreadgroups:MTLSizeMake(nc, Hv_, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    if (prof_ >= 3) {
      gtp_[1] += q_->sync();
      e = q_->encoder();
    }
    [e setComputePipelineState:q_->pipe("gdn_chunk_seq")];
    bind(e, {gk_, gW_, gU_, gA2_, glg_, L.S, core_});
    bind_args(e, 7, ga2);
    [q_->pre(e) dispatchThreadgroups:MTLSizeMake(Dv_ / 32, Hv_, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    if (prof_ >= 3) {
      gtp_[2] += q_->sync();
      if (++gtn_ % 48 == 0) std::fprintf(stderr, "gdn chunk s: conv %.3f prep %.3f seq %.3f\n", gtp_[0], gtp_[1], gtp_[2]);
    }
  } else {
    const bool four = T >= 16;  // prefill: four state rows a simdgroup
    [e setComputePipelineState:q_->pipe(four ? "gdn_recurrent4" : "gdn_recurrent")];
    bind(e, {qkv_, ab_, bb_, L.S, core_});
    GdnArgs g{uint32_t(T), uint32_t(Hk_), uint32_t(Hv_), uint32_t(Dk_), uint32_t(Dv_), uint32_t(cw), verify_ ? 1u : 0u};
    bind_args(e, 5, g);
    [q_->pre(e) dispatchThreadgroups:MTLSizeMake(Hv_ * Dv_ / (four ? 32 : 8), 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  }
  if (prof_) {
    const double tr = q_->sync();
    prof[1] += tr;
    prof[4] += tr;
  }
  if (dump) {
    q_->sync();
    dump(tag + "core", core_, {T, vw});
    dump(tag + "state", L.S, {Hv_, Dv_, Dk_});
  }
  // gated norm: rms over Dv with weight, times silu(z)
  norm(core_, L.gnorm, zb_, gated_, T * Hv_, Dv_, Hv_, vw, Dv_, vw, Dv_, 1, 1.0f, &L.out, T);
  gemv(L.out, gated_, h_, T, vw, hidden_, true);
}

struct RopeArgs {
  uint32_t heads, ldt, ldh, half_rot, pos0;
};
struct PrefAttnArgs {
  uint32_t T, Tp, H, D, ldq, qh, pos0, kvp;
  float scale;
};

struct AttnArgs {
  uint32_t H, KV, D, ldq, qh, pos0, chunk, nchunks;
  float scale;
};

void Model::attention(Layer& L, int T, int pos0, const Dump& dump, const std::string& tag) {
  const int qw = H_ * 2 * D_, kw = KV_ * D_;
  gemv(L.q, xn_, qg_, T, hidden_, qw, false);
  q_->overlap();  // k and v read xn_ like q
  gemv(L.k, xn_, kb_, T, hidden_, kw, false, 0, true);
  q_->overlap();
  gemv(L.v, xn_, vb_, T, hidden_, kw, false, 0, true);
  norm(qg_, L.q_norm, View{}, qg_, T * H_, D_, H_, qw, 2 * D_, qw, 2 * D_, 0);
  q_->overlap();  // k's norm is independent of q's
  norm(kb_, L.k_norm, View{}, kb_, T * KV_, D_, KV_, kw, D_, kw, D_, 0);
  id<MTLComputeCommandEncoder> e = q_->encoder();
  [e setComputePipelineState:q_->pipe("rope_rows")];
  RopeArgs rq{uint32_t(H_), uint32_t(qw), uint32_t(2 * D_), uint32_t(rot_ / 2), uint32_t(pos0)};
  bind(e, {qg_, cos_, sin_});
  bind_args(e, 3, rq);
  [q_->pre(e) dispatchThreads:MTLSizeMake(rot_ / 2, T * H_, 1) threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
  RopeArgs rk{uint32_t(KV_), uint32_t(kw), uint32_t(D_), uint32_t(rot_ / 2), uint32_t(pos0)};
  q_->overlap();  // k's rope is independent of q's
  bind(e, {kb_, cos_, sin_});
  bind_args(e, 3, rk);
  [q_->pre(e) dispatchThreads:MTLSizeMake(rot_ / 2, T * KV_, 1) threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
  if (kv8_) {
    if (D_ != 256) fail("int8 KV needs head dim 256");
    [e setComputePipelineState:q_->pipe("kv_write8")];
    bind(e, {kb_, vb_, L.kc, L.vc, L.ks, L.vs});
    const uint32_t kva[4] = {uint32_t(KV_), uint32_t(D_), uint32_t(pos0), 0};
    [e setBytes:kva length:16 atIndex:6];
    [q_->pre(e) dispatchThreadgroups:MTLSizeMake(KV_, T, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  } else {
    [e setComputePipelineState:q_->pipe("kv_write")];
    bind(e, {kb_, vb_, L.kc, L.vc});
    uint32_t kva[2] = {uint32_t(kw), uint32_t(pos0)};
    [e setBytes:kva length:8 atIndex:4];
    [q_->pre(e) dispatchThreads:MTLSizeMake(kw, T, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  }
  if (dump) {
    q_->sync();
    dump(tag + "qg", qg_, {T, H_, 2 * D_});
    dump(tag + "k", kb_, {T, KV_, D_});
    dump(tag + "v", vb_, {T, KV_, D_});
    e = q_->encoder();
  }
  if (T >= mm_min_ && (attn_flash_ || !S_.empty())) {
    attention_mm(L, T, pos0);
    e = q_->encoder();
  } else if (mq_nch_ && T >= 2 && T <= 10 && !qb_.empty() && !kv8_ && D_ == 256 && H_ == 6 * KV_) {
    // the verify rows: the G * T query rows of a kv head as one 64-row tensor-op tile, keys split in chunks
    const uint32_t Tp = 16, D = uint32_t(D_), g = uint32_t(H_ / KV_), kvlen = uint32_t(pos0 + T);
    PrefAttnArgs pa{uint32_t(T), Tp, uint32_t(H_), D, uint32_t(qw), 2 * D, uint32_t(pos0), uint32_t(pad64(pos0 + T)),
                    1.0f / std::sqrt(float(D_))};
    [e setComputePipelineState:q_->pipe("q_bf16g")];
    bind(e, {qg_, qb_});
    bind_args(e, 2, pa);
    [e setBytes:&g length:4 atIndex:3];
    [q_->pre(e) dispatchThreads:MTLSizeMake(D, Tp, H_) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    const uint32_t nmax = uint32_t(std::max(1, std::min(mq_nch_, C_ * nchunk_max_ / T)));
    uint32_t chunk = (kvlen + nmax - 1) / nmax;
    chunk = std::max(64u, (chunk + 63) / 64 * 64);
    const uint32_t nch = (kvlen + chunk - 1) / chunk, sp[2] = {chunk, nch};
    [e setComputePipelineState:q_->pipe("attn_flash2_6s")];
    bind(e, {qb_, L.kc, L.vc, part_});
    bind_args(e, 4, pa);
    [e setBytes:&flash_dbg_ length:4 atIndex:5];
    [e setBytes:sp length:8 atIndex:6];
    [q_->pre(e) dispatchThreadgroups:MTLSizeMake(nch, KV_, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    AttnArgs a{uint32_t(H_), uint32_t(KV_), D, uint32_t(qw), 2 * D, uint32_t(pos0), chunk, nch, 1.0f / std::sqrt(float(D_))};
    [e setComputePipelineState:q_->pipe("attn_combine")];
    bind(e, {part_, ob_});
    bind_args(e, 2, a);
    [q_->pre(e) dispatchThreadgroups:MTLSizeMake(H_, T, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  } else {
  const int kvlen = pos0 + T;
  const bool fused = attn_gqa_ && D_ == 256 && H_ == 6 * KV_;
  if (kv8_ && !fused) fail("int8 KV needs the fused decode attention (6 query heads a kv head, head dim 256)");
  int chunk;
  if (fused) {
    // part_ holds C * H * nchunk_max_ partials; the fused kernel splits finer when T is small.
    const int nmax = std::max(1, std::min(attn_nch_, C_ * nchunk_max_ / T));
    chunk = std::max(attn_chunk_min_, (kvlen + nmax - 1) / nmax);
    chunk = (chunk + 3) / 4 * 4;
  } else {
    chunk = std::max(64, (kvlen + nchunk_max_ - 1) / nchunk_max_);
    chunk = (chunk + 7) / 8 * 8;
  }
  const int nch = (kvlen + chunk - 1) / chunk;
  AttnArgs a{uint32_t(H_), uint32_t(KV_), uint32_t(D_), uint32_t(qw), uint32_t(2 * D_), uint32_t(pos0),
             uint32_t(chunk), uint32_t(nch), 1.0f / std::sqrt(float(D_))};
  [e setComputePipelineState:q_->pipe(fused ? (kv8_ ? "attn_gqa6_q8" : "attn_gqa6") : "attn_partial")];
  bind(e, {qg_, L.kc, L.vc, part_});
  bind_args(e, 4, a);
  if (fused) {
    const View& ks = kv8_ ? L.ks : L.kc;  // unused by the bfloat kernel
    const View& vs = kv8_ ? L.vs : L.vc;
    [e setBuffer:ks.buf offset:ks.off atIndex:5];
    [e setBuffer:vs.buf offset:vs.off atIndex:6];
  }
  [q_->pre(e) dispatchThreadgroups:MTLSizeMake(nch, fused ? KV_ : H_, T) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [e setComputePipelineState:q_->pipe("attn_combine")];
  bind(e, {part_, ob_});
  bind_args(e, 2, a);
  [q_->pre(e) dispatchThreadgroups:MTLSizeMake(H_, T, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  }
  if (dump) {
    q_->sync();
    dump(tag + "attn", ob_, {T, H_ * D_});
    e = q_->encoder();
  }
  [e setComputePipelineState:q_->pipe("sigmoid_gate")];
  uint32_t hd[4] = {uint32_t(H_), uint32_t(D_), 0, 0};
  bind(e, {ob_, qg_});
  [e setBytes:hd length:16 atIndex:2];
  [q_->pre(e) dispatchThreads:MTLSizeMake(H_ * D_, T, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  gemv(L.o, ob_, h_, T, H_ * D_, hidden_, true);
}

// Prefill attention as matmuls: S = Q K^T over the padded keys, causal softmax, O = P V into ob_ (bfloat inputs).
void Model::attention_mm(Layer& L, int T, int pos0) {
  id<MTLComputeCommandEncoder> e = q_->encoder();
  const uint32_t Tp = uint32_t(pad64(T)), kvp = uint32_t(pad64(pos0 + T)), D = uint32_t(D_), H = uint32_t(H_);
  const uint32_t kw = uint32_t(KV_ * D_), g = uint32_t(H_ / KV_);
  PrefAttnArgs a{uint32_t(T), Tp, H, D, uint32_t(H_ * 2 * D_), 2 * D, uint32_t(pos0), kvp, 1.0f / std::sqrt(float(D_))};
  const bool flash = attn_flash_ != 0;
  [e setComputePipelineState:q_->pipe(flash ? "q_bf16g" : "q_bf16")];
  bind(e, {qg_, qb_});
  bind_args(e, 2, a);
  if (flash) [e setBytes:&g length:4 atIndex:3];
  [q_->pre(e) dispatchThreads:MTLSizeMake(D, Tp, H) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  View kc = L.kc, vc = L.vc;
  if (kv8_) {  // expand this layer's int8 cache to bfloat once per chunk (rows past pos0 + T zero)
    [e setComputePipelineState:q_->pipe("kv_dq8")];
    bind(e, {L.kc, L.vc, L.ks, L.vs, kdq_, vdq_});
    const uint32_t da[4] = {uint32_t(KV_), D, uint32_t(pos0 + T), 0};
    [e setBytes:da length:16 atIndex:6];
    [q_->pre(e) dispatchThreads:MTLSizeMake(kw, kvp, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    kc = kdq_;
    vc = vdq_;
  }
  if (flash) {
    // 2: flash2 (64 rows, 8 simdgroups); 4, 8: 64 query rows a threadgroup (4 or 8 simdgroups); 128, 192: taller tiles
    const char* fn = attn_flash_ == 2 ? "attn_flash2_6" : attn_flash_ == 8 ? "attn_flash6_8" : attn_flash_ == 128 ? "attn_flash6_128"
                     : attn_flash_ == 192 ? "attn_flash6_192" : "attn_flash6";
    const uint32_t br = attn_flash_ >= 128 ? uint32_t(attn_flash_) : 64, nsg = attn_flash_ == 192 ? 12 : attn_flash_ == 4 ? 4 : 8;
    [e setComputePipelineState:q_->pipe(fn)];
    bind(e, {qb_, kc, vc, ob_});
    bind_args(e, 4, a);
    [e setBytes:&flash_dbg_ length:4 atIndex:5];
    const uint32_t sp0[2] = {0, 0};
    [e setBytes:sp0 length:8 atIndex:6];
    [q_->pre(e) dispatchThreadgroups:MTLSizeMake(Tp * g / br, KV_, 1) threadsPerThreadgroup:MTLSizeMake(32 * nsg, 1, 1)];
    return;
  }
  MmArgs s{Tp, kvp, D, D, kw, kvp, Tp * D, D, Tp * kvp, g};
  [e setComputePipelineState:q_->pipe("mm_nt_b")];
  bind(e, {qb_, kc, S_});
  bind_args(e, 3, s);
  [q_->pre(e) dispatchThreadgroups:MTLSizeMake(kvp / 64, Tp / 64, H) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
  [e setComputePipelineState:q_->pipe("softmax_causal")];
  bind(e, {S_, P_});
  bind_args(e, 2, a);
  [q_->pre(e) dispatchThreadgroups:MTLSizeMake(Tp, H, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  MmArgs o{Tp, D, kvp, kvp, kw, H * D, Tp * kvp, D, D, g};
  [e setComputePipelineState:q_->pipe("mm_nn_b")];
  bind(e, {P_, vc, ob_});
  bind_args(e, 3, o);
  [q_->pre(e) dispatchThreadgroups:MTLSizeMake(D / 64, Tp / 64, H) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

void Model::layer(size_t i, int T, int pos0, const Dump& dump) {
  id<MTLComputeCommandEncoder> e;
  Layer& L = layers_[i];
  const bool inner = dump && (i == 0 || i == 3);
  const std::string tag = "l" + std::to_string(i) + ".";
  const bool fold = fold_ && T == 1 && !dump;  // decode: the GEMVs that read xn_ norm the residual row themselves
  fold_in_ = fold && i > 0;  // the previous layer's down GEMV wrote fxw_ and fpart_
  if (!fold_in_) norm(h_, L.in_norm, View{}, xn_, T, hidden_, 1, hidden_, 0, hidden_, 0, 0, 1.0f, L.attn ? &L.q : &L.qkv, T);
  if (inner) {
    q_->sync();
    dump(tag + "in_norm", xn_, {T, hidden_});
  }
  if (fold) fold_out_ = L.post_norm;  // the output projection's residual add
  if (L.attn) attention(L, T, pos0, inner ? dump : nullptr, tag);
  else gdn(L, T, inner ? dump : nullptr, tag);
  fold_out_ = View{};
  fold_in_ = fold;
  if (prof_) prof[L.attn ? 0 : 1] += q_->sync();
  if (!fold) norm(h_, L.post_norm, View{}, xn_, T, hidden_, 1, hidden_, 0, hidden_, 0, 0, 1.0f, &L.gate, T);
  if (inner) {
    q_->sync();
    dump(tag + "post_norm", xn_, {T, hidden_});
  }
  gemv(L.gate, xn_, gb_, T, hidden_, ffn_, false);
  const bool h3 = xh_fuse_ && mm_fused(L.up, T) && mm_fused(L.down, T);
  if (h3) {
    gemv(L.up, xn_, gb_, T, hidden_, ffn_, false, 3, true);  // xh2 = half(silu(gate) * up), down's A
  } else if (mm_fused(L.up, T)) {
    gemv(L.up, xn_, gb_, T, hidden_, ffn_, false, 2, true);  // gb = silu(gate) * up in the matmul's store
  } else {
    q_->overlap();  // up reads xn_ like gate
    gemv(L.up, xn_, ub_, T, hidden_, ffn_, false, 0, true);
    e = q_->encoder();
    [e setComputePipelineState:q_->pipe("silu_mul")];
    bind(e, {gb_, ub_, gb_});
    const uint32_t n = uint32_t(T * ffn_);
    [e setBytes:&n length:4 atIndex:3];
    [q_->pre(e) dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    e = pm_sync(e, 4);
  }
  fold_in_ = false;
  if (fold) fold_out_ = i + 1 < layers_.size() ? layers_[i + 1].in_norm : final_norm_;
  gemv(L.down, gb_, h_, T, ffn_, hidden_, true, 0, false, h3 ? xh2_ : View{});
  fold_out_ = View{};
  if (!feat_.empty() && feat_[i] >= 0) capture(T, pos0, i);
  if (prof_) prof[2] += q_->sync();
  if (dump) {
    q_->sync();
    dump("h" + std::to_string(i), h_, {T, hidden_});
  }
}

void Model::forward(const uint32_t* ids, int T, int pos0, bool all_logits, const Dump& dump) {
  if (T > C_) fail("forward: T exceeds the chunk");
  if (pos0 + T > max_ctx_) fail("forward: context exceeds max_ctx");
  if (ids) std::memcpy(ids_.host<uint32_t>(), ids, size_t(T) * 4);  // null: the ids are already on the device
  q_->concurrent = conc_ && T < mm_min_ && !prof_ && !dump;  // decode: independent small dispatches overlap
  id<MTLComputeCommandEncoder> e = q_->encoder();
  [e setComputePipelineState:q_->pipe("embed_bf16")];
  bind(e, {ids_, embed_, h_});
  const uint32_t dim = uint32_t(hidden_);
  [e setBytes:&dim length:4 atIndex:3];
  [q_->pre(e) dispatchThreads:MTLSizeMake(hidden_, T, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  if (dump) {
    q_->sync();
    dump("embed", h_, {T, hidden_});
  }
  const bool flush = flush_ > 0 && T < mm_min_ && !prof_ && !dump;
  cap_lo_ = pos0;
  cap_hi_ = pos0 + T;
  for (size_t i = 0; i < layers_.size(); ++i) {
    layer(i, T, pos0, dump);
    if (flush && (i + 1) % size_t(flush_) == 0) q_->flush();
  }
  if (has_draft() && !verify_) draft_materialize(T, pos0);  // verify rows: the commit writes the accepted ones
  const int r0 = all_logits ? 0 : T - 1, rows = all_logits ? T : 1;
  View hr = h_.at(size_t(r0) * hidden_ * 4);
  fold_in_ = fold_ && T == 1 && !dump;  // the last down GEMV wrote fxw_ and fpart_ with the final norm's weight
  if (!fold_in_) norm(hr, final_norm_, View{}, xn_, rows, hidden_, 1, hidden_, 0, hidden_, 0, 0);
  if (dump) {
    q_->sync();
    dump("final_norm", xn_, {rows, hidden_});
  }
  gemv(head_, xn_, logits_, rows, hidden_, V_, false);
  fold_in_ = false;
  q_->concurrent = false;  // the encoder in use keeps its kind
  if (verify_) {  // the target's greedy pick of every row, on the GPU
    struct {
      uint32_t n, nb, banned[8];
    } a{uint32_t(V_), uint32_t(gban_.size()), {}};
    std::copy(gban_.begin(), gban_.end(), a.banned);
    for (int r = 0; r < rows; ++r) {
      id<MTLComputeCommandEncoder> e2 = q_->encoder();
      [e2 setComputePipelineState:q_->pipe("top2_ban")];
      bind(e2, {logits_.at(size_t(r) * V_ * 4), vout_.at(size_t(vmax_) * 12 + size_t(r) * 4), vout_.at(size_t(r) * 12)});
      bind_args(e2, 3, a);
      if (r) q_->overlap();
      [q_->pre(e2) dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    }
  }
  if (greedy_) {
    struct {
      uint32_t n, nb, banned[8];
    } a{uint32_t(V_), uint32_t(gban_.size()), {}};
    std::copy(gban_.begin(), gban_.end(), a.banned);
    id<MTLComputeCommandEncoder> e2 = q_->encoder();
    [e2 setComputePipelineState:q_->pipe("top2_ban")];
    bind(e2, {logits_, ids_, gout_.at(size_t(gstep_) * 12)});
    bind_args(e2, 3, a);
    [q_->pre(e2) dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    id<MTLCommandBuffer> cb = q_->cb;
    q_->flush();
    gcb_.push_back(cb);
    return;
  }
  const double th = q_->sync();
  gpu_s += th;
  if (prof_) prof[3] += th;
}

void Model::greedy_begin(uint32_t id0, const std::vector<int>& banned, int steps) {
  if (banned.size() > 8) fail("greedy: more than 8 banned ids");
  gban_.assign(banned.begin(), banned.end());
  if (gout_.empty() || [gout_.buf length] < size_t(steps) * 12) gout_ = q_->alloc(size_t(steps) * 12);
  gcb_.clear();
  gstep_ = 0;
  ids_.host<uint32_t>()[0] = id0;
}

void Model::greedy_step(int pos) {
  if (size_t(gstep_ + 1) * 12 > [gout_.buf length]) fail("greedy: more steps than greedy_begin allowed");
  greedy_ = true;
  forward(nullptr, 1, pos, false);
  greedy_ = false;
  ++gstep_;
}

void Model::greedy_wait(int k) {
  q_->wait_for(gcb_.at(size_t(k)));
}

void Model::prefill(const uint32_t* ids, int P) {
  if (!layer_major_ || P <= C_ || wdq_.empty()) {
    for (int s = 0; s < P; s += C_) forward(ids + s, std::min(C_, P - s), s, false);
    return;
  }
  if (P > max_ctx_) fail("prefill: context exceeds max_ctx");
  auto lins = [](Layer& L) { return std::array<Lin*, 10>{&L.qkv, &L.z, &L.out, &L.q, &L.k, &L.v, &L.o, &L.gate, &L.up, &L.down}; };
  if (wlay_.empty()) {
    size_t cap = 0;
    for (Layer& L : layers_) {
      size_t n = 0;
      for (Lin* l : lins(L))
        if (l->N && use_mm(*l, C_)) n += size_t(l->N) * l->K * 2;
      cap = std::max(cap, n);
    }
    wlay_ = q_->alloc(cap);
  }
  const int nc = (P + C_ - 1) / C_;
  const size_t row = size_t(hidden_) * 4, hb = size_t(nc) * C_ * row;
  if (hall_.empty() || [hall_.buf length] < hb) hall_ = q_->alloc(hb);
  for (int c = 0; c < nc; ++c) {  // ids_ holds one chunk: embed and sync chunk by chunk
    const int T = std::min(C_, P - c * C_);
    std::memcpy(ids_.host<uint32_t>(), ids + size_t(c) * C_, size_t(T) * 4);
    id<MTLComputeCommandEncoder> e = q_->encoder();
    [e setComputePipelineState:q_->pipe("embed_bf16")];
    bind(e, {ids_, embed_, hall_.at(size_t(c) * C_ * row)});
    const uint32_t dim = uint32_t(hidden_);
    [e setBytes:&dim length:4 atIndex:3];
    [q_->pre(e) dispatchThreads:MTLSizeMake(hidden_, T, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    q_->sync();
  }
  const View h0 = h_;
  cap_lo_ = std::max(0, P - cap_rows_);  // the drafter's context: the prompt's last rows only
  cap_hi_ = P;
  for (size_t i = 0; i < layers_.size(); ++i) {
    Layer& L = layers_[i];
    size_t off = 0;
    id<MTLComputeCommandEncoder> e = pm_sync(q_->encoder(), 5);
    for (Lin* l : lins(L))
      if (l->N && use_mm(*l, C_)) {
        dequant(e, *l, wlay_.at(off));
        l->dq_ready = true;
        l->dq_off = off;
        off += size_t(l->N) * l->K * 2;
      }
    pm_sync(e, 0);
    for (int c = 0; c < nc; ++c) {
      h_ = hall_.at(size_t(c) * C_ * row);
      layer(i, std::min(C_, P - c * C_), c * C_, nullptr);
    }
    for (Lin* l : lins(L)) l->dq_ready = false;
    q_->sync();  // one command buffer a layer
  }
  h_ = h0;
  if (has_draft()) draft_materialize(cap_hi_ - cap_lo_, cap_lo_);
  norm(hall_.at(size_t(P - 1) * row), final_norm_, View{}, xn_, 1, hidden_, 1, hidden_, 0, hidden_, 0, 0);
  gemv(head_, xn_, logits_, 1, hidden_, V_, false);
  const double th = q_->sync();
  if (prof_) prof[3] += th;
}


// ---- the DFlash2 drafter ------------------------------------------------------------------------

namespace {
struct CopyArgsH {
  uint32_t cols, lds, ldd;
};
struct GconvArgsH {
  uint32_t C, groups, block, side;
};
struct DRopeArgsH {
  uint32_t ld, pos0;
  float theta, eps;
};
struct DStoreArgsH {
  uint32_t w, pos0, ring;
};
struct DAttnArgsH {
  uint32_t L, B, wl, ring, Hq, KV;
};
}  // namespace

void Model::copy_rows(View src, View dst, int rows, int cols, int lds, int ldd) {
  id<MTLComputeCommandEncoder> e = q_->encoder();
  [e setComputePipelineState:q_->pipe("copy_rows")];
  bind(e, {src, dst});
  CopyArgsH a{uint32_t(cols), uint32_t(lds), uint32_t(ldd)};
  bind_args(e, 2, a);
  [q_->pre(e) dispatchThreads:MTLSizeMake(cols, rows, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

void Model::draft_load(const std::string& dir) {
  NSDictionary* cfg = read_json(dir + "/config.json");
  NSDictionary* dc = cfg[@"dflash_config"];
  if (!dc) fail("drafter: no dflash_config in " + dir + "/config.json");
  auto I = [&](NSString* k) { return [cfg[k] intValue]; };
  dH_ = I(@"hidden_size");
  dI_ = I(@"intermediate_size");
  dHq_ = I(@"num_attention_heads");
  dKV_ = I(@"num_key_value_heads");
  const int hd = I(@"head_dim"), nl = I(@"num_hidden_layers");
  if (dH_ != hidden_) fail("drafter: hidden size does not match the target");
  if (hd != 128 || dHq_ != 4 * dKV_) fail("drafter: the Metal kernels need head_dim 128 and 4 query heads a kv head");
  if (cfg[@"sliding_window"]) dwin_ = I(@"sliding_window");
  deps_ = [cfg[@"rms_norm_eps"] floatValue];
  NSDictionary* rp = cfg[@"rope_parameters"];
  dtheta_ = rp ? [rp[@"rope_theta"] floatValue] : [cfg[@"rope_theta"] floatValue];
  dblock_ = [dc[@"block_size"] intValue];
  dgroups_ = dH_ / [dc[@"conv_group_size"] intValue];
  dmask_ = [dc[@"mask_token_id"] intValue];
  if ([dc[@"conv_kernel_size"] intValue] != 2) fail("drafter: conv_kernel_size must be 2");
  if ([dc[@"selector_top_k"] intValue] != 16 || [dc[@"selector_rank"] intValue] != 256)
    fail("drafter: the Metal selector needs top_k 16 and rank 256");
  NSArray* tl = dc[@"target_layer_ids"];
  nfeat_ = int(tl.count);
  feat_.assign(layers_.size(), -1);
  for (int i = 0; i < nfeat_; ++i) {
    const int li = [tl[i] intValue];
    if (li < 0 || li >= int(layers_.size())) fail("drafter: target layer out of range");
    feat_[size_t(li)] = i;
  }
  dck_.open(*q_, dir);
  dfc_ = lin_of(dck_, "fc");
  if (int(dfc_.K) != nfeat_ * dH_) fail("drafter: fc input does not match the target layers");
  dsel_ = lin_of(dck_, "candidate_selector.hidden_projection");
  dhnorm_ = dck_.f32("hidden_norm.weight");
  dnorm_ = dck_.f32("norm.weight");
  dpred_ = dck_.get("candidate_selector.predecessor_codebook").v;
  dsucc_ = dck_.get("candidate_selector.successor_codebook").v;
  cap_rows_ = std::max(C_, dwin_);
  ring_ = cap_rows_ + 2 * vmax_;
  const size_t kvw = size_t(dKV_) * 128;
  for (int l = 0; l < nl; ++l) {
    const std::string p = "layers." + std::to_string(l) + ".";
    DraftLayer L;
    L.in_norm = dck_.f32(p + "input_layernorm.weight");
    L.post_norm = dck_.f32(p + "post_attention_layernorm.weight");
    L.q_norm = dck_.f32(p + "self_attn.q_norm.weight");
    L.k_norm = dck_.f32(p + "self_attn.k_norm.weight");
    L.attn_base = dck_.f32(p + "attention_conv.base_kernel");
    L.mlp_base = dck_.f32(p + "mlp_conv.base_kernel");
    L.q = lin_of(dck_, p + "self_attn.q_proj");
    L.k = lin_of(dck_, p + "self_attn.k_proj");
    L.v = lin_of(dck_, p + "self_attn.v_proj");
    L.o = lin_of(dck_, p + "self_attn.o_proj");
    L.gate = lin_of(dck_, p + "mlp.gate_proj");
    L.up = lin_of(dck_, p + "mlp.up_proj");
    L.down = lin_of(dck_, p + "mlp.down_proj");
    L.attn_kproj = lin_of(dck_, p + "attention_conv.kernel_projection");
    L.mlp_kproj = lin_of(dck_, p + "mlp_conv.kernel_projection");
    if (int(L.attn_kproj.N) != 4 * dgroups_) fail("drafter: kernel projection is not 2 sides x 2 taps x groups");
    L.kc = q_->alloc(size_t(ring_) * kvw * 4);
    L.vc = q_->alloc(size_t(ring_) * kvw * 4);
    dl_.push_back(L);
  }
  const size_t B = size_t(vmax_), H = size_t(dH_);
  cap_ = q_->alloc(size_t(cap_rows_) * nfeat_ * H * 4);
  dctx_ = q_->alloc(size_t(cap_rows_) * H * 4);
  dk_ = q_->alloc(size_t(cap_rows_) * kvw * 4);
  dv_ = q_->alloc(size_t(cap_rows_) * kvw * 4);
  dres_ = q_->alloc(B * H * 4);
  dh_ = q_->alloc(B * H * 4);
  dh2_ = q_->alloc(B * H * 4);
  dout_ = q_->alloc(B * H * 4);
  dcoef_ = q_->alloc(B * 4 * dgroups_ * 4);
  dq_ = q_->alloc(B * dHq_ * 128 * 4);
  dattn_ = q_->alloc(B * dHq_ * 128 * 4);
  dg_ = q_->alloc(B * dI_ * 4);
  du_ = q_->alloc(B * dI_ * 4);
  dunary_ = q_->alloc(B * 16 * 4);
  dcand_ = q_->alloc(B * 16 * 4);
  dhp_ = q_->alloc(B * 256 * 4);
  dscores_ = q_->alloc(B * 256 * 4);
  dids_ = q_->alloc(B * 4);
  vout_ = q_->alloc(B * 16);
  vpre_.assign(layers_.size(), View{});
  vg_.assign(layers_.size(), View{});
  vbeta_.assign(layers_.size(), View{});
  for (size_t i = 0; i < layers_.size(); ++i)
    if (!layers_[i].attn) {
      vpre_[i] = q_->alloc(B * layers_[i].qkv.N * 4);
      vg_[i] = q_->alloc(B * Hv_ * 4);
      vbeta_[i] = q_->alloc(B * Hv_ * 4);
    }
  const size_t need = size_t(std::max({dI_, dH_, dHq_ * 128})) * 16;
  if (need > xp8_cap_) {
    xp8_cap_ = need;
    xp8_ = q_->alloc(xp8_cap_);
    xp8_buf_ = nullptr;
  }
  std::fprintf(stderr, "metal drafter: %d layers, block %d, window %d, %d target features, %.2f GB mapped\n", nl,
               dblock_, dwin_, nfeat_, dck_.mapped_bytes() / 1e9);
}

void Model::capture(int T, int pos0, size_t layer) {
  const int r0 = std::max(0, cap_lo_ - pos0), r1 = std::min(T, cap_hi_ - pos0);
  if (r1 <= r0) return;
  const size_t H = size_t(hidden_);
  copy_rows(h_.at(size_t(r0) * H * 4), cap_.at((size_t(pos0 + r0 - cap_lo_) * nfeat_ + size_t(feat_[layer])) * H * 4),
            r1 - r0, hidden_, hidden_, nfeat_ * hidden_);
}

void Model::draft_materialize(int n, int pos0) {
  if (n <= 0) return;
  const int H = dH_, kvw = dKV_ * 128;
  gemv(dfc_, cap_, dctx_, n, nfeat_ * H, H, false);
  norm(dctx_, dhnorm_, View{}, dctx_, n, H, 1, H, 0, H, 0, 3);
  for (DraftLayer& L : dl_) {
    gemv(L.k, dctx_, dk_, n, H, kvw, false);
    q_->overlap();
    gemv(L.v, dctx_, dv_, n, H, kvw, false, 0, true);
    id<MTLComputeCommandEncoder> e = q_->encoder();
    [e setComputePipelineState:q_->pipe("draft_qk_rope")];
    bind(e, {dk_, L.k_norm});
    DRopeArgsH ra{uint32_t(kvw), uint32_t(pos0), dtheta_, deps_};
    bind_args(e, 2, ra);
    [q_->pre(e) dispatchThreadgroups:MTLSizeMake(dKV_, n, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [e setComputePipelineState:q_->pipe("draft_store_kv")];
    bind(e, {dk_, dv_, L.kc, L.vc});
    DStoreArgsH sa{uint32_t(kvw), uint32_t(pos0), uint32_t(ring_)};
    bind_args(e, 4, sa);
    [q_->pre(e) dispatchThreads:MTLSizeMake(kvw, n, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  }
}

void Model::draft_forward(int B, uint32_t anchor, int pos) {
  const int H = dH_, E = B - 1, dq = dHq_ * 128, kvw = dKV_ * 128;
  uint32_t* ids = dids_.host<uint32_t>();
  ids[0] = anchor;
  for (int i = 1; i < B; ++i) ids[i] = uint32_t(dmask_);
  id<MTLComputeCommandEncoder> e = q_->encoder();
  [e setComputePipelineState:q_->pipe("embed_bf16")];
  bind(e, {dids_, embed_, dres_});
  const uint32_t dim = uint32_t(H);
  [e setBytes:&dim length:4 atIndex:3];
  [q_->pre(e) dispatchThreads:MTLSizeMake(H, B, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  auto add = [&](View x, View y) {
    id<MTLComputeCommandEncoder> e2 = q_->encoder();
    [e2 setComputePipelineState:q_->pipe("add_rows")];
    bind(e2, {x, y});
    const uint32_t n = uint32_t(B * H);
    [e2 setBytes:&n length:4 atIndex:2];
    [q_->pre(e2) dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  };
  auto gconv = [&](View h, View base, int side, View out) {
    id<MTLComputeCommandEncoder> e2 = q_->encoder();
    [e2 setComputePipelineState:q_->pipe("draft_gconv")];
    bind(e2, {h, dcoef_, base, out});
    GconvArgsH a{uint32_t(H), uint32_t(dgroups_), uint32_t(B), uint32_t(side)};
    bind_args(e2, 4, a);
    [q_->pre(e2) dispatchThreads:MTLSizeMake(H, B, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  };
  auto rope = [&](View x, View w, int heads, int ld) {
    id<MTLComputeCommandEncoder> e2 = q_->encoder();
    [e2 setComputePipelineState:q_->pipe("draft_qk_rope")];
    bind(e2, {x, w});
    DRopeArgsH a{uint32_t(ld), uint32_t(pos), dtheta_, deps_};
    bind_args(e2, 2, a);
    [q_->pre(e2) dispatchThreadgroups:MTLSizeMake(heads, B, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
  };
  auto tick = [&](int k) {  // RAGON_METAL_PROF=6: the draft layers split by kind
    if (prof_ == 6) dpm[k] += q_->sync();
  };
  for (size_t l = 0; l < dl_.size(); ++l) {
    DraftLayer& L = dl_[l];
    if (l > 0) add(dres_, dout_);
    norm(dres_, L.in_norm, View{}, dh_, B, H, 1, H, 0, H, 0, 3);
    gemv(L.attn_kproj, dh_, dcoef_, B, H, 4 * dgroups_, false);
    gconv(dh_, L.attn_base, 0, dh2_);
    tick(3);
    gemv(L.q, dh2_, dq_, B, H, dq, false);
    q_->overlap();
    gemv(L.k, dh2_, dk_, B, H, kvw, false, 0, true);
    q_->overlap();
    gemv(L.v, dh2_, dv_, B, H, kvw, false, 0, true);
    tick(0);
    rope(dq_, L.q_norm, dHq_, dq);
    q_->overlap();
    rope(dk_, L.k_norm, dKV_, kvw);
    e = q_->encoder();
    [e setComputePipelineState:q_->pipe("draft_store_kv")];
    bind(e, {dk_, dv_, L.kc, L.vc});
    DStoreArgsH sa{uint32_t(kvw), uint32_t(pos), uint32_t(ring_)};
    bind_args(e, 4, sa);
    [q_->pre(e) dispatchThreads:MTLSizeMake(kvw, B, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e setComputePipelineState:q_->pipe("draft_attn")];
    bind(e, {dq_, L.kc, L.vc, dattn_});
    DAttnArgsH aa{uint32_t(pos), uint32_t(B), uint32_t(dwin_ - 1), uint32_t(ring_), uint32_t(dHq_), uint32_t(dKV_)};
    bind_args(e, 4, aa);
    [q_->pre(e) dispatchThreadgroups:MTLSizeMake(B, dKV_, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    tick(1);
    gemv(L.o, dattn_, dh2_, B, dq, H, false);
    tick(0);
    gconv(dh2_, L.attn_base, 1, dout_);
    add(dres_, dout_);
    norm(dres_, L.post_norm, View{}, dh_, B, H, 1, H, 0, H, 0, 3);
    gemv(L.mlp_kproj, dh_, dcoef_, B, H, 4 * dgroups_, false);
    gconv(dh_, L.mlp_base, 0, dh2_);
    tick(3);
    gemv(L.gate, dh2_, dg_, B, H, dI_, false);
    q_->overlap();
    gemv(L.up, dh2_, du_, B, H, dI_, false, 0, true);
    tick(2);
    e = q_->encoder();
    [e setComputePipelineState:q_->pipe("silu_mul")];
    bind(e, {dg_, du_, dg_});
    const uint32_t ng = uint32_t(B * dI_);
    [e setBytes:&ng length:4 atIndex:3];
    [q_->pre(e) dispatchThreads:MTLSizeMake(ng, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    tick(3);
    gemv(L.down, dg_, dh2_, B, dI_, H, false);
    tick(2);
    gconv(dh2_, L.mlp_base, 1, dout_);
  }
  add(dres_, dout_);
  norm(dres_.at(size_t(H) * 4), dnorm_, View{}, dh_, E, H, 1, H, 0, H, 0, 3);
  if (prof_ == 5) dpm[2] += q_->sync();
  gemv(head_, dh_, logits_, E, H, V_, false);
  q_->overlap();
  gemv(dsel_, dh_, dhp_, E, H, 256, false);
  if (prof_ == 5) dpm[3] += q_->sync();
  e = q_->encoder();
  [e setComputePipelineState:q_->pipe("topk16_rows")];
  bind(e, {logits_, dunary_, dcand_});
  const uint32_t nv = uint32_t(V_);
  [e setBytes:&nv length:4 atIndex:3];
  [q_->pre(e) dispatchThreadgroups:MTLSizeMake(E, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
  [e setComputePipelineState:q_->pipe("selector_lattice")];
  bind(e, {dhp_, dcand_, dunary_, dpred_, dsucc_, dscores_});
  [e setBytes:&anchor length:4 atIndex:6];
  [q_->pre(e) dispatchThreadgroups:MTLSizeMake(E, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  if (prof_ == 5) dpm[4] += q_->sync();
  else q_->sync();
}

void Model::commit(int n, int pos0) {
  for (size_t i = 0; i < layers_.size(); ++i) {
    Layer& L = layers_[i];
    if (L.attn) continue;
    const int cw = int(L.qkv.N);
    id<MTLComputeCommandEncoder> e = q_->encoder();
    [e setComputePipelineState:q_->pipe("gdn_conv")];
    bind(e, {vpre_[i], L.conv_state, L.conv_w});
    uint32_t ca[4] = {uint32_t(cw), uint32_t(n), 0, 0};
    [e setBytes:ca length:16 atIndex:3];
    [q_->pre(e) dispatchThreads:MTLSizeMake(cw, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e setComputePipelineState:q_->pipe("gdn_recurrent")];
    bind(e, {vpre_[i], vg_[i], vbeta_[i], L.S, core_});
    GdnArgs g{uint32_t(n), uint32_t(Hk_), uint32_t(Hv_), uint32_t(Dk_), uint32_t(Dv_), uint32_t(cw), 0};
    bind_args(e, 5, g);
    [q_->pre(e) dispatchThreadgroups:MTLSizeMake(Hv_ * Dv_ / 8, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  }
  if (prof_ == 5) dpm[0] += q_->sync();
  draft_materialize(n, pos0);
  if (prof_ == 5) dpm[1] += q_->sync();
  q_->flush();  // the GPU starts the commit while the host prepares the next draft
}

namespace {
// The target's sampling distribution for one row from its top-k logits (descending): temperature, top-p, min-p,
// as core's speculative path builds it (target_dist in core/speculate.cpp).
Dist row_dist(const float* vals, const int* ids, int k, const Model::SpecSampling& p) {
  std::vector<double> probs(static_cast<size_t>(k));
  double sum = 0;
  for (int i = 0; i < k; ++i) sum += probs[size_t(i)] = std::exp((double(vals[i]) - vals[0]) / p.temperature);
  for (double& pr : probs) pr /= sum;
  int keep = k;
  if (p.top_p < 1.f) {
    double cum = 0;
    for (int i = 0; i < k; ++i) {
      cum += probs[size_t(i)];
      if (cum >= p.top_p) {
        keep = i + 1;
        break;
      }
    }
  }
  if (p.min_p > 0.f) {
    const double floor = probs[0] * p.min_p;
    int n = 0;
    while (n < keep && probs[size_t(n)] >= floor) ++n;
    keep = std::max(n, 1);
  }
  double kept = 0;
  for (int i = 0; i < keep; ++i) kept += probs[size_t(i)];
  Dist d(static_cast<size_t>(keep));
  for (int i = 0; i < keep; ++i) d[size_t(i)] = {ids[i], probs[size_t(i)] / kept};
  return d;
}

// The k largest logits of a row (descending) and their ids, banned ids left out; false when one is not finite.
bool row_topk(const float* lg, int V, int k, const std::vector<int>& banned, float* vals, int* ids) {
  int n = 0;  // vals[0..n) ascending while scanning
  for (int i = 0; i < V; ++i) {
    const float x = lg[i];
    if (n == k && !(x > vals[0])) continue;
    if (!banned.empty() && std::find(banned.begin(), banned.end(), i) != banned.end()) continue;
    int j = n < k ? n++ : 0;
    if (n == k && j == 0) {  // drop the smallest, sift x up
      while (j + 1 < k && vals[j + 1] < x) {
        vals[j] = vals[j + 1];
        ids[j] = ids[j + 1];
        ++j;
      }
    } else {
      while (j > 0 && vals[j - 1] > x) {
        vals[j] = vals[j - 1];
        ids[j] = ids[j - 1];
        --j;
      }
    }
    vals[j] = x;
    ids[j] = i;
  }
  std::reverse(vals, vals + n);
  std::reverse(ids, ids + n);
  if (n < k) return false;
  for (int i = 0; i < n; ++i)
    if (!std::isfinite(vals[i])) return false;
  return true;
}
}  // namespace

int Model::speculate(uint32_t anchor, int pos, int B, const std::vector<int>& banned, uint32_t* out,
                     const SpecSampling* smp) {
  if (!has_draft()) fail("speculate: no drafter loaded");
  if (B < 2 || B > dblock_ || B > 15 || B > vmax_) fail("speculate: block length out of range");
  if (banned.size() > 8) fail("speculate: more than 8 banned ids");
  if (pos + B > max_ctx_) fail("speculate: context exceeds max_ctx");
  using clk = std::chrono::steady_clock;
  const auto t0 = clk::now();
  draft_forward(B, anchor, pos);
  const int E = B - 1;
  uint32_t rows[16];
  rows[0] = anchor;
  const uint32_t* cand = dcand_.host<uint32_t>();
  const float* sc = dscores_.host<float>();
  if (smp && (smp->top_k < 1 || smp->top_k > 64 || !smp->rng || !(smp->temperature > 0.f)))
    fail("speculate: sampling needs 1 <= top_k <= 64, a temperature above zero and a generator");
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  double q[16][16] = {};  // q[e][c]: the draft's probability of candidate c at drafted position e
  int prev = 0;
  for (int e = 0; e < E; ++e) {
    const float* s = sc + (size_t(e) * 16 + size_t(e == 0 ? 0 : prev)) * 16;
    int pick = 0;
    if (!smp) {
      for (int c = 1; c < 16; ++c)
        if (s[c] > s[pick]) pick = c;  // ties to the lower index
    } else {
      const double T = std::max(double(smp->temperature), 1e-5);
      double mx = s[0], sum = 0;
      for (int c = 1; c < 16; ++c) mx = std::max(mx, double(s[c]));
      for (int c = 0; c < 16; ++c) sum += q[e][c] = std::exp((s[c] - mx) / T);
      const double u = uni(*smp->rng) * sum;
      double cum = 0;
      pick = 15;
      for (int c = 0; c < 16; ++c) {
        cum += q[e][c];
        if (u < cum) {
          pick = c;
          break;
        }
      }
      for (int c = 0; c < 16; ++c) q[e][c] /= sum;
    }
    rows[e + 1] = cand[e * 16 + pick];
    prev = pick;
  }
  const auto t1 = clk::now();
  gban_.assign(banned.begin(), banned.end());
  verify_ = true;
  forward(rows, B, pos, true);
  verify_ = false;
  const auto t2 = clk::now();
  const uint32_t* vo = vout_.host<uint32_t>();
  for (int i = 0; i < B; ++i) {
    float top;
    std::memcpy(&top, vo + size_t(i) * 3 + 1, 4);
    if (!std::isfinite(top)) return -1;
  }
  int a = 0;
  uint32_t next = 0;
  if (!smp) {
    while (a < E && rows[a + 1] == vo[size_t(a) * 3]) ++a;
    next = vo[size_t(a) * 3];
  } else {
    const int k = smp->top_k;
    std::vector<float> vals(size_t(B) * size_t(k));
    std::vector<int> ids(size_t(B) * size_t(k));
    bool finite[16];
    // one row per core: blocks capture by value, so the block gets plain pointers
    const float* lg = logits_.host<float>();
    float* vp = vals.data();
    int* ip = ids.data();
    bool* fp = finite;
    const std::vector<int>* bp = &banned;
    const int V = V_;
    dispatch_apply(size_t(B), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t r) {
      fp[r] = row_topk(lg + r * size_t(V), V, k, *bp, vp + r * size_t(k), ip + r * size_t(k));
    });
    for (int r = 0; r < B; ++r)
      if (!finite[r]) return -1;
    std::vector<Dist> P(static_cast<size_t>(B));
    for (int r = 0; r < B; ++r) P[size_t(r)] = row_dist(vals.data() + size_t(r) * k, ids.data() + size_t(r) * k, k, *smp);
    auto q_of = [&](int e, int token) {
      double qt = 0;
      for (int c = 0; c < 16; ++c)
        if (int(cand[e * 16 + c]) == token) qt += q[e][c];
      return qt;
    };
    const auto [acc, tok] = accept_sampled(P, std::vector<int>(rows + 1, rows + B), q_of, *smp->rng);
    a = acc;
    next = uint32_t(tok);
  }
  const int n = a + 1;
  for (int i = 0; i < a; ++i) out[i] = rows[i + 1];
  out[a] = next;
  commit(n, pos);
  const auto t3 = clk::now();
  spec_s[0] += std::chrono::duration<double>(t1 - t0).count();
  spec_s[1] += std::chrono::duration<double>(t2 - t1).count();
  spec_s[2] += std::chrono::duration<double>(t3 - t2).count();
  ++spec_steps;
  spec_tokens += n;
  return n;
}

}  // namespace ragon::metal
