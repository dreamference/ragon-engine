#include "backend/metal/model.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
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
  NSDictionary* idx = read_json(dir + "/model.safetensors.index.json");
  NSDictionary* wm = idx[@"weight_map"];
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

Lin Model::lin(const std::string& prefix) {
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
  if (const char* mm = std::getenv("RAGON_METAL_MM_MIN")) mm_min_ = std::max(1, std::atoi(mm));
  mm_old_ = std::getenv("RAGON_METAL_MM_OLD") != nullptr;
  // Decode attention: the GQA-fused kernel by default; RAGON_METAL_ATTN_OLD=1 keeps the per-head kernel for A/B checks.
  attn_gqa_ = std::getenv("RAGON_METAL_ATTN_OLD") == nullptr;
  if (const char* v = std::getenv("RAGON_METAL_ATTN_NCH")) attn_nch_ = std::max(1, std::atoi(v));
  if (const char* v = std::getenv("RAGON_METAL_ATTN_CHUNK_MIN")) attn_chunk_min_ = std::max(4, std::atoi(v));
  // Prefill attention: the flash kernel (4 simdgroups) by default; RAGON_METAL_ATTN_FLASH=0 keeps the S/P matmuls.
  if (const char* v = std::getenv("RAGON_METAL_ATTN_FLASH")) attn_flash_ = std::atoi(v);
  if (attn_flash_ != 0 && attn_flash_ != 4 && attn_flash_ != 8 && attn_flash_ != 128 && attn_flash_ != 192)
    fail("RAGON_METAL_ATTN_FLASH must be 0, 4, 8, 128 or 192");
  if (D_ != 256 || H_ != 6 * KV_) attn_flash_ = 0;
  if (const char* v = std::getenv("RAGON_METAL_FLASH_DBG")) flash_dbg_ = uint32_t(std::atoi(v));
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
  [e dispatchThreads:MTLSizeMake(ng, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
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

void Model::gemv(const Lin& l, View x, View y, int T, int ldx, int ldy, bool add, int mode, bool reuse_x) {
  id<MTLComputeCommandEncoder> e = q_->encoder();
  if (use_mm(l, T)) {
    const uint32_t Mp = uint32_t(pad64(T));
    e = pm_sync(e, 5);
    if (l.fmt == Lin::kNvfp4 || l.fmt == Lin::kQ4) {
      const uint32_t n32 = l.N * (l.K / 32);
      [e setComputePipelineState:q_->pipe(l.fmt == Lin::kQ4 ? "dq_q4" : "dq_nvfp4")];
      bind(e, {l.w, l.sc, wdq_});
      [e setBytes:&n32 length:4 atIndex:3];
      [e dispatchThreads:MTLSizeMake(n32, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    } else {
      const uint32_t n16 = l.N * (l.K / 16);
      [e setComputePipelineState:q_->pipe("dq_fp8")];
      bind(e, {l.w, wdq_});
      [e setBytes:&n16 length:4 atIndex:2];
      [e dispatchThreads:MTLSizeMake(n16, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    e = pm_sync(e, 0);
    const bool have_x = reuse_x && !mm_old_ && xh_buf_ == (__bridge const void*)x.buf && xh_off_ == x.off &&
                        xh_k_ == l.K && xh_t_ == uint32_t(T) && xh_ld_ == uint32_t(ldx);
    if (!have_x) {
      const uint32_t xa[4] = {l.K, uint32_t(T), uint32_t(ldx), 0};
      [e setComputePipelineState:q_->pipe("x_half")];
      bind(e, {x, xh_});
      [e setBytes:xa length:16 atIndex:2];
      [e dispatchThreads:MTLSizeMake(l.K, Mp, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
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
      const uint32_t md = mode == 2 ? 2u : add ? 1u : 0u;
      struct { uint32_t M, N, K, T, ldy, mode; float scale; } ea = {Mp, l.N, l.K, uint32_t(T), uint32_t(ldy), md, l.scale};
      [e setComputePipelineState:q_->pipe(wide ? "mm_ep_64x128" : "mm_ep_64x64")];
      bind(e, {xh_, wdq_, y});
      [e setBytes:&ea length:sizeof(ea) atIndex:3];
      [e dispatchThreadgroups:MTLSizeMake(l.N / (wide ? 128 : 64), Mp / 64, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      e = pm_sync(e, 2);
      return;
    }
    MmArgs m{Mp, l.N, l.K, l.K, l.K, l.N, 0, 0, 0, 1};
    [e setComputePipelineState:q_->pipe("mm_nt_h")];
    bind(e, {xh_, wdq_, mmc_});
    bind_args(e, 3, m);
    [e dispatchThreadgroups:MTLSizeMake(l.N / 64, Mp / 64, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    e = pm_sync(e, 2);
    const uint32_t sa[4] = {l.N, uint32_t(T), uint32_t(ldy), add ? 1u : 0u};
    const float f[2] = {l.scale, 0};
    [e setComputePipelineState:q_->pipe("mm_store")];
    bind(e, {mmc_, y});
    [e setBytes:sa length:16 atIndex:2];
    [e setBytes:f length:8 atIndex:3];
    [e dispatchThreads:MTLSizeMake(l.N, T, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    e = pm_sync(e, 3);
    return;
  }
  const char* k = l.fmt == Lin::kNvfp4 ? "gemv_nvfp4"
                  : l.fmt == Lin::kQ4  ? "gemv_q4"
                  : l.fmt == Lin::kFp8 ? "gemv_fp8"
                                       : "gemv_bf16";
  [e setComputePipelineState:q_->pipe(k, T == 1 ? 1 : 8)];
  GemvArgs a{l.N, l.K, uint32_t(ldx), uint32_t(ldy), uint32_t(T), l.scale, add ? 1u : 0u};
  if (l.fmt == Lin::kNvfp4 || l.fmt == Lin::kQ4) {
    bind(e, {x, l.w, l.sc, y});
    bind_args(e, 4, a);
  } else {
    bind(e, {x, l.w, y});
    bind_args(e, 3, a);
  }
  // one dispatch: threadgroup (x, y) takes output rows 8x.. and activation rows 8y..
  [e dispatchThreadgroups:MTLSizeMake((l.N + 7) / 8, (T + 7) / 8, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

struct NormArgs {
  uint32_t n, rpt, ldt, ldh, ldo_t, ldo_h;
  float eps;
  uint32_t mode;
  float scale;
};

void Model::norm(View x, View w, View z, View out, int rows, int n, int rpt, int ldt, int ldh, int ldo_t,
                 int ldo_h, int mode, float scale) {
  id<MTLComputeCommandEncoder> e = q_->encoder();
  [e setComputePipelineState:q_->pipe("rmsnorm_rows")];
  NormArgs a{uint32_t(n), uint32_t(rpt), uint32_t(ldt), uint32_t(ldh), uint32_t(ldo_t), uint32_t(ldo_h), eps_,
             uint32_t(mode), scale};
  bind(e, {x, w.empty() ? x : w, z.empty() ? x : z, out});
  bind_args(e, 4, a);
  [e dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

struct GdnArgs {
  uint32_t T, Hk, Hv, Dk, Dv, ldc;
};

void Model::gdn(Layer& L, int T, const Dump& dump, const std::string& tag) {
  const int cw = int(L.qkv.N), vw = Hv_ * Dv_;
  gemv(L.qkv, xn_, qkv_, T, hidden_, cw, false);
  gemv(L.z, xn_, zb_, T, hidden_, vw, false, 0, true);
  gemv(L.a, xn_, ab_, T, hidden_, Hv_, false, 0, true);
  gemv(L.b, xn_, bb_, T, hidden_, Hv_, false, 0, true);
  if (dump) {
    q_->sync();
    dump(tag + "qkv", qkv_, {T, cw});
  }
  id<MTLComputeCommandEncoder> e = q_->encoder();
  [e setComputePipelineState:q_->pipe("gdn_conv")];
  bind(e, {qkv_, L.conv_state, L.conv_w});
  uint32_t ca[2] = {uint32_t(cw), uint32_t(T)};
  [e setBytes:ca length:8 atIndex:3];
  [e dispatchThreads:MTLSizeMake(cw, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [e setComputePipelineState:q_->pipe("gdn_gates")];
  bind(e, {ab_, bb_, L.A_log, L.dt_bias});
  uint32_t ga[2] = {uint32_t(Hv_), uint32_t(T)};
  [e setBytes:ga length:8 atIndex:4];
  [e dispatchThreads:MTLSizeMake(Hv_ * T, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
  if (dump) {
    q_->sync();
    dump(tag + "conv", qkv_, {T, cw});
    dump(tag + "g", ab_, {T, Hv_});
    dump(tag + "beta", bb_, {T, Hv_});
    e = q_->encoder();
  }
  if (prof_) {
    prof[1] += q_->sync();
    e = q_->encoder();
  }
  const bool four = T >= 16;  // prefill: four state rows a simdgroup
  [e setComputePipelineState:q_->pipe(four ? "gdn_recurrent4" : "gdn_recurrent")];
  bind(e, {qkv_, ab_, bb_, L.S, core_});
  GdnArgs g{uint32_t(T), uint32_t(Hk_), uint32_t(Hv_), uint32_t(Dk_), uint32_t(Dv_), uint32_t(cw)};
  bind_args(e, 5, g);
  [e dispatchThreadgroups:MTLSizeMake(Hv_ * Dv_ / (four ? 32 : 8), 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
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
  norm(core_, L.gnorm, zb_, gated_, T * Hv_, Dv_, Hv_, vw, Dv_, vw, Dv_, 1);
  gemv(L.out, gated_, h_, T, vw, hidden_, true);
}

struct RopeArgs {
  uint32_t heads, ldt, ldh, half_rot, pos0;
};
struct AttnArgs {
  uint32_t H, KV, D, ldq, qh, pos0, chunk, nchunks;
  float scale;
};

void Model::attention(Layer& L, int T, int pos0, const Dump& dump, const std::string& tag) {
  const int qw = H_ * 2 * D_, kw = KV_ * D_;
  gemv(L.q, xn_, qg_, T, hidden_, qw, false);
  gemv(L.k, xn_, kb_, T, hidden_, kw, false, 0, true);
  gemv(L.v, xn_, vb_, T, hidden_, kw, false, 0, true);
  norm(qg_, L.q_norm, View{}, qg_, T * H_, D_, H_, qw, 2 * D_, qw, 2 * D_, 0);
  norm(kb_, L.k_norm, View{}, kb_, T * KV_, D_, KV_, kw, D_, kw, D_, 0);
  id<MTLComputeCommandEncoder> e = q_->encoder();
  [e setComputePipelineState:q_->pipe("rope_rows")];
  RopeArgs rq{uint32_t(H_), uint32_t(qw), uint32_t(2 * D_), uint32_t(rot_ / 2), uint32_t(pos0)};
  bind(e, {qg_, cos_, sin_});
  bind_args(e, 3, rq);
  [e dispatchThreads:MTLSizeMake(rot_ / 2, T * H_, 1) threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
  RopeArgs rk{uint32_t(KV_), uint32_t(kw), uint32_t(D_), uint32_t(rot_ / 2), uint32_t(pos0)};
  bind(e, {kb_, cos_, sin_});
  bind_args(e, 3, rk);
  [e dispatchThreads:MTLSizeMake(rot_ / 2, T * KV_, 1) threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
  if (kv8_) {
    if (D_ != 256) fail("int8 KV needs head dim 256");
    [e setComputePipelineState:q_->pipe("kv_write8")];
    bind(e, {kb_, vb_, L.kc, L.vc, L.ks, L.vs});
    const uint32_t kva[4] = {uint32_t(KV_), uint32_t(D_), uint32_t(pos0), 0};
    [e setBytes:kva length:16 atIndex:6];
    [e dispatchThreadgroups:MTLSizeMake(KV_, T, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  } else {
    [e setComputePipelineState:q_->pipe("kv_write")];
    bind(e, {kb_, vb_, L.kc, L.vc});
    uint32_t kva[2] = {uint32_t(kw), uint32_t(pos0)};
    [e setBytes:kva length:8 atIndex:4];
    [e dispatchThreads:MTLSizeMake(kw, T, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
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
  [e dispatchThreadgroups:MTLSizeMake(nch, fused ? KV_ : H_, T) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [e setComputePipelineState:q_->pipe("attn_combine")];
  bind(e, {part_, ob_});
  bind_args(e, 2, a);
  [e dispatchThreadgroups:MTLSizeMake(H_, T, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
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
  [e dispatchThreads:MTLSizeMake(H_ * D_, T, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  gemv(L.o, ob_, h_, T, H_ * D_, hidden_, true);
}

struct PrefAttnArgs {
  uint32_t T, Tp, H, D, ldq, qh, pos0, kvp;
  float scale;
};

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
  [e dispatchThreads:MTLSizeMake(D, Tp, H) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  View kc = L.kc, vc = L.vc;
  if (kv8_) {  // expand this layer's int8 cache to bfloat once per chunk (rows past pos0 + T zero)
    [e setComputePipelineState:q_->pipe("kv_dq8")];
    bind(e, {L.kc, L.vc, L.ks, L.vs, kdq_, vdq_});
    const uint32_t da[4] = {uint32_t(KV_), D, uint32_t(pos0 + T), 0};
    [e setBytes:da length:16 atIndex:6];
    [e dispatchThreads:MTLSizeMake(kw, kvp, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    kc = kdq_;
    vc = vdq_;
  }
  if (flash) {
    // 4, 8: 64 query rows a threadgroup (4 or 8 simdgroups); 128, 192: taller tiles that read K/V fewer times
    const char* fn = attn_flash_ == 8 ? "attn_flash6_8" : attn_flash_ == 128 ? "attn_flash6_128"
                     : attn_flash_ == 192 ? "attn_flash6_192" : "attn_flash6";
    const uint32_t br = attn_flash_ >= 128 ? uint32_t(attn_flash_) : 64, nsg = attn_flash_ == 192 ? 12 : attn_flash_ == 4 ? 4 : 8;
    [e setComputePipelineState:q_->pipe(fn)];
    bind(e, {qb_, kc, vc, ob_});
    bind_args(e, 4, a);
    [e setBytes:&flash_dbg_ length:4 atIndex:5];
    [e dispatchThreadgroups:MTLSizeMake(Tp * g / br, KV_, 1) threadsPerThreadgroup:MTLSizeMake(32 * nsg, 1, 1)];
    return;
  }
  MmArgs s{Tp, kvp, D, D, kw, kvp, Tp * D, D, Tp * kvp, g};
  [e setComputePipelineState:q_->pipe("mm_nt_b")];
  bind(e, {qb_, kc, S_});
  bind_args(e, 3, s);
  [e dispatchThreadgroups:MTLSizeMake(kvp / 64, Tp / 64, H) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
  [e setComputePipelineState:q_->pipe("softmax_causal")];
  bind(e, {S_, P_});
  bind_args(e, 2, a);
  [e dispatchThreadgroups:MTLSizeMake(Tp, H, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  MmArgs o{Tp, D, kvp, kvp, kw, H * D, Tp * kvp, D, D, g};
  [e setComputePipelineState:q_->pipe("mm_nn_b")];
  bind(e, {P_, vc, ob_});
  bind_args(e, 3, o);
  [e dispatchThreadgroups:MTLSizeMake(D / 64, Tp / 64, H) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

void Model::forward(const uint32_t* ids, int T, int pos0, bool all_logits, const Dump& dump) {
  if (T > C_) fail("forward: T exceeds the chunk");
  if (pos0 + T > max_ctx_) fail("forward: context exceeds max_ctx");
  std::memcpy(ids_.host<uint32_t>(), ids, size_t(T) * 4);
  id<MTLComputeCommandEncoder> e = q_->encoder();
  [e setComputePipelineState:q_->pipe("embed_bf16")];
  bind(e, {ids_, embed_, h_});
  const uint32_t dim = uint32_t(hidden_);
  [e setBytes:&dim length:4 atIndex:3];
  [e dispatchThreads:MTLSizeMake(hidden_, T, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  if (dump) {
    q_->sync();
    dump("embed", h_, {T, hidden_});
  }
  for (size_t i = 0; i < layers_.size(); ++i) {
    Layer& L = layers_[i];
    const bool inner = dump && (i == 0 || i == 3);
    const std::string tag = "l" + std::to_string(i) + ".";
    norm(h_, L.in_norm, View{}, xn_, T, hidden_, 1, hidden_, 0, hidden_, 0, 0);
    if (inner) {
      q_->sync();
      dump(tag + "in_norm", xn_, {T, hidden_});
    }
    if (L.attn) attention(L, T, pos0, inner ? dump : nullptr, tag);
    else gdn(L, T, inner ? dump : nullptr, tag);
    if (prof_) prof[L.attn ? 0 : 1] += q_->sync();
    norm(h_, L.post_norm, View{}, xn_, T, hidden_, 1, hidden_, 0, hidden_, 0, 0);
    if (inner) {
      q_->sync();
      dump(tag + "post_norm", xn_, {T, hidden_});
    }
    gemv(L.gate, xn_, gb_, T, hidden_, ffn_, false);
    if (mm_fused(L.up, T)) {
      gemv(L.up, xn_, gb_, T, hidden_, ffn_, false, 2, true);  // gb = silu(gate) * up in the matmul's store
    } else {
      gemv(L.up, xn_, ub_, T, hidden_, ffn_, false, 0, true);
      e = q_->encoder();
      [e setComputePipelineState:q_->pipe("silu_mul")];
      bind(e, {gb_, ub_, gb_});
      const uint32_t n = uint32_t(T * ffn_);
      [e setBytes:&n length:4 atIndex:3];
      [e dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      e = pm_sync(e, 4);
    }
    gemv(L.down, gb_, h_, T, ffn_, hidden_, true);
    if (prof_) prof[2] += q_->sync();
    if (dump) {
      q_->sync();
      dump("h" + std::to_string(i), h_, {T, hidden_});
    }
  }
  const int r0 = all_logits ? 0 : T - 1, rows = all_logits ? T : 1;
  View hr = h_.at(size_t(r0) * hidden_ * 4);
  norm(hr, final_norm_, View{}, xn_, rows, hidden_, 1, hidden_, 0, hidden_, 0, 0);
  if (dump) {
    q_->sync();
    dump("final_norm", xn_, {rows, hidden_});
  }
  gemv(head_, xn_, logits_, rows, hidden_, V_, false);
  const double th = q_->sync();
  if (prof_) prof[3] += th;
}

}  // namespace ragon::metal
