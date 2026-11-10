// ragon-kvq8-tests: the FP8 KV read rule (--kv-fp8, specs LOWBIT §8). R(v) = bf16(e4m3_satfinite(v / s) * s) is
// idempotent: every one of the 254 finite E4M3 codes, read back at a scale and quantized again, gives the same code,
// and R(R(v)) == R(v) for random values across scales. The attention kernels with a KvQ8 read the keys of 32-key unit
// u as R(v) for a row at position p iff u < p / 32: each row's output is compared bit for bit against the plain kernel
// over a cache whose keys that row reads rounded were rounded beforehand (rows 1-256 at contexts across units, chunks
// and pages; a graph's device position; a tree verify whose rows straddle a unit). attention_rows_seqs and the cascade
// with a KvQ8 against attention_rows per sequence. M2: the same rows with the E4M3 page mirror (kv_seal) read for the
// tiles every row of a block takes rounded, all pages or every other one sealed. Exits 0 when everything is identical.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "core/attention_body.cuh"

namespace k = ragon::kernels;

namespace {

template <typename T>
T* upload(const std::vector<T>& h) {
  T* d = nullptr;
  cudaMalloc(&d, std::max<size_t>(h.size(), 1) * sizeof(T));
  cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}
template <typename T>
std::vector<T> download(const T* d, size_t n) {
  std::vector<T> h(n);
  cudaDeviceSynchronize();
  cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
  return h;
}

constexpr int Hq = 24, Hkv = 4, D = 256, PT = k::kKvPageTokens;
std::mt19937 rng(4041);
std::normal_distribution<float> normal(0.f, 1.f);
std::vector<float> randv(size_t n, float scale = 1.f) {
  std::vector<float> v(n);
  for (auto& x : v) x = normal(rng) * scale;
  return v;
}
std::vector<__nv_bfloat16> randb(size_t n, float scale = 1.f) {
  std::vector<__nv_bfloat16> v(n);
  for (auto& x : v) x = __float2bfloat16(normal(rng) * scale);
  return v;
}

int failures = 0;
void report(const char* what, bool ok) {
  std::printf("%-100s %s\n", what, ok ? "identical" : "DIFFERENT");
  if (!ok) ++failures;
}
bool cuda_ok() {
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) std::printf("CUDA error: %s\n", cudaGetErrorString(e));
  return e == cudaSuccess;
}

// ---- the codes ----
__global__ void codes_kernel(float s, float inv, int* bad) {
  const int c = threadIdx.x;  // 0 .. 255
  if ((c & 0x7f) == 0x7f) return;  // the two NaN codes
  const __nv_fp8x2_storage_t e = static_cast<__nv_fp8x2_storage_t>(c | (c << 8));
  const __nv_bfloat162 x = k::kv_q8_decode2(e, s);
  const __nv_fp8x2_storage_t e2 = k::kv_q8_encode2(__bfloat1622float2(x), inv);
  const __nv_bfloat162 x2 = k::kv_q8_decode2(e2, s);
  // Codes that decode to the same value (only +0 and -0 times a scale) may come back as either.
  if (e2 != e && *reinterpret_cast<const unsigned*>(&x2) != *reinterpret_cast<const unsigned*>(&x)) atomicAdd(bad, 1);
}
__global__ void idem_kernel(const __nv_bfloat162* v, int n, float s, float inv, int* bad) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const __nv_bfloat162 r1 = k::kv_q8_round2(v[i], s, inv), r2 = k::kv_q8_round2(r1, s, inv);
  if (*reinterpret_cast<const unsigned*>(&r1) != *reinterpret_cast<const unsigned*>(&r2)) atomicAdd(bad, 1);
}

// ---- a cache ----
// One layer's KV, flat ([Hkv][max_ctx][D]) or in pages of a pool (table: element offsets, kv_off's addressing).
struct Cache {
  int max_ctx = 0;
  bool paged = false;
  size_t hs = 0, elems = 0;
  __nv_bfloat16 *k = nullptr, *v = nullptr;        // the bases the kernels take
  __nv_bfloat16 *k0 = nullptr, *v0 = nullptr;      // pristine copies
  uint64_t* table = nullptr;
};
// Rounds the K and V of every head at the given positions (kv_off's addressing).
__global__ void round_positions(__nv_bfloat16* kc, __nv_bfloat16* vc, const uint64_t* pages, const int* pos, size_t hs,
                                float ks, float kinv, float vs, float vinv) {
  const int p = pos[blockIdx.x], h = blockIdx.y;
  const size_t off = k::kv_off(pages, h, p, hs, D);
  for (int d = 2 * threadIdx.x; d < D; d += 2 * blockDim.x) {
    auto* kp = reinterpret_cast<__nv_bfloat162*>(kc + off + d);
    auto* vp = reinterpret_cast<__nv_bfloat162*>(vc + off + d);
    *kp = k::kv_q8_round2(*kp, ks, kinv);
    *vp = k::kv_q8_round2(*vp, vs, vinv);
  }
}
void layout(const Cache& c) { k::set_kv_layout(c.hs, D); }
void restore(const Cache& c) {
  cudaMemcpy(c.k, c.k0, c.elems * 2, cudaMemcpyDeviceToDevice);
  cudaMemcpy(c.v, c.v0, c.elems * 2, cudaMemcpyDeviceToDevice);
}
void round_at(const Cache& c, const std::vector<int>& pos, const k::KvQ8& q) {
  if (pos.empty()) return;
  int* dp = upload(pos);
  round_positions<<<dim3(pos.size(), Hkv), 64>>>(c.k, c.v, c.table, dp, c.hs, q.ks, q.kinv, q.vs, q.vinv);
  cudaDeviceSynchronize();
  cudaFree(dp);
}
// Flat: the KV of max_ctx positions. Paged: the same in a pool of shuffled pages; `share` (optional) maps its first
// `shared` pages to the pages of another paged cache (the cascade's runs).
Cache make_cache(int max_ctx, bool paged, k::KvQ8* scales) {
  Cache c;
  c.max_ctx = max_ctx;
  c.paged = paged;
  const int np = (max_ctx + PT - 1) / PT;
  c.elems = paged ? size_t(np + 1) * Hkv * PT * D : size_t(Hkv) * max_ctx * D;
  c.hs = paged ? size_t(np + 1) * PT * D : size_t(max_ctx) * D;
  const std::vector<__nv_bfloat16> hk = randb(c.elems, 1.5f), hv = randb(c.elems, 0.8f);
  if (scales) {  // the layer's scales from its maxima, as bench/quant/kv_calib.py sets them
    float mk = 0.f, mv = 0.f;
    for (size_t i = 0; i < c.elems; ++i) {
      mk = std::max(mk, std::fabs(__bfloat162float(hk[i])));
      mv = std::max(mv, std::fabs(__bfloat162float(hv[i])));
    }
    const float ks = mk / 448.f, vs = mv / 448.f;
    *scales = k::KvQ8{ks, vs, 1.f / ks, 1.f / vs};
  }
  c.k = upload(hk);
  c.v = upload(hv);
  c.k0 = upload(hk);
  c.v0 = upload(hv);
  if (paged) {  // head-major pool ([K or V][head][page][PT][D]); the table in units of one head's run, page 0 unused
    std::vector<int> ids(np);
    for (int i = 0; i < np; ++i) ids[i] = i + 1;
    std::shuffle(ids.begin(), ids.end(), rng);
    std::vector<uint64_t> t(np);
    for (int i = 0; i < np; ++i) t[i] = uint64_t(ids[i]) * PT * D;
    c.table = upload(t);
  }
  return c;
}

// The rows' output under q8, and the reference: per row, the plain kernel over the cache with the keys that row reads
// rounded (committed keys below its unit; with a tree, its path's rows whose positions are below it) rounded first.
bool rows_case(const Cache& c, const float* dq, int P, int M, const k::KvQ8& q8, float* scratch, float* out,
               const k::TreeDesc* htd = nullptr, const k::TreeDesc* dtd = nullptr, bool graph = false) {
  layout(c);
  const size_t rowf = size_t(Hq) * D, n = size_t(M) * rowf;
  int* dpos = graph ? upload(std::vector<int>{P}) : nullptr;
  const k::DevPos pos = graph ? k::DevPos(-1, dpos, 0) : k::DevPos(P);
  const int fixed = graph ? c.max_ctx : 0;
  restore(c);
  k::attention_rows(dq, c.k, c.v, pos, M, Hq, Hkv, D, scratch, out, nullptr, fixed, dtd, c.table, q8);
  const std::vector<float> got = download(out, n);
  bool ok = cuda_ok();
  // Rows grouped by their unit: one reference launch per unit.
  std::vector<int> units;
  for (int m = 0; m < M; ++m) units.push_back((P + (htd ? htd->depth[m] : m)) & ~31);
  std::vector<int> uniq = units;
  std::sort(uniq.begin(), uniq.end());
  uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
  for (int U : uniq) {
    restore(c);
    std::vector<int> pos_r;
    // Every key below the row's unit, the pass's own rows' included (a plain pass's row m sits at position P + m); a
    // tree's slots past P are not positions, so there the committed keys below the unit, then the path's rows below.
    for (int p = 0; p < (htd ? std::min(U, P) : U); ++p) pos_r.push_back(p);
    if (htd)  // the tree's rows sit at slots P + m; a row at depth d is at position P + d
      for (int m = 0; m < M; ++m)
        if (P + htd->depth[m] < U) pos_r.push_back(P + m);
    round_at(c, pos_r, q8);
    cudaMemset(out, 0, n * 4);
    k::attention_rows(dq, c.k, c.v, pos, M, Hq, Hkv, D, scratch, out, nullptr, fixed, dtd, c.table);
    const std::vector<float> ref = download(out, n);
    ok &= cuda_ok();
    for (int m = 0; m < M; ++m)
      if (units[m] == U && std::memcmp(&got[m * rowf], &ref[m * rowf], rowf * 4) != 0) {
        if (ok) std::printf("  P %d M %d: row %d (unit %d) differs\n", P, M, m, U);
        ok = false;
      }
  }
  restore(c);
  if (dpos) cudaFree(dpos);
  return ok;
}

}  // namespace

int main() {
  k::prepare_kernels();
  int* bad = upload(std::vector<int>{0});
  // ---- the codes and R's idempotence ----
  {
    bool ok = true;
    for (float s : {1e-3f, 0.0123f, 0.37f, 1.f, 3.3f, 17.5f}) {
      cudaMemset(bad, 0, 4);
      codes_kernel<<<1, 256>>>(s, 1.f / s, bad);
      ok &= download(bad, 1)[0] == 0 && cuda_ok();
    }
    report("every finite E4M3 code read back at six scales and quantized again: the same code", ok);
  }
  {
    bool ok = true;
    const int n = 1 << 20;
    for (float s : {1e-3f, 0.0123f, 0.37f, 1.f, 3.3f}) {
      for (float spread : {0.01f, 1.f, 100.f}) {
        std::vector<__nv_bfloat16> h = randb(2 * n, spread * s * 64.f);
        auto* dv = upload(h);
        cudaMemset(bad, 0, 4);
        idem_kernel<<<(n + 255) / 256, 256>>>(reinterpret_cast<const __nv_bfloat162*>(dv), n, s, 1.f / s, bad);
        ok &= download(bad, 1)[0] == 0 && cuda_ok();
        cudaFree(dv);
      }
    }
    report("R(R(v)) == R(v) for 2M random values at five scales and three spreads (saturation included)", ok);
  }

  const int kMaxCtx = 8192 + 64;
  k::KvQ8 q8;
  Cache flat = make_cache(kMaxCtx, false, &q8);
  Cache paged = make_cache(kMaxCtx, true, nullptr);
  float* dq = upload(randv(size_t(256) * Hq * D, 2.f));
  float *scratch, *out;
  cudaMalloc(&scratch, k::attention_rows_scratch_floats(256, Hq, D, kMaxCtx) * sizeof(float));
  cudaMalloc(&out, size_t(256) * Hq * D * sizeof(float));
  k::set_attention_bulk(false);

  for (const Cache* c : {&flat, &paged}) {
    for (int pf : {0, 2}) {
      k::set_attention_prefetch(pf);
      bool ok = true;
      for (int ctx : {1, 31, 32, 33, 63, 64, 65, 100, 4095, 4096, 4097, 4127, 8192})
        for (int M : {1, 5, 16, 32, 40})
          if (ctx >= M) ok &= rows_case(*c, dq, ctx - M, M, q8, scratch, out);
      char what[160];
      std::snprintf(what, sizeof what, "attention_rows with FP8 reads, %s, prefetch %d, 13 contexts x rows 1-40",
                    c->paged ? "paged" : "flat", pf);
      report(what, ok);
    }
  }
  k::set_attention_prefetch(2);
  {
    bool ok = true;
    for (int P : {0, 3900, 4000, 7900}) ok &= rows_case(paged, dq, P, 256, q8, scratch, out);
    report("attention_rows with FP8 reads over a 256-row prefill slice (units and a key range crossed)", ok);
  }
  {
    bool ok = true;
    for (int P : {60, 4090, 7001}) ok &= rows_case(paged, dq, P, 16, q8, scratch, out, nullptr, nullptr, true);
    report("attention_rows with FP8 reads as a graph launches it (device position, fixed ranges)", ok);
  }
  {  // rows 0-4 one variant (depths 0-4), 5-7 another (depths 2-4)
    k::TreeDesc td;
    td.n = 8;
    const int depth[8] = {0, 1, 2, 3, 4, 2, 3, 4}, var[8] = {0, 0, 0, 0, 0, 1, 1, 1};
    for (int m = 0; m < 8; ++m) td.depth[m] = depth[m], td.var[m] = var[m];
    td.nvar = 2;
    const int p0[5] = {0, 1, 2, 3, 4}, p1[5] = {0, 1, 5, 6, 7};
    for (int j = 0; j < 5; ++j) td.path[0][j] = p0[j], td.path[1][j] = p1[j];
    k::TreeDesc* dtd = upload(std::vector<k::TreeDesc>{td});
    bool ok = true;
    for (const Cache* c : {&flat, &paged})
      for (int P : {28, 30, 32, 58, 62, 4093}) ok &= rows_case(*c, dq, P, 8, q8, scratch, out, &td, dtd);
    report("attention_rows with FP8 reads, a tree verify whose rows straddle a unit", ok);
    cudaFree(dtd);
  }
  // ---- M2: the E4M3 page mirror (kv_seal) read for the tiles every row takes as R(v), against the same reference ----
  {
    layout(paged);
    const int np = (kMaxCtx + PT - 1) / PT;
    // kv_seal's pool: one layer, [K, V][head][page][PT][D]; the cache's K and V copied into it.
    __nv_bfloat16* pool = nullptr;
    cudaMalloc(&pool, paged.elems * 2 * 2);
    cudaMemcpy(pool, paged.k0, paged.elems * 2, cudaMemcpyDeviceToDevice);
    cudaMemcpy(pool + paged.elems, paged.v0, paged.elems * 2, cudaMemcpyDeviceToDevice);
    uint8_t* mirror = nullptr;
    cudaMalloc(&mirror, paged.elems * 2);
    cudaMemset(mirror, 0xff, paged.elems * 2);  // NaN codes where nothing was sealed
    uint8_t* flags = upload(std::vector<uint8_t>(np + 1, 0));
    k::KvQ8 m8 = q8;
    m8.k8 = mirror;
    m8.v8 = mirror + paged.elems;
    m8.sealed = flags;
    m8.unit = size_t(PT) * D;
    const float kinv[1] = {q8.kinv}, vinv[1] = {q8.vinv};
    std::vector<int> all, half;
    for (int i = 1; i <= np; ++i) {
      all.push_back(i);
      if (i % 2) half.push_back(i);
    }
    for (int pass = 0; pass < 2; ++pass) {
      const std::vector<int>& ids = pass ? all : half;
      k::kv_seal(pool, mirror, flags, ids.data(), static_cast<int>(ids.size()), 0, 1, Hkv, D, m8.unit, kinv, vinv, 0);
      cudaDeviceSynchronize();
      for (int pf : {0, 2}) {
        k::set_attention_prefetch(pf);
        bool ok = cuda_ok();
        for (int ctx : {1, 33, 64, 65, 100, 129, 4095, 4096, 4097, 4127, 8192})
          for (int M : {1, 5, 16, 40})
            if (ctx >= M) ok &= rows_case(paged, dq, ctx - M, M, m8, scratch, out);
        char what[160];
        std::snprintf(what, sizeof what, "attention_rows with the E4M3 mirror, %s pages sealed, prefetch %d", pass ? "all" : "odd", pf);
        report(what, ok);
      }
    }
    k::set_attention_prefetch(2);
    {
      bool ok = true;
      for (int P : {0, 3900, 4000, 7900}) ok &= rows_case(paged, dq, P, 256, m8, scratch, out);
      for (int P : {60, 4090, 7001}) ok &= rows_case(paged, dq, P, 16, m8, scratch, out, nullptr, nullptr, true);
      report("attention_rows with the E4M3 mirror: a 256-row prefill slice, and as a graph launches it", ok);
    }
    {  // the tree of the test above, over sealed pages
      k::TreeDesc td;
      td.n = 8;
      const int depth[8] = {0, 1, 2, 3, 4, 2, 3, 4}, var[8] = {0, 0, 0, 0, 0, 1, 1, 1};
      for (int m = 0; m < 8; ++m) td.depth[m] = depth[m], td.var[m] = var[m];
      td.nvar = 2;
      const int p0[5] = {0, 1, 2, 3, 4}, p1[5] = {0, 1, 5, 6, 7};
      for (int j = 0; j < 5; ++j) td.path[0][j] = p0[j], td.path[1][j] = p1[j];
      k::TreeDesc* dtd = upload(std::vector<k::TreeDesc>{td});
      bool ok = true;
      for (int P : {62, 126, 4093, 8000}) ok &= rows_case(paged, dq, P, 8, m8, scratch, out, &td, dtd);
      report("attention_rows with the E4M3 mirror, a tree verify", ok);
      cudaFree(dtd);
    }
    {  // kv_unseal clears exactly the flags named
      k::kv_unseal(flags, half.data(), static_cast<int>(half.size()), 0);
      const std::vector<uint8_t> f = download(flags, np + 1);
      bool ok = cuda_ok() && f[0] == 0;
      for (int i = 1; i <= np; ++i) ok &= f[i] == (i % 2 ? 0 : 1);
      report("kv_unseal clears the flags of the pages named and no others", ok);
    }
    cudaFree(pool);
    cudaFree(mirror);
    cudaFree(flags);
  }
  // ---- several sequences: attention_rows_seqs and the cascade against attention_rows per sequence ----
  {
    // Three sequences over one pool: 0 and 1 share their first 40 pages (a run), 2 has its own.
    const int np = 72, shared = 40;
    const size_t hs = size_t(3 * np + 1) * PT * D, elems = size_t(Hkv) * hs;
    __nv_bfloat16* kp = upload(randb(elems, 1.5f));
    __nv_bfloat16* vp = upload(randb(elems, 0.8f));
    std::vector<std::vector<uint64_t>> t(3);
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < np; ++j) t[i].push_back(uint64_t(i == 1 && j < shared ? 1 + j : 1 + i * np + j) * PT * D);
    k::set_kv_layout(hs, D);
    const int pos[3] = {4100, 3333, 61}, rows[3] = {6, 3, 8};
    k::SeqBatch sb;
    sb.n = 3;
    for (int i = 0; i < 3; ++i) sb.row0[i + 1] = sb.row0[i] + rows[i], sb.pos[i] = pos[i];
    k::SeqPtrs<const __nv_bfloat16> kc, vc;
    k::SeqPtrs<const uint64_t> pg;
    std::vector<uint64_t*> dt;
    for (int i = 0; i < 3; ++i) {
      kc.p[i] = kp, vc.p[i] = vp;
      dt.push_back(upload(t[i]));
      pg.p[i] = dt.back();
    }
    const int M = sb.row0[3];
    const size_t rowf = size_t(Hq) * D;
    std::vector<float> per(size_t(M) * rowf);
    bool ok = true;
    for (int i = 0; i < 3; ++i) {
      k::attention_rows(dq + sb.row0[i] * rowf, kp, vp, pos[i], rows[i], Hq, Hkv, D, scratch, out, nullptr, 0, nullptr,
                        dt[i], q8);
      const std::vector<float> o = download(out, rows[i] * rowf);
      std::memcpy(&per[sb.row0[i] * rowf], o.data(), o.size() * 4);
    }
    k::attention_rows_seqs(dq, kc, vc, sb, Hq, Hkv, D, scratch, out, nullptr, 0, pg, q8);
    std::vector<float> got = download(out, M * rowf);
    ok &= cuda_ok() && std::memcmp(got.data(), per.data(), got.size() * 4) == 0;
    report("attention_rows_seqs with FP8 reads, three sequences, against attention_rows per sequence", ok);
    // The cascade: sequences 0 and 1 a group whose keys below the cut (on the tile grid, under both positions) are shared.
    k::SeqGroups g;
    g.n = 1;
    g.seq0[0] = 0, g.seq1[0] = 2, g.cut[0] = 40 * PT;  // 2560: below 3333 and 4100, the end of the 40 shared pages
    cudaMemset(out, 0, M * rowf * 4);
    k::attention_rows_cascade(dq, kc, vc, sb, g, Hq, Hkv, D, scratch, out, nullptr, pg, q8);
    got = download(out, M * rowf);
    const bool okc = cuda_ok() && std::memcmp(got.data(), per.data(), got.size() * 4) == 0;
    report("attention_rows_cascade with FP8 reads (a group of two over shared pages) against attention_rows", okc);
    for (uint64_t* p : dt) cudaFree(p);
    cudaFree(kp);
    cudaFree(vp);
  }
  // ---- a KvQ8 that is off leaves the plain path: the default instantiation, the same bits ----
  {
    layout(paged);
    const size_t n = size_t(40) * Hq * D;
    k::attention_rows(dq, paged.k, paged.v, 5000, 40, Hq, Hkv, D, scratch, out, nullptr, 0, nullptr, paged.table);
    const std::vector<float> a = download(out, n);
    k::attention_rows(dq, paged.k, paged.v, 5000, 40, Hq, Hkv, D, scratch, out, nullptr, 0, nullptr, paged.table,
                      k::KvQ8{});
    const std::vector<float> b = download(out, n);
    report("a KvQ8 with no scale: the BF16 path", cuda_ok() && std::memcmp(a.data(), b.data(), n * 4) == 0);
  }
  std::printf("%s\n", failures ? "FAILED" : "all identical");
  return failures ? 1 : 0;
}
