// ragon-lora-kernel-tests: the LoRA kernels (src/core/lora_kernels.cuh) against a host reference in double, and the
// properties the engine relies on bit for bit: a row's delta is the same alone or in a batch of any size beside rows
// on other adapters or none (row invariance, so a verify's rows equal the decode's); a row on no adapter is never
// written; a graph's device slots give what the eager slots give; a site the adapter does not touch is skipped.
// Exits 0 when everything holds.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "core/kernels.cuh"
#include "core/lora_kernels.cuh"

namespace k = ragon::kernels;

namespace {

int failures = 0;

void report(const char* what, bool ok) {
  std::printf("%-100s %s\n", what, ok ? "ok" : "FAILED");
  if (!ok) ++failures;
}

template <typename T>
T* upload(const std::vector<T>& v) {
  T* d = nullptr;
  cudaMalloc(&d, v.size() * sizeof(T));
  cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}

template <typename T>
std::vector<T> download(const T* d, size_t n) {
  std::vector<T> v(n);
  cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
  return v;
}

uint16_t bf16_bits(float f) {
  const __nv_bfloat16 b = __float2bfloat16(f);
  uint16_t u;
  std::memcpy(&u, &b, 2);
  return u;
}

float bf16_value(uint16_t u) {
  __nv_bfloat16 b;
  std::memcpy(&b, &u, 2);
  return __bfloat162float(b);
}

constexpr int K = 512;
const int kN[3] = {96, 32, 32};  // the site's three outputs (q_proj's, k_proj's and v_proj's, scaled down)
constexpr int kMaxN = 96;

// One slot's site on the host: its A, B (BF16 bits) and segments, and its device copy.
struct HostSite {
  int rank = 0, blocks = 0;
  float scale = 0.f;
  std::vector<uint16_t> a, b;
  std::vector<k::LoraSegDev> segs;
};

HostSite make_site(std::mt19937& rng, int rank, const std::vector<std::pair<int, int>>& outs_blocks, float scale) {
  std::normal_distribution<float> nd(0.f, 0.05f);
  HostSite s;
  s.rank = rank;
  s.scale = scale;
  for (const auto& [out, block] : outs_blocks) s.blocks = std::max(s.blocks, block + 1);
  s.a.resize(size_t(s.blocks) * rank * K);
  for (auto& v : s.a) v = bf16_bits(nd(rng));
  for (const auto& [out, block] : outs_blocks) {
    k::LoraSegDev g;
    g.out = out;
    g.n0 = 0;
    g.n1 = kN[out];
    g.block = block;
    g.b_off = static_cast<long long>(s.b.size());
    for (int i = 0; i < kN[out] * rank; ++i) s.b.push_back(bf16_bits(nd(rng)));
    s.segs.push_back(g);
  }
  return s;
}

k::LoraSiteDev to_device(const HostSite& s, std::vector<void*>& owned) {
  k::LoraSiteDev d;
  if (s.rank == 0) return d;
  __nv_bfloat16* a = reinterpret_cast<__nv_bfloat16*>(upload(s.a));
  __nv_bfloat16* b = reinterpret_cast<__nv_bfloat16*>(upload(s.b));
  owned.push_back(a);
  owned.push_back(b);
  d.a = a;
  d.b = b;
  d.K = K;
  d.rank = s.rank;
  d.blocks = s.blocks;
  d.nsegs = static_cast<int>(s.segs.size());
  d.scale = s.scale;
  for (int i = 0; i < d.nsegs; ++i) d.segs[i] = s.segs[i];
  return d;
}

// The host reference: y[r] += scale * B (A x[r]), in double.
void reference(const HostSite& s, const float* x, std::vector<double>* y) {
  std::vector<double> t(size_t(s.blocks) * s.rank);
  for (size_t j = 0; j < t.size(); ++j) {
    double v = 0;
    for (int kk = 0; kk < K; ++kk) v += double(x[kk]) * bf16_value(s.a[j * K + kk]);
    t[j] = v;
  }
  for (const k::LoraSegDev& g : s.segs)
    for (int n = g.n0; n < g.n1; ++n) {
      double v = 0;
      for (int j = 0; j < s.rank; ++j)
        v += double(bf16_value(s.b[g.b_off + size_t(n - g.n0) * s.rank + j])) * t[size_t(g.block) * s.rank + j];
      y[g.out][n] += s.scale * v;
    }
}

// The device state of one run: x, t and the three outputs over M rows, with a slot table of `sites`.
struct Run {
  int M;
  float *x, *t, *y[3];
  k::LoraSiteDev* table;
  int slots;
  Run(const std::vector<float>& hx, const std::vector<float> hy[3], int M_, const std::vector<k::LoraSiteDev>& sites)
      : M(M_), slots(static_cast<int>(sites.size())) {
    x = upload(hx);
    cudaMalloc(&t, size_t(M) * k::kLoraRankRows * sizeof(float));
    cudaMemset(t, 0, size_t(M) * k::kLoraRankRows * sizeof(float));
    for (int o = 0; o < 3; ++o) y[o] = upload(hy[o]);
    table = upload(sites);
  }
  ~Run() {
    cudaFree(x);
    cudaFree(t);
    for (float* p : y) cudaFree(p);
    cudaFree(table);
  }
  void go(const k::LoraRows& rows, bool skip_q = false) {
    k::LoraOuts outs;
    for (int o = 0; o < 3; ++o) {
      outs.y[o] = skip_q && o == 0 ? nullptr : y[o];
      outs.ld[o] = kN[o];
    }
    k::lora_shrink(x, K, M, rows, table, 1, k::kLoraRankRows, t, nullptr);
    k::lora_expand(t, M, rows, table, 1, outs, nullptr);
  }
  std::vector<float> out(int o) {
    cudaDeviceSynchronize();
    return download(y[o], size_t(M) * kN[o]);
  }
};

// A site for the prefill GEMM: segments {out, n0, n1, block} of any place in the outputs.
HostSite make_fold_site(std::mt19937& rng, int rank, const std::vector<std::array<int, 4>>& segs, float scale) {
  std::normal_distribution<float> nd(0.f, 0.05f);
  HostSite s;
  s.rank = rank;
  s.scale = scale;
  for (const auto& g : segs) s.blocks = std::max(s.blocks, g[3] + 1);
  s.a.resize(size_t(s.blocks) * rank * K);
  for (auto& v : s.a) v = bf16_bits(nd(rng));
  for (const auto& g : segs) {
    k::LoraSegDev d;
    d.out = g[0];
    d.n0 = g[1];
    d.n1 = g[2];
    d.block = g[3];
    d.b_off = static_cast<long long>(s.b.size());
    for (int i = 0; i < (g[2] - g[1]) * rank; ++i) s.b.push_back(bf16_bits(nd(rng)));
    s.segs.push_back(d);
  }
  return s;
}

// A tiled FP8 weight [N][K] of random values, none of them a NaN.
uint8_t* fold_weight(std::mt19937& rng, int N) {
  std::vector<uint8_t> h(size_t(N) * K);
  for (auto& v : h) {
    v = static_cast<uint8_t>(rng());
    if ((v & 0x7f) >= 0x70) v &= 0x8f;  // no exponent of 14 or 15: the products stay small
  }
  uint8_t* d = upload(h);
  uint8_t* tiled = nullptr;
  cudaMalloc(&tiled, k::tiled_bytes_fp8(N, K));
  k::retile_fp8(d, tiled, N, K, nullptr);
  cudaDeviceSynchronize();
  cudaFree(d);
  return tiled;
}

// Both outputs of one site over M rows: the prefill GEMM with the site's parts, or the base's alone with no site.
void fold_go(const std::vector<float>& hx, int M, const k::LoraSiteDev* site, uint8_t* const (&w)[2], const int (&N)[2],
             bool accumulate, const std::vector<float> (&y0)[2], std::vector<float> (&out)[2]) {
  constexpr float kAlpha = 0.01f;
  float* x = upload(hx);
  uint8_t* xq = nullptr;
  cudaMalloc(&xq, size_t(M) * K);
  k::quant_act_fp8(x, M, K, 1.f, xq, nullptr);
  float* t = nullptr;
  cudaMalloc(&t, size_t(M) * k::kLoraRankRows * sizeof(float));
  cudaMemset(t, 0, size_t(M) * k::kLoraRankRows * sizeof(float));
  k::LoraGemm parts[k::kLoraOuts];
  k::LoraSiteDev* table = nullptr;
  if (site) {
    table = upload(std::vector<k::LoraSiteDev>{*site});
    k::LoraRows rows;
    rows.n = 1;
    rows.row0[1] = M;
    rows.slot[0] = 0;
    k::lora_shrink_prefill(x, K, M, rows, table, 1, t, nullptr);
    k::lora_gemm_parts(*site, t, parts);
  }
  for (int o = 0; o < 2; ++o) {
    float* y = upload(y0[o]);
    if (site) k::prefill_gemm_fp8_lora(xq, M, w[o], kAlpha, y, N[o], N[o], K, accumulate, parts[o], nullptr);
    else k::prefill_gemm_fp8(xq, M, w[o], kAlpha, y, N[o], N[o], K, accumulate, nullptr);
    cudaDeviceSynchronize();
    out[o] = download(y, y0[o].size());
    cudaFree(y);
  }
  cudaFree(x);
  cudaFree(xq);
  cudaFree(t);
  if (table) cudaFree(table);
}

}  // namespace

int main() {
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.f, 1.f);
  // Slot 0: q, k and v at rank 16 (three blocks); slot 1: q and v at rank 8 (two blocks, rsLoRA's scale); slot 2: the
  // adapter does not touch this site.
  std::vector<HostSite> host = {make_site(rng, 16, {{0, 0}, {1, 1}, {2, 2}}, 2.f),
                                make_site(rng, 8, {{0, 0}, {2, 1}}, 16.f / std::sqrt(8.f)), HostSite{}};
  std::vector<void*> owned;
  std::vector<k::LoraSiteDev> sites;
  for (const HostSite& s : host) sites.push_back(to_device(s, owned));

  // Four sequences of 5, 1, 9 and 3 rows on slots 1, -1, 0 and 2: 18 rows, tiles that straddle sequences.
  const int seq_rows[4] = {5, 1, 9, 3}, seq_slot[4] = {1, -1, 0, 2};
  const int M = 18;
  std::vector<float> hx(size_t(M) * K);
  for (auto& v : hx) v = nd(rng);
  std::vector<float> hy[3];
  for (int o = 0; o < 3; ++o) {
    hy[o].resize(size_t(M) * kN[o]);
    for (auto& v : hy[o]) v = nd(rng);
  }
  k::LoraRows rows;
  rows.n = 4;
  for (int i = 0; i < 4; ++i) {
    rows.row0[i + 1] = rows.row0[i] + seq_rows[i];
    rows.slot[i] = seq_slot[i];
  }
  std::vector<float> batch[3];
  {
    Run r(hx, hy, M, sites);
    r.go(rows);
    for (int o = 0; o < 3; ++o) batch[o] = r.out(o);
  }

  // Against the reference; rows on no adapter, and on slot 2's untouched site, bit for bit as they were.
  {
    double worst = 0;
    bool untouched = true;
    for (int seq = 0; seq < 4; ++seq)
      for (int r = rows.row0[seq]; r < rows.row0[seq + 1]; ++r) {
        std::vector<double> want[3];
        for (int o = 0; o < 3; ++o) want[o].assign(hy[o].begin() + size_t(r) * kN[o], hy[o].begin() + size_t(r + 1) * kN[o]);
        const int s = seq_slot[seq];
        if (s >= 0 && host[s].rank > 0) reference(host[s], hx.data() + size_t(r) * K, want);
        for (int o = 0; o < 3; ++o)
          for (int n = 0; n < kN[o]; ++n) {
            const float got = batch[o][size_t(r) * kN[o] + n];
            if (s < 0 || host[s].rank == 0) untouched &= got == hy[o][size_t(r) * kN[o] + n];
            worst = std::max(worst, std::abs(got - want[o][n]) / (1.0 + std::abs(want[o][n])));
          }
      }
    std::printf("worst relative error against the double reference: %.3g\n", worst);
    report("shrink and expand over four sequences on slots 1, -1, 0, 2 match the reference (1e-5)", worst < 1e-5);
    report("the base sequence's rows and the untouched site's rows are not written", untouched);
  }

  // Row invariance: each row alone (M = 1, its own slot) gives the batch's bits.
  {
    bool same = true;
    for (int seq = 0; seq < 4; ++seq)
      for (int r = rows.row0[seq]; r < rows.row0[seq + 1]; ++r) {
        std::vector<float> x1(hx.begin() + size_t(r) * K, hx.begin() + size_t(r + 1) * K), y1[3];
        for (int o = 0; o < 3; ++o) y1[o].assign(hy[o].begin() + size_t(r) * kN[o], hy[o].begin() + size_t(r + 1) * kN[o]);
        Run one(x1, y1, 1, sites);
        k::LoraRows r1;
        r1.n = 1;
        r1.row0[1] = 1;
        r1.slot[0] = seq_slot[seq];
        one.go(r1);
        for (int o = 0; o < 3; ++o)
          same &= std::memcmp(one.out(o).data(), batch[o].data() + size_t(r) * kN[o], kN[o] * sizeof(float)) == 0;
      }
    report("every row alone (M = 1) is bit for bit its row of the 18-row batch", same);
  }

  // The prefill path: the shrink and the GEMM that adds the expand in its epilogue. Base GEMM + the reference, to BF16's
  // precision (x and t are rounded on the way in); ranks 16 and 8; two blocks in one output, a gap, a second output. Both
  // block shapes of the GEMM: 83 rows take 128 x 128, 6100 rows 128 x 256.
  for (const int big : {83, 6100}) {
    std::mt19937 frng(77);
    std::normal_distribution<float> nd(0.f, 0.05f);
    const int fN[2] = {512, 128};
    std::vector<float> bx(size_t(big) * K);
    for (auto& v : bx) v = nd(frng);
    uint8_t* w[2];
    for (int o = 0; o < 2; ++o) w[o] = fold_weight(frng, fN[o]);
    std::vector<float> y0[2];
    for (int o = 0; o < 2; ++o) {
      y0[o].resize(size_t(big) * fN[o]);
      for (auto& v : y0[o]) v = nd(frng);
    }
    for (const int rank : {16, 8}) {
      const HostSite fs = make_fold_site(frng, rank, {{0, 0, 256, 0}, {0, 256, 496, 1}, {1, 0, 128, 2}}, 0.75f);
      const k::LoraSiteDev fd = to_device(fs, owned);
      for (const bool accumulate : {false, true}) {
        std::vector<float> base[2], got[2];
        fold_go(bx, big, nullptr, w, fN, accumulate, y0, base);
        fold_go(bx, big, &fd, w, fN, accumulate, y0, got);
        double worst = 0;
        bool gap_same = true;
        for (int r = 0; r < big; r += big > 100 ? 37 : 1) {
          std::vector<double> want[2];
          for (int o = 0; o < 2; ++o)
            want[o].assign(base[o].begin() + size_t(r) * fN[o], base[o].begin() + size_t(r + 1) * fN[o]);
          reference(fs, bx.data() + size_t(r) * K, want);
          for (int o = 0; o < 2; ++o)
            for (int n = 0; n < fN[o]; ++n) {
              const float v = got[o][size_t(r) * fN[o] + n];
              worst = std::max(worst, std::abs(v - want[o][n]) / (1.0 + std::abs(want[o][n])));
              if (o == 0 && n >= 496) gap_same &= v == base[o][size_t(r) * fN[o] + n];
            }
        }
        std::printf("fold, %d rows, rank %d, accumulate %d: worst error against base + reference: %.3g\n", big, rank,
                    accumulate, worst);
        report("the GEMM with the adapter's expand matches base + reference to BF16's precision (1e-2)", worst < 1e-2);
        report("the GEMM with the adapter's expand leaves a column outside every segment as the base's", gap_same);
      }
      // A prompt's state must not depend on how it was chunked: a row of a chunk has the bits of the row alone.
      {
        std::vector<float> got[2];
        fold_go(bx, big, &fd, w, fN, true, y0, got);
        bool same = true;
        for (int r = 0; r < big; r += big > 100 ? 211 : 1) {
          const std::vector<float> x1(bx.begin() + size_t(r) * K, bx.begin() + size_t(r + 1) * K);
          std::vector<float> y1[2], one[2];
          for (int o = 0; o < 2; ++o) y1[o].assign(y0[o].begin() + size_t(r) * fN[o], y0[o].begin() + size_t(r + 1) * fN[o]);
          fold_go(x1, 1, &fd, w, fN, true, y1, one);
          for (int o = 0; o < 2; ++o)
            same &= std::memcmp(one[o].data(), got[o].data() + size_t(r) * fN[o], fN[o] * sizeof(float)) == 0;
        }
        report("the GEMM with the adapter's expand gives a row of a chunk the bits of the row alone", same);
      }
      // Every B zero: the base's values, whatever the weights.
      {
        HostSite z = fs;
        std::fill(z.b.begin(), z.b.end(), uint16_t(0));
        const k::LoraSiteDev zd = to_device(z, owned);
        std::vector<float> base[2], got[2];
        fold_go(bx, big, nullptr, w, fN, true, y0, base);
        fold_go(bx, big, &zd, w, fN, true, y0, got);
        report("the GEMM with every B zero gives the base's values", got[0] == base[0] && got[1] == base[1]);
      }
    }
    // An output with no segment is the base's own kernel.
    {
      const HostSite fs = make_fold_site(frng, 16, {{1, 0, 128, 0}}, 1.f);
      const k::LoraSiteDev fd = to_device(fs, owned);
      std::vector<float> base[2], got[2];
      fold_go(bx, big, nullptr, w, fN, true, y0, base);
      fold_go(bx, big, &fd, w, fN, true, y0, got);
      report("an output with no segment stays the base's, bit for bit",
             std::memcmp(base[0].data(), got[0].data(), base[0].size() * sizeof(float)) == 0);
    }
    for (uint8_t* p : w) cudaFree(p);
  }

  // A graph's slots (slot_dev) against the eager ones; then the slots changed on the device under the same launch.
  {
    int* dslot = upload(std::vector<int>(seq_slot, seq_slot + 4));
    k::LoraRows g = rows;
    for (int& s : g.slot) s = -7;  // must not be read
    g.slot_dev = dslot;
    Run r(hx, hy, M, sites);
    r.go(g);
    bool same = true;
    for (int o = 0; o < 3; ++o) same &= std::memcmp(r.out(o).data(), batch[o].data(), batch[o].size() * 4) == 0;
    report("device slots (a captured graph's) give the eager slots' bits", same);
    const std::vector<int> base(4, -1);
    cudaMemcpy(dslot, base.data(), 4 * sizeof(int), cudaMemcpyHostToDevice);
    Run r2(hx, hy, M, sites);
    r2.go(g);
    bool none = true;
    for (int o = 0; o < 3; ++o) none &= std::memcmp(r2.out(o).data(), hy[o].data(), hy[o].size() * 4) == 0;
    report("every slot -1 on the device: no output written", none);
    cudaFree(dslot);
  }

  // A null output (the drafter's context rows: k and v, no q) is skipped; the others get the batch's bits.
  {
    Run r(hx, hy, M, sites);
    r.go(rows, true);
    const bool q_kept = r.out(0) == hy[0];
    bool same = true;
    for (int o = 1; o < 3; ++o) same &= std::memcmp(r.out(o).data(), batch[o].data(), batch[o].size() * 4) == 0;
    report("a null output is skipped, the site's other outputs as in the batch", q_kept && same);
  }

  for (void* p : owned) cudaFree(p);
  const cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) {
    std::printf("CUDA error: %s\n", cudaGetErrorString(e));
    ++failures;
  }
  std::printf(failures ? "LORA KERNEL TESTS FAILED (%d)\n" : "all lora kernel tests passed\n", failures);
  return failures ? 1 : 0;
}
