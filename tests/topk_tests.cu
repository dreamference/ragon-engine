// topk_rows and topk_rows_masked on any backend (BACKEND §6, §7): the same inputs on CUDA and on HIP, the outputs
// checked bit for bit against a host reference and hashed, so two runs and two backends compare byte for byte.
//
// The inputs come from std::mt19937 (its sequence is fixed by the standard) and integer arithmetic only, so every
// platform and standard library builds the same rows; coarse values make many ties, so the index tie-break is
// exercised. A third set mixes +inf, +-0 and subnormals into a row.
//
//   ragon-topk-tests                  host reference + the recorded CUDA hash (kCudaHash)
//   ragon-topk-tests --out FILE       also write the raw outputs (for cmp across runs and backends)
//   ragon-topk-tests --time N         also time topk_rows (K=64, 5 rows) over N launches, median in us
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "core/kernels.cuh"

#if defined(RAGON_BACKEND_HIP)
#define GPU(name) hip##name
#define GPU_H2D hipMemcpyHostToDevice
#define GPU_D2H hipMemcpyDeviceToHost
#define GPU_PROP hipDeviceProp_t
#else
#define GPU(name) cuda##name
#define GPU_H2D cudaMemcpyHostToDevice
#define GPU_D2H cudaMemcpyDeviceToHost
#define GPU_PROP cudaDeviceProp
#endif

namespace {

// Every runtime call is checked: a failure aborts with its line (HIP's calls are also [[nodiscard]]).
#define CK(call)                                                                       \
  do {                                                                                 \
    if ((call) != GPU(Success)) std::fprintf(stderr, "line %d: %s\n", __LINE__, #call), std::abort(); \
  } while (0)

// FNV-1a 64 over every output of every case, in order: the CUDA build's value recorded on a GB10 (sm_121a).
constexpr uint64_t kCudaHash = 0xbc18c88dc89f0d68ull;  // 12120 bytes, sha256 8e33e6c3…502704eae

constexpr int V = 248320;
const int kKs[] = {1, 16, 20, 64};

std::vector<uint8_t> g_out;  // every output, in order

void emit(const void* p, size_t n) {
  const auto* b = static_cast<const uint8_t*>(p);
  g_out.insert(g_out.end(), b, b + n);
}

uint64_t fnv1a(const std::vector<uint8_t>& b) {
  uint64_t h = 1469598103934665603ull;
  for (uint8_t c : b) h = (h ^ c) * 1099511628211ull;
  return h;
}

uint32_t bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}

// Values k/64 for k in [-768, 768]: 1537 distinct values over 248320 ids, so every row is full of ties.
std::vector<float> coarse_rows(uint32_t seed, int rows) {
  std::mt19937 rng(seed);
  std::vector<float> x(size_t(rows) * V);
  for (auto& v : x) v = float(int(rng() % 1537u) - 768) / 64.f;
  return x;
}

template <typename T>
T* upload(const std::vector<T>& h) {
  T* d = nullptr;
  CK(GPU(Malloc)(reinterpret_cast<void**>(&d), h.size() * sizeof(T)));
  CK(GPU(Memcpy)(d, h.data(), h.size() * sizeof(T), GPU_H2D));
  return d;
}

template <typename T>
std::vector<T> download(const T* d, size_t n) {
  std::vector<T> h(n);
  CK(GPU(Memcpy)(h.data(), d, n * sizeof(T), GPU_D2H));
  return h;
}

// The reference order: larger value first, the lower id first among equal values; -0 equals +0.
void ref_sort(const float* row, std::vector<int>& idx, int n) {
  std::partial_sort(idx.begin(), idx.begin() + n, idx.end(),
                    [&](int a, int b) { return row[a] > row[b] || (row[a] == row[b] && a < b); });
}

bool run_plain(const char* name, const std::vector<float>& x, int rows) {
  float* dx = upload(x);
  bool ok = true;
  for (int K : kKs) {
    float *scratch, *vals;
    int* ids;
    CK(GPU(Malloc)(reinterpret_cast<void**>(&scratch), ragon::kernels::topk_scratch_floats(rows, K) * sizeof(float)));
    CK(GPU(Malloc)(reinterpret_cast<void**>(&vals), size_t(rows) * K * sizeof(float)));
    CK(GPU(Malloc)(reinterpret_cast<void**>(&ids), size_t(rows) * K * sizeof(int)));
    ragon::kernels::topk_rows(dx, rows, V, K, scratch, vals, ids, nullptr);
    const auto gv = download(vals, size_t(rows) * K);
    const auto gi = download(ids, size_t(rows) * K);
    emit(gv.data(), gv.size() * 4), emit(gi.data(), gi.size() * 4);
    bool good = true;
    for (int r = 0; r < rows; ++r) {
      std::vector<int> idx(V);
      std::iota(idx.begin(), idx.end(), 0);
      const float* row = x.data() + size_t(r) * V;
      ref_sort(row, idx, K);
      for (int k = 0; k < K; ++k)
        good &= gi[size_t(r) * K + k] == idx[k] && bits(gv[size_t(r) * K + k]) == bits(row[idx[k]]);
    }
    ok &= good;
    std::printf("%s K=%-2d %s\n", name, K, good ? "ok" : "FAIL");
    CK(GPU(Free)(scratch)); CK(GPU(Free)(vals)); CK(GPU(Free)(ids));
  }
  CK(GPU(Free)(dx));
  return ok;
}

bool run_masked() {
  const int rows = 8, W = (V + 31) / 32;
  const auto x = coarse_rows(11, rows);
  std::mt19937 rng(12);
  std::vector<int32_t> mask(size_t(rows) * W, 0);
  const int density[rows] = {0, 1, 3, 17, 70, 1000, 30000, -1};  // allowed ids per row; -1 = all
  for (int r = 0; r < rows; ++r) {
    int32_t* m = mask.data() + size_t(r) * W;
    if (density[r] < 0) {
      for (int w = 0; w < W; ++w) m[w] = -1;
      continue;
    }
    for (int i = 0; i < density[r]; ++i) {
      const int id = r == 2 && i == 0 ? V - 1 : int(rng() % uint32_t(V));  // the last id too
      m[id >> 5] |= int32_t(1u << (id & 31));
    }
  }
  float* dx = upload(x);
  int32_t* dm = upload(mask);
  bool ok = true;
  for (int K : kKs) {
    float *scratch, *vals;
    int* ids;
    CK(GPU(Malloc)(reinterpret_cast<void**>(&scratch), ragon::kernels::topk_scratch_floats(rows, K) * sizeof(float)));
    CK(GPU(Malloc)(reinterpret_cast<void**>(&vals), size_t(rows) * K * sizeof(float)));
    CK(GPU(Malloc)(reinterpret_cast<void**>(&ids), size_t(rows) * K * sizeof(int)));
    ragon::kernels::topk_rows_masked(dx, rows, V, K, dm, W, scratch, vals, ids, nullptr);
    const auto gv = download(vals, size_t(rows) * K);
    const auto gi = download(ids, size_t(rows) * K);
    emit(gv.data(), gv.size() * 4), emit(gi.data(), gi.size() * 4);
    bool good = true;
    for (int r = 0; r < rows; ++r) {
      const float* row = x.data() + size_t(r) * V;
      const int32_t* m = mask.data() + size_t(r) * W;
      std::vector<int> idx;
      for (int id = 0; id < V; ++id)
        if ((uint32_t(m[id >> 5]) >> (id & 31)) & 1u) idx.push_back(id);
      const int n = std::min<int>(K, int(idx.size()));
      ref_sort(row, idx, n);
      for (int k = 0; k < K; ++k) {
        const size_t o = size_t(r) * K + k;
        if (k < n)
          good &= gi[o] == idx[k] && bits(gv[o]) == bits(row[idx[k]]);
        else
          good &= gi[o] == 0x7fffffff && bits(gv[o]) == bits(-FLT_MAX);
      }
    }
    ok &= good;
    std::printf("masked K=%-2d %s\n", K, good ? "ok" : "FAIL");
    CK(GPU(Free)(scratch)); CK(GPU(Free)(vals)); CK(GPU(Free)(ids));
  }
  CK(GPU(Free)(dx)); CK(GPU(Free)(dm));
  return ok;
}

// Coarse rows salted with +inf, -0 and +0 (equal, the lower id first) and subnormals (no flush to zero allowed).
std::vector<float> special_rows() {
  auto x = coarse_rows(13, 2);
  std::mt19937 rng(14);
  for (int i = 0; i < 40; ++i) x[rng() % uint32_t(V)] = INFINITY;  // row 0: 40 (or fewer) +inf
  for (size_t i = V; i < x.size(); ++i) {  // row 1: subnormals of both signs, zeros of both signs, the rest < 0
    const uint32_t u = rng() % 8u;
    float f = -std::fabs(x[i]);
    if (u == 0) f = 0.f;
    if (u == 1) f = -0.f;
    if (u == 2) f = std::ldexp(float(1 + rng() % 1000u), -140);   // +subnormal
    if (u == 3) f = -std::ldexp(float(1 + rng() % 1000u), -140);  // -subnormal
    x[i] = f;
  }
  return x;
}

double time_topk(int iters) {
  const int rows = 5, K = 64;
  const auto x = coarse_rows(5, rows);
  float* dx = upload(x);
  float *scratch, *vals;
  int* ids;
  CK(GPU(Malloc)(reinterpret_cast<void**>(&scratch), ragon::kernels::topk_scratch_floats(rows, K) * sizeof(float)));
  CK(GPU(Malloc)(reinterpret_cast<void**>(&vals), size_t(rows) * K * sizeof(float)));
  CK(GPU(Malloc)(reinterpret_cast<void**>(&ids), size_t(rows) * K * sizeof(int)));
  GPU(Event_t) a, b;
  CK(GPU(EventCreate)(&a)); CK(GPU(EventCreate)(&b));
  for (int i = 0; i < 20; ++i) ragon::kernels::topk_rows(dx, rows, V, K, scratch, vals, ids, nullptr);
  std::vector<float> us;
  for (int i = 0; i < iters; ++i) {
    CK(GPU(EventRecord)(a, nullptr));
    ragon::kernels::topk_rows(dx, rows, V, K, scratch, vals, ids, nullptr);
    CK(GPU(EventRecord)(b, nullptr));
    CK(GPU(EventSynchronize)(b));
    float ms = 0;
    CK(GPU(EventElapsedTime)(&ms, a, b));
    us.push_back(ms * 1000.f);
  }
  std::sort(us.begin(), us.end());
  CK(GPU(EventDestroy)(a)); CK(GPU(EventDestroy)(b));
  CK(GPU(Free)(dx)); CK(GPU(Free)(scratch)); CK(GPU(Free)(vals)); CK(GPU(Free)(ids));
  return us.empty() ? 0.0 : us[us.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
  const char* out = nullptr;
  int time_iters = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--out" && i + 1 < argc) out = argv[++i];
    else if (a == "--time" && i + 1 < argc) time_iters = std::atoi(argv[++i]);
    else return std::fprintf(stderr, "usage: %s [--out FILE] [--time N]\n", argv[0]), 2;
  }
  GPU_PROP prop;
  if (GPU(GetDeviceProperties)(&prop, 0) != GPU(Success)) return std::fprintf(stderr, "no device\n"), 1;
  std::printf("device %s (%s), warp %d\n", prop.name, ragon::gpu::kName, prop.warpSize);
  if (prop.warpSize != 32) return std::fprintf(stderr, "the kernels assume 32-lane warps\n"), 1;

  bool ok = run_plain("plain", coarse_rows(5, 5), 5);
  ok &= run_masked();
  ok &= run_plain("special", special_rows(), 2);

  const uint64_t h = fnv1a(g_out);
  std::printf("outputs %zu bytes, fnv1a %016llx\n", g_out.size(), static_cast<unsigned long long>(h));
  if (kCudaHash != 0) {
    const bool same = h == kCudaHash;
    std::printf("against the CUDA outputs (%016llx): %s\n", static_cast<unsigned long long>(kCudaHash),
                same ? "identical" : "DIFFERENT");
    ok &= same;
  }
  if (out) {
    std::ofstream f(out, std::ios::binary);
    f.write(reinterpret_cast<const char*>(g_out.data()), std::streamsize(g_out.size()));
    if (!f) return std::fprintf(stderr, "cannot write %s\n", out), 1;
  }
  if (time_iters > 0) std::printf("topk_rows K=64 rows=5: median %.2f us over %d\n", time_topk(time_iters), time_iters);
  std::printf("%s\n", ok ? "all ok" : "FAILED");
  return ok ? 0 : 1;
}
