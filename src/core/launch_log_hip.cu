// RAGON_LAUNCH_LOG (hip only; launch.cuh): each launch timed alone, the slowest per name printed at exit.
#include "core/launch.cuh"

#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace ragon::kernels {
namespace {

struct Entry {
  float max_ms = 0;
  double total_ms = 0;
  long n = 0;
};
struct Log {
  std::mutex mu;
  std::map<std::string, Entry> by_name;
  ~Log() {
    if (by_name.empty()) return;
    std::vector<std::pair<float, std::string>> order;
    for (const auto& [name, e] : by_name) order.push_back({e.max_ms, name});
    std::sort(order.rbegin(), order.rend());
    std::fprintf(stderr, "launch log: the slowest launch of each kernel (ms; launches, total ms)\n");
    for (const auto& [ms, name] : order) {
      const Entry& e = by_name[name];
      std::fprintf(stderr, "launch log: %-40s max %10.3f  n %8ld  total %12.1f\n", name.c_str(), ms, e.n, e.total_ms);
    }
  }
};
Log& log() {
  static Log l;
  return l;
}
struct Pair {
  hipEvent_t a, b;
};

}  // namespace

bool launch_log_enabled() {
  static const bool on = [] {
    const char* e = std::getenv("RAGON_LAUNCH_LOG");
    return e && std::atoi(e) != 0;
  }();
  return on;
}

void* launch_log_begin(gpu::Stream s) {
  hipStreamCaptureStatus st = hipStreamCaptureStatusNone;
  if (hipStreamIsCapturing(s, &st) != hipSuccess || st != hipStreamCaptureStatusNone) return nullptr;
  log();  // constructed before the first record, so it is destroyed (and prints) after every user
  auto* p = new Pair;
  (void)hipEventCreate(&p->a), (void)hipEventCreate(&p->b);
  (void)hipEventRecord(p->a, s);
  return p;
}

void launch_log_end(void* h, const char* what, gpu::Stream s) {
  auto* p = static_cast<Pair*>(h);
  (void)hipEventRecord(p->b, s);
  (void)hipEventSynchronize(p->b);
  float ms = 0;
  (void)hipEventElapsedTime(&ms, p->a, p->b);
  (void)hipEventDestroy(p->a), (void)hipEventDestroy(p->b);
  delete p;
  std::lock_guard<std::mutex> g(log().mu);
  Entry& e = log().by_name[what];
  e.max_ms = std::max(e.max_ms, ms), e.total_ms += ms, ++e.n;
}

}  // namespace ragon::kernels
