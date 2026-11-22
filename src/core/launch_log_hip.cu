// RAGON_LAUNCH_LOG (hip only; launch.cuh): each launch timed alone, the slowest per name printed at exit.
// RAGON_LAUNCH_LOG=2 times every launch between two events WITHOUT waiting (no host sync, so the stream keeps
// its normal back-to-back schedule): the events are read later once they completed, kernels are keyed by
// "<range>/<name>#<grid x>x<grid y>" (a GEMM's shapes apart), and the stream's idle time before each launch
// (the previous launch's end to this one's start: copies, host syncs, unlogged work) is logged as
// "<range>/(gap)", so a range's rows sum to its GPU wall time.
#include "core/launch.cuh"

#include <deque>

#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <cstring>
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
  std::map<std::string, long> range_n;  // mode 2: how often each range opened (steps per phase)
  ~Log() {
    for (const auto& [name, n] : range_n) std::fprintf(stderr, "launch log: range %-30s n %8ld\n", name.c_str(), n);
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
int mode() {
  static const int m = [] {
    const char* e = std::getenv("RAGON_LAUNCH_LOG");
    return e ? std::atoi(e) : 0;
  }();
  return m;
}
// Mode 2: event pool, the pending launches in record order, and the last recorded end event per stream.
struct Rec {
  std::string key, gap_key;
  hipEvent_t a, b, prev;  // prev: the stream's previous end event (nullptr for its first launch)
};
std::vector<hipEvent_t> pool;
std::deque<Rec> pending;
std::map<gpu::Stream, hipEvent_t> last_end;
std::mutex q_mu;  // pool, pending, last_end
hipEvent_t take() {
  if (!pool.empty()) {
    hipEvent_t e = pool.back();
    pool.pop_back();
    return e;
  }
  hipEvent_t e = nullptr;
  (void)hipEventCreate(&e);
  return e;
}
void add(const std::string& key, float ms) {
  std::lock_guard<std::mutex> g(log().mu);
  Entry& e = log().by_name[key];
  e.max_ms = std::max(e.max_ms, ms), e.total_ms += ms, ++e.n;
}
// Reads every completed launch at the head of the queue, never waiting (q_mu held). The launches still pending at
// exit (the last step's tail) are not read.
void harvest() {
  while (!pending.empty()) {
    Rec& r = pending.front();
    if (hipEventQuery(r.b) != hipSuccess) break;
    float ms = 0;
    if (hipEventElapsedTime(&ms, r.a, r.b) == hipSuccess) add(r.key, ms);
    if (r.prev) {
      float g = 0;
      if (hipEventElapsedTime(&g, r.prev, r.a) == hipSuccess) add(r.gap_key, g);
      pool.push_back(r.prev);  // only this record still referred to it
    }
    pool.push_back(r.a);
    // r.b stays alive: it is the next record's prev (or the stream's last end), released when that one is read
    pending.pop_front();
  }
}
// The open profiler ranges of this thread (trace.hpp): a timed launch is logged under "<innermost range>/<name>".
thread_local std::vector<const char*> ranges;

}  // namespace

bool launch_log_enabled() {
  static const bool on = [] {
    const char* e = std::getenv("RAGON_LAUNCH_LOG");
    return e && std::atoi(e) != 0;
  }();
  return on;
}

// The comma-separated parts of an environment variable (empty parts dropped).
static std::vector<std::string> name_list(const char* var) {
  std::vector<std::string> v;
  const char* e = std::getenv(var);
  std::string s = e ? e : "";
  for (size_t a = 0; a < s.size();) {
    size_t b = s.find(',', a);
    if (b == std::string::npos) b = s.size();
    if (b > a) v.push_back(s.substr(a, b - a));
    a = b + 1;
  }
  return v;
}

// RAGON_LAUNCH_LOG_ONLY=a,b (mode 2): only launches whose name contains one of the comma-separated parts are timed;
// the others run without events and fall into the (gap) rows, so the timed kernels carry less event overhead.
bool launch_log_wants(const char* what) {
  static const std::vector<std::string> only = name_list("RAGON_LAUNCH_LOG_ONLY");
  if (only.empty()) return true;
  for (const std::string& o : only)
    if (std::strstr(what, o.c_str())) return true;
  return false;
}

bool launch_skip_wants(const char* what) {
  static const std::vector<std::string> skip = name_list("RAGON_LAUNCH_SKIP");
  for (const std::string& d : skip)
    if (d == what) return true;
  return false;
}

bool launch_dup_wants(const char* what) {
  static const std::vector<std::string> dup = name_list("RAGON_LAUNCH_DUP");
  for (const std::string& d : dup)
    if (d == what) return true;
  return false;
}

void launch_log_range_push(const char* name) {
  if (!launch_log_enabled()) return;
  ranges.push_back(name);
  if (mode() == 2) {
    std::lock_guard<std::mutex> g(log().mu);
    ++log().range_n[name];
  }
}

void launch_log_range_pop() {
  if (launch_log_enabled() && !ranges.empty()) ranges.pop_back();
  if (mode() == 2) {
    std::lock_guard<std::mutex> g(q_mu);
    harvest();
  }
}

void launch_log_end_grid(void* h, const char* what, gpu::Stream s, unsigned gx, unsigned gy) {
  if (mode() != 2) return launch_log_end(h, what, s);
  auto* p = static_cast<Pair*>(h);
  const std::string range = ranges.empty() ? std::string() : std::string(ranges.back()) + "/";
  Rec r;
  r.key = range + what + "#" + std::to_string(gx) + "x" + std::to_string(gy);
  r.gap_key = range + "(gap)";
  r.a = p->a;
  delete p;
  std::lock_guard<std::mutex> g(q_mu);
  r.b = take();
  (void)hipEventRecord(r.b, s);
  auto it = last_end.find(s);
  r.prev = it == last_end.end() ? nullptr : it->second;
  last_end[s] = r.b;
  pending.push_back(std::move(r));
  if (pending.size() > 512) harvest();
}

void* launch_log_begin(gpu::Stream s) {
  hipStreamCaptureStatus st = hipStreamCaptureStatusNone;
  if (hipStreamIsCapturing(s, &st) != hipSuccess || st != hipStreamCaptureStatusNone) return nullptr;
  log();  // constructed before the first record, so it is destroyed (and prints) after every user
  auto* p = new Pair;
  if (mode() == 2) {
    std::lock_guard<std::mutex> g(q_mu);
    p->a = take(), p->b = nullptr;
  } else {
    (void)hipEventCreate(&p->a), (void)hipEventCreate(&p->b);
  }
  (void)hipEventRecord(p->a, s);
  return p;
}

void launch_log_end(void* h, const char* what, gpu::Stream s) {
  if (mode() == 2) return launch_log_end_grid(h, what, s, 0, 0);
  auto* p = static_cast<Pair*>(h);
  (void)hipEventRecord(p->b, s);
  (void)hipEventSynchronize(p->b);
  float ms = 0;
  (void)hipEventElapsedTime(&ms, p->a, p->b);
  (void)hipEventDestroy(p->a), (void)hipEventDestroy(p->b);
  delete p;
  const std::string key = ranges.empty() ? std::string(what) : std::string(ranges.back()) + "/" + what;
  std::lock_guard<std::mutex> g(log().mu);
  Entry& e = log().by_name[key];
  e.max_ms = std::max(e.max_ms, ms), e.total_ms += ms, ++e.n;
}

int kick_every() {
  static const int n = [] {
    const char* e = std::getenv("RAGON_HIP_KICK_EVERY");
    return e && *e ? std::max(0, std::atoi(e)) : 0;
  }();
  return n;
}

void launch_kick(gpu::Stream s) {
  const int n = kick_every();
  if (n == 0) return;
  thread_local int count = 0;
  if (++count < n) return;
  count = 0;
  hipStreamCaptureStatus st = hipStreamCaptureStatusNone;
  if (hipStreamIsCapturing(s, &st) != hipSuccess || st != hipStreamCaptureStatusNone) return;
  (void)hipStreamQuery(s);  // NotReady is the expected answer
}

}  // namespace ragon::kernels
