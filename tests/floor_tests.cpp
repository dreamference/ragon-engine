// The decode-rate floor (serve/floor.hpp, C3 design A2 and A3); host only.
#include <chrono>
#include <cstdio>

#include "serve/floor.hpp"

namespace {

using namespace ragon::serve;

int failures = 0;

void expect(bool ok, const char* what) {
  if (!ok) {
    std::printf("FAILED: %s\n", what);
    ++failures;
  }
}

FloorInput input(double floor, int left, std::vector<double> rates, bool held = false, double held_s = 0) {
  FloorInput in;
  in.floor = floor;
  in.max_hold = 3;
  in.left = left;
  in.held = held;
  in.held_s = held_s;
  in.rates = std::move(rates);
  return in;
}

void verdicts() {
  // Off: the legacy rule alone decides, whatever the rates.
  expect(floor_verdict(input(0, 30000, {10})) == FloorVerdict::Off, "floor 0 is off");
  // One decoder below the floor holds a prime (at one decoder or more, C3 §1 cause 1).
  expect(floor_verdict(input(36, 30000, {24.4})) == FloorVerdict::Hold, "one slow decoder holds a prime");
  expect(floor_verdict(input(36, 30000, {60, 35.9})) == FloorVerdict::Hold, "any slow decoder holds");
  expect(floor_verdict(input(36, 30000, {36, 80})) == FloorVerdict::Rate, "decoders at the floor let it go");
  expect(floor_verdict(input(36, 30000, {})) == FloorVerdict::Rate, "no decoder: it goes");
  expect(floor_verdict(input(36, 30000, {-1, -1})) == FloorVerdict::Rate, "no opinion: it goes");
  // A warm append is never held.
  expect(floor_verdict(input(36, kWarmAppend, {5})) == FloorVerdict::Warm, "2,048 left is a warm append");
  expect(floor_verdict(input(36, kWarmAppend + 1, {5})) == FloorVerdict::Hold, "2,049 left is held");
  // The cap: a prefill held for max_hold seconds goes whatever the rates.
  expect(floor_verdict(input(36, 30000, {5}, true, 2.9)) == FloorVerdict::Hold, "held under the cap waits");
  expect(floor_verdict(input(36, 30000, {5}, true, 3.0)) == FloorVerdict::Cap, "held for the cap goes");
  expect(floor_verdict(input(36, 30000, {5}, false, 9)) == FloorVerdict::Hold, "the cap counts only while held");
}

void legacy() {
  // floor 0 = legacy hold_side (C1): only with two or more others decoding, for `share` passes after the mark.
  for (int share : {0, 1, 16})
    for (long mark : {-1L, 0L, 5L, 40L})
      for (long passes : {0L, 5L, 10L, 20L, 21L, 56L})
        for (int others : {0, 1, 2, 3}) {
          const bool want = share > 0 && mark >= 0 && others >= 2 && passes - mark < share;
          expect(share_hold(share, mark, passes, others) == want, "share_hold is the legacy hold_side");
        }
}

void chunk() {
  expect(floor_side_chunk(0, 1024, true, 30000) == 0, "floor off: full chunks");
  expect(floor_side_chunk(36, 1024, true, 30000) == 1024, "floor on beside a decoder: cut");
  expect(floor_side_chunk(36, 1024, false, 30000) == 0, "no decoder beside: full");
  expect(floor_side_chunk(36, 1024, true, kWarmAppend) == 0, "a warm append: full");
  expect(floor_side_chunk(36, 0, true, 30000) == 0, "--floor-side-chunk 0: full");
}

void rates() {
  using namespace std::chrono;
  const auto t0 = FloorClock::now();
  RateWindow w;
  for (int i = 0; i < kRateMin - 1; ++i) w.record(t0 + milliseconds(100 * i), 7 * i);
  expect(w.rate(t0 + milliseconds(300), 21) < 0, "fewer than kRateMin steps: no opinion");
  w.record(t0 + milliseconds(300), 21);
  // 4 marks: tokens 0..21 at 0..300 ms; at 400 ms with 28 tokens: 28 / 0.4 = 70 tok/s.
  const double r = w.rate(t0 + milliseconds(400), 28);
  expect(r > 69.9 && r < 70.1, "rate over the window to now");
  // Waiting lowers it: no step for 400 ms more.
  expect(w.rate(t0 + milliseconds(800), 28) < 36, "a waiting decoder's rate falls");
  for (int i = 4; i < 40; ++i) w.record(t0 + milliseconds(100 * i), 7 * i);
  expect(static_cast<int>(w.marks.size()) == kRateWindow, "the window keeps kRateWindow steps");
  expect(w.marks.front().second == 7 * (40 - kRateWindow), "the oldest steps leave the window");
  w.clear();
  expect(w.rate(t0, 0) < 0, "cleared: no opinion");
}

// A one-decoder simulation (C3 §2 Expected): alone steps of 100 ms at 7 tokens, beside a chunk 250 ms; a chunk takes
// chunk_ms. The decoder's rate over the run stays near the floor (a chunk goes when the trailing rate reaches it, so
// the run's rate sits a little under it) and the prime still moves. Returns the decoder's rate.
double simulate(double floor, int chunk_ms) {
  using namespace std::chrono;
  auto t = FloorClock::now();
  RateWindow w;
  int produced = 0, chunks = 0, cap = 0;
  bool held = false;
  FloorClock::time_point held_t0{};
  FloorClock::time_point chunk_end{};
  bool in_flight = false;
  const auto start = t;
  for (int step = 0; step < 2000; ++step) {
    if (!in_flight) {
      FloorInput in = input(floor, 30000, {w.rate(t, produced)}, held, held ? duration<double>(t - held_t0).count() : 0);
      const FloorVerdict v = floor_verdict(in);
      if (v == FloorVerdict::Hold) {
        if (!held) held = true, held_t0 = t;
      } else {
        held = false;
        in_flight = true;
        chunk_end = t + milliseconds(chunk_ms);
        ++chunks;
        cap += v == FloorVerdict::Cap;
      }
    }
    w.record(t, produced);
    t += milliseconds(in_flight ? 250 : 100);
    produced += 7;
    if (in_flight && t >= chunk_end) in_flight = false;
  }
  const double s = duration<double>(t - start).count();
  const double rate = produced / s;
  std::printf("simulated one decoder beside a prime, floor %.0f, %d ms chunks: %.1f tok/s, %d chunks in %.0f s (%d by the cap)\n",
              floor, chunk_ms, rate, chunks, s, cap);
  expect(rate >= 0.9 * floor, "one decoder stays near the floor");
  expect(chunks >= s / (3 + chunk_ms / 1e3 + 0.3), "the prime keeps a chunk per cap and chunk at least");
  return rate;
}

}  // namespace

int main() {
  verdicts();
  legacy();
  chunk();
  rates();
  simulate(36, 1200);
  simulate(36, 600);
  simulate(42, 1200);
  if (failures) {
    std::printf("%d failures\n", failures);
    return 1;
  }
  std::printf("floor: all passed\n");
  return 0;
}
