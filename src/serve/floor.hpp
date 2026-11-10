// The decode-rate floor (C3, design A2 and A3): whether the side stream holds a prefill's next chunk so the
// decoding conversations keep their speed.
//
// Legacy (--side-share N, C1): while two or more other requests decode, N decode passes run on the main stream
// alone after the last side chunk before the next one starts. With one decoder there is no hold, so the last
// conversation decoding beside a long prompt took every step beside a chunk (C3 §1, cause 1).
//
// The floor (--decode-floor F): while any decoding request's trailing rate (its tokens over the wall time since the
// oldest of its last kRateWindow steps, measured to now) is below F tokens/s, the next side chunk waits, at one
// decoder or more. Two bounds keep C1's results: a prefill with at most kWarmAppend tokens left (a warm append, the
// follow-up's whole time to its first token) is never held, and a held prefill takes its next chunk after
// --prefill-max-hold seconds whatever the rates, so a prime keeps a minimum share and a floor that cannot be met does
// not starve it. A request with fewer than kRateMin steps in its window gives no opinion (its rate is not known yet).
// A3: while the floor is on and a decoder runs beside it, a side chunk of a prompt longer than a warm append is cut
// to --floor-side-chunk tokens (the prefill is chunk-size invariant, BATCHING §5.5), so a step beside it is shorter.
//
// Only the order and size of prefill units change: every unit runs in its own session's state, as it would alone.
// Host only: no Engine, so tests/floor_tests.cpp checks the policy without a GPU.
#pragma once

#include <chrono>
#include <deque>
#include <vector>

namespace ragon::serve {

inline constexpr int kRateWindow = 16;     // the trailing steps a decoding request's rate is taken over
inline constexpr int kRateMin = 4;         // fewer steps in the window: no opinion
inline constexpr int kWarmAppend = 2048;   // a prefill with at most this many tokens left is never held
inline constexpr double kDecodeFloorOn = 36;  // --decode-floor on: the bar (34.6, BATCHING §4 item 1) plus a margin

using FloorClock = std::chrono::steady_clock;

// A decoding request's trailing rate: (time, tokens produced so far) at the start of each of its last kRateWindow
// steps.
struct RateWindow {
  std::deque<std::pair<FloorClock::time_point, int>> marks;
  void record(FloorClock::time_point t, int produced) {
    marks.emplace_back(t, produced);
    while (static_cast<int>(marks.size()) > kRateWindow) marks.pop_front();
  }
  void clear() { marks.clear(); }
  // Tokens per second from the oldest mark to `now` (the tokens of the steps since that mark over their wall time,
  // the time it has been waiting since its last step included); negative when there is no opinion yet.
  double rate(FloorClock::time_point now, int produced) const {
    if (static_cast<int>(marks.size()) < kRateMin) return -1;
    const double s = std::chrono::duration<double>(now - marks.front().first).count();
    return s > 0 ? (produced - marks.front().second) / s : -1;
  }
};

// The inputs of one hold decision for the request whose side chunk is next.
struct FloorInput {
  double floor = 0;                // --decode-floor (0: off)
  double max_hold = 3;             // --prefill-max-hold, seconds
  int left = 0;                    // its prompt tokens still to prefill
  bool held = false;               // it is already held, since held_s seconds
  double held_s = 0;
  std::vector<double> rates;       // the other decoding requests' trailing rates (negative: no opinion)
};

enum class FloorVerdict {
  Off,     // the floor is off: the legacy rule alone decides
  Warm,    // a warm append: never held
  Rate,    // every decoder is at or above the floor (or none has an opinion): the chunk goes
  Cap,     // held for max_hold: the chunk goes whatever the rates
  Hold,    // a decoder is below the floor: the chunk waits
};

inline FloorVerdict floor_verdict(const FloorInput& in) {
  if (in.floor <= 0) return FloorVerdict::Off;
  if (in.left <= kWarmAppend) return FloorVerdict::Warm;
  if (in.held && in.held_s >= in.max_hold) return FloorVerdict::Cap;
  for (double r : in.rates)
    if (r >= 0 && r < in.floor) return FloorVerdict::Hold;
  return FloorVerdict::Rate;
}

// The legacy --side-share rule (C1), as Scheduler::hold_side applied it: holds while two or more others decode and
// fewer than `share` decode passes have run since the last side chunk finished (mark < 0: none yet).
inline bool share_hold(int share, long mark, long passes, int others_decoding) {
  if (share <= 0 || mark < 0 || others_decoding < 2) return false;
  return passes - mark < share;
}

// The tokens a side chunk launch asks for (0: a full chunk): cut to `cut` while the floor is on, a decoder runs
// beside it and the prompt is longer than a warm append.
inline int floor_side_chunk(double floor, int cut, bool contended, int left) {
  return floor > 0 && cut > 0 && contended && left > kWarmAppend ? cut : 0;
}

inline const char* floor_verdict_name(FloorVerdict v) {
  switch (v) {
    case FloorVerdict::Off: return "off";
    case FloorVerdict::Warm: return "warm";
    case FloorVerdict::Rate: return "rate";
    case FloorVerdict::Cap: return "cap";
    case FloorVerdict::Hold: return "hold";
  }
  return "?";
}

}  // namespace ragon::serve
