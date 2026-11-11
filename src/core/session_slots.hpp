// Which slot a new session takes (M4 §1, §7 question 6): the policy behind Engine::free_session and the scheduler's
// lifting of the protection window, with no Engine, so tests/session_slots_tests.cpp checks it without a GPU.
//
// A session bound with a prompt_cache_key (an agent's thread) whose last request ran less than the protection window
// ago is protected: a new session does not take its slot while its tool runs. Protection holds against a new
// session's request only while that request has waited less than the window. Once it has waited the window or
// longer, it takes the least recently used slot that no running request holds and no request queued ahead of it
// continues, protected or not. Without that, a full table of keyed sessions whose agents keep coming back within the
// window starves every other conversation: with one slot, a second agent's first request waited the whole hour of a
// task (eng-session-protect-starve, 2026-11-11). Only which slot a session takes changes, never what a request computes.
#pragma once

#include <cstddef>
#include <vector>

namespace ragon {

// One slot as the policy sees it.
struct SlotState {
  bool owned = false;  // it holds a session
  bool keyed = false;  // that session is bound with a prompt_cache_key
  double idle_s = 0;   // seconds since the session's last request
};

// A slot for a new session: the first one that holds none and is not pinned, else the least recently used one that
// is neither pinned nor protected (keyed and idle less than protect_s); -1 when there is none.
inline int lru_slot(const std::vector<SlotState>& slots, double protect_s, const std::vector<bool>& pinned) {
  auto is_pinned = [&](size_t i) { return i < pinned.size() && pinned[i]; };
  for (size_t i = 0; i < slots.size(); ++i)
    if (!slots[i].owned && !is_pinned(i)) return static_cast<int>(i);
  int best = -1;
  for (size_t i = 0; i < slots.size(); ++i) {
    if (is_pinned(i)) continue;
    if (slots[i].keyed && slots[i].idle_s < protect_s) continue;
    if (best < 0 || slots[i].idle_s > slots[best].idle_s) best = static_cast<int>(i);
  }
  return best;
}

// True when a new session's request has waited long enough to take a protected slot.
inline bool protect_lift_due(double protect_s, double waited_s) { return protect_s > 0 && waited_s >= protect_s; }

// The slot a new session's request takes in the scheduler: lru_slot under protection and `pinned` (the running
// requests' sessions and those any queued request continues); when there is none and the request has waited the
// window, lru_slot without protection under `pinned_ahead` (the running requests' sessions and those the requests
// queued ahead of it continue). `lifted` is set when the slot comes from the second look.
inline int new_session_slot(const std::vector<SlotState>& slots, double protect_s, double waited_s,
                            const std::vector<bool>& pinned, const std::vector<bool>& pinned_ahead, bool* lifted) {
  if (lifted) *lifted = false;
  int s = lru_slot(slots, protect_s, pinned);
  if (s >= 0 || !protect_lift_due(protect_s, waited_s)) return s;
  s = lru_slot(slots, 0, pinned_ahead);
  if (lifted) *lifted = s >= 0;
  return s;
}

}  // namespace ragon
