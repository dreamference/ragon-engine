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

// TRUE_FORK, fork admission: which queued requests fork an idle resident session into a slot of their own
// (Engine::fork_session, then a prefill of their own suffix only, which is what continuing the session in place
// computes) instead of moving that session on. Without it the first of k branches of one context continues the
// session that holds the context, the others wait for it and then find only checkpoints: they re-read the context
// (39 s per branch at 100K, eng-fork-vs-sglang-1). Only where a request runs changes, never what it computes.
//
// For each source session f: the request that continues f in place is the thread f is bound with (its owner) when it
// is queued, else the last queued request that continues f and is not another thread's; every other request whose
// source is f forks it. A request bound with another thread's key always forks (the owner's next turn, or its next
// branch, comes back to f). The continuer of f, and every other request that would continue f in place, yields while
// a fork of f waits (so f is not moved on before its branches start), for at most kForkYieldSeconds of that fork's
// wait. With a single slot nothing forks.
constexpr double kForkYieldSeconds = 30;

enum class ForkAdmit { kPlain, kFork, kYield };

// One queued request as the policy sees it.
struct ForkQuery {
  int source = -1;         // the idle session it could fork (Engine::fork_source), -1 none
  int continues = -1;      // the session it continues in place (Engine::match_session), -1 a new session
  bool owner = false;      // it is the thread `continues` is bound with (both keyed with the same key)
  bool other_key = false;  // it and `source` are bound with keys, different ones
  double waited_s = 0;     // seconds it has been queued
};

inline std::vector<ForkAdmit> fork_admit(const std::vector<ForkQuery>& q, int sessions) {
  const size_t n = q.size();
  std::vector<ForkAdmit> out(n, ForkAdmit::kPlain);
  if (sessions < 2) return out;
  auto wants = [&](size_t k, int f) { return q[k].source == f || q[k].continues == f; };
  for (size_t j = 0; j < n; ++j) {
    const int f = q[j].source;
    if (f < 0 || q[j].owner) continue;
    if (q[j].other_key) {
      out[j] = ForkAdmit::kFork;
      continue;
    }
    // Not another thread's: it forks unless it is f's continuer (no owner queued, and the last such request).
    bool continuer = true;
    for (size_t k = 0; k < n && continuer; ++k) {
      if (k == j) continue;
      if (q[k].owner && q[k].continues == f) continuer = false;
      if (k > j && wants(k, f) && !q[k].owner && !q[k].other_key) continuer = false;
    }
    if (!continuer) out[j] = ForkAdmit::kFork;
  }
  for (size_t j = 0; j < n; ++j) {
    if (out[j] == ForkAdmit::kFork) continue;
    const int f = q[j].continues >= 0 ? q[j].continues : q[j].source;
    if (f < 0) continue;
    for (size_t k = 0; k < n; ++k)
      if (out[k] == ForkAdmit::kFork && q[k].source == f && q[k].waited_s < kForkYieldSeconds) {
        out[j] = ForkAdmit::kYield;
        break;
      }
  }
  return out;
}

}  // namespace ragon
