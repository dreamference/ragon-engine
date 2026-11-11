// Which slot a new session takes (core/session_slots.hpp): the protection window and its lifting for a request that has
// waited it out (eng-session-protect-starve, 2026-11-11); host only.
#include <cstdio>
#include <vector>

#include "core/session_slots.hpp"

namespace {

using ragon::lru_slot;
using ragon::new_session_slot;
using ragon::protect_lift_due;
using ragon::SlotState;

int failures = 0;

void expect(bool ok, const char* what) {
  if (!ok) {
    std::printf("FAILED: %s\n", what);
    ++failures;
  }
}

SlotState keyed(double idle_s) { return {true, true, idle_s}; }
SlotState unkeyed(double idle_s) { return {true, false, idle_s}; }

void free_slots_first() {
  const std::vector<SlotState> s = {keyed(100), SlotState{}, SlotState{}};
  expect(lru_slot(s, 30, {}) == 1, "an empty slot is taken before any session is evicted");
  expect(lru_slot(s, 30, {false, true, false}) == 2, "a pinned empty slot is passed over");
}

void least_recently_used() {
  const std::vector<SlotState> s = {keyed(40), keyed(90), unkeyed(5), keyed(60)};
  expect(lru_slot(s, 30, {}) == 1, "the least recently used unprotected session is evicted");
  expect(lru_slot(s, 30, {false, true, false, false}) == 3, "a pinned session is never evicted");
  expect(lru_slot(s, 0, {}) == 1, "with protection off the order is the same");
  const std::vector<SlotState> tie = {keyed(50), keyed(50)};
  expect(lru_slot(tie, 30, {}) == 0, "a tie goes to the first slot, as before");
}

void protection() {
  const std::vector<SlotState> s = {keyed(10), keyed(20), unkeyed(1)};
  expect(lru_slot(s, 30, {}) == 2, "an unkeyed session is never protected, however recent");
  expect(lru_slot(s, 30, {false, false, true}) == -1, "every keyed session used within the window is protected");
  expect(lru_slot(s, 15, {false, false, true}) == 1, "a keyed session idle past the window is evicted");
  expect(lru_slot(s, 0, {false, false, true}) == 1, "protection off: the least recently used one");
}

void lifting() {
  expect(!protect_lift_due(30, 29.9), "not lifted before the window");
  expect(protect_lift_due(30, 30), "lifted at the window");
  expect(!protect_lift_due(0, 1e9), "nothing to lift with protection off");

  // One slot, held by an agent that keeps coming back within the window (the night recipe's --sessions 1).
  const std::vector<SlotState> one = {keyed(2)};
  bool lifted = true;
  expect(new_session_slot(one, 30, 5, {}, {}, &lifted) == -1 && !lifted, "a new session waits within the window");
  expect(new_session_slot(one, 30, 31, {}, {}, &lifted) == 0 && lifted, "after the window it takes the protected slot");
  expect(new_session_slot(one, 30, 31, {true}, {true}, &lifted) == -1 && !lifted,
         "a running request's session is never taken");
  // The agent's next turn is queued behind the waiting request: it pins the slot for the first look, not the second.
  expect(new_session_slot(one, 30, 31, {true}, {false}, &lifted) == 0 && lifted,
         "a turn queued behind the waiting request does not hold the slot");
  expect(new_session_slot(one, 0, 31, {true}, {false}, &lifted) == -1 && !lifted,
         "protection off: the pins hold as before");

  // Four slots of agents: the lift takes the least recently used one that is not running or continued ahead.
  const std::vector<SlotState> four = {keyed(3), keyed(12), keyed(25), keyed(8)};
  const std::vector<bool> running = {false, false, true, false};
  expect(new_session_slot(four, 30, 10, running, running, &lifted) == -1 && !lifted, "within the window it waits");
  expect(new_session_slot(four, 30, 30, running, running, &lifted) == 1 && lifted,
         "after it, the least recently used idle session");
  expect(new_session_slot(four, 30, 30, running, {false, true, true, false}, &lifted) == 3 && lifted,
         "a session a job ahead continues is passed over");
  // A free slot or an unprotected session is taken at once, lifted or not.
  const std::vector<SlotState> mixed = {keyed(3), keyed(45)};
  expect(new_session_slot(mixed, 30, 0, {}, {}, &lifted) == 1 && !lifted, "an expired protection needs no lift");
}

}  // namespace

int main() {
  free_slots_first();
  least_recently_used();
  protection();
  lifting();
  if (failures == 0) std::printf("session slot tests: all passed\n");
  return failures == 0 ? 0 : 1;
}
