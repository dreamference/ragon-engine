// The chain walk over the drafter's lattice (SGLang's sample_path), with structured output's filter (SPECULATION
// "Structured output, stage 2"). Host only, so tests/spec_tests.cpp runs it against a toy constraint.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "core/constraint.hpp"
#include "core/sampling.hpp"

namespace ragon {

struct ChainWalk {
  // In: E rows of K candidates (cand, E*K) and the selector's scores (row e, given the previous pick: K*K each).
  int E = 0, K = 0;
  const int* cand = nullptr;
  const float* scores = nullptr;
  bool greedy = true;
  double temperature = 1.0;
  // The constraint (nullptr: none), whether it is armed, and the token that arms it; masks has E + 1 rows of words.
  TokenConstraint* con = nullptr;
  bool armed = false;
  int arm_token = -1;
  int32_t* masks = nullptr;
  int words = 0;
  // Out: each row's pick (its token and index) and the draft distribution q; cut, the drafts that may be accepted
  // (E unconstrained); masked, whether any row has a mask (then every one of the E + 1 rows holds one).
  std::vector<int> drafts, picks;
  std::vector<std::vector<double>> q;
  int cut = 0;
  bool masked = false;
};

// The constraint walks the chain as a scratch copy (accept, then roll back), each row's mask the state after the
// anchor and the drafts before it. A pick is the best (greedy) or a draw (sampled, q renormalised) among the
// candidates the row allows; at the first row that allows none the chain is cut, and the rows after it keep the
// drafter's own picks under all-ones masks, never to be accepted (plan_accept). Unconstrained, the walk is the
// plain one. Throws ConstraintError when the constraint refuses a token its own mask allowed.
inline void walk_chain(ChainWalk& w, std::mt19937_64& rng) {
  const int E = w.E, K = w.K;
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  w.drafts.assign(E, 0);
  w.picks.assign(E, 0);
  w.q.assign(E, std::vector<double>(K, 0.0));
  w.cut = E;
  w.masked = false;
  TokenConstraint* con = w.con;
  bool armed = con && w.armed, live = con != nullptr;
  int advanced = 0;
  std::vector<char> filled(con ? E + 1 : 0, 0);
  auto fill_row = [&](int e) -> const int32_t* {  // row e's mask, or nullptr when the row is not constrained
    if (!live || !armed || con->terminated()) return nullptr;
    int32_t* m = w.masks + size_t(e) * w.words;
    con->fill(m);
    filled[e] = 1;
    w.masked = true;
    return m;
  };
  int prev = 0;
  for (int e = 0; e < E; ++e) {
    const float* sc = w.scores + (size_t(e) * K + (e == 0 ? 0 : prev)) * K;
    std::vector<double>& q = w.q[e];
    const int* cand_e = w.cand + size_t(e) * K;
    const int32_t* m = con ? fill_row(e) : nullptr;
    if (m) {
      bool any = false;
      for (int k = 0; k < K && !any; ++k) any = mask_allows(m, cand_e[k]);
      if (!any) {  // the chain stops here: this row's mask stays for the next token, the drafts are not accepted
        w.cut = e;
        live = false;
        m = nullptr;
      }
    }
    auto ok = [&](int k) { return !m || mask_allows(m, cand_e[k]); };
    int pick = 0;
    if (w.greedy) {
      while (!ok(pick)) ++pick;
      for (int k = pick + 1; k < K; ++k)
        if (ok(k) && sc[k] > sc[pick]) pick = k;  // ties to the lower index
      q[pick] = 1.0;
    } else {
      const double T = std::max(w.temperature, 1e-5);
      double mx = -std::numeric_limits<double>::infinity();
      for (int k = 0; k < K; ++k)
        if (ok(k)) mx = std::max(mx, double(sc[k]));
      double sum = 0;
      for (int k = 0; k < K; ++k) sum += q[k] = ok(k) ? std::exp((sc[k] - mx) / T) : 0.0;
      for (double& v : q) v /= sum;
      const double u = uni(rng);
      double cum = 0;
      pick = K - 1;
      while (pick > 0 && !ok(pick)) --pick;  // the rounding fallback: the last candidate the row allows
      for (int k = 0; k < K; ++k) {
        cum += q[k];
        if (u < cum) {
          pick = k;
          break;
        }
      }
    }
    w.drafts[e] = cand_e[pick];
    w.picks[e] = pick;
    prev = pick;
    if (con && live) {
      const int t = w.drafts[e];
      if (!armed) {
        armed = t == w.arm_token;
      } else if (!con->terminated()) {
        if (!con->accept(t)) throw ConstraintError("structured output: a draft the mask allows was refused");
        ++advanced;
        if (con->terminated()) {  // the grammar's end: nothing after this draft is accepted
          w.cut = e + 1;
          live = false;
        }
      }
    }
  }
  if (!con) return;
  fill_row(E);  // the row after the last draft (when the chain was not cut)
  con->rollback(advanced);
  if (w.masked)
    for (int e = 0; e <= E; ++e)
      if (!filled[e]) std::fill_n(w.masks + size_t(e) * w.words, w.words, int32_t(-1));
}

}  // namespace ragon
