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
  std::vector<char> filled;  // the rows that hold the constraint's own mask (the others all-ones when masked)
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
  std::vector<char>& filled = w.filled;
  filled.assign(con ? E + 1 : 0, 0);
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

// A fixed proposal under the constraint (SPECULATION "Structured output, stage 3"): the context lookup's chain, q = 1.
// Row e's mask is the state after the anchor and tokens[0, e); the proposal is cut at its first token the mask
// refuses (cut = e, that row's mask stays for the token after), or after the token that ends the grammar (cut = e + 1).
// The constraint is rolled back; the rows after the cut hold all-ones masks when any row is masked.
struct FixedWalk {
  const int* tokens = nullptr;  // E proposed tokens; masks has E + 1 rows of words
  int E = 0;
  TokenConstraint* con = nullptr;
  bool armed = false;
  int arm_token = -1;
  int32_t* masks = nullptr;
  int words = 0;
  int cut = 0;
  bool masked = false;
};

inline void walk_fixed(FixedWalk& w) {
  const int E = w.E;
  TokenConstraint* con = w.con;
  w.cut = E;
  w.masked = false;
  std::vector<char> filled(E + 1, 0);
  bool armed = w.armed;
  int advanced = 0;
  for (int e = 0; e <= E; ++e) {
    const int32_t* m = nullptr;
    if (armed && !con->terminated()) {
      int32_t* row = w.masks + size_t(e) * w.words;
      con->fill(row);
      filled[e] = 1;
      w.masked = true;
      m = row;
    }
    if (e == E) break;
    const int t = w.tokens[e];
    if (!armed) {
      armed = t == w.arm_token;
      continue;
    }
    if (!m) continue;  // the grammar ended before the step: as the chain walk, nothing to mask
    if (!mask_allows(m, t)) {
      w.cut = e;
      break;
    }
    if (!con->accept(t)) throw ConstraintError("structured output: a draft the mask allows was refused");
    ++advanced;
    if (con->terminated()) {
      w.cut = e + 1;
      break;
    }
  }
  con->rollback(advanced);
  if (w.masked)
    for (int e = 0; e <= E; ++e)
      if (!filled[e]) std::fill_n(w.masks + size_t(e) * w.words, w.words, int32_t(-1));
}

// The tree's masks (SPECULATION "Structured output, stage 3"): a depth-first walk from the anchor, the constraint
// accepting each node's token on the way down and rolling it back on the way up, so each row's mask is the state
// after its node's path (the anchor's row: the state now). A child whose token its parent's mask refuses is dead,
// as is every child of a node that ends the grammar; accept_tree_* never enter a dead node, and the rows below one are
// never visited (all-ones). The chain's rows 1 .. E reuse the chain walk's masks (chain_filled, the same states).
struct TreeMaskWalk {
  const std::vector<int>* token = nullptr;                // per tree row
  const std::vector<std::vector<int>>* children = nullptr;  // per tree row
  int E = 0;                                              // the tree's chain rows 1 .. E
  const std::vector<char>* chain_filled = nullptr;        // the chain walk's filled rows (masks rows 0 .. E hold them)
  TokenConstraint* con = nullptr;
  bool armed = false;
  int arm_token = -1;
  int32_t* masks = nullptr;  // one row per tree row
  int words = 0;
  std::vector<char> dead;  // out: per tree row
  bool masked = false;
};

namespace detail {
inline void tree_visit(TreeMaskWalk& w, std::vector<char>& filled, int u, bool armed) {
  TokenConstraint* con = w.con;
  const int32_t* m = nullptr;
  if (armed && !con->terminated()) {
    int32_t* row = w.masks + size_t(u) * w.words;
    const bool reuse = u <= w.E && w.chain_filled && size_t(u) < w.chain_filled->size() && (*w.chain_filled)[u];
    if (!reuse) con->fill(row);
    filled[u] = 1;
    w.masked = true;
    m = row;
  }
  for (int c : (*w.children)[u]) {
    const int t = (*w.token)[c];
    if (!armed) {
      tree_visit(w, filled, c, t == w.arm_token);
      continue;
    }
    if (!m || !mask_allows(m, t)) {  // after the grammar's end, or a token the grammar refuses here
      w.dead[c] = 1;
      continue;
    }
    if (!con->accept(t)) throw ConstraintError("structured output: a draft the mask allows was refused");
    tree_visit(w, filled, c, true);
    con->rollback(1);
  }
}
}  // namespace detail

inline void walk_tree(TreeMaskWalk& w) {
  const int n = static_cast<int>(w.token->size());
  w.dead.assign(n, 0);
  w.masked = false;
  std::vector<char> filled(n, 0);
  detail::tree_visit(w, filled, 0, w.armed);
  if (w.masked)
    for (int u = 0; u < n; ++u)
      if (!filled[u]) std::fill_n(w.masks + size_t(u) * w.words, w.words, int32_t(-1));
}

}  // namespace ragon
