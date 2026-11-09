#include "core/sampling.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <numeric>
#include <unordered_set>
#include <vector>

namespace ragon {

bool all_finite(std::span<const float> values) {
  return std::all_of(values.begin(), values.end(), [](float v) { return std::isfinite(v); });
}

void require_finite(std::span<const float> logits) {
  if (!all_finite(logits)) throw NonFiniteLogits();
}

void require_finite_topk(std::span<const float> vals, std::span<const int> ids, int vocab) {
  for (size_t i = 0; i < vals.size() && i < ids.size(); ++i)
    if (!std::isfinite(vals[i]) || vals[i] == -FLT_MAX || ids[i] < 0 || ids[i] >= vocab) throw NonFiniteLogits();
}

int require_finite_topk_masked(std::span<const float> vals, std::span<const int> ids, int vocab) {
  const size_t k = std::min(vals.size(), ids.size());
  size_t n = 0;
  while (n < k && !(vals[n] == -FLT_MAX && ids[n] == 0x7fffffff)) ++n;
  if (n == 0) throw NonFiniteLogits();
  require_finite_topk(vals.subspan(0, n), ids.subspan(0, n), vocab);
  for (size_t i = n; i < k; ++i)
    if (!(vals[i] == -FLT_MAX && ids[i] == 0x7fffffff)) throw NonFiniteLogits();
  return static_cast<int>(n);
}

int sample_logits(std::span<const float> logits, const SamplingParams& p, std::span<const int> output,
                  std::mt19937_64& rng) {
  require_finite(logits);
  const int V = static_cast<int>(logits.size());
  std::vector<float> l(logits.begin(), logits.end());
  if (p.presence_penalty != 0.f || p.repetition_penalty != 1.f) {
    std::unordered_set<int> seen(output.begin(), output.end());
    for (int t : seen) {
      if (t < 0 || t >= V) continue;
      if (p.repetition_penalty != 1.f) l[t] = l[t] > 0 ? l[t] / p.repetition_penalty : l[t] * p.repetition_penalty;
      l[t] -= p.presence_penalty;
    }
  }
  if (p.temperature <= 0.f) return static_cast<int>(std::max_element(l.begin(), l.end()) - l.begin());
  const int k = (p.top_k > 0 && p.top_k < V) ? p.top_k : V;
  std::vector<int> idx(V);
  std::iota(idx.begin(), idx.end(), 0);
  std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return l[a] > l[b]; });
  idx.resize(k);
  std::vector<float> top(k);
  for (int i = 0; i < k; ++i) top[i] = l[idx[i]];
  return sample_topk(top, idx, p, rng);
}

int sample_logits_masked(std::span<const float> logits, const SamplingParams& p, std::span<const int> output,
                         std::mt19937_64& rng, const int32_t* mask) {
  require_finite(logits);
  const int V = static_cast<int>(logits.size());
  std::vector<float> l(logits.begin(), logits.end());
  if (p.presence_penalty != 0.f || p.repetition_penalty != 1.f) {
    std::unordered_set<int> seen(output.begin(), output.end());
    for (int t : seen) {
      if (t < 0 || t >= V) continue;
      if (p.repetition_penalty != 1.f) l[t] = l[t] > 0 ? l[t] / p.repetition_penalty : l[t] * p.repetition_penalty;
      l[t] -= p.presence_penalty;
    }
  }
  std::vector<int> idx;
  for (int t = 0; t < V; ++t)
    if (mask_allows(mask, t)) idx.push_back(t);
  if (idx.empty()) throw ConstraintError("structured output: the grammar allows no token here");
  if (p.temperature <= 0.f) {
    int best = idx[0];
    for (int t : idx)
      if (l[t] > l[best]) best = t;  // ties to the lower id, like max_element
    return best;
  }
  const int n = static_cast<int>(idx.size());
  const int k = (p.top_k > 0 && p.top_k < n) ? p.top_k : n;
  std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return l[a] > l[b]; });
  idx.resize(k);
  std::vector<float> top(k);
  for (int i = 0; i < k; ++i) top[i] = l[idx[i]];
  return sample_topk(top, idx, p, rng);
}

int sample_topk(std::span<const float> vals, std::span<const int> ids, const SamplingParams& p, std::mt19937_64& rng) {
  const int k = static_cast<int>(vals.size());
  std::vector<double> probs(k);
  const double mx = vals[0];
  double sum = 0;
  for (int i = 0; i < k; ++i) sum += probs[i] = std::exp((vals[i] - mx) / p.temperature);
  for (double& pr : probs) pr /= sum;
  int keep = k;
  if (p.top_p < 1.f) {
    double cum = 0;
    for (int i = 0; i < k; ++i) {
      cum += probs[i];
      if (cum >= p.top_p) {
        keep = i + 1;
        break;
      }
    }
  }
  if (p.min_p > 0.f) {
    const double floor = probs[0] * p.min_p;
    int n = 0;
    while (n < keep && probs[n] >= floor) ++n;
    keep = std::max(n, 1);
  }
  std::discrete_distribution<int> dist(probs.begin(), probs.begin() + keep);
  return ids[dist(rng)];
}

}  // namespace ragon
