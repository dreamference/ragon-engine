// ragon-admission-tests: the start's memory plan with the KV page pool (serve/admission.hpp, P4d), on the host. The one
// plan the start check, auto-size and --plan-only share (start_plan), the line --plan-only prints (plan_json), and
// auto-size in paged mode (choose_size with KvPlanInputs). BATCHING §5.3 and §5.6, with §3.5's node-b figures.
// Exits 0 when everything holds.
#include <cstdio>
#include <nlohmann/json.hpp>

#include "serve/admission.hpp"

using namespace ragon::serve;

namespace {

int failures = 0;
void check(bool ok, const char* what) {
  if (!ok) {
    std::printf("FAIL: %s\n", what);
    ++failures;
  }
}

ragon::ModelConfig qwen() {  // Qwen3.8-27B's shapes (MODEL §5)
  ragon::ModelConfig c;
  c.layers = 64;
  c.kv_heads = 4;
  c.head_dim = 256;
  c.lin_k_heads = 16;
  c.lin_v_heads = 48;
  c.lin_k_dim = 128;
  c.lin_v_dim = 128;
  c.conv_kernel = 4;
  for (int i = 0; i < 64; ++i) c.full_attention.push_back(i % 4 == 3);
  return c;
}

// node-b (§3.5): 114.7 GB available, weights 23.47 GB with the drafter, 4 GB of scratch, the 16,384 MiB reserve.
const long kNodeB = static_cast<long>(114.7e9 / (1 << 20)), kReserve = 16384;
const size_t kWeights = size_t(23.47e9);

// --plan-only reports the plan the start runs: with --kv-pages the pool is in planned_bytes and in its own fields.
void plan_only() {
  const ragon::ModelConfig c = qwen();
  const KvPlanInputs paged{true, 0, -1};
  const StartPlan p = start_plan(c, kWeights, kScratchBytes, true, 4, 262144, kNodeB, kReserve, paged);
  check(p.refusal.empty(), "paged: four sessions at 262,144 start on node-b");
  check(p.plan.kv_pool > 0 && p.plan.kv_pool_pages > 0, "paged: the plan has a pool");
  const nlohmann::json j = nlohmann::json::parse(plan_json(4, 4, 262144, p.plan, true, kReserve));
  check(j["sessions"] == 4 && j["batch"] == 4 && j["max_context"] == 262144, "what runs");
  check(j["planned_bytes"].get<size_t>() == p.plan.total(), "planned_bytes is the start's plan");
  check(j["kv_pages"] == true && j["kv_pool_bytes"].get<size_t>() == p.plan.kv_pool &&
            j["kv_pool_pages"].get<long>() == p.plan.kv_pool_pages,
        "the pool's fields");
  check(j["planned_bytes"].get<size_t>() >= p.plan.kv_pool + kWeights, "planned_bytes counts the pool");
  check(j["host_reserve_bytes"].get<size_t>() == size_t(kReserve) << 20, "the reserve the pool leaves");
  check(j["kv_store_bytes"] == 0 && j["scratch_bytes"].get<size_t>() == kScratchBytes, "no store; the scratch");
  // P4f: the prefix KV store is allocated before the pool and counted in planned_bytes; the pool takes what remains.
  const KvPlanInputs paged_store{true, size_t(4096) << 20, -1};
  const StartPlan ps = start_plan(c, kWeights, kScratchBytes, true, 4, 262144, kNodeB, kReserve, paged_store);
  const nlohmann::json js = nlohmann::json::parse(plan_json(4, 4, 262144, ps.plan, true, kReserve));
  check(js["kv_store_bytes"].get<size_t>() == size_t(4096) << 20, "P4f: the store's field");
  check(js["planned_bytes"].get<size_t>() == ps.plan.total() &&
            ps.plan.total() >= kWeights + ps.plan.kv_pool + (size_t(4096) << 20),
        "P4f: planned_bytes counts the store");
  check(ps.plan.kv_pool_pages < p.plan.kv_pool_pages && p.plan.kv_pool_pages - ps.plan.kv_pool_pages <= 1025,
        "P4f: the pool gives the store its 1,024 pages");
  // The pool takes what the reserve leaves: planned + reserve is MemAvailable within a page (and the reserved page 0).
  const size_t have = size_t(kNodeB) << 20, planned = p.plan.total() + (size_t(kReserve) << 20);
  check(planned <= have && have - planned < 2 * kv_page_bytes(c), "the pool fills what the reserve leaves");
  // Without --kv-pages: the flat plan, no pool, and the same contract fields.
  const StartPlan f = start_plan(c, kWeights, kScratchBytes, true, 2, 65536, kNodeB, kReserve);
  const nlohmann::json g = nlohmann::json::parse(plan_json(2, 2, 65536, f.plan, false, kReserve));
  check(f.refusal.empty() && g["kv_pages"] == false && g["kv_pool_bytes"] == 0 && g["kv_pool_pages"] == 0,
        "flat: no pool");
  check(g["planned_bytes"].get<size_t>() == plan_memory(c, kWeights, 2, 65536, true).total(), "flat: the flat plan");
  check(plan_json(4, 8, 65536, f.plan, false, 0).find("\"batch\":4,") != std::string::npos, "batch never > sessions");
  // The start's refusals, in order: flat four at 262,144 do not fit node-b; a too-small auto pool is refused; an
  // explicit --kv-pool-mb is judged only by refuse_start; with no reserve nothing is judged.
  check(!start_plan(c, kWeights, kScratchBytes, true, 4, 262144, kNodeB, kReserve).refusal.empty(), "flat 4: refused");
  check(start_plan(c, kWeights, kScratchBytes, true, 4, 262144, 50000, kReserve, paged).refusal.find("KV page pool") !=
            std::string::npos,
        "paged on a small host: the pool's refusal");
  check(start_plan(c, kWeights, kScratchBytes, true, 4, 262144, kNodeB, kReserve, {true, 0, 8192}).refusal.empty(),
        "--kv-pool-mb 8192 fits");
  check(!start_plan(c, kWeights, kScratchBytes, true, 4, 262144, kNodeB, kReserve, {true, 0, 80000}).refusal.empty(),
        "--kv-pool-mb 80000 is refused");
  check(start_plan(c, kWeights, kScratchBytes, true, 8, 262144, 1000, 0, paged).refusal.empty(), "no reserve: no check");
}

// Auto-size with --kv-pages: given sessions fit when the pool holds one conversation at max_context and n x max_context / 2
// positions (kPagedFitShare); sessions chosen here ask for the full pool, n x max_context positions (kAutoPagedFitShare,
// P4e), so an auto-sized paged start picks what flat picks.
void paged_auto_size() {
  const ragon::ModelConfig c = qwen();
  const size_t vis = kScratchBytes + (size_t(1100 + 512) << 20);  // images on, the default 512 MB cache
  const KvPlanInputs paged{true, 0, -1}, paged_store{true, size_t(1024) << 20, -1};
  auto choose = [&](long avail, size_t scratch, const KvPlanInputs& kv, std::optional<int> s = {},
                    std::optional<int> ctx = {}) {
    return choose_size(c, kWeights, scratch, true, avail, kReserve, 262144, s, ctx, true, kv);
  };
  // §3.5's two node-b figures: 114.7 GB available, and 109,423 MiB with nothing else on the GPU (api_tests).
  for (const long avail : {kNodeB, 109423L})
    for (const size_t scratch : {kScratchBytes, vis}) {
      const SizeChoice flat = choose(avail, scratch, {});
      const SizeChoice p = choose(avail, scratch, paged), q = choose(avail, scratch, paged_store);
      const StartPlan sp = start_plan(c, kWeights, scratch, true, p.sessions, p.max_context, avail, kReserve, paged);
      std::printf("avail %ld MiB, images %s: flat %d x %d; paged %d x %d (pool %s); store 1 GB: %d x %d\n", avail,
                  scratch == vis ? "on" : "off", flat.sessions, flat.max_context, p.sessions, p.max_context,
                  describe_kv_pool(sp.plan).c_str(), q.sessions, q.max_context);
      check(flat.max_context == 262144 && flat.sessions == 3, "flat: three sessions at 262,144 on node-b");
      check(p.sized && p.sessions_auto && p.context_auto, "paged: chosen here");
      check(p.sessions == 3 && p.max_context == 262144, "paged: three sessions at 262,144 on node-b, as flat");
      check(q.sessions == 3 && q.max_context == 262144, "paged with a 1 GB store: three at 262,144");
      check(sp.plan.kv_pool_pages == 3L * 4096, "paged: the pool is the three sessions' full context");
      check(sp.refusal.empty(), "paged: the choice starts");
    }
  // P4f: the start's measured terms (the engine's fixed part, the vision tower) and a 4 GiB store: still three at
  // 262,144 with nothing else on the GPU, and the plan leaves the reserve.
  {
    const size_t honest = kScratchBytes + kEngineFixedBytes + (size_t(1100 + 512) << 20) + kVisionTowerBytes;
    const KvPlanInputs store4{true, size_t(4096) << 20, -1};
    const SizeChoice h = choose(109423L, honest, store4);
    const StartPlan hp = start_plan(c, kWeights, honest, true, h.sessions, h.max_context, 109423L, kReserve, store4);
    std::printf("measured terms and a 4 GiB store: %d x %d, planned %s\n", h.sessions, h.max_context,
                gb(hp.plan.total()).c_str());
    check(h.sessions == 3 && h.max_context == 262144, "P4f: three at 262,144 with the measured terms and the store");
    check(hp.refusal.empty() && static_cast<long>(hp.plan.total() >> 20) + kReserve <= 109423L,
          "P4f: and the reserve is left");
  }
  // The rule bites: a host whose pool holds one conversation at 262,144 but not two runs one; four given still start.
  // Pool pages = (avail - weights - n x ~0.5 GB - scratch - reserve) / 4 MiB; 8,192 pages are 32 GiB.
  const long tight = 16384 + 22383 + 4096 + 2048 + 24576;  // reserve, weights, scratch, sessions' fixed, 24 GiB pool
  const SizeChoice t = choose(tight, kScratchBytes, paged);
  const StartPlan tp = start_plan(c, kWeights, kScratchBytes, true, t.sessions, t.max_context, tight, kReserve, paged);
  std::printf("avail %ld MiB: paged %d x %d (pool %s)\n", tight, t.sessions, t.max_context,
              describe_kv_pool(tp.plan).c_str());
  check(t.max_context == 262144 && t.sessions == 1, "a smaller pool: one session at 262,144");
  check(tp.plan.kv_pool_pages >= long(t.sessions) * 4096 / kAutoPagedFitShare, "the chosen sessions satisfy the rule");
  check(!paged_sessions_fit(start_plan(c, kWeights, kScratchBytes, true, 2, 262144, tight, kReserve, paged).plan, 2,
                            262144, kAutoPagedFitShare),
        "two do not fill their full context");
  const SizeChoice tg = choose(tight, kScratchBytes, paged, 4);
  check(tg.sessions == 4 && tg.max_context == 131072, "four given on the smaller pool: 131,072 at half the pool (as before)");
  check(!paged_sessions_fit(start_plan(c, kWeights, kScratchBytes, true, 4, 262144, tight, kReserve, paged).plan, 4,
                            262144),
        "four do not");
  check(start_plan(c, kWeights, kScratchBytes, true, 4, 262144, tight, kReserve, paged).refusal.empty(),
        "but four given at 262,144 still start (the scheduler requeues)");
  // Given values: both given is a check only; given sessions take the longest context they fit at.
  const SizeChoice g = choose(kNodeB, kScratchBytes, paged, 4, 262144);
  check(g.sessions == 4 && g.max_context == 262144 && !g.sessions_auto && !g.context_auto, "both given: as given");
  const SizeChoice eight = choose(kNodeB, kScratchBytes, paged, 8);
  std::printf("eight given on node-b: %d x %d\n", eight.sessions, eight.max_context);
  check(eight.sessions == 8 && eight.max_context == 131072, "eight given: 131,072 (eight halves of 262,144 do not fit the pool)");
  // One session, paged: its pool is capped at its own max_context and is never "too small" for it.
  const SizeChoice one = choose(kNodeB, kScratchBytes, paged, 1);
  const StartPlan op = start_plan(c, kWeights, kScratchBytes, true, 1, 262144, kNodeB, kReserve, paged);
  check(one.sessions == 1 && one.max_context == 262144, "one session given: 262,144");
  check(op.plan.kv_pool_pages == 4096 && op.refusal.empty(), "one session at 262,144 starts with 4,096 pages");
  // Without --kv-pages auto-size is unchanged (the flat plan).
  check(choose(kNodeB, kScratchBytes, {}).sessions == 3, "flat unchanged");
}

}  // namespace

int main() {
  plan_only();
  paged_auto_size();
  if (failures) {
    std::printf("%d check(s) failed\n", failures);
    return 1;
  }
  std::printf("admission tests passed\n");
  return 0;
}
