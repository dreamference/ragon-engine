// ragon-serve: the HTTP front of ragon-engine (proxygen). One engine thread (the scheduler) runs up to --batch
// requests at once, each in its own resident session, taking turns one prefill chunk or decode step at a
// time; the HTTP threads parse requests and stream server-sent events back.
//
//   ragon-serve --model DIR [--host 0.0.0.0] [--port 8000] [--served-model-name NAME] [--max-context N]
//              [--draft DIR [--drafter ID=DIR ...] [--drafter-routes FILE] [--draft-block N] [--lookup 0|1|2] [--lookup-min N] [--draft-off-below X]] [--graphs] [--pdl]
//              [--attn-bulk] [--attn-prefetch N] [--keepalive-seconds N] [--prefill-bf16]
//              [--sessions N] [--session-protect-seconds S] [--batch N] [--prefill-chunk-contended N]
//              [--side-prefill N] [--side-share N] [--fair-pairing 1|0] [--host-reserve-mb N] [--admit-reserve-mb N]
//              [--auto-size 1|0] [--draft-bf16] [--repetition-detection P,p,R,K|off] [--repetition-report incomplete|completed]
//              [--kv-pages [--kv-pool-mb N] | --no-kv-pages]
//              [--fanout-plain-above N] [--fanout-gang-above N] [--fanout-default on|off] [--fanout-draft-block N]
//
// --sessions N (default: chosen at start, --auto-size below; 2 without it; M4 §1) keeps N resident sessions, each with
// its own KV cache and state, so N agent tasks that alternate keep their histories warm. A request runs in the session
// it extends (its prompt_cache_key's session is looked at first; the token prefix decides), else in a new session,
// which takes a free slot or evicts the least recently used one. A session bound with a prompt_cache_key whose last
// request ran less than --session-protect-seconds ago (default 30, 0 off; the owner's decision of 2026-11-01) is never
// evicted: a new session's request waits in the queue instead, while requests that continue a resident session run.
//
// --batch N (default: the sessions when they are chosen at start, else 2; at most --sessions; M4 §3, stage 2) requests
// run at once, each in its own session, interleaved: the running requests take turns, one prefill chunk or one decode
// step each. While another request decodes, a prefill runs in chunks of --prefill-chunk-contended N tokens (default
// 256, a multiple of 32 up to the prefill chunk of 2,048), so the decoding request's step period stays near twice its
// normal step during a long prefill. A request that continues a running request's session waits for it. Nothing numeric
// changes: every unit runs in its own session's state exactly as it would alone.
//
// --side-prefill N (drafts/M4-prefill-beside-decode.md, option B; default 40, 0 off): while another
// request decodes, a prefill runs in full chunks on the engine's second stream, on a green context of N SMs (-1: a
// plain second stream), at the same time as the other request's steps on the main stream, instead of alternating
// contended chunks with them. The same kernels on the same inputs (ragon-run --interleave-check --side-prefill N).
// --side-share N (BATCHING §5.5, C1; default 16, 0 off): while two or more other requests decode, the side stream
// waits after each prefill chunk (any request's) until N decode passes have run on the main stream alone, so the
// decoding conversations take some steps at their speed alone instead of every step beside a chunk. At four warm
// agent conversations (node-b, 2026-11-04) 16 raised the pooled decode from 25.7 to 35.6 tok/s with the rounds'
// time unchanged (139.3 / 140.9 s). Only the order of units changes, as with the shortest prefill first.
// --fair-pairing 1|0 (C3, design A1; default 0; RAGON_FAIR_PAIRING overrides): the request that joins a batched step
// is the one with the fewest decode passes in the trailing 16 (serve/pairing.hpp), not the first in running order,
// so at four decoding conversations each takes 2 passes in 4 instead of 4/2/1/1. Only which requests step together
// changes. RAGON_STEP_TRACE=FILE writes one JSON line per scheduler unit (serve/step_trace.hpp).
//
// --draft-off-below X (BACKLOG 10; default 1.5, 0 off; RAGON_DRAFT_OFF_BELOW overrides it): the drafter-off guard. A
// session whose running mean of tokens per speculative step falls below X decodes one plain row per step, with no
// drafter pass, and probes the drafter once every 16 plain steps until the mean is back at X + 0.25
// (core/draft_guard.hpp). Greedy output is the same token for token; counted in /spec_stats (drafter_off_steps,
// drafter_switches) and /metrics (ragon:drafter_off_steps_total, ragon:drafter_switches_total).
//
// Fan-out (BATCHING P4d and §5.8): --fanout-plain-above N (RAGON_FANOUT_PLAIN_ABOVE; default 6 with a drafter, 1
// without, 0 never) has every decoding request take plain batched steps above N decoding requests; --fanout-gang-above
// N (RAGON_FANOUT_GANG_ABOVE; default the plain threshold) runs the short question prefills together above N live
// requests. --fanout-default on|off (RAGON_FANOUT_DEFAULT 1/0; default on, stage 11): with a drafter, above 7
// drafting requests every one speculates with the gang block, --fanout-draft-block N (RAGON_FANOUT_DRAFT_BLOCK;
// default 2, 0 for --draft-block, at most --draft-block), and the plain threshold holds there only when
// --fanout-plain-above is given; at 7 or fewer nothing changes. Off is the server before stage 11. Greedy output is
// the same token for token either way.
//
// --graphs runs each speculative step's draft, verify and commit as captured CUDA graphs (M3); --pdl launches
// the rows path's kernels with programmatic dependent launch (M3); --attn-bulk loads the rows path's attention
// tiles with bulk copies (M3); --attn-prefetch N sets how many KV tiles ahead its attention prefetches into L2
// (default 2, 0 off; M3). --keepalive-seconds N (default 15, 0 off; RAGON_KEEPALIVE_SECONDS overrides it) is
// how often a streamed request that has produced no token yet is sent a keep-alive (API §3, §4).
// --prefill-bf16 keeps the prefill's DeltaNet intermediates in BF16 as production does (B34; not exact, off
// by default: MILESTONES §14).
// With --kv-pages the full-attention KV is kept in 64-position pages behind a page table per session (BATCHING
// §5.3, P4; off by default, the flat slots, --no-kv-pages says so explicitly; exact: only addresses change;
// RAGON_KV_PAGES=0/1 overrides). The pages come from one pool allocated
// at start: what MemAvailable leaves after the memory plan, the prefix KV store's budget and --host-reserve-mb, at most
// --sessions x --max-context positions; or --kv-pool-mb N, refused at start when the plan with it does not fit. A
// session then costs its fixed part (~0.5 GB) and 64 KB for each position it holds, so --sessions 4 --max-context
// 262144 starts on a 128 GB node.
// --draft-bf16 reads the drafter's fc and conv kernel projections in BF16, as its file has them, instead of the
// NVFP4 copies rounded at load (BACKLOG 33; the default since 2026-11-03; RAGON_DRAFT_NVFP4=0/1 overrides). Only the
// drafter's proposals change, so only acceptance can move: the verify decides every token either way.
//
// --repetition-detection P,p,R,K (API §9; default 256,2,12,64; `off` or 0,0,0,0 switches it off) is the repetition
// stop rule a request runs with unless its `repetition_detection` says otherwise: an output (reasoning included) that
// ends in a block of p .. P tokens repeated R times, or in one token repeated K times, ends with finish reason
// "repetition". --repetition-report completed reports such a stop on /v1/responses as a completed response with
// `stop_reason: "repetition"` instead of the default `response.incomplete` (reason "repetition"), which Codex retries.
//
// --host-reserve-mb N (default 16384; RAGON_HOST_RESERVE_MB overrides) and --admit-reserve-mb N (default 8192;
// RAGON_ADMIT_RESERVE_MB overrides) are the admission control on host memory (serve/admission.hpp; BATCHING §5.6,
// phase 0): ragon-serve refuses to start (exit 3, before allocating anything) when its weights, sessions and scratch
// would leave less than the first of MemAvailable, and answers a generation request HTTP 503 (code host_memory_low)
// while MemAvailable is below the second. 0 switches either off. Neither changes what an admitted request computes.
//
// --drafter ID=DIR (repeatable) and --drafter-routes FILE (MULTI_DRAFTER §5): resident drafters beside `--draft DIR`,
// which is `general` (`--drafter general=DIR` says the same). All are loaded at start, counted by the start plan, and
// must share general's shape. The routes file is the workload table, {"work":"work","ask":"ask","*":"general"}; without
// one every request goes to general. A request's drafter comes from its hint (`"ragon": {"workload": ...}`, or
// `ragon_workload` in Codex's x-codex-turn-metadata), else from its tools and last item. The output never depends on
// it. The response says which in `x-ragon-drafter`; /spec_stats has per-drafter counters (`drafters`), how each was
// chosen (`drafter_chosen`) and the resident sessions taken over to another drafter (`drafter_changes`); /metrics
// labels ragon:spec_drafted_tokens_total and ragon:spec_accepted_tokens_total with `drafter`.
//
// --plan-only: print the start plan (sessions, batch, context, planned_bytes; with paged KV the page pool, which
// planned_bytes includes, and the host reserve) as one JSON line on stdout and exit 0 before
// anything is allocated; the launcher's pre-load host-safety check runs it so it checks the engine's own plan.
// --auto-size 1|0 (default 1; RAGON_AUTO_SIZE overrides; BATCHING §5.6, the owner's decision of 2026-11-04): what is
// not given of --sessions and --max-context is chosen at start from MemAvailable by the same plan the start check
// uses: the model's full context (config.json's max_position_embeddings) and as many sessions as fit, at most 4; the
// context is halved, down to 32,768, only when one session does not fit. --batch, when not given, is then the chosen
// sessions. Given values are kept; with both given, or --auto-size 0, or --host-reserve-mb 0, or MemAvailable
// unreadable, the defaults are --sessions 2, --batch 2, --max-context 65,536 as before. One start line says what
// was chosen and why. Placement only: a server sized so computes what the same three flags given by hand compute.
//
// With --draft, requests decode speculatively with the DFlash2 drafter (requests that set penalties
// decode plainly). GET /metrics exports SGLang's counter names, so M0's replay harness reads them as is.
#include <folly/SocketAddress.h>
#include <folly/init/Init.h>
#include <folly/io/IOBufQueue.h>
#include <folly/io/async/EventBase.h>
#include <folly/io/async/EventBaseManager.h>
#include <proxygen/httpserver/HTTPServer.h>
#include <proxygen/httpserver/RequestHandler.h>
#include <proxygen/httpserver/RequestHandlerFactory.h>
#include <proxygen/httpserver/ResponseBuilder.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <fstream>
#include <iterator>
#include <map>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>

#include "core/engine.hpp"
#include "serve/admission.hpp"
#include "serve/api.hpp"
#include "tokenizer/tokenizer.hpp"
#include "serve/image_input.hpp"
#include "serve/kv_policy.hpp"
#include "serve/pairing.hpp"
#include "serve/step_trace.hpp"

namespace ragon::serve {
namespace {

using proxygen::HTTPMessage;
using proxygen::ResponseBuilder;

// Prometheus counters under SGLang's names (the ones bench/replay/replay.py reads).
struct Metrics {
  std::mutex mu;
  double prompt_tokens = 0, cached_tokens = 0, generation_tokens = 0, verify_calls = 0;
  double e2e_sum = 0, e2e_count = 0, ttft_sum = 0, ttft_count = 0;
  double accepted_drafts = 0, drafted = 0;
  // Per resident drafter (MULTI_DRAFTER §2.5), by id; empty without a drafter. The ids are entered before the server
  // starts, so every series exists from zero.
  struct DrafterCounters {
    double requests = 0, steps = 0, drafted = 0, accepted = 0, draft_seconds = 0, commit_seconds = 0;
  };
  std::map<std::string, DrafterCounters> by_drafter;
  // A resident session taken over to another drafter (§2.3): how many, and those requests' prefills (the new turn's
  // tokens included). Not the drafter-off guard's `drafter_switches`.
  double drafter_changes = 0, drafter_change_tokens = 0, drafter_change_seconds = 0;
  double chosen_declared = 0, chosen_inferred = 0, chosen_default = 0;
  double nonfinite = 0;  // requests ended because the logits were not finite
  double repetition_stops = 0;  // requests ended by the repetition stop rule (API §9)
  double memory_refusals = 0;   // requests answered 503 because host memory was below the admission reserve
  SpecStats spec;  // the engine's speculation counters after the last request
  // Gauges the benchmark harness reads (its admission and its parallelism): requests running (up to
  // --batch at once, M4 §3) and queued, and the token pool a night's tasks are budgeted
  // against: the resident sessions times the context size (M4 §1, SESSIONS §11).
  int running = 0, queued = 0, max_total_tokens = 0;
  // The scheduler (M4 §3): requests that run at once, the prefill chunk size beside a decode, and the
  // prefill chunks run between another request's steps so far.
  int batch = 0, prefill_chunk_contended = 0;
  double prefill_chunks_interleaved = 0;
  // Stage 3: the rows of the last decode pass (a batched verify's sum over its sequences, or one sequence's), and
  // the batched steps so far (one per batch, whatever its size).
  int batch_rows = 0;
  double batched_steps = 0;
  // The session pool (M4 §1): slots, slots holding a session, evictions so far, the protection window.
  int sessions = 0, sessions_resident = 0;
  double session_evictions = 0, protect_seconds = 0;
  // P4b (--kv-pages): the KV page pool's pages and free pages, and the requests ended by its last resort.
  int kv_pages_total = 0, kv_pages_free = 0;
  double kv_pool_exhausted = 0, kv_evictions = 0;

  json spec_json() {
    std::lock_guard<std::mutex> l(mu);
    json j = {{"steps", spec.steps}, {"drafted", spec.drafted}, {"accepted", spec.accepted},
              {"accept_histogram", spec.accept_hist}, {"draft_seconds", spec.draft_seconds},
              {"verify_seconds", spec.verify_seconds}, {"commit_seconds", spec.commit_seconds},
              {"lookup_steps", spec.lookup_steps}, {"lookup_accepted", spec.lookup_accepted},
              {"batched_steps", spec.batched_steps}, {"fanout_steps", spec.fanout_steps}, {"tree_steps", spec.tree_steps},
              {"tree_branch_steps", spec.tree_branch_steps}, {"tree_off_chain", spec.tree_off_chain},
              {"sparse_steps", spec.sparse_steps}, {"sparse_fallbacks", spec.sparse_fallbacks},
              {"drafter_off_steps", spec.drafter_off_steps}, {"drafter_switches", spec.drafter_switches},
              {"ride_steps", spec.ride_steps}, {"shadow_total_steps", spec.shadow_total_steps}, {"shadow_dflash_all", spec.shadow_dflash_all},
              {"shadow_best", spec.shadow_best}};
    for (int b = 0; b < 5; ++b) {
      j["shadow_steps"].push_back(spec.shadow_steps[b]);
      j["shadow_lookup"].push_back(spec.shadow_lookup[b]);
      j["shadow_dflash"].push_back(spec.shadow_dflash[b]);
    }
    if (!by_drafter.empty()) {
      json per = json::object();
      for (const auto& [id, c] : by_drafter)
        per[id] = {{"requests", c.requests}, {"steps", c.steps}, {"drafted", c.drafted}, {"accepted", c.accepted},
                   {"draft_seconds", c.draft_seconds}, {"commit_seconds", c.commit_seconds}};
      j["drafters"] = per;
      j["drafter_changes"] = {{"count", drafter_changes}, {"prefill_tokens", drafter_change_tokens},
                              {"prefill_seconds", drafter_change_seconds}};
      j["drafter_chosen"] = {{"declared", chosen_declared}, {"inferred", chosen_inferred}, {"default", chosen_default}};
    }
    return j;
  }

  std::string render(const std::string& model) {
    std::lock_guard<std::mutex> l(mu);
    const std::string lab = "{model_name=\"" + model + "\"}";
    std::string out;
    auto counter = [&](const char* name, double v, const char* help) {
      out += std::string("# HELP ") + name + " " + help + "\n# TYPE " + name + " counter\n";
      char buf[64];
      std::snprintf(buf, sizeof buf, "%.6f", v);
      out += std::string(name) + lab + " " + buf + "\n";
    };
    counter("sglang:prompt_tokens_total", prompt_tokens, "Prompt tokens.");
    counter("sglang:cached_tokens_total", cached_tokens, "Prompt tokens served from the cache.");
    counter("sglang:generation_tokens_total", generation_tokens, "Generated tokens.");
    counter("sglang:spec_verify_calls_total", verify_calls, "Decode passes (speculative verify steps, or plain steps).");
    counter("sglang:e2e_request_latency_seconds_sum", e2e_sum, "Request latency, sum.");
    counter("sglang:e2e_request_latency_seconds_count", e2e_count, "Requests.");
    counter("sglang:time_to_first_token_seconds_sum", ttft_sum, "Time to first token, sum.");
    counter("sglang:time_to_first_token_seconds_count", ttft_count, "Time to first token, count.");
    // With a drafter loaded these two carry its id (MULTI_DRAFTER §2.5), one series per resident drafter.
    auto per_drafter = [&](const char* name, double DrafterCounters::*field, double total, const char* help) {
      if (by_drafter.empty()) return counter(name, total, help);
      out += std::string("# HELP ") + name + " " + help + "\n# TYPE " + name + " counter\n";
      for (const auto& [id, c] : by_drafter) {
        char buf[64];
        std::snprintf(buf, sizeof buf, "%.6f", c.*field);
        out += std::string(name) + "{model_name=\"" + model + "\",drafter=\"" + id + "\"} " + buf + "\n";
      }
    };
    per_drafter("ragon:spec_drafted_tokens_total", &DrafterCounters::drafted, drafted,
                "Drafted tokens offered to the verify.");
    per_drafter("ragon:spec_accepted_tokens_total", &DrafterCounters::accepted, accepted_drafts,
                "Drafted tokens accepted.");
    if (!by_drafter.empty())
      counter("ragon:drafter_changes_total", drafter_changes, "Resident sessions taken over to another drafter.");
    counter("ragon:nonfinite_logits_total", nonfinite, "Requests ended because the logits were NaN or inf.");
    counter("ragon:repetition_stops_total", repetition_stops, "Requests ended by the repetition stop rule.");
    counter("ragon:memory_refusals_total", memory_refusals,
            "Generation requests answered 503 because host memory was below --admit-reserve-mb.");
    counter("ragon:session_evictions_total", session_evictions, "Resident sessions replaced by a new session.");
    counter("ragon:prefill_chunks_interleaved_total", prefill_chunks_interleaved,
            "Prefill chunks run between another request's decode steps.");
    counter("ragon:batched_steps_total", batched_steps,
            "Decode steps that verified two or more requests' rows in one pass.");
    if (kv_pages_total > 0) {
      counter("ragon:kv_pool_exhausted_total", kv_pool_exhausted,
              "Requests ended at finish_reason length because the KV page pool was exhausted (the last resort).");
      counter("ragon:kv_session_evictions_total", kv_evictions, "Idle sessions whose KV pages the pool took back.");
    }
    counter("ragon:drafter_off_steps_total", spec.drafter_off_steps,
            "Plain one-row steps taken because the drafter-off guard had switched a session's drafter off.");
    counter("ragon:drafter_switches_total", spec.drafter_switches,
            "Times the drafter-off guard switched a session's drafter off or back on.");
    auto gauge = [&](const char* name, double v, const char* help) {
      out += std::string("# HELP ") + name + " " + help + "\n# TYPE " + name + " gauge\n";
      char buf[64];
      std::snprintf(buf, sizeof buf, "%.6f", v);
      out += std::string(name) + lab + " " + buf + "\n";
    };
    gauge("sglang:num_running_reqs", running, "Requests being prefilled or generated (at most --batch).");
    gauge("sglang:num_queue_reqs", queued, "Requests waiting for the engine.");
    gauge("sglang:max_total_num_tokens", max_total_tokens,
          "Tokens the resident sessions can hold (sessions times the context size).");
    gauge("ragon:sessions", sessions, "Resident session slots.");
    if (kv_pages_total > 0) {
      gauge("ragon:kv_pages_total", kv_pages_total, "Pages of the KV page pool (64 positions each, --kv-pages).");
      gauge("ragon:kv_pages_free", kv_pages_free, "Free pages of the KV page pool.");
    }
    gauge("ragon:sessions_resident", sessions_resident, "Slots holding a session.");
    gauge("ragon:batch", batch, "Requests that run at once (each in its own session).");
    gauge("ragon:prefill_chunk_contended", prefill_chunk_contended,
          "Prefill chunk size, in tokens, while another request decodes.");
    gauge("ragon:batch_rows", batch_rows, "Rows of the last decode pass (all its requests' rows when batched).");
    gauge("ragon:session_protect_seconds", protect_seconds,
          "How long a keyed session is protected from eviction after its last request.");
    return out;
  }
};

struct ServerContext {
  Engine* engine = nullptr;
  ragon::serve::DrafterRoutes drafter_routes;  // --drafter-routes (MULTI_DRAFTER §5.2)
  Tokenizer* tok = nullptr;
  std::string model_name;
  int im_end = -1;
  // The token ids that end the thinking span: `</think>`, then the blank line the template puts after it
  // ("\n\n" as one token, or "\n" twice). -1 when the tokenizer has no such single token.
  int think_end = -1, newline = -1, blank_line = -1;
  // How often a streamed request that has produced no token yet is sent a keep-alive (0: never).
  int keepalive_ms = 15000;
  // The repetition stop rule's default (--repetition-detection), and whether /v1/responses reports its stop as a
  // completed response (--repetition-report completed) rather than response.incomplete.
  RepetitionDetection repetition = kRepetitionDefault;
  bool repetition_completed = false;
  // Admission control (serve/admission.hpp): a generation request is refused with HTTP 503 while MemAvailable is
  // below this many MB (0: never).
  long admit_reserve_mb = 8192;
  Metrics metrics;
  // Images (VISION §20.3 (c)): on when the engine has the tower; <|image_pad|>'s id, each image's id in prompts, the
  // processor's settings.
  bool vision = false;
  int image_pad = -1;
  ImageRegistry* images = nullptr;
  ProcessorConfig processor;
  // LoRA adapters (LORA §4): the loaded set (nullptr: no --lora-library), read and changed on the HTTP threads under
  // lora_mu; the GPU slots, the engine thread's only; the key the adapter routes ask for (empty: they are off).
  LoraLibrary* lora_lib = nullptr;
  std::mutex lora_mu;
  LoraSlots* lora_slots = nullptr;
  std::string api_key;
};

// What a job reports back, called on the engine thread.
struct JobSink {
  std::function<void(const std::string&)> text;
  // reasoning_tokens: the thinking span's share of completion_tokens, counted from the token ids as they
  // are produced (up to and including `</think>` and the blank line after it; every token when the answer
  // was cut inside the span; 0 with thinking off), so an accepted draft block that crosses `</think>` is
  // split exactly and the count never exceeds completion_tokens (vllm#49711, sglang#39826).
  std::function<void(const std::string& finish, int prompt_tokens, int completion_tokens, int cached_tokens,
                     int reasoning_tokens)>
      done;
  // The message, and the code the client sees (server_error, or context_length_exceeded).
  std::function<void(const std::string& message, const std::string& code)> error;
  // Per produced token, only when the request asked for logprobs: the token's own logprob first, then the
  // top k (top_logprobs(), api.hpp). Called before the token's text, which the decoder or the parser may
  // still hold back.
  std::function<void(const std::vector<TokenLogprob>& entries)> logprobs;
};

struct Job {
  std::chrono::steady_clock::time_point received = std::chrono::steady_clock::now();
  Request req;
  std::vector<int> prompt;
  std::vector<PromptImage> images;  // the prompt's images (VISION §20.3 (c))
  JobSink sink;
  std::shared_ptr<std::atomic<bool>> cancelled;
  int session = -1;    // the session it runs in (bound when the scheduler admits it)
  bool fresh = false;  // a new session in that slot
  // Its adapter (nullptr: the base), held until it ends even if the adapter is unloaded meanwhile; the GPU slot it
  // holds from admission on (-1: none), and whether the slot still has to be filled with the adapter's weights.
  std::shared_ptr<const LoraAdapter> lora;
  int lora_slot = -1;
  bool lora_upload = false;
  // BATCHING P4d, a request's other choices (`n`): each starts as a fork of this job's session when its prompt is done.
  std::vector<Job> forks;
  bool fork = false;  // one of those: never continues a session (the identical prompt would wait for its parent)
};

// P4b (--kv-pages): a prompt whose KV pages (with the two of growth it is admitted with) exceed the whole page pool
// could never be admitted; it is answered at once, as a context overflow.
class KvPoolTooSmall : public std::invalid_argument {
 public:
  KvPoolTooSmall(size_t prompt_tokens, int pages)
      : std::invalid_argument("the prompt's " + std::to_string(prompt_tokens) +
                              " tokens need more KV pages than this server's pool holds (" + std::to_string(pages) +
                              " pages of 64 positions): use a shorter prompt, or a larger --kv-pool-mb") {}
};

// Step periods of one decoding request: the time from the start of one of its steps (or its first token) to
// the start of the next, split by whether another request was prefilling meanwhile (M4 §3: a prefill chunk
// runs between two of its steps).
struct Periods {
  double sum = 0, max = 0;
  int n = 0;
  void add(double s) {
    sum += s;
    max = std::max(max, s);
    ++n;
  }
  double mean_ms() const { return n ? 1e3 * sum / n : 0.0; }
};

// A request the scheduler is running (M4 §3): its prefill cursor, then what one decode step needs (what
// generate() kept in locals when one request ran at a time).
struct Running {
  Job job;
  Engine::PrefillCursor cur;
  bool prefilling = true;
  int limit = 0;
  bool spec = false;
  std::vector<float> logits;  // the prefill's logits (the first token), then a plain step's
  std::vector<float> rows;    // the verify's rows, when the request asked for logprobs
  std::vector<int> pending;   // chosen tokens not yet emitted
  std::optional<StreamDecoder> dec;
  std::optional<StopScanner> stops;
  std::optional<RepetitionDetector> repeat;  // the request's repetition stop rule, when on (API §9)
  std::string finish = "length";
  int produced = 0, passes = 0, reasoning = 0;
  bool in_think = false;
  int newlines_after_think = 0;  // of the blank line after </think>, still to count
  bool ended_by_eos = false;
  double ttft = -1;
  std::chrono::steady_clock::time_point t_decode, last_step;
  long steps0 = 0, drafted0 = 0, accepted0 = 0;
  double draft_seconds0 = 0, commit_seconds0 = 0;
  bool drafter_changed = false;  // its request took a resident session over to another drafter (MULTI_DRAFTER §2.3)
  int chunks_beside = 0;  // prefill chunks run while another request was decoding (contended size, or on the side)
  bool side_flight = false;  // one of its chunks is in flight on the engine's side stream
  bool yielding = false;     // its prefill is letting a shorter one go first, since yield_t0 (C1)
  std::chrono::steady_clock::time_point yield_t0;
  int batched = 0;       // its decode steps taken in a batch with another request (M4 stage 3)
  int step_rows = 0;     // the rows its last batched step verified (a block, or its tree)
  long admitted = 0;     // the order of admission (P4b: the last resort ends the request admitted last)
  bool kv_wait = false;  // its last decode turn waited for KV pages (P4b)
  Periods alone, beside;  // step periods without and with another request's prefill in between
  PassWindow window;      // the decode passes it took part in, in the trailing kFairWindow (--fair-pairing)
};

// The engine thread (M4 §3). Up to `batch` requests run at once, each in its own resident session: a request
// is admitted when it can be bound (its session is not another running request's, or a slot is free or
// evictable), then the running requests take turns, one unit each: a prefill chunk while its prompt is not
// done, else one decode step (speculative, or plain). A prefill chunk that runs while another request is
// decoding is at most `contended` tokens, which bounds the decoding request's step period during a long
// prefill. Nothing numeric changes: each unit runs in its own session's state, as it would alone.
// Stage 3: when the request whose turn it is decodes speculatively and another one does too, they take one
// batched step together (Engine::speculate_batch: one verify pass over both requests' rows), which is bit for bit
// each one's step alone. A request that cannot join (plain decoding, no room for a full draft block) takes its own
// one-row step on its turn; a prefill chunk still alternates with the decode steps.
class Scheduler {
 public:
  Scheduler(ServerContext& ctx, double protect_seconds, int batch, int contended, int side_share = 0,
            bool fair_pairing = false, int fanout_plain_above = 0, int fanout_gang_above = -1,
            bool fanout_plain_explicit = true, int fanout_wide_above = 0, int fanout_wide_block = 0)
      : ctx_(ctx),
        protect_seconds_(protect_seconds),
        batch_(batch),
        contended_(contended),
        side_share_(side_share),
        fair_pairing_(fair_pairing),
        fanout_plain_above_(fanout_plain_above),
        fanout_gang_above_(fanout_gang_above < 0 ? fanout_plain_above : fanout_gang_above),
        fanout_plain_explicit_(fanout_plain_explicit),
        fanout_wide_above_(fanout_wide_above),
        base_block_(ctx.engine->draft_block()),
        wide_block_(fanout_wide_block > 0 ? fanout_wide_block : ctx.engine->draft_block()),
        thread_([this] { run(); }) {}
  ~Scheduler() {
    {
      std::lock_guard<std::mutex> l(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    thread_.join();
  }
  void submit(Job job) {
    std::lock_guard<std::mutex> l(mu_);
    queue_.push_back(std::move(job));
    dirty_ = true;
    {  // under mu_ as well, so the scheduler's decrement cannot precede this increment
      std::lock_guard<std::mutex> m(ctx_.metrics.mu);
      ctx_.metrics.queued += 1;
    }
    cv_.notify_one();
  }

 private:
  // Under mu_. Drops the jobs whose client has gone, then takes the first job that can run now and binds
  // its session (M4 §1): the resident session it continues, else a slot for a new session. Sessions that a
  // running request holds or a queued job continues are pinned, so a new session never takes them; a job
  // that continues a running request's session waits for it (one request per session at a time), and when
  // every other slot is pinned or protected a new session's job waits while the jobs behind it still run.
  // P4b (--kv-pages): a job is admitted only when its prompt's KV pages and two more are free (kv_make_room may evict
  // idle sessions for them); otherwise it waits and the jobs behind it may still run.
  bool pick(Job& out) {
    Engine& eng = *ctx_.engine;
    kv_wait_ = false;
    for (auto it = queue_.begin(); it != queue_.end();) {
      if (!it->cancelled->load()) {
        ++it;
        continue;
      }
      it = queue_.erase(it);
      std::lock_guard<std::mutex> m(ctx_.metrics.mu);  // mu_ then metrics.mu, as submit() takes them
      ctx_.metrics.queued -= 1;
    }
    std::vector<bool> busy(eng.sessions(), false);
    for (const auto& r : running_) busy[r->job.session] = true;
    std::vector<int> match(queue_.size());
    std::vector<bool> pinned = busy;
    for (size_t j = 0; j < queue_.size(); ++j) {
      match[j] = queue_[j].fork ? -1 : eng.match_session(queue_[j].prompt, queue_[j].req.session_key, queue_[j].req.share_scope);
      if (match[j] >= 0) pinned[match[j]] = true;
    }
    for (size_t j = 0; j < queue_.size(); ++j) {
      int s = match[j];
      if (s >= 0 && busy[s]) continue;
      const bool fresh = s < 0;
      if (fresh) s = eng.free_session(protect_seconds_, pinned);
      if (s < 0) continue;
      if (eng.kv_paged() && eng.kv_admit_pages(queue_[j].prompt.size()) <= eng.kv_pages_total()) {
        // (A prompt whose pages the whole pool could never hold is admitted to be answered at once: start().)
        bool evicted = false;
        if (!kv_admit_room(s, queue_[j], &evicted)) {
          kv_wait_ = true;
          // An eviction emptied sessions the other jobs' matches above may name: look again from the start.
          if (evicted) return false;
          continue;
        }
      }
      // LORA §3: a job on an adapter is admitted with the adapter's GPU slot, and waits while every slot holds another
      // adapter a running request uses (max_loras_per_batch); the jobs behind it may still run.
      int lslot = -1;
      bool upload = false;
      if (queue_[j].lora && (lslot = ctx_.lora_slots->acquire(queue_[j].lora->id, &upload)) < 0) continue;
      out = std::move(queue_[j]);
      out.session = s;
      out.fresh = fresh;
      out.lora_slot = lslot;
      out.lora_upload = upload;
      queue_.erase(queue_.begin() + j);
      std::lock_guard<std::mutex> m(ctx_.metrics.mu);
      ctx_.metrics.queued -= 1;
      return true;
    }
    return false;
  }

  // Waits while nothing runs and nothing can be admitted; admits at most one job. Returns false to stop.
  bool admit() {
    Job job;
    {
      std::unique_lock<std::mutex> l(mu_);
      while (true) {
        if (stop_) return false;
        const auto now = std::chrono::steady_clock::now();
        // A job that could not be placed is looked at again when something changed (a new job, a request
        // ended) or once a second (a protection expires, a waiting client leaves).
        // A job waiting for KV pages that are being freed (P4b) is looked at again at every turn.
        const bool look = static_cast<int>(running_.size()) < batch_ && !queue_.empty() &&
                          (dirty_ || kv_wait_ || now - last_look_ >= std::chrono::seconds(1));
        if (look) {
          dirty_ = false;
          last_look_ = now;
          if (pick(job)) {
            dirty_ = !queue_.empty();  // a burst: the jobs behind this one are looked at on the next turn, not in a second
            break;
          }
        }
        if (!running_.empty()) return true;
        kv_gauges();  // the pool's gauges stay current while nothing runs
        if (queue_.empty()) cv_.wait(l);
        else cv_.wait_for(l, std::chrono::seconds(1));
      }
    }
    start(std::move(job));
    // Fan-out (BATCHING P4d): the rest of a burst of short prefills (forks' questions after a shared context) is admitted
    // in this turn too, one job at a time as above, so the gang's joint pass (joint_prefill) finds them together instead
    // of prefilling each one before the next is admitted. Only with fan-out on and while the last job admitted is such a
    // short prefill; RAGON_JOINT_PREFILL=0 keeps one admission a turn.
    while (burst_more()) {
    }
    return true;
  }

  bool burst_more() {
    static const bool on = [] {
      const char* e = std::getenv("RAGON_JOINT_PREFILL");
      return !e || *e != '0';
    }();
    if (!on || fanout_gang_above_ <= 0 || running_.empty()) return false;
    const Running& last = *running_.back();
    if (!last.prefilling || prefill_left(last) > ctx_.engine->prefill_chunk_size()) return false;
    Job job;
    {
      std::lock_guard<std::mutex> l(mu_);
      if (stop_ || static_cast<int>(running_.size()) >= batch_ || queue_.empty() || !pick(job)) return false;
      dirty_ = !queue_.empty();
    }
    const size_t before = running_.size();
    start(std::move(job));
    return running_.size() > before;  // a job start() refused ends the burst (the next turn looks again)
  }

  // Binds the admitted job's session and readies its prefill; a job refused here is answered at once.
  void start(Job job) {
    Engine& eng = *ctx_.engine;
    auto r = std::make_unique<Running>();
    r->job = std::move(job);
    Job& j = r->job;
    r->admitted = ++admitted_;
    try {
      if (exceeds_context(j.prompt.size(), eng.max_context())) throw ContextOverflow(j.prompt.size(), eng.max_context());
      if (eng.kv_paged() && eng.kv_admit_pages(j.prompt.size()) > eng.kv_pages_total())
        throw KvPoolTooSmall(j.prompt.size(), eng.kv_pages_total());
      if (j.lora_upload) {
        eng.lora_upload(j.lora_slot, *j.lora);
        j.lora_upload = false;
      }
      r->drafter_changed =
          eng.activate_session(j.session, j.req.session_key, j.fresh, j.req.share_scope, j.lora_slot, j.req.drafter);
      try {
        r->cur = eng.begin_prefill(j.prompt, false, j.images.empty() ? nullptr : &j.images);
      } catch (const KvPoolExhausted&) {
        // P4c: admission's count was short after all (begin_prefill changed nothing). While nothing runs nothing else
        // will free pages: make room again (everything idle may go: the prompt fits once it has) and try once more.
        if (!running_.empty()) throw;
        kv_admit_room(j.session, j);
        r->cur = eng.begin_prefill(j.prompt, false, j.images.empty() ? nullptr : &j.images);
      }
    } catch (const KvPoolExhausted& e) {
      // P4c: the pool is short after all (pages shared at admission's count were let go of meanwhile): while a
      // request runs, the job goes back to the front of the queue and is admitted again once one ends.
      const int session = j.session;
      release_lora(j);  // taken again when it is admitted again
      if (running_.empty()) {
        fail(*r, e);
      } else {
        std::fprintf(stderr, "KV pool: a prompt of %zu tokens waits (%s)\n", j.prompt.size(), e.what());
        std::lock_guard<std::mutex> l(mu_);
        queue_.push_front(std::move(j));
        std::lock_guard<std::mutex> m(ctx_.metrics.mu);
        ctx_.metrics.queued += 1;
      }
      eng.touch_session(session);
      return;
    } catch (const std::exception& e) {
      fail(*r, e);
      release_lora(j);
      eng.touch_session(j.session);
      return;
    }
    running_.push_back(std::move(r));
    std::lock_guard<std::mutex> m(ctx_.metrics.mu);
    ctx_.metrics.running = static_cast<int>(running_.size());
    ctx_.metrics.sessions_resident = eng.sessions_resident();
    ctx_.metrics.session_evictions = static_cast<double>(eng.session_evictions());
  }

  // BATCHING P4d: the request's other choices start now, each a fork of its session: no prefill, the prompt's logits
  // (still the engine's: its last prefill chunk just ran), its pages shared. A fork that finds no room (no session, no
  // pages, --batch) goes to the front of the queue and later runs as a request with the same prompt: the same tokens,
  // resumed from its parent's history instead of forked. The active session is the parent's again on return.
  void start_forks(Running& parent) {
    if (parent.job.forks.empty()) return;
    Engine& eng = *ctx_.engine;
    const int from = parent.job.session;
    std::vector<Job> forks = std::move(parent.job.forks);
    parent.job.forks.clear();
    std::vector<Job> later;
    for (Job& j : forks)
      if (!later.empty() || !start_fork(j, from)) later.push_back(std::move(j));
    eng.select_session(from);
    {
      std::lock_guard<std::mutex> l(mu_);
      for (auto it = later.rbegin(); it != later.rend(); ++it) queue_.push_front(std::move(*it));
      if (!later.empty()) dirty_ = true;
      std::lock_guard<std::mutex> m(ctx_.metrics.mu);
      ctx_.metrics.queued += static_cast<double>(later.size());
      ctx_.metrics.running = static_cast<int>(running_.size());
      ctx_.metrics.sessions_resident = eng.sessions_resident();
      ctx_.metrics.session_evictions = static_cast<double>(eng.session_evictions());
    }
  }

  // Starts one fork; false, with j as it was, when it has to wait.
  bool start_fork(Job& j, int from) {
    Engine& eng = *ctx_.engine;
    if (static_cast<int>(running_.size()) >= batch_) return false;
    int s;
    {
      std::lock_guard<std::mutex> l(mu_);  // pick()'s pins: running requests, and the sessions queued jobs continue
      std::vector<bool> pinned(eng.sessions(), false);
      for (const auto& o : running_) pinned[o->job.session] = true;
      for (const Job& q : queue_) {
        const int m = q.fork ? -1 : eng.match_session(q.prompt, q.req.session_key, q.req.share_scope);
        if (m >= 0) pinned[m] = true;
      }
      s = eng.free_session(protect_seconds_, pinned);
    }
    if (s < 0) return false;
    if (eng.kv_paged()) {
      Engine::KvSource src;
      src.donor = from;
      const KvNeed need = [&](const Engine::KvGone* gone, bool) { return eng.kv_fork_need(from, s, gone); };
      if (!kv_make_room(need, s, src)) return false;
    }
    int lslot = -1;
    bool upload = false;
    if (j.lora && (lslot = ctx_.lora_slots->acquire(j.lora->id, &upload)) < 0) return false;
    j.session = s;
    j.fresh = true;
    j.lora_slot = lslot;
    j.lora_upload = upload;
    try {
      if (j.lora_upload) {  // (its parent holds the adapter, so the slot is filled: this is only the general rule)
        eng.lora_upload(j.lora_slot, *j.lora);
        j.lora_upload = false;
      }
      eng.activate_session(s, j.req.session_key, true, j.req.share_scope, j.lora_slot, j.req.drafter);  // the parent's
      eng.fork_session(from);
    } catch (const KvPoolExhausted& e) {  // nothing changed: the count above was short, it waits
      std::fprintf(stderr, "KV pool: a fork of %zu tokens waits (%s)\n", j.prompt.size(), e.what());
      release_lora(j);
      j.session = -1;
      j.fresh = false;
      eng.touch_session(s);
      return false;
    } catch (const std::exception& e) {
      auto failed = std::make_unique<Running>();
      failed->job = std::move(j);
      fail(*failed, e);
      release_lora(failed->job);
      eng.touch_session(s);
      return true;
    }
    auto r = std::make_unique<Running>();
    r->job = std::move(j);
    r->admitted = ++admitted_;
    r->prefilling = false;
    r->cur.t0 = std::chrono::steady_clock::now();
    r->cur.stats.reused_tokens = static_cast<int>(r->job.prompt.size());
    bool done;
    try {
      begin_decode(*r);
      done = emit(*r) && end(*r);
    } catch (const std::exception& e) {
      fail(*r, e);
      done = true;
    }
    if (done) {
      release_lora(r->job);
      eng.touch_session(s);
      return true;
    }
    running_.push_back(std::move(r));
    return true;
  }

  // Gives back the job's GPU slot (it ended, failed or went back to the queue); a slot whose upload did not finish
  // holds no adapter.
  void release_lora(Job& j) {
    if (j.lora_slot < 0) return;
    ctx_.lora_slots->release(j.lora_slot);
    if (j.lora_upload) ctx_.lora_slots->forget(j.lora_slot);
    j.lora_slot = -1;
    j.lora_upload = false;
  }

  void fail(Running& r, const std::exception& e) {
    if (const auto* o = dynamic_cast<const ContextOverflow*>(&e)) {
      r.job.sink.error(o->what(), kContextLengthExceeded);
      return;
    }
    if (const auto* o = dynamic_cast<const KvPoolTooSmall*>(&e)) {
      r.job.sink.error(o->what(), kContextLengthExceeded);
      return;
    }
    if (dynamic_cast<const NonFiniteLogits*>(&e)) {
      {
        std::lock_guard<std::mutex> l(ctx_.metrics.mu);
        ctx_.metrics.nonfinite += 1;
      }
      std::fprintf(stderr, "request failed: %s (%zu prompt tokens)\n", e.what(), r.job.prompt.size());
    }
    r.job.sink.error(e.what(), "server_error");
  }

  void run() {
    while (admit()) {
      if (running_.empty()) continue;
      kv_gauges();
      fanout_regime();  // before kv_room and batch_with: both read the draft block
      size_t k = turn_++ % running_.size();
      // A side chunk whose pass is done is finished (and the next one launched) before anything else, so the side
      // stream does not wait out another request's step.
      if (ctx_.engine->side_busy() && ctx_.engine->side_ready())
        for (size_t j = 0; j < running_.size(); ++j)
          if (running_[j]->side_flight) {
            k = j;
            break;
          }
      gang_unit_ = false;
      if (Running* g = gang_prefill()) {
        k = static_cast<size_t>(std::find_if(running_.begin(), running_.end(), [&](const auto& p) { return p.get() == g; }) -
                                running_.begin());
        gang_unit_ = true;
      }
      Running& r = *running_[k];
      std::vector<Running*> ended;
      // P4b: a decode step whose KV pages are not there waits while the others run; when every request waits, the
      // last resort may end one (kv_stalled).
      const bool kv_blocked = ctx_.engine->kv_paged() && !r.prefilling && !kv_room(r);
      r.kv_wait = kv_blocked;
      cand_.clear();
      std::vector<Running*> batch = kv_blocked ? std::vector<Running*>() : batch_with(r);
      const double t0 = trace_.on() ? trace_.now() : 0;
      unit_kind_ = nullptr;
      if (kv_blocked) {
        kv_stalled(ended);
      } else if (gang_unit_ && joint_prefill(batch, ended)) {
        unit_kind_ = "joint";
      } else if (batch.size() >= 2) {
        batch_step(batch, ended);
        unit_kind_ = "decode";
      } else {
        bool done = false;
        try {
          done = unit(r);
        } catch (const std::exception& e) {
          fail(r, e);
          done = true;
        }
        if (done) ended.push_back(&r);
        batch.assign(1, &r);
      }
      if (trace_.on() && unit_kind_) trace_unit(t0, k, batch);
      if (ended.empty()) continue;
      for (Running* e : ended) {
        if (e->side_flight) ctx_.engine->side_abandon();  // its chunk's pass finishes before the session is reused
        ctx_.engine->touch_session(e->job.session);  // the session's clock restarts however the request ended
        release_lora(e->job);
        running_.erase(std::find_if(running_.begin(), running_.end(), [&](const auto& p) { return p.get() == e; }));
      }
      if (running_.empty()) trace_.flush();
      {
        std::lock_guard<std::mutex> l(mu_);
        dirty_ = true;  // its session may be what a queued job waits for
      }
      std::lock_guard<std::mutex> m(ctx_.metrics.mu);
      ctx_.metrics.running = static_cast<int>(running_.size());
    }
  }

  // Whether r's next decode step can join a batch: past its prompt, speculative, room for a full draft block.
  bool batchable(const Running& r) const {
    return !r.prefilling && r.spec && ctx_.engine->can_batch(r.job.session, r.job.req.sampling);
  }
  // Fan-out (BATCHING P4d): above --fanout-plain-above decoding requests, a speculative block no longer fits enough
  // of them in one pass, and one row each decodes more in total. Every decoder then takes plain steps, together;
  // greedy output is unchanged (a plain row's logits are a verify row's) and a sampled one keeps its distribution.
  // Without a drafter nothing speculates, so the threshold is 1: two decoding requests already share a pass.
  bool fanout_plain() const {
    if (fanout_plain_above_ <= 0 || (wide_ && !fanout_plain_explicit_)) return false;
    int decoding = 0;
    for (const auto& o : running_) decoding += !o->prefilling;
    return decoding > fanout_plain_above_;
  }
  // The wide regime (BATCHING §5.8 stage 11, the default since 2026-11-08): above fanout_wide_above_ drafting requests
  // (7; decoding, with a drafter for their sampling) every one of them speculates, gang-sized: the plain threshold no
  // longer holds unless --fanout-plain-above was given, and the engine's draft block is the gang block (2: 32 requests
  // x 2 rows fill a 64-row verify). At fewer the server is what it was (plain above 6, the global block, trees). A
  // change of block drops the captured graphs (Engine::set_draft_block), so it waits while a side chunk is in flight
  // and the regime with it. Called once a turn, before anything reads the block. --fanout-default off: never wide.
  void fanout_regime() {
    if (fanout_wide_above_ <= 0) return;
    int drafting = 0;
    for (const auto& o : running_) drafting += !o->prefilling && o->spec;
    Engine& eng = *ctx_.engine;
    const bool before = wide_;
    wide_ = drafting > fanout_wide_above_;
    const int block = wide_ && !fanout_plain() ? wide_block_ : base_block_;
    if (block == eng.draft_block()) return;
    if (eng.side_busy()) {
      wide_ = before;
      return;
    }
    eng.set_draft_block(block);
  }
  // Fan-out's question prefills (BATCHING P4d): above --fanout-gang-above live requests (by default the plain
  // threshold), the short prefills waiting (at most a chunk left: a fork's question after a shared context) run ahead
  // of the decode passes, shortest first and alone on the main stream, so the forks start decoding together instead of
  // each waiting out the others' steps; a pass over 32 rows is what pays. A hold lasts kGangHold at most, then a
  // decode pass runs. None while a side chunk is in flight: the decoders go on beside it.
  static constexpr auto kGangHold = std::chrono::seconds(10);
  Running* gang_prefill() {
    if (fanout_gang_above_ <= 0 || static_cast<int>(running_.size()) <= fanout_gang_above_) return nullptr;
    Engine& eng = *ctx_.engine;
    int live = 0;
    Running* next = nullptr;
    for (const auto& o : running_) {
      if (!o->prefilling) {
        ++live;
      } else if (prefill_left(*o) <= eng.prefill_chunk_size()) {
        ++live;
        if (!next || prefill_left(*o) < prefill_left(*next)) next = o.get();
      }
    }
    const auto now = std::chrono::steady_clock::now();
    if (!next || live <= fanout_gang_above_ || eng.side_busy()) {
      gang_holding_ = false;
      return nullptr;
    }
    if (!gang_holding_) {
      gang_holding_ = true;
      gang_t0_ = now;
    } else if (now - gang_t0_ >= kGangHold) {
      gang_holding_ = false;
      return nullptr;
    }
    return next;
  }
  // The held question prefills together (BATCHING P4d): every short prefill the engine can take in a joint pass
  // (Engine::joint_eligible), shortest first, in as few passes as its row cap allows; each request then goes on as after
  // its last chunk, alone. Each question's logits and state are what its own chunk would have left, bit for bit.
  // RAGON_JOINT_PREFILL=0 keeps the one-by-one order. False when fewer than two qualify (the gang's unit runs instead).
  bool joint_prefill(std::vector<Running*>& batch, std::vector<Running*>& ended) {
    static const bool on = [] {
      const char* e = std::getenv("RAGON_JOINT_PREFILL");
      return !e || *e != '0';
    }();
    Engine& eng = *ctx_.engine;
    if (!on || eng.side_busy()) return false;
    std::vector<Running*> rs;
    for (const auto& o : running_)
      if (o->prefilling && !o->side_flight && eng.joint_eligible(o->cur)) rs.push_back(o.get());
    if (rs.size() < 2) return false;
    std::stable_sort(rs.begin(), rs.end(),
                     [](const Running* a, const Running* b) { return prefill_left(*a) < prefill_left(*b); });
    std::vector<Engine::PrefillCursor*> curs;
    for (Running* r : rs) curs.push_back(&r->cur);
    std::vector<std::exception_ptr> errs;
    try {
      errs = eng.prefill_joint(curs);
    } catch (...) {  // before any pass (a precondition): every request fails as its own chunk would have
      errs.assign(rs.size(), std::current_exception());
    }
    for (size_t k = 0; k < rs.size(); ++k) {
      Running& r = *rs[k];
      bool done = false;
      try {
        if (errs[k]) std::rethrow_exception(errs[k]);
        eng.joint_select(static_cast<int>(k));
        r.prefilling = false;
        begin_decode(r);
        start_forks(r);
        done = emit(r) && end(r);
      } catch (const std::exception& e) {
        fail(r, e);
        done = true;
      }
      if (done) ended.push_back(&r);
    }
    batch = rs;
    return true;
  }
  // Whether r's next step speculates.
  bool spec_turn(const Running& r) const { return r.spec && !fanout_plain(); }
  // Whether r's next step drafts: a speculative turn whose drafter guard is off takes a plain step, and such requests
  // step together in one plain pass (step_batch) instead of one unit each. RAGON_DRAFTER_OFF_ALONE=1 keeps one unit each,
  // for a comparison.
  bool spec_step(const Running& r) const {
    static const bool alone = [] {
      const char* e = std::getenv("RAGON_DRAFTER_OFF_ALONE");
      return e && std::atoi(e) != 0;
    }();
    return spec_turn(r) && (alone || !ctx_.engine->drafter_off(r.job.session, r.job.req.sampling));
  }
  // BATCHING §5.8 stage 10: a speculative turn whose drafter guard is off (a rider) joins a batch of drafting requests
  // as one row of its verify (Engine::speculate_batch) instead of a plain pass of its own. RAGON_RIDE_ALONG=0 keeps the
  // separate plain pass (step_batch), for a comparison.
  static bool ride_along() {
    static const bool on = [] {
      const char* e = std::getenv("RAGON_RIDE_ALONG");
      return !e || std::atoi(e) != 0;
    }();
    return on;
  }
  bool rider(const Running& r) const { return spec_turn(r) && !spec_step(r); }
  // Whether r's next decode step can join the batch of the kind it would take alone.
  bool batchable_now(const Running& r) const {
    if (r.prefilling) return false;
    return spec_step(r) ? batchable(r) : ctx_.engine->can_step_batch(r.job.session);
  }
  // The requests that step with r (r first, then the others in running order) while their rows fit one pass;
  // just r, or nothing, when there is no batch to form.
  // P4b: only requests whose step's KV pages are free join (r's are: kv_room).
  std::vector<Running*> batch_with(Running& r) const {
    std::vector<Running*> b;
    if (trace_.on()) snapshot_candidates();
    if (!batchable_now(r)) return b;
    b.push_back(&r);
    Engine& eng = *ctx_.engine;
    if (ride_along() && spec_turn(r)) {
      if (ride_batch(b).size() >= 2 && spec_step(*b[0])) return b;
      b.resize(1);  // no drafter to ride with: the batch of r's own kind, as before
    }
    // A batch is all speculative or all plain; a plain one takes one row per request.
    const bool spec = spec_step(r);
    const int fit = spec ? kernels::kMaxStreamRows / eng.draft_block() : std::min(kernels::kMaxStreamRows, kernels::kMaxSeqs);
    int kv_need = eng.kv_paged() ? eng.kv_step_need(r.job.session, spec) : 0;
    const int kv_free = eng.kv_paged() ? eng.kv_pages_free() : 0;
    for (size_t i : partner_order()) {
      const auto& o = running_[i];
      if (o.get() != &r && static_cast<int>(b.size()) < fit && spec_step(*o) == spec && batchable_now(*o)) {
        if (eng.kv_paged()) {
          const int n = eng.kv_step_need(o->job.session, spec);
          if (kv_need + n > kv_free) {
            if (trace_.on()) cand_[i].state = 2;
            continue;
          }
          kv_need += n;
        }
        b.push_back(o.get());
      }
    }
    return b;
  }
  // batch_with for a speculative turn when riders join (b holds r): the drafting partners in partner order while their
  // blocks fit, then the riders in the rows left (one each), at most kMaxSeqs requests; the drafters first, as
  // speculate_batch takes them. batch_with forms the batch of r's own kind instead when none of them drafts.
  std::vector<Running*>& ride_batch(std::vector<Running*>& b) const {
    Engine& eng = *ctx_.engine;
    Running& r = *b[0];
    const auto rows_of = [&](const Running& o) { return spec_step(o) ? eng.draft_block() : 1; };
    int rows = rows_of(r);
    int kv_need = eng.kv_paged() ? eng.kv_step_need(r.job.session, spec_step(r)) : 0;
    const int kv_free = eng.kv_paged() ? eng.kv_pages_free() : 0;
    const std::vector<size_t> order = partner_order();
    for (const bool drafting : {true, false})
      for (size_t i : order) {
        const auto& o = running_[i];
        if (o.get() == &r || static_cast<int>(b.size()) >= kernels::kMaxSeqs || !spec_turn(*o) ||
            spec_step(*o) != drafting || !batchable_now(*o) || rows + rows_of(*o) > kernels::kMaxStreamRows)
          continue;
        if (eng.kv_paged()) {
          const int n = eng.kv_step_need(o->job.session, drafting);
          if (kv_need + n > kv_free) {
            if (trace_.on()) cand_[i].state = 2;
            continue;
          }
          kv_need += n;
        }
        rows += rows_of(*o);
        b.push_back(o.get());
      }
    std::stable_partition(b.begin(), b.end(), [&](const Running* o) { return spec_step(*o); });
    return b;
  }
  // The order the running requests are tried in as partners (serve/pairing.hpp): running order, or with
  // --fair-pairing 1 the fewest decode passes in the trailing kFairWindow first.
  std::vector<size_t> partner_order() const {
    std::vector<PairCandidate> c(running_.size());
    if (fair_pairing_)
      for (size_t i = 0; i < running_.size(); ++i)
        c[i] = {false, running_[i]->window.count(decode_passes_), running_[i]->window.last()};
    return pair_order(c, fair_pairing_);
  }
  // The running requests as the partner choice sees them, for the step trace's `cand` (serve/step_trace.hpp).
  void snapshot_candidates() const {
    cand_.clear();
    for (const auto& o : running_)
      cand_.push_back({o->job.session, o->admitted, batchable_now(*o) ? 1 : 0, o->window.count(decode_passes_),
                       o->window.last()});
  }

  // One line of the step trace (RAGON_STEP_TRACE) for the unit just run: `batch` holds its requests.
  void trace_unit(double t0, size_t turn, const std::vector<Running*>& batch) {
    StepRecord s;
    s.t = trace_.now();
    s.ms = (s.t - t0) * 1e3;
    s.pass = decode_passes_;
    s.kind = unit_kind_;
    s.turn = turn;
    const bool decode = std::strcmp(unit_kind_, "decode") == 0;
    int alone_rows = 0;
    if (decode && batch.size() == 1) {
      std::lock_guard<std::mutex> m(ctx_.metrics.mu);
      alone_rows = ctx_.metrics.batch_rows;
    }
    for (Running* r : batch) {
      s.sessions.push_back(r->job.session);
      s.rows.push_back(!decode ? 0 : batch.size() == 1 ? alone_rows : r->step_rows);
      s.tokens.push_back(decode ? static_cast<int>(r->pending.size()) : 0);
      s.left.push_back(r->prefilling ? prefill_left(*r) : 0);
    }
    if (decode) s.cand = cand_;
    s.side = ctx_.engine->side_busy();
    trace_.write(s);
  }

  // ---- The KV page pool (P4b, --kv-pages; serve/kv_policy.hpp). ----
  // The pool as the policy sees it; the idle sessions (no running request, not `keep` or `donor`) that hold pages,
  // least recently used first, into idle when given. v.idle counts what evicting all of them and the prefix KV store
  // (its nodes other than `store_keep`) frees, jointly (P4c: a page shared by two of them counted once).
  KvPoolView kv_view(int keep, std::vector<int>* idle = nullptr, int donor = -1,
                     const std::vector<int>& store_keep = {}) const {
    Engine& eng = *ctx_.engine;
    std::vector<bool> excl(eng.sessions(), false);
    for (const auto& r : running_) excl[r->job.session] = true;
    if (keep >= 0) excl[keep] = true;
    if (donor >= 0) excl[donor] = true;
    KvPoolView v;
    v.free = eng.kv_pages_free();
    v.pending = eng.kv_pages_pending();
    std::vector<int> lru;
    for (int s; (s = eng.kv_lru_idle(excl)) >= 0;) {
      lru.push_back(s);
      excl[s] = true;
    }
    v.idle = eng.kv_reclaimable(lru, true, store_keep);
    if (idle) *idle = std::move(lru);
    return v;
  }
  // Whether the pages for a unit of session `keep` (an admission, a decode step; need() its pages, planned again after
  // every eviction) are free now, evicting what kv_reclaim (serve/kv_policy.hpp) says: prefix KV store entries and
  // idle sessions, least recently used first (--session-protect-seconds does not hold against the pool's need), the
  // unit's own source (`src`: the store path and the session an admission resumes from) last. Evicted pages, and
  // pages already released, are free once the streams have passed their release (a later turn), or at once after a
  // synchronization when nothing runs. *evicted: whether it evicted anything.
  // need(gone, from_scratch): the unit's need planned now (nullptr), or with those holders gone (Engine::KvGone; an
  // admission from position 0 with from_scratch).
  using KvNeed = std::function<int(const Engine::KvGone*, bool)>;
  bool kv_make_room(const KvNeed& need_with, int keep, const Engine::KvSource& src, bool* evicted = nullptr) {
    Engine& eng = *ctx_.engine;
    const std::function<int()> need = [&] { return need_with(nullptr, false); };
    if (need() > eng.kv_pages_free() + eng.kv_pages_pending()) {
      Engine::KvGone gone_kept{{}, true, src.store_path}, gone_all{{}, true, {}};
      const KvPoolView kept = kv_view(keep, &gone_kept.sessions, src.donor, src.store_path),
                       all = kv_view(keep, &gone_all.sessions);
      std::vector<bool> excl(eng.sessions(), false), excl_kept;
      for (const auto& r : running_) excl[r->job.session] = true;
      if (keep >= 0) excl[keep] = true;
      excl_kept = excl;
      if (src.donor >= 0) excl_kept[src.donor] = true;
      int dropped = 0;
      auto store = [&](const std::vector<int>& keep_nodes) {
        return [&eng, &dropped, &keep_nodes] {
          if (!eng.kv_store_evict(keep_nodes)) return false;
          ++dropped;
          return true;
        };
      };
      auto session = [&](const std::vector<bool>& busy) {
        return [this, &eng, &busy, &need, evicted] {
          const int s = eng.kv_lru_idle(busy);
          if (s < 0) return false;
          if (evicted) *evicted = true;
          std::fprintf(stderr, "KV pool: idle session %d evicted (%d pages its own) for %d pages; %d free, %d pending\n",
                       s, eng.kv_session_pages(s), need(), eng.kv_pages_free(), eng.kv_pages_pending());
          eng.kv_evict(s);
          std::lock_guard<std::mutex> m(ctx_.metrics.mu);
          ctx_.metrics.kv_evictions += 1;
          return true;
        };
      };
      const std::vector<int> no_nodes;
      const std::vector<KvEvictNext> kept_order = {store(src.store_path), session(excl_kept)};
      const std::vector<KvEvictNext> source_order = {store(no_nodes), session(excl)};
      kv_reclaim(need, [&eng] { return eng.kv_pages_free() + eng.kv_pages_pending(); }, kept_order,
                 need_with(&gone_kept, false), kept.free + kept.pending + kept.idle, source_order,
                 need_with(&gone_all, true), all.free + all.pending + all.idle, running_.empty() && src.cold > 0);
      if (dropped > 0) {
        if (evicted) *evicted = true;
        std::fprintf(stderr, "KV pool: %d prefix KV store entries evicted for %d pages; %d free, %d pending\n", dropped,
                     need(), eng.kv_pages_free(), eng.kv_pages_pending());
      }
    }
    if (running_.empty()) eng.kv_retire(true);
    return need() <= eng.kv_pages_free();
  }
  // Room for job j's admission in session s (kv_make_room, its need planned by Engine::kv_admit_need each time).
  bool kv_admit_room(int s, const Job& j, bool* evicted = nullptr) {
    Engine& eng = *ctx_.engine;
    Engine::KvSource src;
    eng.kv_admit_need(s, j.prompt, j.req.share_scope, &src, nullptr, false, j.req.drafter);
    return kv_make_room(
        [&](const Engine::KvGone* gone, bool from_scratch) {
          return eng.kv_admit_need(s, j.prompt, j.req.share_scope, nullptr, gone, from_scratch, j.req.drafter);
        },
        s, src, evicted);
  }
  // Whether r's next decode step has its KV pages (Engine::kv_step_need), making room as kv_make_room does.
  bool kv_room(Running& r) {
    Engine& eng = *ctx_.engine;
    return kv_make_room([&](const Engine::KvGone* gone, bool) { return eng.kv_step_need(r.job.session, spec_turn(r), gone); },
                        r.job.session, Engine::KvSource{});
  }
  // A decode turn waited for pages. When every running request waits: pages being freed are waited for, else the
  // last resort ends the request admitted last as if it had reached the context limit (finish_reason "length") and
  // frees its pages; it is added to `ended`.
  void kv_stalled(std::vector<Running*>& ended) {
    Engine& eng = *ctx_.engine;
    std::vector<KvRunner> rs;
    for (const auto& o : running_) {
      if (o->prefilling || !o->kv_wait) return;  // someone still makes progress
      rs.push_back({o->admitted, true, eng.kv_step_need(o->job.session, spec_turn(*o))});
    }
    const KvPoolView v = kv_view(-1);
    if (v.pending > 0) {  // nothing else can run: wait for the released pages' readers
      eng.kv_retire(true);
      return;
    }
    const int i = kv_last_resort(rs, v);
    if (i < 0) return;
    Running& victim = *running_[i];
    std::fprintf(stderr, "KV pool exhausted: every running request waits for a page; session %d's request (%d tokens "
                         "produced) ends at length\n", victim.job.session, victim.produced);
    victim.finish = "length";
    end(victim);
    eng.kv_evict(victim.job.session);
    {
      std::lock_guard<std::mutex> m(ctx_.metrics.mu);
      ctx_.metrics.kv_pool_exhausted += 1;
    }
    ended.push_back(&victim);
  }
  void kv_gauges() {
    Engine& eng = *ctx_.engine;
    if (!eng.kv_paged()) return;
    const int total = eng.kv_pages_total(), free = eng.kv_pages_free();
    std::lock_guard<std::mutex> m(ctx_.metrics.mu);
    ctx_.metrics.kv_pages_total = total;
    ctx_.metrics.kv_pages_free = free;
  }

  // One batched step of the requests in `batch` (speculative, or plain: batch_with), then each one's tokens emitted
  // as unit() does;
  // the requests that end are added to `ended`. A non-finite row fails only its own request.
  void batch_step(const std::vector<Running*>& batch, std::vector<Running*>& ended) {
    Engine& eng = *ctx_.engine;
    const bool spec = spec_step(*batch[0]);  // a drafter first, if the batch has one (ride_batch)
    std::vector<Engine::SeqStep> steps(batch.size());
    const auto now = std::chrono::steady_clock::now();
    for (size_t i = 0; i < batch.size(); ++i) {
      Running& r = *batch[i];
      (other_prefilling(r) ? r.beside : r.alone).add(std::chrono::duration<double>(now - r.last_step).count());
      r.last_step = now;
      ++r.passes;
      ++r.batched;
      r.kv_wait = false;
      r.window.record(decode_passes_ + 1);  // this pass
      steps[i].session = r.job.session;
      steps[i].anchor = r.pending.back();
      steps[i].params = &r.job.req.sampling;
      steps[i].row_logits = r.job.req.logprobs ? &r.rows : nullptr;
      steps[i].drafter_off = rider(r);  // in a speculative batch, a rider (speculate_batch)
    }
    ++decode_passes_;
    try {
      if (spec)
        eng.speculate_batch(steps);
      else
        eng.step_batch(steps);
    } catch (const std::exception& e) {  // not one sequence's rows: the step itself failed
      for (Running* r : batch) {
        fail(*r, e);
        ended.push_back(r);
      }
      return;
    }
    {
      std::lock_guard<std::mutex> m(ctx_.metrics.mu);
      ctx_.metrics.batched_steps += 1;
      int rows = 0;  // each sequence's block, or its tree
      for (size_t i = 0; i < batch.size(); ++i) rows += batch[i]->step_rows = steps[i].rows;
      ctx_.metrics.batch_rows = rows;
    }
    for (size_t i = 0; i < batch.size(); ++i) {
      Running& r = *batch[i];
      bool done = false;
      try {
        if (steps[i].error) std::rethrow_exception(steps[i].error);
        r.pending = std::move(steps[i].out);
        if (r.job.req.logprobs) require_finite(r.rows);  // as unit(): a row reported as logprobs is checked whole
        done = emit(r) && end(r);
      } catch (const std::exception& e) {
        fail(r, e);
        done = true;
      }
      if (done) ended.push_back(&r);
    }
  }

  bool other_decoding(const Running& self) const {
    for (const auto& r : running_)
      if (r.get() != &self && !r->prefilling) return true;
    return false;
  }
  int others_decoding(const Running& self) const {
    int n = 0;
    for (const auto& r : running_)
      if (r.get() != &self && !r->prefilling) ++n;
    return n;
  }
  // Whether r's next side chunk waits (--side-share N): while two or more other requests decode, N decode passes
  // run on the main stream alone after the last side chunk (any request's) before the next one starts.
  bool hold_side(const Running& r) const {
    if (side_share_ <= 0 || side_mark_ < 0 || others_decoding(r) < 2) return false;
    return decode_passes_ - side_mark_ < side_share_;
  }
  bool other_prefilling(const Running& self) const {
    for (const auto& r : running_)
      if (r.get() != &self && r->prefilling) return true;
    return false;
  }

  // The shortest prefill first (BATCHING §5.5, C1): of the requests waiting to prefill, the one with the fewest
  // tokens left takes the next chunk. A warm follow-up's append is that request's whole time to first token, while a
  // cold prompt of tens of thousands of tokens loses a chunk's time to it. Before, a long prefill on the side stream
  // relaunched its next chunk the moment one finished, so a follow-up arriving meanwhile waited out the whole cold
  // prompt (up to 55 s at four streams, W11). A request that has yielded for kMaxYield takes its next chunk anyway,
  // so a long prompt still moves while short ones keep arriving. Only the order of units changes: every chunk runs
  // in its own session's state, as it would alone.
  static constexpr auto kMaxYield = std::chrono::seconds(5);
  static int prefill_left(const Running& r) { return static_cast<int>(r.cur.prompt.size()) - r.cur.next; }
  // Whether r lets a shorter prefill that is waiting take the next chunk (and r's yield clock).
  bool yield_prefill(Running& r) const {
    const int left = prefill_left(r);
    bool shorter = false;
    for (const auto& o : running_)
      if (o.get() != &r && o->prefilling && !o->side_flight && prefill_left(*o) < left) shorter = true;
    const auto now = std::chrono::steady_clock::now();
    if (!shorter || (r.yielding && now - r.yield_t0 >= kMaxYield)) {
      r.yielding = false;
      return false;
    }
    if (!r.yielding) {
      r.yielding = true;
      r.yield_t0 = now;
    }
    return true;
  }

  // One prefill chunk or one decode step of `r`. Returns true when the request has ended (and was answered).
  bool unit(Running& r) {
    Engine& eng = *ctx_.engine;
    if (r.prefilling) {
      const bool contended = !gang_unit_ && other_decoding(r);
      if (eng.side_enabled() && (r.side_flight || (contended && !eng.side_busy()))) {
        // A full chunk on the side stream; the decoding request's turns go on while it runs.
        if (!r.side_flight) {
          if (hold_side(r)) {
            unit_kind_ = "hold";
            return false;
          }
          if (yield_prefill(r)) {
            unit_kind_ = "yield";
            return false;
          }
          eng.prefill_chunk_launch(r.cur, 0);
          r.side_flight = true;
          count_interleaved(r);
          unit_kind_ = "side_launch";
          return false;
        }
        if (contended && !eng.side_ready()) return false;
        r.side_flight = false;
        const bool last_chunk = eng.prefill_chunk_finish();
        side_mark_ = decode_passes_;
        unit_kind_ = "side_finish";
        if (!last_chunk) {
          if (other_decoding(r) && !hold_side(r) && !yield_prefill(r)) {  // the next chunk at once, unless a short prefill waits
            eng.prefill_chunk_launch(r.cur, 0);
            r.side_flight = true;
            count_interleaved(r);
          }
          return false;
        }
        r.prefilling = false;
        begin_decode(r);
        start_forks(r);
        return emit(r) && end(r);
      }
      if (eng.side_busy()) return false;  // another request's chunk is in flight: no prefill beside it
      if (yield_prefill(r)) {
        unit_kind_ = "yield";
        return false;
      }
      unit_kind_ = "chunk";
      const bool last = eng.prefill_chunk(r.cur, contended ? contended_ : 0);
      if (contended) count_interleaved(r);
      if (!last) return false;
      r.prefilling = false;
      begin_decode(r);
      start_forks(r);
      return emit(r) && end(r);
    }
    const auto now = std::chrono::steady_clock::now();
    (other_prefilling(r) ? r.beside : r.alone).add(std::chrono::duration<double>(now - r.last_step).count());
    r.last_step = now;
    eng.select_session(r.job.session);
    ++r.passes;
    ++decode_passes_;
    r.window.record(decode_passes_);
    unit_kind_ = "decode";
    {
      std::lock_guard<std::mutex> m(ctx_.metrics.mu);
      ctx_.metrics.batch_rows = spec_turn(r) && !eng.drafter_off(r.job.session, r.job.req.sampling)
                                    ? std::min(eng.draft_block(), eng.max_context() - static_cast<int>(eng.history().size()))
                                    : 1;
    }
    if (spec_turn(r)) {
      r.pending = eng.speculate(r.pending.back(), r.job.req.sampling, r.job.req.logprobs ? &r.rows : nullptr);
      // The verify checks only each row's top K for NaN and inf (the plain path's sample() checks the
      // whole row); a row reported as logprobs is checked whole, so a NaN never reaches the client.
      if (r.job.req.logprobs) require_finite(r.rows);
    } else {
      r.logits = eng.step(r.pending.back());
      r.rows.clear();  // emit reads r.logits, not an earlier batched step's row
      r.pending = {eng.sample(r.logits, r.job.req.sampling, eng.history())};
    }
    return emit(r) && end(r);
  }

  void count_interleaved(Running& r) {
    ++r.chunks_beside;
    std::lock_guard<std::mutex> m(ctx_.metrics.mu);
    ctx_.metrics.prefill_chunks_interleaved += 1;
  }

  // After the last prefill chunk (the engine's active session is the request's, its logits the prompt's).
  void begin_decode(Running& r) {
    Engine& eng = *ctx_.engine;
    const Job& j = r.job;
    const int ctx_left = eng.max_context() - static_cast<int>(j.prompt.size());
    r.limit = j.req.max_tokens > 0 ? std::min(j.req.max_tokens, ctx_left) : ctx_left;
    r.spec = eng.can_speculate(j.req.sampling);
    const SpecStats& s = eng.session_spec_stats(j.session);
    r.steps0 = s.steps;
    r.drafted0 = s.drafted;
    r.accepted0 = s.accepted;
    r.draft_seconds0 = s.draft_seconds;
    r.commit_seconds0 = s.commit_seconds;
    r.t_decode = r.last_step = std::chrono::steady_clock::now();
    r.dec.emplace(*ctx_.tok);
    r.stops.emplace(j.req.stop);
    if (j.req.repetition.on()) r.repeat.emplace(j.req.repetition);
    r.in_think = j.req.reasoning;  // the prompt ends inside <think>
    r.logits = eng.logits();
    // `pending` holds chosen tokens not yet emitted: the first from the prefill's logits, then a
    // speculative step's accepted drafts and its next token, or a plain step's sample.
    r.pending = {eng.sample(r.logits, j.req.sampling, eng.history())};
    r.rows.clear();
  }

  // Emits r's pending tokens, as generate() did per token. Returns true when the request is done.
  bool emit(Running& r) {
    Job& job = r.job;
    const int eos = ctx_.engine->config().eos_token;
    // Logprobs (API §3), only when asked: each pending token's logprobs come from the row it was chosen
    // from, `logits` on the plain path and the prefill, or the verify's rows that speculate() copies to
    // `rows` (one per pending token, in order). Nothing here runs otherwise.
    const size_t vocab = ctx_.engine->config().vocab;
    for (size_t j = 0; j < r.pending.size(); ++j) {
      const int t = r.pending[j];
      if (r.ttft < 0) r.ttft = std::chrono::duration<double>(std::chrono::steady_clock::now() - job.received).count();
      if (job.cancelled->load()) {
        r.finish = "cancelled";
        return true;
      }
      if (t == eos || t == ctx_.im_end) {
        r.finish = "stop";
        r.ended_by_eos = true;
        return true;
      }
      ++r.produced;
      if (job.req.logprobs) {
        const float* row = r.rows.empty() ? r.logits.data() : r.rows.data() + j * vocab;
        job.sink.logprobs(top_logprobs(std::span<const float>(row, vocab), t, job.req.top_logprobs));
      }
      if (r.in_think) {
        ++r.reasoning;
        if (t == ctx_.think_end) {
          r.in_think = false;
          r.newlines_after_think = 2;
        }
      } else if (r.newlines_after_think > 0) {
        if (t == ctx_.blank_line && r.newlines_after_think == 2) {
          ++r.reasoning;
          r.newlines_after_think = 0;
        } else if (t == ctx_.newline) {
          ++r.reasoning;
          --r.newlines_after_think;
        } else {
          r.newlines_after_think = 0;
        }
      }
      const std::string ready = r.stops->push(r.dec->push(t));
      if (!ready.empty()) job.sink.text(ready);
      if (r.stops->stopped()) {
        r.finish = "stop";
        return true;
      }
      // The repetition stop rule (API §9): on the produced tokens, reasoning included; the token that completes the
      // loop has been sent, nothing after it is.
      if (r.repeat && r.repeat->push(t)) {
        r.finish = "repetition";
        std::lock_guard<std::mutex> m(ctx_.metrics.mu);
        ctx_.metrics.repetition_stops += 1;
        return true;
      }
      if (r.produced >= r.limit) return true;
    }
    return false;
  }

  // The request's end: what the decoder and the stop scanner still hold, the counters, the log line, done.
  bool end(Running& r) {
    Engine& eng = *ctx_.engine;
    Job& job = r.job;
    if (!r.stops->stopped()) {
      std::string rest = r.stops->push(r.dec->flush());
      if (!r.stops->stopped()) rest += r.stops->finish();
      if (!rest.empty()) job.sink.text(rest);
    }
    const auto now = std::chrono::steady_clock::now();
    const double decode_s = std::chrono::duration<double>(now - r.t_decode).count();
    const SpecStats& s = eng.session_spec_stats(job.session);
    const long steps = s.steps - r.steps0;
    const EngineStats& stats = r.cur.stats;
    {
      Metrics& m = ctx_.metrics;
      std::lock_guard<std::mutex> l(m.mu);
      m.prompt_tokens += job.prompt.size();
      m.cached_tokens += stats.reused_tokens;
      m.generation_tokens += r.produced + (r.ended_by_eos ? 1 : 0);  // SGLang counts the stop token
      m.verify_calls += r.passes;
      m.e2e_sum += std::chrono::duration<double>(now - job.received).count();
      m.e2e_count += 1;
      if (r.ttft >= 0) {
        m.ttft_sum += r.ttft;
        m.ttft_count += 1;
      }
      m.drafted += s.drafted - r.drafted0;
      m.accepted_drafts += s.accepted - r.accepted0;
      if (!job.req.drafter_id.empty()) {
        Metrics::DrafterCounters& d = m.by_drafter[job.req.drafter_id];
        d.requests += 1;
        d.steps += steps;
        d.drafted += s.drafted - r.drafted0;
        d.accepted += s.accepted - r.accepted0;
        d.draft_seconds += s.draft_seconds - r.draft_seconds0;
        d.commit_seconds += s.commit_seconds - r.commit_seconds0;
        const std::string how = job.req.drafter_how;
        (how == "declared" ? m.chosen_declared : how == "inferred" ? m.chosen_inferred : m.chosen_default) += 1;
      }
      if (r.drafter_changed) {
        m.drafter_changes += 1;
        m.drafter_change_tokens += stats.prefill_tokens;
        m.drafter_change_seconds += stats.prefill_seconds;
      }
      m.spec = eng.spec_stats_total();
    }
    // Step periods (M4 §3): mean and max, apart from those with another request's prefill chunk between.
    char periods[200] = "";
    if (r.alone.n + r.beside.n > 0)
      std::snprintf(periods, sizeof periods, "; step period %.1f ms mean, %.1f max over %d; beside a prefill %.1f ms mean, %.1f max over %d",
                    r.alone.mean_ms(), 1e3 * r.alone.max, r.alone.n, r.beside.mean_ms(), 1e3 * r.beside.max, r.beside.n);
    // Only with several drafters, so a one-drafter server's log reads as it always did.
    const std::string drafter_note = eng.drafters() > 1 ? ", drafter " + job.req.drafter_id + " (" + job.req.drafter_how +
                                                              (r.drafter_changed ? ", changed)" : ")")
                                                        : "";
    std::fprintf(stderr,
                 "request: session %d%s%s, %zu prompt tokens (%d reused%s), prefill %.2f s (%d chunks, %d beside a decode, "
                 "%.0f tok/s of compute); %d tokens in %.2f s (%.1f tok/s), %s; %d passes (%d batched)%s%s\n",
                 job.session, job.fresh ? " (new)" : "", drafter_note.c_str(), job.prompt.size(), stats.reused_tokens,
                 stats.store_tokens > 0 ? ", from the prefix KV store" : "", stats.prefill_seconds,
                 stats.chunks, r.chunks_beside,
                 stats.compute_seconds > 0 ? stats.prefill_tokens / stats.compute_seconds : 0.0, r.produced, decode_s,
                 decode_s > 0 ? r.produced / decode_s : 0.0, r.finish.c_str(), r.passes, r.batched,
                 r.spec && steps > 0 ? (", " + std::to_string(double(r.produced) / std::max(1, r.passes)).substr(0, 4) +
                                        " tokens/pass").c_str()
                                     : "",
                 periods);
    job.sink.done(r.finish == "cancelled" ? "stop" : r.finish, static_cast<int>(job.prompt.size()), r.produced,
                  stats.reused_tokens, r.reasoning);
    return true;
  }

  ServerContext& ctx_;
  double protect_seconds_;
  int batch_, contended_, side_share_;
  bool fair_pairing_;  // --fair-pairing: partners by fewest recent decode passes (serve/pairing.hpp)
  int fanout_plain_above_;  // --fanout-plain-above: fanout_plain(), 0 for never
  int fanout_gang_above_;   // --fanout-gang-above: gang_prefill() and burst_more(), 0 for never, -1 as plain
  bool fanout_plain_explicit_;  // --fanout-plain-above was given: it holds in the wide regime too
  int fanout_wide_above_;       // the wide regime above this many drafting requests (fanout_regime()), 0 for never
  int base_block_, wide_block_;  // the draft block below and in the wide regime (--draft-block, --fanout-draft-block)
  bool wide_ = false;            // this turn is in the wide regime (fanout_regime())
  bool gang_unit_ = false;     // this turn's unit is a gang_prefill()
  bool gang_holding_ = false;  // the decode passes are holding for short prefills, since gang_t0_
  std::chrono::steady_clock::time_point gang_t0_;
  StepTrace trace_{std::getenv("RAGON_STEP_TRACE")};  // the per-step trace, when the variable names a file
  const char* unit_kind_ = nullptr;  // what the last unit did, for the trace (null: nothing worth a line)
  mutable std::vector<StepCandidate> cand_;  // the partner choice's view for the trace (filled while it is on)
  long decode_passes_ = 0;  // decode passes run (a batched step counts once), for --side-share
  long side_mark_ = -1;     // decode_passes_ when the last side chunk finished (--side-share)
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<Job> queue_;
  bool stop_ = false;
  bool dirty_ = false;  // the queue or the running set changed since pick() last looked
  bool kv_wait_ = false;  // pick() left a job waiting for KV pages that eviction or retirement frees (P4b)
  long admitted_ = 0;     // requests admitted so far (Running::admitted)
  std::chrono::steady_clock::time_point last_look_{};
  // Touched by the engine thread only.
  std::vector<std::unique_ptr<Running>> running_;
  size_t turn_ = 0;
  std::thread thread_;
};

// Lives as long as either the HTTP handler or the job needs it; touched only on the handler's thread.
struct Exchange {
  proxygen::ResponseHandler* downstream = nullptr;  // null once the client is gone
  folly::EventBase* evb = nullptr;
  std::shared_ptr<std::atomic<bool>> cancelled = std::make_shared<std::atomic<bool>>(false);
  bool keepalive = false;  // a keep-alive timer is armed (arm_keepalive); cleared by stop_keepalive
  int seq = 0;             // the Responses route's sequence numbers, stamped in write order (emit_event)
};

void post(const std::shared_ptr<Exchange>& ex, std::function<void(proxygen::ResponseHandler*)> fn) {
  ex->evb->runInEventBaseThread([ex, fn = std::move(fn)] {
    if (ex->downstream) fn(ex->downstream);
  });
}

// Keep-alives (API §3, §4). While a streamed request has produced nothing (waiting in the worker's queue,
// or in prefill) the request's event base writes one every `interval_ms`, so a client with an idle
// timeout (Codex ends a stream after 300 s without an event) keeps the stream through a long prefill or
// a queue ahead. Armed on the handler's thread right before the job is submitted. Stopped by
// stop_keepalive, which the engine thread posts before its first write (the first token, done, or
// error): posts from one thread run in order, so a tick never follows that write, and since every write
// runs on the event base a tick never interleaves with one. A tick after EOM, or after a disconnect,
// finds `downstream` null (Handler::finish) and does nothing. The timer itself is not cancelled
// (runAfterDelay has no handle): a cleared flag makes the next tick a no-op that does not re-arm, which
// releases its closure within one interval.
void keepalive_tick(std::shared_ptr<Exchange> ex, int interval_ms, std::function<void(proxygen::ResponseHandler*)> write) {
  ex->evb->tryRunAfterDelay(
      [ex, interval_ms, write] {
        if (!ex->keepalive || !ex->downstream) return;
        write(ex->downstream);
        keepalive_tick(ex, interval_ms, write);
      },
      static_cast<uint32_t>(interval_ms));
}

void arm_keepalive(const std::shared_ptr<Exchange>& ex, int interval_ms, std::function<void(proxygen::ResponseHandler*)> write) {
  if (interval_ms <= 0) return;
  ex->keepalive = true;
  keepalive_tick(ex, interval_ms, std::move(write));
}

void stop_keepalive(const std::shared_ptr<Exchange>& ex) {
  ex->evb->runInEventBaseThread([ex] { ex->keepalive = false; });
}

// Writes one Responses event, numbered in write order. The event base's thread only, with a client.
void emit_event(Exchange& ex, const std::string& type, json payload) {
  payload["type"] = type;
  payload["sequence_number"] = ex.seq++;
  ResponseBuilder(ex.downstream).body("event: " + type + "\ndata: " + dump_json(payload) + "\n\n").send();
}

void send_json(proxygen::ResponseHandler* d, int status, const json& body, const std::string& drafter = "") {
  ResponseBuilder out(d);
  out.status(status, status == 200   ? "OK"
                     : status == 400 ? "Bad Request"
                     : status == 401 ? "Unauthorized"
                     : status == 403 ? "Forbidden"
                     : status == 404 ? "Not Found"
                     : status == 413 ? "Payload Too Large"
                                     : "Error")
      .header("Content-Type", "application/json");
  if (!drafter.empty()) out.header("x-ragon-drafter", drafter);  // MULTI_DRAFTER §2.5: outside the body
  out.body(dump_json(body)).sendWithEOM();
}

json error_body(const std::string& msg, const std::string& type, const std::string& code = "") {
  json e = {{"message", msg}, {"type", type}};
  if (!code.empty()) e["code"] = code;
  return json{{"error", e}};
}

// The type that goes with an error code.
std::string error_type(const std::string& code) {
  return code == kContextLengthExceeded ? "invalid_request_error" : "server_error";
}

class Handler : public proxygen::RequestHandler {
 public:
  Handler(ServerContext& ctx, Scheduler& scheduler) : ctx_(ctx), scheduler_(scheduler) {}

  void onRequest(std::unique_ptr<HTTPMessage> headers) noexcept override {
    method_ = headers->getMethodString();
    path_ = headers->getPath();
    share_scope_ = share_scope_of(headers->getHeaders().getSingleOrEmpty("X-Ragon-Share-Scope"));
    authorization_ = headers->getHeaders().getSingleOrEmpty("Authorization");
    ex_->downstream = downstream_;
    ex_->evb = folly::EventBaseManager::get()->getEventBase();
  }
  void onBody(std::unique_ptr<folly::IOBuf> body) noexcept override {
    body_.append(std::move(body));
    if (body_.chainLength() > (64u << 20)) too_large_ = true;
  }
  void onEOM() noexcept override {
    try {
      route();
    } catch (const std::exception& e) {
      send_json(downstream_, 500, error_body(e.what(), "server_error"));
    }
  }
  void onUpgrade(proxygen::UpgradeProtocol) noexcept override {}
  void requestComplete() noexcept override { finish(); }
  void onError(proxygen::ProxygenError) noexcept override { finish(); }

 private:
  void finish() {
    ex_->downstream = nullptr;
    ex_->cancelled->store(true);
    delete this;
  }

  // Admission control on host memory: answers 503 before anything is queued or streamed while MemAvailable is below
  // the reserve (serve/admission.hpp). Returns whether it did.
  bool refuse_for_memory() {
    if (ctx_.admit_reserve_mb <= 0) return false;
    const std::string why = refuse_request(mem_available_mb(), ctx_.admit_reserve_mb);
    if (why.empty()) return false;
    {
      std::lock_guard<std::mutex> l(ctx_.metrics.mu);
      ctx_.metrics.memory_refusals += 1;
    }
    std::fprintf(stderr, "request refused: %s\n", why.c_str());
    ResponseBuilder(downstream_)
        .status(503, "Service Unavailable")
        .header("Content-Type", "application/json")
        .header("Retry-After", "5")
        .body(dump_json(error_body(why, "server_error", "host_memory_low")))
        .sendWithEOM();
    return true;
  }

  void route() {
    if (method_ == "GET" && (path_ == "/health" || path_ == "/healthz")) {
      ResponseBuilder(downstream_).status(200, "OK").body("ok").sendWithEOM();
    } else if (method_ == "GET" && path_ == "/v1/models") {
      json model = {{"id", ctx_.model_name}, {"object", "model"}, {"created", std::time(nullptr)},
                    {"owned_by", "ragon-engine"}, {"root", ctx_.model_name},
                    {"max_model_len", ctx_.engine->max_context()}};
      json data = json::array({model});
      if (ctx_.lora_lib) {  // each loaded adapter as base+name (LORA §4), after the base the launcher reads first
        std::lock_guard<std::mutex> l(ctx_.lora_mu);
        for (const std::string& n : ctx_.lora_lib->names()) {
          json m = model;
          m["id"] = ctx_.model_name + "+" + n;
          m["parent"] = ctx_.model_name;
          data.push_back(m);
        }
      }
      send_json(downstream_, 200, json{{"object", "list"}, {"data", data}});
    } else if (method_ == "GET" && (path_ == "/model_info" || path_ == "/get_model_info")) {
      // SGLang's answer, which Ragon's launcher reads to advertise image input (VISION §20.6).
      send_json(downstream_, 200,
                json{{"model_path", ctx_.model_name}, {"is_generation", true}, {"has_image_understanding", ctx_.vision}});
    } else if (method_ == "GET" && path_ == "/metrics") {
      ResponseBuilder(downstream_)
          .status(200, "OK")
          .header("Content-Type", "text/plain; version=0.0.4")
          .body(ctx_.metrics.render(ctx_.model_name))
          .sendWithEOM();
    } else if (method_ == "GET" && path_ == "/spec_stats") {
      send_json(downstream_, 200, ctx_.metrics.spec_json());
    } else if (method_ == "POST" && (path_ == "/load_lora_adapter" || path_ == "/v1/load_lora_adapter" ||
                                     path_ == "/unload_lora_adapter" || path_ == "/v1/unload_lora_adapter")) {
      lora_route(path_.find("unload") == std::string::npos);
    } else if (method_ == "POST" && (path_ == "/v1/responses" || path_ == "/v1/chat/completions" ||
                                     path_ == "/v1/completions") &&
               refuse_for_memory()) {
      return;
    } else if (method_ == "POST" && path_ == "/v1/responses") {
      generate_responses();
    } else if (method_ == "POST" && (path_ == "/v1/chat/completions" || path_ == "/v1/completions")) {
      generate(path_ == "/v1/chat/completions");
    } else {
      send_json(downstream_, 404, error_body("no route for " + method_ + " " + path_, "not_found"));
    }
  }

  // Whether the request carries the server's key (--api-key) as a Bearer token; answers 403 (no key: the routes are
  // off) or 401 when it does not.
  bool authorized() {
    if (ctx_.api_key.empty()) {
      send_json(downstream_, 403, error_body("the adapter routes are off: the server was started without --api-key",
                                             "invalid_request_error", "forbidden"));
      return false;
    }
    const std::string want = "Bearer " + ctx_.api_key;
    size_t diff = want.size() ^ authorization_.size();  // compared in full, whatever differs
    for (size_t i = 0; i < want.size(); ++i)
      diff |= size_t(static_cast<unsigned char>(want[i]) ^ (i < authorization_.size() ? static_cast<unsigned char>(authorization_[i]) : 0u));
    if (diff == 0) return true;
    send_json(downstream_, 401, error_body("a wrong or missing API key", "invalid_request_error", "invalid_api_key"));
    return false;
  }

  // POST /load_lora_adapter {"lora_name": N[, "lora_path": D]} and /unload_lora_adapter {"lora_name": N} (LORA §4,
  // SGLang's routes; also under /v1/, vLLM's). A load reads the adapter from the library's directory D (default N):
  // one name in the library, never a path the request gives, and the library's realpath check keeps symlinks in it.
  void lora_route(bool load) {
    if (!authorized()) return;
    if (!ctx_.lora_lib) {
      send_json(downstream_, 400,
                error_body("the engine was started without an adapter library (--lora-library)", "invalid_request_error"));
      return;
    }
    std::string name, dir;
    try {
      std::string text;
      if (auto buf = body_.move()) text = buf->moveToFbString().toStdString();
      const json body = json::parse(text);
      name = body.value("lora_name", std::string());
      dir = body.value("lora_path", name);
    } catch (const std::exception& e) {
      send_json(downstream_, 400, error_body(std::string("bad request body: ") + e.what(), "invalid_request_error"));
      return;
    }
    if (!lora_name_ok(name) || !lora_name_ok(dir)) {
      send_json(downstream_, 400,
                error_body("lora_name and lora_path are each one name in the adapter library (1 to 64 of A-Z a-z 0-9 . _ -)",
                           "invalid_request_error"));
      return;
    }
    std::lock_guard<std::mutex> l(ctx_.lora_mu);
    if (!load) {
      if (!ctx_.lora_lib->unload(name)) {
        send_json(downstream_, 404, error_body("adapter " + name + " is not loaded", "invalid_request_error", "model_not_found"));
        return;
      }
      std::fprintf(stderr, "lora: unloaded %s\n", name.c_str());
      send_json(downstream_, 200, json{{"success", true}, {"message", "unloaded " + name}, {"loaded_adapters", ctx_.lora_lib->names()}});
      return;
    }
    try {
      const auto a = ctx_.lora_lib->load(name, ctx_.lora_lib->root() + "/" + dir);
      std::fprintf(stderr, "lora: loaded %s (rank %d, %zu MB%s)\n", name.c_str(), a->target.rank, a->bytes() >> 20,
                   a->has_drafter ? ", with the drafter's" : "");
    } catch (const std::exception& e) {
      send_json(downstream_, 400, error_body(e.what(), "invalid_request_error"));
      return;
    }
    send_json(downstream_, 200, json{{"success", true}, {"message", "loaded " + name}, {"loaded_adapters", ctx_.lora_lib->names()}});
  }

  // The request's adapter from its model (adapter_of, LORA §4): nullptr for the base. Throws std::invalid_argument (a
  // malformed name) or ModelNotFound (another base, an adapter not loaded); mixes the adapter into the share scope, so
  // its sessions and prefix entries are its own.
  std::shared_ptr<const LoraAdapter> resolve_adapter(Request& req) {
    const std::string name = adapter_of(req.model, ctx_.model_name);
    if (name.empty()) return nullptr;
    std::shared_ptr<const LoraAdapter> a;
    if (ctx_.lora_lib) {
      std::lock_guard<std::mutex> l(ctx_.lora_mu);
      a = ctx_.lora_lib->find(name);
    }
    if (!a) throw ModelNotFound("the model " + req.model + " does not exist: no adapter " + name + " is loaded");
    req.share_scope = lora_scope(req.share_scope, a.get());
    return a;
  }

  // The model a response names: the base, or base+adapter.
  std::string model_of(const Job& job) const {
    return job.lora ? ctx_.model_name + "+" + job.lora->name : ctx_.model_name;
  }

  // Images (VISION §20.3 (c), §20.6): each URL decoded and preprocessed here, on the HTTP thread; each <|image_pad|>
  // of the tokenized prompt becomes the image's tokens, all its id (§20.7). Throws ImageError or invalid_argument.
  void attach_images(Job& job) {
    // Without the tower an image part rendered as `[image]` and no URL was collected: nothing to do (a literal
    // <|image_pad|> in the text stays a token, as before).
    if (!ctx_.vision) return;
    int pads = 0;
    for (int t : job.prompt) pads += t == ctx_.image_pad;
    if (job.req.image_urls.empty() && pads == 0) return;
    serve::attach_images(job.prompt, ctx_.image_pad, job.req.image_urls, *ctx_.images, ctx_.processor, job.images);
  }

  void generate(bool chat) {
    if (too_large_) {
      send_json(downstream_, 413, error_body("request body too large", "invalid_request_error"));
      return;
    }
    Job job;
    try {
      std::string text;
      if (auto buf = body_.move()) text = buf->moveToFbString().toStdString();
      json body = json::parse(text);
      job.req = parse_request(body, chat, ctx_.vision);
      job.req.share_scope = share_scope_;
      job.lora = resolve_adapter(job.req);
      choose_drafter(job);
      job.req.repetition = resolve_repetition(job.req, ctx_.repetition);
      job.prompt = job.req.prompt_ids.empty() ? ctx_.tok->encode(job.req.prompt_text) : job.req.prompt_ids;
      if (!job.req.prompt_ids.empty())
        for (int t : job.prompt)
          if (t < 0 || t >= ctx_.engine->config().vocab)
            throw std::invalid_argument("prompt token " + std::to_string(t) + " is outside the vocabulary");
      attach_images(job);
    } catch (const ImageError& e) {
      send_json(downstream_, 400, error_body(e.what(), "invalid_request_error", e.code));
      return;
    } catch (const ModelNotFound& e) {
      send_json(downstream_, 404, error_body(e.what(), "invalid_request_error", "model_not_found"));
      return;
    } catch (const std::exception& e) {
      send_json(downstream_, 400, error_body(e.what(), "invalid_request_error"));
      return;
    }
    if (exceeds_context(job.prompt.size(), ctx_.engine->max_context())) {  // refused before any header
      send_json(downstream_, 400, context_overflow_error(job.prompt.size(), ctx_.engine->max_context()));
      return;
    }
    job.cancelled = ex_->cancelled;
    const std::string id = new_id(chat ? "chatcmpl-" : "cmpl-");
    const long created = std::time(nullptr);
    const std::string model = model_of(job);
    auto ex = ex_;
    const Request req = job.req;

    if (req.stream) {
      ResponseBuilder out(downstream_);
      out.status(200, "OK").header("Content-Type", "text/event-stream").header("Cache-Control", "no-cache");
      if (!req.drafter_id.empty()) out.header("x-ragon-drafter", req.drafter_id);  // MULTI_DRAFTER §2.5
      out.send();
    }
    // Shared by the job's callbacks (engine thread only).
    struct State {
      OutputParser parser;
      std::string content, reasoning;
      std::vector<ToolCall> calls;
      bool first = true;
      bool keepalive = false;  // keep-alives are armed and nothing has been written yet
      // Logprobs in the route's shape (API §3): chat, the `content` entries; completions, the object with
      // `tokens`, `token_logprobs`, `top_logprobs` and `text_offset`. Streamed, what the next chunk takes.
      json logprobs;
      int text_chars = 0;  // characters of output text so far (completions text_offset)
      State(bool r, const json& t) : parser(r, t) {}
    };
    // What a request's choices (`n`, BATCHING P4d) share: one response (engine thread only).
    struct Fan {
      int left = 1;            // choices not done yet
      bool keepalive = false;  // keep-alives are armed and nothing has been written yet
      bool closed = false;     // an error answered for every choice
      json choices = json::array();
      int completion = 0, reasoning = 0, cached = 0;
    };
    auto fan = std::make_shared<Fan>();
    fan->left = req.n;
    for (int i = 0; i < req.n; ++i) fan->choices.push_back(nullptr);
    // One choice's callbacks: its index in the response, its own parser and logprobs.
    auto sink_for = [&](const int index) {
      JobSink sink;
      auto st = std::make_shared<State>(chat && req.reasoning, req.tools);
      auto fresh_logprobs = [chat]() {
        return chat ? json(json::array())
                    : json{{"tokens", json::array()}, {"token_logprobs", json::array()}, {"top_logprobs", json::array()},
                           {"text_offset", json::array()}};
      };
      st->logprobs = fresh_logprobs();
      auto pending_logprobs = [=]() { return chat ? !st->logprobs.empty() : !st->logprobs["tokens"].empty(); };
      auto take_logprobs = [=]() {  // choices[0].logprobs from what has been collected, which starts afresh
        json out = chat ? json{{"content", st->logprobs}} : st->logprobs;
        st->logprobs = fresh_logprobs();
        return out;
      };

      auto sse = [ex, fan](const json& chunk) {
        if (fan->closed) return;  // an error answered for every choice
        std::string data = "data: " + dump_json(chunk) + "\n\n";
        post(ex, [data](proxygen::ResponseHandler* d) { ResponseBuilder(d).body(data).send(); });
      };
      // Before the first write of any kind (the first token, done, or error): no keep-alive after it.
      auto stop_ticks = [=]() {
        if (!fan->keepalive) return;
        fan->keepalive = false;
        stop_keepalive(ex);
      };
      auto chunk_of = [=](json delta, json finish, json logprobs = json()) {
        json c = {{"id", id}, {"object", chat ? "chat.completion.chunk" : "text_completion"}, {"created", created},
                  {"model", model}};
        json choice = {{"index", index}};
        if (chat) choice["delta"] = std::move(delta);
        else choice["text"] = delta.is_string() ? delta : json("");
        if (!logprobs.is_null()) choice["logprobs"] = std::move(logprobs);
        choice["finish_reason"] = std::move(finish);
        c["choices"] = json::array({choice});
        return c;
      };
      // A streamed chunk carries the logprobs of the tokens produced since the previous chunk (the text of
      // a token may follow its entry by a chunk when the decoder or the parser held it back).
      auto chunk_logprobs = [=]() { return req.logprobs ? take_logprobs() : json(); };
      auto deliver = [=](const OutputParser::Delta& d) {
        st->content += d.content;
        st->reasoning += d.reasoning;
        if (!req.stream) {
          st->calls.insert(st->calls.end(), d.tool_calls.begin(), d.tool_calls.end());
          return;
        }
        if (!chat) {
          if (!d.content.empty()) sse(chunk_of(json(d.content), nullptr, chunk_logprobs()));
          return;
        }
        json delta = json::object();
        if (st->first) {
          delta["role"] = "assistant";
          st->first = false;
        }
        if (!d.reasoning.empty()) delta["reasoning_content"] = d.reasoning;
        if (!d.content.empty()) delta["content"] = d.content;
        if (!d.tool_calls.empty()) {
          json calls = json::array();
          for (const auto& c : d.tool_calls) {
            calls.push_back({{"index", static_cast<int>(st->calls.size())},
                             {"id", new_id("call_")},
                             {"type", "function"},
                             {"function", {{"name", c.name}, {"arguments", c.arguments}}}});
            st->calls.push_back(c);
          }
          delta["tool_calls"] = calls;
        }
        if (delta.size() > 0) sse(chunk_of(delta, nullptr, chunk_logprobs()));
      };

      sink.text = [=](const std::string& piece) {
        stop_ticks();
        deliver(chat ? st->parser.push(piece) : OutputParser::Delta{{}, piece, {}});
      };
      if (req.logprobs) {
        // One entry per produced token, in vLLM's shapes. `token` is the token's text (U+FFFD where a
        // character is split across tokens, as dump_json writes it) and `bytes` its exact bytes, so a client
        // can reassemble a character from its pieces. text_offset counts characters of the output text (a
        // byte that is not a continuation byte starts one), so a character split across tokens counts once.
        const Tokenizer* tok = ctx_.tok;
        auto token_json = [tok](const TokenLogprob& e) {
          const std::string text = tok->token_bytes(e.token);
          json bytes = json::array();
          for (const unsigned char c : text) bytes.push_back(static_cast<int>(c));
          return json{{"token", text}, {"logprob", e.logprob}, {"bytes", bytes}};
        };
        const int k = req.top_logprobs;
        sink.logprobs = [=](const std::vector<TokenLogprob>& entries) {
          if (entries.empty()) return;
          if (chat) {
            json e = token_json(entries[0]);
            json top = json::array();
            for (size_t i = 0; i < entries.size() && i < static_cast<size_t>(k); ++i) top.push_back(token_json(entries[i]));
            e["top_logprobs"] = std::move(top);
            st->logprobs.push_back(std::move(e));
            return;
          }
          const std::string text = tok->token_bytes(entries[0].token);
          json top = json::object();
          for (const TokenLogprob& e : entries) top[tok->token_bytes(e.token)] = e.logprob;
          st->logprobs["tokens"].push_back(text);
          st->logprobs["token_logprobs"].push_back(entries[0].logprob);
          st->logprobs["top_logprobs"].push_back(std::move(top));
          st->logprobs["text_offset"].push_back(st->text_chars);
          for (const unsigned char c : text) st->text_chars += (c & 0xC0) != 0x80;
        };
      }
      sink.error = [=](const std::string& msg, const std::string& code) {
        stop_ticks();
        if (fan->closed) return;
        const json error = error_body(msg, error_type(code), code);
        if (req.stream) {  // the headers are out: end the stream properly after the error
          const std::string tail = stream_error_tail(error, chunk_of(json::object(), kStreamErrorFinish));
          post(ex, [tail](proxygen::ResponseHandler* d) { ResponseBuilder(d).body(tail).sendWithEOM(); });
        } else {
          const int status = code == kContextLengthExceeded ? 400 : 500;
          post(ex, [error, status](proxygen::ResponseHandler* d) { send_json(d, status, error); });
        }
        fan->closed = true;
        if (req.n > 1) ex->cancelled->store(true);  // the other choices stop: the response is over
      };
      sink.done = [=](const std::string& finish_in, int prompt_tokens, int completion_tokens, int cached,
                          int reasoning_tokens) {
        stop_ticks();
        if (fan->closed) return;
        if (chat) deliver(st->parser.finish());
        const std::string finish = !st->calls.empty() && finish_in == "stop" ? "tool_calls" : finish_in;
        // The response's usage counts the prompt once, as its first choice saw it, and every choice's output.
        fan->completion += completion_tokens;
        fan->reasoning += reasoning_tokens;
        if (index == 0) fan->cached = cached;
        const bool last = --fan->left == 0;
        json usage = {{"prompt_tokens", prompt_tokens},
                      {"completion_tokens", fan->completion},
                      {"total_tokens", prompt_tokens + fan->completion},
                      {"prompt_tokens_details", {{"cached_tokens", fan->cached}}}};
        if (chat) usage["completion_tokens_details"] = {{"reasoning_tokens", fan->reasoning}};
        if (req.stream) {
          // Entries still pending (their text held back to the end) ride on the finish chunk.
          sse(chunk_of(json::object(), finish, req.logprobs && pending_logprobs() ? take_logprobs() : json()));
          if (!last) return;
          if (req.include_usage) {
            json u = {{"id", id}, {"object", chat ? "chat.completion.chunk" : "text_completion"}, {"created", created},
                      {"model", model}, {"choices", json::array()}, {"usage", usage}};
            sse(u);
          }
          post(ex, [](proxygen::ResponseHandler* d) { ResponseBuilder(d).body("data: [DONE]\n\n").sendWithEOM(); });
          return;
        }
        json choice = {{"index", index}, {"finish_reason", finish}};
        if (chat) {
          json msg = {{"role", "assistant"}, {"content", st->content.empty() && !st->calls.empty() ? json() : json(st->content)}};
          if (!st->reasoning.empty()) msg["reasoning_content"] = st->reasoning;
          if (!st->calls.empty()) {
            json calls = json::array();
            for (const auto& c : st->calls)
              calls.push_back({{"id", new_id("call_")}, {"type", "function"},
                               {"function", {{"name", c.name}, {"arguments", c.arguments}}}});
            msg["tool_calls"] = calls;
          }
          choice["message"] = msg;
        } else {
          choice["text"] = st->content;
        }
        if (req.logprobs) choice["logprobs"] = take_logprobs();
        fan->choices[index] = std::move(choice);
        if (!last) return;
        json body = {{"id", id}, {"object", chat ? "chat.completion" : "text_completion"}, {"created", created},
                     {"model", model}, {"choices", fan->choices}, {"usage", usage}};
        post(ex, [body, drafter = req.drafter_id](proxygen::ResponseHandler* d) { send_json(d, 200, body, drafter); });
      };
      return sink;
    };
    job.sink = sink_for(0);
    for (int i = 1; i < req.n; ++i) {
      Job fork;
      fork.received = job.received;
      fork.req = req;
      fork.req.n = 1;
      fork.req.session_key.clear();
      // Not seed + i: a token's seed is the request's seed plus the history's length, which would give one choice's
      // stream shifted by a token. (A seed of 0 is a random one: every session draws its own.)
      if (req.sampling.seed != 0) fork.req.sampling.seed = req.sampling.seed + uint64_t(i) * 0x9E3779B97F4A7C15ull;
      fork.prompt = job.prompt;
      fork.images = job.images;
      fork.lora = job.lora;
      fork.cancelled = job.cancelled;
      fork.sink = sink_for(i);
      fork.fork = true;
      job.forks.push_back(std::move(fork));
    }
    if (req.stream && ctx_.keepalive_ms > 0) {  // an SSE comment, which every conforming parser ignores (API §3)
      fan->keepalive = true;
      arm_keepalive(ex, ctx_.keepalive_ms, [](proxygen::ResponseHandler* d) {
        ResponseBuilder(d).body(std::string(": keep-alive\n\n")).send();
      });
    }
    scheduler_.submit(std::move(job));
  }

  // The Responses API, streamed as the event sequence Ragon's agent parses: response.created, then
  // per output item output_item.added, its text deltas and output_item.done, then response.completed.
  void generate_responses() {
    if (too_large_) {
      send_json(downstream_, 413, error_body("request body too large", "invalid_request_error"));
      return;
    }
    Job job;
    try {
      std::string text;
      if (auto buf = body_.move()) text = buf->moveToFbString().toStdString();
      job.req = parse_responses_request(json::parse(text), ctx_.vision);
      job.req.share_scope = share_scope_;
      job.lora = resolve_adapter(job.req);
      choose_drafter(job);
      job.req.repetition = resolve_repetition(job.req, ctx_.repetition);
      job.prompt = ctx_.tok->encode(job.req.prompt_text);
      attach_images(job);
    } catch (const ImageError& e) {
      send_json(downstream_, 400, error_body(e.what(), "invalid_request_error", e.code));
      return;
    } catch (const ModelNotFound& e) {
      send_json(downstream_, 404, error_body(e.what(), "invalid_request_error", "model_not_found"));
      return;
    } catch (const std::exception& e) {
      send_json(downstream_, 400, error_body(e.what(), "invalid_request_error"));
      return;
    }
    job.cancelled = ex_->cancelled;
    const Request req = job.req;
    const bool repetition_completed = ctx_.repetition_completed;
    auto ex = ex_;
    const std::string model = model_of(job);
    const long created = std::time(nullptr);

    // Events are numbered when they are written, on the event base (emit_event), so a keep-alive written
    // by the timer and an event posted by the engine thread share one counter and stay consecutive.
    auto event = [=](const std::string& type, json payload) {
      if (!req.stream) return;
      post(ex, [ex, type, payload](proxygen::ResponseHandler*) { emit_event(*ex, type, payload); });
    };
    struct State {
      OutputParser parser;
      ResponseItems items;  // the output items and their events
      std::string id = new_id("resp_");
      bool keepalive = false;  // keep-alives are armed and nothing has been written yet
      State(bool r, const json& t, std::vector<std::string> c, ResponseItems::Emit e)
          : parser(r, t), items(std::move(c), std::move(e)) {}
    };
    auto st = std::make_shared<State>(req.reasoning, req.tools, req.custom_tools, event);
    auto response_obj = [=](const std::string& status) {
      return json{{"id", st->id}, {"object", "response"}, {"created_at", created}, {"status", status},
                  {"model", model}, {"output", st->items.output()}, {"tools", json::array()}};
    };
    // Before the first write of any kind (the first token, done, or error): no keep-alive after it.
    auto stop_ticks = [=]() {
      if (!st->keepalive) return;
      st->keepalive = false;
      stop_keepalive(ex);
    };
    auto deliver = [=](const OutputParser::Delta& d) { st->items.deliver(d); };

    if (req.stream) {
      ResponseBuilder out(downstream_);
      out.status(200, "OK").header("Content-Type", "text/event-stream").header("Cache-Control", "no-cache");
      if (!req.drafter_id.empty()) out.header("x-ragon-drafter", req.drafter_id);  // MULTI_DRAFTER §2.5
      out.send();
    }
    event("response.created", {{"response", response_obj("in_progress")}});

    // Streamed, an error is response.failed carrying its code (context_length_exceeded is what makes
    // Codex compact; server_error is retried); not streamed, an HTTP error with the same code.
    auto fail = [=](const std::string& msg, const std::string& code) {
      stop_ticks();
      if (req.stream) {
        json failed = response_obj("failed");
        failed["error"] = {{"code", code}, {"message", msg}};
        event("response.failed", {{"response", failed}});
        post(ex, [](proxygen::ResponseHandler* d) { ResponseBuilder(d).sendWithEOM(); });
      } else {
        const json error = error_body(msg, error_type(code), code);
        const int status = code == kContextLengthExceeded ? 400 : 500;
        post(ex, [error, status](proxygen::ResponseHandler* d) { send_json(d, status, error); });
      }
    };
    if (exceeds_context(job.prompt.size(), ctx_.engine->max_context())) {
      fail(context_overflow_message(job.prompt.size(), ctx_.engine->max_context()), kContextLengthExceeded);
      return;
    }

    job.sink.text = [=](const std::string& piece) {
      stop_ticks();
      deliver(st->parser.push(piece));
    };
    job.sink.error = fail;
    job.sink.done = [=](const std::string& finish, int prompt_tokens, int completion_tokens, int cached,
                        int reasoning_tokens) {
      stop_ticks();
      deliver(st->parser.finish());
      st->items.close();
      // A repetition stop (API §9) is response.incomplete with reason "repetition" (the ragon launcher's loop guard
      // reads it; Codex alone retries it), or with --repetition-report completed a completed response; either way
      // the response says stop_reason "repetition", and every delta up to the loop's last token went out first.
      const bool repeated = finish == "repetition", rep_incomplete = repeated && !repetition_completed;
      json resp = response_obj(finish == "length" || rep_incomplete ? "incomplete" : "completed");
      if (finish == "length") resp["incomplete_details"] = {{"reason", "max_output_tokens"}};
      if (rep_incomplete) resp["incomplete_details"] = {{"reason", "repetition"}};
      if (repeated) resp["stop_reason"] = "repetition";
      resp["usage"] = {{"input_tokens", prompt_tokens},
                       {"input_tokens_details", {{"cached_tokens", cached}}},
                       {"output_tokens", completion_tokens},
                       {"output_tokens_details", {{"reasoning_tokens", reasoning_tokens}}},
                       {"total_tokens", prompt_tokens + completion_tokens}};
      if (req.stream) {
        // A response cut at max_output_tokens still ends with response.completed: the agent treats
        // response.incomplete as a failed turn and discards what was written.
        event(rep_incomplete ? "response.incomplete" : "response.completed", {{"response", resp}});
        post(ex, [](proxygen::ResponseHandler* d) { ResponseBuilder(d).sendWithEOM(); });
      } else {
        post(ex, [resp, drafter = req.drafter_id](proxygen::ResponseHandler* d) { send_json(d, 200, resp, drafter); });
      }
    };
    if (req.stream && ctx_.keepalive_ms > 0) {
      // response.in_progress with the response object as response.created sent it (the output is still
      // empty, and it is taken here, before the engine thread can touch it); Codex ignores the kind.
      st->keepalive = true;
      const json in_progress = {{"response", response_obj("in_progress")}};
      arm_keepalive(ex, ctx_.keepalive_ms, [ex, in_progress](proxygen::ResponseHandler*) {
        emit_event(*ex, "response.in_progress", in_progress);
      });
    }
    scheduler_.submit(std::move(job));
  }

  ServerContext& ctx_;
  Scheduler& scheduler_;
  std::string method_, path_;
  // Which resident drafter serves the request (MULTI_DRAFTER §2): from the request alone (§2.4). An adapter with a
  // drafter part was trained on `general`, so its requests stay there.
  void choose_drafter(Job& job) const {
    const Engine& eng = *ctx_.engine;
    Request& q = job.req;
    if (!eng.has_drafter()) return;
    ragon::serve::DrafterChoice c =
        ctx_.drafter_routes.choose(q.workload, !q.tools.empty(), ragon::serve::tools_include_shell(q.tools), q.last_item);
    if (job.lora && job.lora->has_drafter) c = {ragon::serve::kGeneralDrafter, "default"};
    q.drafter = 0;
    for (int i = 0; i < eng.drafters(); ++i)
      if (eng.drafter_id(i) == c.id) q.drafter = i;
    q.drafter_id = eng.drafter_id(q.drafter);
    q.drafter_how = c.how;
  }

  uint64_t share_scope_ = 0;  // X-Ragon-Share-Scope (P4c, share_scope_of)
  std::string authorization_;  // the Authorization header (the adapter routes' key)
  folly::IOBufQueue body_{folly::IOBufQueue::cacheChainLength()};
  bool too_large_ = false;
  std::shared_ptr<Exchange> ex_ = std::make_shared<Exchange>();
};

class Factory : public proxygen::RequestHandlerFactory {
 public:
  Factory(ServerContext& ctx, Scheduler& scheduler) : ctx_(ctx), scheduler_(scheduler) {}
  void onServerStart(folly::EventBase*) noexcept override {}
  void onServerStop() noexcept override {}
  proxygen::RequestHandler* onRequest(proxygen::RequestHandler*, HTTPMessage*) noexcept override {
    return new Handler(ctx_, scheduler_);
  }

 private:
  ServerContext& ctx_;
  Scheduler& scheduler_;
};

}  // namespace
}  // namespace ragon::serve

// Above this many decoding requests a server with a drafter stops speculating (BATCHING P4d, measured there).
constexpr long kFanoutPlainAboveDrafting = 6;
// The wide regime's defaults (BATCHING §5.8 stage 11): above this many drafting requests every one speculates with
// the gang block, whatever the plain threshold says (--fanout-default off: never).
constexpr int kFanoutWideAboveDrafting = 7;
constexpr long kFanoutWideBlock = 2;

int main(int argc, char** argv) try {
  std::string model_dir, host = "0.0.0.0", name;
  double draft_off_below = 1.5;  // the drafter-off guard (BACKLOG 10)
  int port = 8000, max_context = 0, draft_block = 12, lookup = 1, lookup_min = 8, checkpoints = 8, tree_nodes = 7, prefix_kv_mb = -1;
  std::string draft_dir, pretokenizer = "production";
  std::vector<std::string> drafter_args;  // --drafter ID=DIR, repeatable (MULTI_DRAFTER §5.3)
  std::string drafter_routes_file;
  bool graphs = false, pdl = false, attn_bulk = false, prefill_bf16 = false;
  bool kv_pages = false;  // the flat slots by default (P4d kept them: paged passes are slower, BATCHING §5.3)
  long kv_pool_mb = -1;  // P4b: -1, the pool from the memory plan
  bool plan_only = false;  // print the start plan as one JSON line and exit 0 before anything is allocated
  int attn_prefetch = 2, keepalive_seconds = 15, sessions = 0, batch = 0, contended = 256, side_prefill = 40, side_share = 16;
  // --sessions, --max-context and --batch as given; what is not given is chosen at start (--auto-size).
  std::optional<int> sessions_arg, max_context_arg, batch_arg;
  long host_reserve_mb = 16384, admit_reserve_mb = 8192, auto_size = 1, fair_pairing = 0, fanout_plain_above = -1,
       fanout_gang_above = -1, fanout_default = 1, fanout_draft_block = -1;
  double protect_seconds = 30;
  bool draft_bf16 = false;
  std::string repetition_arg, repetition_report = "incomplete";
  std::string draft_head = "sparse", sparse_head_dir;  // the drafter's head (BACKLOG 19)
  int sparse_head_dynamic = 16384;
  std::string vision = "on", jpeg = "gpu";  // images (VISION §20.3)
  long vision_max_pixels = 16777216;
  int vision_cache_mb = 512;
  // LoRA adapters (LORA §4): the library, the adapters loaded at start (name=dir, each dir one name in the library),
  // the adapters in one batch (the base counts as one, as in SGLang: max-1 GPU slots), those held in host memory.
  std::string lora_library, lora_paths, api_key, api_key_file;
  int max_loras_per_batch = 4, max_loaded_loras = 8;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::cerr << a << " needs a value\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--model") model_dir = next();
    else if (a == "--host") host = next();
    else if (a == "--port") port = std::stoi(next());
    else if (a == "--served-model-name") name = next();
    else if (a == "--max-context") max_context_arg = std::stoi(next());
    else if (a == "--vision") vision = next();
    else if (a == "--vision-max-pixels") vision_max_pixels = std::stol(next());
    else if (a == "--vision-cache-mb") vision_cache_mb = std::stoi(next());
    else if (a == "--jpeg-decoder") jpeg = next();
    else if (a == "--draft") draft_dir = next();
    else if (a == "--drafter") drafter_args.push_back(next());
    else if (a == "--drafter-routes") drafter_routes_file = next();
    else if (a == "--draft-block") draft_block = std::stoi(next());
    else if (a == "--tree") tree_nodes = std::stoi(next());
    else if (a == "--draft-off-below") draft_off_below = std::stod(next());
    else if (a == "--lookup") lookup = std::stoi(next());
    else if (a == "--lookup-min") lookup_min = std::stoi(next());
    else if (a == "--pretokenizer") pretokenizer = next();
    else if (a == "--prefix-checkpoints") checkpoints = std::stoi(next());
    else if (a == "--prefix-kv-mb") prefix_kv_mb = std::stoi(next());
    else if (a == "--graphs") graphs = true;
    else if (a == "--pdl") pdl = true;
    else if (a == "--attn-bulk") attn_bulk = true;
    else if (a == "--attn-prefetch") attn_prefetch = std::stoi(next());
    else if (a == "--keepalive-seconds") keepalive_seconds = std::stoi(next());
    else if (a == "--prefill-bf16") prefill_bf16 = true;
    else if (a == "--kv-pages") kv_pages = true;
    else if (a == "--no-kv-pages") kv_pages = false;
    else if (a == "--kv-pool-mb") kv_pool_mb = std::stol(next());
    else if (a == "--draft-bf16") draft_bf16 = true;
    else if (a == "--draft-head") draft_head = next();  // sparse (default) or full (BACKLOG 19)
    else if (a == "--sparse-head-dir") sparse_head_dir = next();
    else if (a == "--sparse-head-dynamic") sparse_head_dynamic = std::stoi(next());
    else if (a == "--sessions") sessions_arg = std::stoi(next());
    else if (a == "--session-protect-seconds") protect_seconds = std::stod(next());
    else if (a == "--batch") batch_arg = std::stoi(next());
    else if (a == "--prefill-chunk-contended") contended = std::stoi(next());
    else if (a == "--side-prefill") side_prefill = std::stoi(next());
    else if (a == "--side-share") side_share = std::stoi(next());
    else if (a == "--host-reserve-mb") host_reserve_mb = std::stol(next());
    else if (a == "--admit-reserve-mb") admit_reserve_mb = std::stol(next());
    else if (a == "--auto-size") auto_size = std::stol(next());
    else if (a == "--fair-pairing") fair_pairing = std::stol(next());
    else if (a == "--fanout-plain-above") fanout_plain_above = std::stol(next());
    else if (a == "--fanout-gang-above") fanout_gang_above = std::stol(next());
    else if (a == "--fanout-draft-block") fanout_draft_block = std::stol(next());
    else if (a == "--fanout-default") {
      const std::string v = next();
      if (v != "on" && v != "off" && v != "1" && v != "0") {
        std::cerr << "--fanout-default must be on or off\n";
        return 2;
      }
      fanout_default = v == "on" || v == "1";
    }
    else if (a == "--plan-only") plan_only = true;
    else if (a == "--repetition-detection") repetition_arg = next();
    else if (a == "--repetition-report") repetition_report = next();
    else if (a == "--lora-library") lora_library = next();
    else if (a == "--lora-paths") lora_paths = next();
    else if (a == "--max-loras-per-batch") max_loras_per_batch = std::stoi(next());
    else if (a == "--max-loaded-loras") max_loaded_loras = std::stoi(next());
    else if (a == "--api-key") api_key = next();
    else if (a == "--api-key-file") api_key_file = next();
    else {
      std::cerr << "unknown argument " << a << "\n";
      return 2;
    }
  }
  if (model_dir.empty()) {
    std::cerr << "usage: ragon-serve --model DIR [--host H] [--port P] [--served-model-name N] [--max-context N] [--draft DIR [--drafter ID=DIR ...] [--drafter-routes FILE] [--draft-block N] [--tree N] [--draft-off-below X] [--draft-head sparse|full [--sparse-head-dir DIR] [--sparse-head-dynamic N]]] [--pretokenizer production|checkpoint] [--prefix-checkpoints N] [--prefix-kv-mb N] [--keepalive-seconds N] [--sessions N] [--session-protect-seconds S] [--batch N] [--prefill-chunk-contended N] [--side-prefill N] [--side-share N] [--fair-pairing 1|0] [--host-reserve-mb N] [--admit-reserve-mb N] [--auto-size 1|0] [--plan-only] [--draft-bf16] [--repetition-detection P,p,R,K|off] [--repetition-report incomplete|completed] [--kv-pages [--kv-pool-mb N] | --no-kv-pages] [--fanout-plain-above N] [--fanout-gang-above N] [--fanout-default on|off] [--fanout-draft-block N] [--vision on|off] [--vision-max-pixels N] [--vision-cache-mb N] [--jpeg-decoder gpu|cpu] [--lora-library DIR [--lora-paths NAME=DIR,...] [--max-loras-per-batch N] [--max-loaded-loras N]] [--api-key KEY | --api-key-file F]\n";
    return 2;
  }
  for (auto [flag, v] : {std::pair<const char*, std::optional<int>>{"--sessions", sessions_arg},
                         {"--max-context", max_context_arg}, {"--batch", batch_arg}})
    if (v && *v < 1) {
      std::cerr << flag << " must be at least 1\n";
      return 2;
    }
  if (const char* e = std::getenv("RAGON_KEEPALIVE_SECONDS")) {  // overrides the flag, as the engine's variables do
    try {
      keepalive_seconds = std::stoi(e);
    } catch (const std::exception&) {
      std::cerr << "RAGON_KEEPALIVE_SECONDS must be an integer\n";
      return 2;
    }
  }
  int fake_argc = 1;
  char** fake_argv = argv;
  folly::Init init(&fake_argc, &fake_argv, false);

  // Prompts are tokenized as production tokenizes them (tokenizer.hpp, kProductionPretokenizer) unless
  // --pretokenizer checkpoint asks for the checkpoint's own pattern.
  if (pretokenizer != "production" && pretokenizer != "checkpoint") {
    std::cerr << "--pretokenizer is production or checkpoint\n";
    return 2;
  }
  ragon::serve::RepetitionDetection repetition = ragon::serve::kRepetitionDefault;
  if (!repetition_arg.empty()) {
    if (repetition_arg == "off") {
      repetition = {0, 0, 0, 0};
    } else {
      int v[4];
      char tail = 0;
      if (std::sscanf(repetition_arg.c_str(), "%d,%d,%d,%d%c", &v[0], &v[1], &v[2], &v[3], &tail) != 4) {
        std::cerr << "--repetition-detection is P,p,R,K (max and min pattern size, min count, max run) or off\n";
        return 2;
      }
      repetition = {v[0], v[1], v[2], v[3]};
    }
    try {
      ragon::serve::validate(repetition);
    } catch (const std::exception& e) {
      std::cerr << "--repetition-detection: " << e.what() << "\n";
      return 2;
    }
  }
  if (repetition_report != "incomplete" && repetition_report != "completed") {
    std::cerr << "--repetition-report is incomplete or completed\n";
    return 2;
  }
  ragon::Tokenizer tok(model_dir + "/tokenizer.json", pretokenizer == "production" ? ragon::kProductionPretokenizer : "");
  // The resident drafters (MULTI_DRAFTER §5.3): `--draft DIR` is `general` alone, and every other id comes beside a
  // `general`. A routes file that names a drafter not loaded is refused here, not on the first request it would send.
  std::vector<std::pair<std::string, std::string>> extra_drafters;
  ragon::serve::DrafterRoutes drafter_routes;
  try {
    for (const std::string& arg : drafter_args) {
      const ragon::serve::DrafterArg d = ragon::serve::parse_drafter_arg(arg);
      if (d.id == ragon::serve::kGeneralDrafter) {
        if (!draft_dir.empty()) throw std::invalid_argument("--draft is the drafter general: give it once");
        draft_dir = d.dir;
      } else {
        extra_drafters.emplace_back(d.id, d.dir);
      }
    }
    if (!extra_drafters.empty() && draft_dir.empty())
      throw std::invalid_argument("no drafter is general: add --draft DIR or --drafter general=DIR");
    if (!drafter_routes_file.empty()) {
      if (draft_dir.empty()) throw std::invalid_argument("--drafter-routes needs a drafter");
      std::ifstream in(drafter_routes_file);
      if (!in) throw std::invalid_argument("cannot read " + drafter_routes_file);
      drafter_routes = ragon::serve::DrafterRoutes::parse(std::string(std::istreambuf_iterator<char>(in), {}));
      for (const std::string& id : drafter_routes.targets()) {
        const bool loaded = id == ragon::serve::kGeneralDrafter ||
                            std::any_of(extra_drafters.begin(), extra_drafters.end(),
                                        [&](const auto& d) { return d.first == id; });
        if (!loaded) throw std::invalid_argument("the drafter routes name the drafter " + id + ", which is not loaded");
      }
    }
  } catch (const std::invalid_argument& e) {
    std::cerr << "ragon-serve: " << e.what() << "\n";
    return 2;
  }
  ragon::EngineOptions opts;
  opts.draft_dir = draft_dir;
  opts.extra_drafters = extra_drafters;
  opts.draft_block = draft_block;
  opts.tree_nodes = tree_nodes;  // BACKLOG 26: 7 by default, 0 the chain alone (RAGON_TREE overrides)
  opts.draft_off_below = draft_off_below;  // BACKLOG 10: 1.5 by default, 0 the guard off (RAGON_DRAFT_OFF_BELOW overrides)
  opts.lookup_mode = lookup;
  opts.lookup_min_match = lookup_min;
  opts.prefix_checkpoints = checkpoints;
  if (prefix_kv_mb >= 0) opts.prefix_kv_mb = prefix_kv_mb;
  opts.boundary_token = tok.token_id("<|im_end|>");
  opts.step_graphs = graphs;
  opts.pdl = pdl;
  opts.attention_bulk = attn_bulk;
  opts.attention_prefetch = attn_prefetch;
  opts.prefill_bf16 = prefill_bf16;
  opts.kv_pages = kv_pages;
  opts.draft_nvfp4 = !draft_bf16;
  opts.draft_head = draft_head;
  opts.sparse_head_dir = sparse_head_dir;
  opts.sparse_head_dynamic = sparse_head_dynamic;
  opts.side_prefill = side_prefill;
  if (vision != "on" && vision != "off") {
    std::cerr << "--vision is on or off\n";
    return 2;
  }
  if (jpeg != "gpu" && jpeg != "cpu") {
    std::cerr << "--jpeg-decoder is gpu or cpu\n";
    return 2;
  }
  ragon::serve::set_jpeg_decoder(jpeg == "gpu" ? ragon::serve::JpegDecoder::Gpu : ragon::serve::JpegDecoder::Cpu);
  opts.vision = vision == "on";
  opts.vision_max_pixels = vision_max_pixels;
  if (!lora_library.empty() && (max_loras_per_batch < 2 || max_loras_per_batch > 9 || max_loaded_loras < 1)) {
    std::cerr << "--max-loras-per-batch is 2 to 9 (the base counts as one), --max-loaded-loras at least 1\n";
    return 2;
  }
  if (!lora_paths.empty() && lora_library.empty()) {
    std::cerr << "--lora-paths needs --lora-library\n";
    return 2;
  }
  opts.lora_slots = lora_library.empty() ? 0 : max_loras_per_batch - 1;
  if (const char* e = std::getenv("RAGON_API_KEY"); e && api_key.empty()) api_key = e;  // kept out of the process list
  if (!api_key_file.empty()) {
    std::ifstream f(api_key_file);
    if (!f || !std::getline(f, api_key) || api_key.empty()) {
      std::cerr << "--api-key-file " << api_key_file << ": no key in it\n";
      return 2;
    }
  }
  opts.vision_cache_mb = vision_cache_mb;
  for (auto [var, dst] : {std::pair<const char*, long*>{"RAGON_HOST_RESERVE_MB", &host_reserve_mb},
                          std::pair<const char*, long*>{"RAGON_ADMIT_RESERVE_MB", &admit_reserve_mb},
                          std::pair<const char*, long*>{"RAGON_AUTO_SIZE", &auto_size},
                          std::pair<const char*, long*>{"RAGON_FAIR_PAIRING", &fair_pairing},
                          std::pair<const char*, long*>{"RAGON_FANOUT_PLAIN_ABOVE", &fanout_plain_above},
                          std::pair<const char*, long*>{"RAGON_FANOUT_GANG_ABOVE", &fanout_gang_above},
                          std::pair<const char*, long*>{"RAGON_FANOUT_DEFAULT", &fanout_default},
                          std::pair<const char*, long*>{"RAGON_FANOUT_DRAFT_BLOCK", &fanout_draft_block}})
    if (const char* e = std::getenv(var)) {
      try {
        *dst = std::stol(e);
      } catch (const std::exception&) {
        std::cerr << var << " must be an integer\n";
        return 2;
      }
    }
  // P4b: the engine reads RAGON_KV_PAGES itself; the plan must agree with it.
  if (const char* e = std::getenv("RAGON_KV_PAGES")) kv_pages = std::atoi(e) != 0;
  if (kv_pool_mb >= 0 && !kv_pages) {
    std::cerr << "--kv-pool-mb needs --kv-pages (or RAGON_KV_PAGES=1)\n";
    return 2;
  }
  ragon::serve::MemoryPlan plan;  // the start's plan, which --plan-only reports
  {  // sizing and admission control at start (serve/admission.hpp; BATCHING §5.6), before anything is allocated
    namespace as = ragon::serve;
    const ragon::ModelConfig cfg = ragon::ModelConfig::from_file(model_dir + "/config.json");
    // Every resident drafter's weights, from the declared budget (MULTI_DRAFTER §5.4): the reserve and the prefix store
    // are not reduced for them. A session holds one drafter ring whichever drafter it runs, so the rings stay per session.
    size_t weights = as::checkpoint_bytes(model_dir) + as::checkpoint_bytes(draft_dir);
    for (const auto& [id, dir] : extra_drafters) {
      const size_t bytes = as::checkpoint_bytes(dir);
      weights += bytes;
      std::cerr << "admission: drafter " << id << " counts " << bytes / 1e9 << " GB\n";
    }
    // With images on (VISION §20.4) the tower's scratch at the largest image and the embedding cache come on top of
    // the checkpoint's bytes (which hold the tower's weights): 1,100 MB at 65,536 patches, an upper bound below it.
    const size_t scratch = as::kScratchBytes + (opts.vision ? size_t(1100 + vision_cache_mb) << 20 : 0) +
                           size_t(opts.lora_slots) * as::kLoraSlotPlanBytes;
    const long avail = as::mem_available_mb();
    const int full_context = as::model_full_context(model_dir + "/config.json");  // 0: sized from 65,536
    // With paged KV the pool takes what the rest of the plan, the prefix KV store's budget and the reserve leave, or
    // --kv-pool-mb; auto-size sizes by the same plan. Since P4c the store's arena holds DeltaNet and conv states only
    // (its KV is pool pages); the engine allocates it after the pool at min(prefix_kv_mb, free above its own reserve),
    // so counting prefix_kv_mb here is what it takes, not a second count.
    as::KvPlanInputs kv{kv_pages, 0, kv_pool_mb};
    if (kv_pages) {
      long store_mb = opts.prefix_kv_mb;
      if (const char* e = std::getenv("RAGON_PREFIX_KV_MB")) store_mb = std::atol(e);
      if (checkpoints <= 0 || store_mb < 0) store_mb = 0;
      kv.store_bytes = size_t(store_mb) << 20;
    }
    const as::SizeChoice size = as::choose_size(cfg, weights, scratch, !draft_dir.empty(), avail, host_reserve_mb,
                                                full_context, sessions_arg, max_context_arg, auto_size != 0, kv);
    sessions = size.sessions;
    max_context = size.max_context;
    batch = batch_arg.value_or(size.sessions_auto ? sessions : as::kDefaultBatch);
    const as::StartPlan start = as::start_plan(cfg, weights, scratch, !draft_dir.empty(), sessions, max_context, avail,
                                               host_reserve_mb, kv);
    plan = start.plan;
    const as::MemoryPlan& mp = plan;
    if (!start.refusal.empty()) {
      std::cerr << "ragon-serve: refused to start: " << start.refusal << "\n";
      return 3;
    }
    if (kv_pages) {
      opts.kv_pool_pages = static_cast<int>(mp.kv_pool_pages);
      std::cerr << "KV page pool: " << as::describe_kv_pool(mp)
                << (kv_pool_mb >= 0 ? " (--kv-pool-mb)" : " (from the memory plan)") << "\n";
    }
    // One line: what runs, whether each was chosen here (auto), given, or a default (and why no sizing), and the
    // plan's numbers it was chosen by.
    const char* off = auto_size == 0          ? "default, --auto-size 0"
                      : host_reserve_mb <= 0 ? "default, --host-reserve-mb 0"
                      : avail < 0            ? "default, MemAvailable unknown"
                                             : "default";
    auto why = [&](bool given, bool chosen) { return given ? "given" : chosen ? "auto" : off; };
    std::cerr << "size: " << sessions << " sessions (" << why(bool(sessions_arg), size.sessions_auto) << "), batch "
              << std::min(batch, sessions) << " (" << why(bool(batch_arg), size.sessions_auto) << "), context "
              << max_context << " (" << why(bool(max_context_arg), size.context_auto) << "); memory: available "
              << (avail >= 0 ? as::gb(size_t(avail) << 20) : std::string("unknown")) << ", weights and scratch "
              << as::gb(mp.weights + mp.scratch) << ", " << as::gb(mp.per_session) << " a session, kept free "
              << (host_reserve_mb > 0 ? as::gb(size_t(host_reserve_mb) << 20) : std::string("none")) << ", planned "
              << as::gb(mp.total()) << "; requests refused below " << admit_reserve_mb << " MB\n";
  }
  if (plan_only) {  // the host-safety check of the launcher asks for this before it loads (BATCHING §5.6)
    std::cout << ragon::serve::plan_json(sessions, batch, max_context, plan, kv_pages, host_reserve_mb) << "\n";
    return 0;
  }
  opts.max_context = max_context;
  opts.sessions = sessions;
  std::cerr << "loading " << model_dir << " ...\n";
  ragon::Engine engine(model_dir, opts);
  std::cerr << "loaded " << engine.model().device_bytes() / 1e9 << " GB of weights\n";

  ragon::serve::ServerContext ctx;
  ctx.engine = &engine;
  ctx.drafter_routes = drafter_routes;
  for (int i = 0; i < engine.drafters(); ++i) ctx.metrics.by_drafter[engine.drafter_id(i)];
  ctx.tok = &tok;
  ctx.im_end = tok.token_id("<|im_end|>");
  static ragon::serve::ImageRegistry image_registry;
  ctx.vision = engine.has_vision();
  ctx.image_pad = tok.token_id("<|image_pad|>");
  ctx.images = &image_registry;
  ctx.processor.max_pixels = vision_max_pixels;
  ctx.metrics.max_total_tokens = engine.sessions() * max_context;
  if (engine.kv_paged())  // P4b: the positions the pool can hold, when that is fewer
    ctx.metrics.max_total_tokens = static_cast<int>(
        std::min<long>(ctx.metrics.max_total_tokens, long(engine.kv_pages_total()) * ragon::kernels::kKvPageTokens));
  ctx.metrics.sessions = engine.sessions();
  ctx.metrics.protect_seconds = protect_seconds;
  ctx.admit_reserve_mb = admit_reserve_mb;
  batch = std::min(batch, engine.sessions());  // one running request per session
  if (contended < 32 || contended > engine.prefill_chunk_size()) {
    std::cerr << "--prefill-chunk-contended must be in [32, " << engine.prefill_chunk_size() << "]\n";
    return 2;
  }
  ctx.metrics.batch = batch;
  ctx.metrics.prefill_chunk_contended = contended;
  ctx.keepalive_ms = keepalive_seconds > 0 ? keepalive_seconds * 1000 : 0;
  ctx.repetition = repetition;
  ctx.repetition_completed = repetition_report == "completed";
  if (ctx.repetition.on())
    std::cerr << "repetition stop: blocks of " << ctx.repetition.min_pattern_size << "-" << ctx.repetition.max_pattern_size
              << " tokens repeated " << ctx.repetition.min_count << " times, runs of " << ctx.repetition.max_run << "\n";
  ctx.think_end = tok.token_id("</think>");
  {
    const std::vector<int> one = tok.encode("\n"), two = tok.encode("\n\n");
    ctx.newline = one.size() == 1 ? one[0] : -1;
    ctx.blank_line = two.size() == 1 ? two[0] : -1;
  }
  if (name.empty()) {
    name = model_dir;
    while (!name.empty() && name.back() == '/') name.pop_back();
    name = name.substr(name.find_last_of('/') + 1);
  }
  ctx.model_name = name;
  ctx.api_key = api_key;
  std::unique_ptr<ragon::LoraLibrary> lora_lib;
  std::unique_ptr<ragon::LoraSlots> lora_slots;
  if (!lora_library.empty()) {
    try {
      lora_lib = std::make_unique<ragon::LoraLibrary>(lora_library, max_loaded_loras, engine.lora_shape(),
                                                     engine.lora_draft_shape());
      for (size_t at = 0; at < lora_paths.size();) {
        const size_t comma = std::min(lora_paths.find(',', at), lora_paths.size());
        const std::string item = lora_paths.substr(at, comma - at);
        at = comma + 1;
        const size_t eq = item.find('=');
        const std::string n = item.substr(0, eq), dir = eq == std::string::npos ? n : item.substr(eq + 1);
        if (!ragon::lora_name_ok(dir)) throw std::runtime_error("--lora-paths " + item + ": DIR is one name in the library");
        const auto a = lora_lib->load(n, lora_lib->root() + "/" + dir);
        std::cerr << "lora: loaded " << n << " (rank " << a->target.rank << ", " << (a->bytes() >> 20) << " MB)\n";
      }
    } catch (const std::exception& e) {
      std::cerr << "lora: " << e.what() << "\n";
      return 2;
    }
    lora_slots = std::make_unique<ragon::LoraSlots>(engine.lora_slots());
    ctx.lora_lib = lora_lib.get();
    ctx.lora_slots = lora_slots.get();
    std::cerr << "lora: library " << lora_lib->root() << ", " << engine.lora_slots() << " GPU slots, at most "
              << max_loaded_loras << " loaded; adapter routes " << (api_key.empty() ? "off (no --api-key)" : "on")
              << "\n";
  }
  std::fprintf(stderr, "pairing: %s\n", fair_pairing ? "fair (--fair-pairing 1)" : "running order (--fair-pairing 0)");
  const bool plain_explicit = fanout_plain_above >= 0;
  if (fanout_plain_above < 0) fanout_plain_above = draft_dir.empty() ? 1 : kFanoutPlainAboveDrafting;
  // The fork prefills gang above the same count unless told otherwise (BATCHING §5.8: speculation under fan-out needs
  // the forks started together as much as plain steps do).
  if (fanout_gang_above < 0) fanout_gang_above = fanout_plain_above;
  // The wide regime (BATCHING §5.8 stage 11): with a drafter and --fanout-default on, above 7 drafting requests every
  // one speculates with the gang block (--fanout-draft-block, 2; 0 keeps --draft-block), unless --fanout-plain-above
  // was given. The drafter's buffers are sized for --draft-block at start, so the gang block cannot exceed it.
  if (fanout_draft_block < 0) fanout_draft_block = kFanoutWideBlock;
  if (fanout_draft_block == 0) fanout_draft_block = draft_block;
  if (fanout_draft_block < 2 || fanout_draft_block > draft_block) {
    std::cerr << "--fanout-draft-block must be 0 or in [2, --draft-block]\n";
    return 2;
  }
  const int wide_above = fanout_default && !draft_dir.empty() ? kFanoutWideAboveDrafting : 0;
  if (wide_above > 0)
    std::fprintf(stderr, "fanout: plain above %ld decoding%s, gang above %ld; above %d drafting: draft block %ld%s\n",
                 fanout_plain_above, plain_explicit ? "" : " up to the wide regime", fanout_gang_above, wide_above,
                 fanout_draft_block, plain_explicit ? "" : ", never plain");
  else
    std::fprintf(stderr, "fanout: plain above %ld decoding, gang above %ld; no wide regime\n", fanout_plain_above,
                 fanout_gang_above);
  ragon::serve::Scheduler scheduler(ctx, protect_seconds, batch, contended, side_share, fair_pairing != 0,
                                     static_cast<int>(fanout_plain_above), static_cast<int>(fanout_gang_above),
                                     plain_explicit, wide_above, static_cast<int>(fanout_draft_block));

  proxygen::HTTPServerOptions options;
  options.threads = 2;
  options.idleTimeout = std::chrono::minutes(30);
  options.shutdownOn = {SIGINT, SIGTERM};
  options.enableContentCompression = false;
  options.handlerFactories =
      proxygen::RequestHandlerChain().addThen<ragon::serve::Factory>(ctx, scheduler).build();
  std::vector<proxygen::HTTPServer::IPConfig> ips = {
      {folly::SocketAddress(host, static_cast<uint16_t>(port), true), proxygen::HTTPServer::Protocol::HTTP}};
  proxygen::HTTPServer server(std::move(options));
  server.bind(ips);
  std::cerr << "ragon-serve: serving " << name << " on " << host << ":" << port << "\n";
  std::thread t([&] { server.start(); });
  t.join();
  return 0;
} catch (const std::exception& e) {  // a start that cannot go on (a drafter of another shape, say) is a refusal, not a core dump
  std::cerr << "ragon-serve: " << e.what() << "\n";
  return 2;
}
