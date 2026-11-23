// ragon-serve on the Metal backend (the Mac node, BACKEND §8): the OpenAI-compatible HTTP surface of
// src/serve/main.cpp (Responses, chat completions, completions, models, health) over ragon::metal::Model.
//
//   ragon-serve --model DIR [--host 127.0.0.1] [--port 8000] [--served-model-name NAME] [--max-context N]
//               [--chunk C] [--keepalive-seconds N] [--repetition-detection P,p,R,K|off]
//               [--repetition-report incomplete|completed] [--api-key KEY] [--pretokenizer production|checkpoint]
//
// The request handling is the CUDA serve's own (serve/api.hpp: parse_request, parse_responses_request,
// OutputParser, ResponseItems, StopScanner, RepetitionDetector, the context-overflow errors), and so are the
// response and event shapes; what differs is the engine and the HTTP layer:
// - one engine thread runs one request at a time (the Metal model holds one sequence). The sequence stays in the
//   model between requests: a prompt that extends it (an agent's next turn) feeds only the new tokens; any other
//   prompt starts over (the recurrent state cannot be rewound).
// - plain greedy (temperature 0, no penalties) runs the GPU pick loop, two steps in flight (Model::greedy_step);
//   RAGON_METAL_SERVE_HOST_LOOP=1 forces the host loop. Otherwise the serve's sampler (core/sampling.hpp) on
//   the host logits.
// - the HTTP layer is plain POSIX sockets, HTTP/1.1, one thread a connection, `Connection: close`. The engine
//   thread never writes to a socket: it appends to the request's outbox, which the connection thread drains, so a
//   slow client never stalls decoding; a failed write cancels the request.
// - Responses requests take `type: namespace` function tools (Codex's multi_agent_v1) as plain functions and
//   return their calls with the `namespace` field (parse_responses_request's `namespaces`). Hosted tools
//   (`web_search`) are not offered: the server has nothing to run them with.
// - n > 1, logprobs, images, adapters and structured output are refused or not supported (see the checks).
// RAGON_METAL_IGNORE_EOS=1 bans the end tokens (benchmarks only: the decode-speed recipe runs 400 tokens).

#import <Foundation/Foundation.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <bit>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "backend/metal/model.hpp"
#include "core/sampling.hpp"
#include "serve/api.hpp"
#include "tokenizer/tokenizer.hpp"

using ragon::SamplingParams;
using ragon::Tokenizer;
using ragon::metal::Model;
using ragon::metal::Queue;
using namespace ragon::serve;

namespace {

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// ---------------------------------------------------------------------------------------------------------------
// HTTP

struct HttpRequest {
  std::string method, path, body;
  std::map<std::string, std::string> headers;  // lower-case names
};

const char* reason_of(int status) {
  switch (status) {
    case 200: return "OK";
    case 400: return "Bad Request";
    case 401: return "Unauthorized";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 413: return "Payload Too Large";
    case 415: return "Unsupported Media Type";
    case 503: return "Service Unavailable";
    default: return "Internal Server Error";
  }
}

std::string head_of(int status, const std::string& content_type, long length) {
  std::string h = "HTTP/1.1 " + std::to_string(status) + " " + reason_of(status) + "\r\nContent-Type: " + content_type +
                  "\r\nConnection: close\r\n";
  if (length >= 0) h += "Content-Length: " + std::to_string(length) + "\r\n";
  else h += "Cache-Control: no-cache\r\n";
  return h + "\r\n";
}

std::string json_response(int status, const json& body) {
  const std::string b = dump_json(body);
  return head_of(status, "application/json", long(b.size())) + b;
}

bool write_all(int fd, const char* p, size_t n) {
  while (n) {
    const ssize_t w = ::write(fd, p, n);
    if (w < 0 && errno == EINTR) continue;
    if (w <= 0) return false;
    p += w;
    n -= size_t(w);
  }
  return true;
}
bool write_all(int fd, const std::string& s) { return write_all(fd, s.data(), s.size()); }

constexpr size_t kMaxBody = 64u << 20;

// Reads one request. Returns false on a closed or malformed connection; sets `status` for a refusal to send.
bool read_request(int fd, HttpRequest& r, int& status) {
  std::string buf;
  char tmp[65536];
  size_t end = std::string::npos;
  while ((end = buf.find("\r\n\r\n")) == std::string::npos) {
    if (buf.size() > (1u << 20)) return status = 413, false;
    const ssize_t n = ::read(fd, tmp, sizeof tmp);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return false;
    buf.append(tmp, size_t(n));
  }
  const size_t line_end = buf.find("\r\n");
  const std::string line = buf.substr(0, line_end);
  const size_t s1 = line.find(' '), s2 = line.find(' ', s1 + 1);
  if (s1 == std::string::npos || s2 == std::string::npos) return status = 400, false;
  r.method = line.substr(0, s1);
  r.path = line.substr(s1 + 1, s2 - s1 - 1);
  if (const size_t q = r.path.find('?'); q != std::string::npos) r.path.resize(q);
  for (size_t p = line_end + 2; p < end;) {
    const size_t e = buf.find("\r\n", p);
    const std::string h = buf.substr(p, e - p);
    p = e + 2;
    const size_t c = h.find(':');
    if (c == std::string::npos) continue;
    std::string name = h.substr(0, c), value = h.substr(c + 1);
    for (char& ch : name) ch = char(std::tolower(static_cast<unsigned char>(ch)));
    value.erase(0, value.find_first_not_of(" \t"));
    while (!value.empty() && (value.back() == ' ' || value.back() == '\t')) value.pop_back();
    r.headers[name] = value;
  }
  if (r.headers.count("transfer-encoding")) return status = 400, false;  // clients here send Content-Length
  if (auto ce = r.headers.find("content-encoding"); ce != r.headers.end() && ce->second != "identity")
    return status = 415, false;
  size_t length = 0;
  if (auto cl = r.headers.find("content-length"); cl != r.headers.end()) length = std::strtoull(cl->second.c_str(), nullptr, 10);
  if (length > kMaxBody) return status = 413, false;
  if (auto ex = r.headers.find("expect"); ex != r.headers.end() && length > 0) write_all(fd, "HTTP/1.1 100 Continue\r\n\r\n");
  r.body = buf.substr(end + 4);
  while (r.body.size() < length) {
    const ssize_t n = ::read(fd, tmp, std::min(sizeof tmp, length - r.body.size()));
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return false;
    r.body.append(tmp, size_t(n));
  }
  r.body.resize(length);
  return true;
}

json error_body(const std::string& msg, const std::string& type, const std::string& code = "") {
  json e = {{"message", msg}, {"type", type}};
  if (!code.empty()) e["code"] = code;
  return json{{"error", e}};
}
std::string error_type(const std::string& code) {
  return code == kContextLengthExceeded ? "invalid_request_error" : "server_error";
}

// What the engine thread writes for one request, drained by its connection thread.
struct Outbox {
  std::mutex mu;
  std::condition_variable cv;
  std::string buf;
  bool closed = false;
  bool wrote = false;  // the engine has written (keep-alives stop)
  void put(const std::string& s, bool from_engine = true) {
    {
      std::lock_guard<std::mutex> l(mu);
      buf += s;
      wrote = wrote || from_engine;
    }
    cv.notify_one();
  }
  void close() {
    {
      std::lock_guard<std::mutex> l(mu);
      closed = true;
    }
    cv.notify_one();
  }
};

// ---------------------------------------------------------------------------------------------------------------
// The engine

struct Sink {
  std::function<void(const std::string&)> text;
  std::function<void(const std::string& finish, int prompt_tokens, int completion_tokens, int cached, int reasoning)> done;
  std::function<void(const std::string& msg, const std::string& code)> error;
};

struct Job {
  Request req;
  std::vector<int> prompt;
  Sink sink;
  std::shared_ptr<std::atomic<bool>> cancelled = std::make_shared<std::atomic<bool>>(false);
  std::string route;
};

struct Specials {
  int think_end = -1, im_end = -1, newline = -1, blank_line = -1;
};

class Engine {
 public:
  Engine(Model& m, const Tokenizer& tok, Specials sp, int max_ctx, int chunk, bool ignore_eos, int spec_block)
      : m_(m), tok_(tok), sp_(sp), max_ctx_(max_ctx), chunk_(chunk), spec_block_(m.has_draft() ? spec_block : 0) {
    stop_ids_ = m.eos_ids;
    if (sp.im_end >= 0 && std::find(stop_ids_.begin(), stop_ids_.end(), sp.im_end) == stop_ids_.end())
      stop_ids_.push_back(sp.im_end);
    if (ignore_eos) banned_ = stop_ids_;
    thread_ = std::thread([this] { loop(); });
  }
  int max_context() const { return max_ctx_; }
  void submit(std::shared_ptr<Job> j) {
    {
      std::lock_guard<std::mutex> l(mu_);
      q_.push_back(std::move(j));
    }
    cv_.notify_one();
  }

 private:
  void loop() {
    for (;;) {
      std::shared_ptr<Job> j;
      {
        std::unique_lock<std::mutex> l(mu_);
        cv_.wait(l, [&] { return !q_.empty(); });
        j = std::move(q_.front());
        q_.pop_front();
      }
      @autoreleasepool {
        try {
          run(*j);
        } catch (const std::exception& e) {
          fed_.clear();  // the model's sequence is unknown now
          std::fprintf(stderr, "request failed: %s\n", e.what());
          j->sink.error(e.what(), "server_error");
        }
      }
    }
  }

  int pick(const std::vector<int>& out, const SamplingParams& sp, std::mt19937_64& rng) {
    const float* lg = m_.logits();
    const int V = m_.vocab();
    if (sp.temperature <= 0.f && sp.presence_penalty == 0.f && sp.repetition_penalty == 1.f) {
      int best = -1;
      float v = -INFINITY;
      for (int i = 0; i < V; ++i) {
        if (!(lg[i] > v)) continue;
        if (!banned_.empty() && std::find(banned_.begin(), banned_.end(), i) != banned_.end()) continue;
        v = lg[i];
        best = i;
      }
      if (best < 0 || !std::isfinite(v)) throw ragon::NonFiniteLogits();
      return best;
    }
    if (banned_.empty()) return ragon::sample_logits(std::span<const float>(lg, size_t(V)), sp, out, rng);
    row_.assign(lg, lg + V);
    for (int b : banned_) row_[size_t(b)] = -1e30f;  // finite: the sampler refuses non-finite logits
    return ragon::sample_logits(row_, sp, out, rng);
  }

  void run(Job& j) {
    const Request& req = j.req;
    const int P = int(j.prompt.size());
    if (j.cancelled->load()) return;
    if (P == 0) throw std::invalid_argument("the prompt is empty");
    if (exceeds_context(size_t(P), max_ctx_)) {  // behind the handler's own check
      j.sink.error(context_overflow_message(size_t(P), max_ctx_), kContextLengthExceeded);
      return;
    }
    std::vector<uint32_t> ids(j.prompt.begin(), j.prompt.end());
    size_t reuse = 0;
    if (!fed_.empty() && fed_.size() < ids.size() && std::equal(fed_.begin(), fed_.end(), ids.begin())) reuse = fed_.size();
    const double tp = now_s();
    fed_.clear();
    if (reuse) {
      for (int s = int(reuse); s < P; s += chunk_) m_.forward(ids.data() + s, std::min(chunk_, P - s), s, false);
    } else {
      m_.reset();
      m_.prefill(ids.data(), P);
    }
    fed_ = ids;
    const double prefill_s = now_s() - tp;

    const int ctx_left = max_ctx_ - P;
    const int limit = req.max_tokens > 0 ? std::min(req.max_tokens, ctx_left) : ctx_left;
    ragon::StreamDecoder dec(tok_);
    StopScanner stops(req.stop);
    std::optional<RepetitionDetector> repeat;
    if (req.repetition.on()) repeat.emplace(req.repetition);
    std::mt19937_64 rng(req.sampling.seed ? req.sampling.seed : std::random_device{}());
    std::vector<int> out;
    std::string finish = "length";
    bool in_think = req.reasoning;
    int newlines_after_think = 0, reasoning = 0, produced = 0;
    const double td = now_s();
    int steps = 0;
    // accept(t): one produced token through the stop ids, the think accounting, the stop strings and the
    // repetition detector; false ends the request (finish says why).
    auto accept = [&](int t) -> bool {
      if (j.cancelled->load()) {
        finish = "cancelled";
        return false;
      }
      if (std::find(stop_ids_.begin(), stop_ids_.end(), t) != stop_ids_.end()) {
        finish = "stop";
        return false;
      }
      ++produced;
      out.push_back(t);
      if (in_think) {
        ++reasoning;
        if (t == sp_.think_end) {
          in_think = false;
          newlines_after_think = 2;
        }
      } else if (newlines_after_think > 0) {
        if (t == sp_.blank_line && newlines_after_think == 2) {
          ++reasoning;
          newlines_after_think = 0;
        } else if (t == sp_.newline) {
          ++reasoning;
          --newlines_after_think;
        } else {
          newlines_after_think = 0;
        }
      }
      const std::string ready = stops.push(dec.push(t));
      if (!ready.empty()) j.sink.text(ready);
      if (stops.stopped()) {
        finish = "stop";
        return false;
      }
      if (repeat && repeat->push(t)) {
        finish = "repetition";
        return false;
      }
      return produced < limit;
    };
    const SamplingParams& spm = req.sampling;
    const bool gpu_greedy = spm.temperature <= 0.f && spm.presence_penalty == 0.f && spm.repetition_penalty == 1.f &&
                            banned_.size() <= 8 && !std::getenv("RAGON_METAL_SERVE_HOST_LOOP");
    int t = pick(out, spm, rng);
    long spec_steps0 = m_.spec_steps, spec_tokens0 = m_.spec_tokens;
    // Speculation as core's can_speculate allows it: no penalties (they need the whole vocabulary), and greedy or
    // a top-k of at most 64 (the accept step builds the target's distribution from each row's top-k logits).
    const bool spec = spec_block_ > 0 && spm.presence_penalty == 0.f && spm.repetition_penalty == 1.f &&
                      banned_.size() <= 8 && (spm.temperature <= 0.f || (spm.top_k > 0 && spm.top_k <= 64));
    Model::SpecSampling smp;
    smp.temperature = spm.temperature;
    smp.top_p = spm.top_p;
    smp.min_p = spm.min_p;
    smp.top_k = spm.top_k;
    smp.rng = &rng;
    const Model::SpecSampling* sampled = spm.temperature > 0.f ? &smp : nullptr;
    if (spec) {
      // Speculative decode: the drafter drafts spec_block_ rows from the last token, the target verifies them in
      // one forward; greedy keeps the longest matching prefix plus the target's own next token (the same ids as
      // plain greedy decode), sampling accepts by the standard rule (Model::speculate). The accepted inputs are committed, so fed_ holds every row the model
      // has taken, including drafted rows past a stop.
      uint32_t buf[16];
      bool more = accept(t);
      while (more) {
        const int pos = int(fed_.size());
        if (pos + spec_block_ > max_ctx_) {  // no room for a whole block: one plain step
          const uint32_t u = uint32_t(t);
          m_.forward(&u, 1, pos, false);
          fed_.push_back(u);
          ++steps;
          t = pick(out, spm, rng);
          more = accept(t);
          continue;
        }
        const int n = m_.speculate(uint32_t(t), pos, spec_block_, banned_, buf, sampled);
        if (n < 0) {
          fed_.clear();
          throw ragon::NonFiniteLogits();
        }
        fed_.push_back(uint32_t(t));
        for (int i = 0; i + 1 < n; ++i) fed_.push_back(buf[i]);
        ++steps;
        for (int i = 0; i < n && more; ++i) {
          t = int(buf[i]);
          more = accept(t);
        }
      }
    } else if (gpu_greedy) {
      // The GPU picks each next id itself (Model::greedy_step), two steps in flight: step k feeds the token
      // produced last at position fed_.size() and leaves the next id in greedy_out(k). The host handles token k
      // while step k+1 runs, and encodes step k+2 only once token k is accepted, so a stop leaves exactly one step
      // in flight; that step's input (the stopping token) is then part of the model's sequence, and fed_ says so.
      if (accept(t)) {
        const int base = int(fed_.size());
        const int nsteps = limit - produced;  // step k produces token produced+k+1
        m_.greedy_begin(uint32_t(t), banned_, nsteps);
        int enq = 0;
        auto enqueue = [&] {
          m_.greedy_step(base + enq);
          ++enq;
        };
        enqueue();
        if (nsteps > 1) enqueue();
        uint32_t fed_tok = uint32_t(t);  // the input of the oldest unfinished step
        for (int k = 0;; ++k) {
          m_.greedy_wait(k);
          fed_.push_back(fed_tok);
          ++steps;
          const uint32_t* o = m_.greedy_out(k);
          const float top = std::bit_cast<float>(o[1]);
          if (!std::isfinite(top)) {
            if (enq > k + 1) m_.greedy_wait(k + 1);
            fed_.clear();
            throw ragon::NonFiniteLogits();
          }
          t = int(o[0]);
          fed_tok = uint32_t(t);
          const bool more = accept(t);
          if (!more) {
            if (enq > k + 1) {  // the step that took t as its input
              m_.greedy_wait(k + 1);
              fed_.push_back(fed_tok);
            }
            break;
          }
          if (enq < nsteps) enqueue();
        }
      }
    } else {
      while (accept(t)) {
        const uint32_t u = uint32_t(t);
        m_.forward(&u, 1, int(fed_.size()), false);
        fed_.push_back(u);
        ++steps;
        t = pick(out, spm, rng);
      }
    }
    const double decode_s = now_s() - td;
    if (finish == "cancelled") {
      std::fprintf(stderr, "[%s] cancelled after %d tokens\n", j.route.c_str(), produced);
      return;
    }
    if (!stops.stopped()) {
      std::string rest = stops.push(dec.flush());
      if (!stops.stopped()) rest += stops.finish();
      if (!rest.empty()) j.sink.text(rest);
    }
    char spec_note[96] = "";
    if (const long ns = m_.spec_steps - spec_steps0; ns > 0)
      std::snprintf(spec_note, sizeof spec_note, " (speculative: %.2f tokens a step)",
                    double(m_.spec_tokens - spec_tokens0) / double(ns));
    std::fprintf(stderr,
                 "[%s] prompt %d reused %zu prefill %d tok in %.3f s (%.0f tok/s) | out %d (reasoning %d) steps %d%s in "
                 "%.3f s = %.2f tok/s | finish %s\n",
                 j.route.c_str(), P, reuse, P - int(reuse), prefill_s, prefill_s > 0 ? (P - reuse) / prefill_s : 0.0,
                 produced, reasoning, steps, spec_note, decode_s, produced > 0 ? produced / decode_s : 0.0,
                 finish.c_str());
    std::fflush(stderr);
    j.sink.done(finish, P, produced, int(reuse), reasoning);
  }

  Model& m_;
  const Tokenizer& tok_;
  Specials sp_;
  int max_ctx_, chunk_, spec_block_;
  std::vector<int> stop_ids_, banned_;
  std::vector<uint32_t> fed_;  // the sequence in the model (prompt and the tokens fed after it)
  std::vector<float> row_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<std::shared_ptr<Job>> q_;
  std::thread thread_;
};

// ---------------------------------------------------------------------------------------------------------------
// The server

struct Server {
  Engine* engine = nullptr;
  const Tokenizer* tok = nullptr;
  std::string model_name, model_dir, api_key;
  RepetitionDetection repetition = kRepetitionDefault;
  bool repetition_completed = false;
  int keepalive_ms = 15000;
  long started = std::time(nullptr);
};

// Drains the job's outbox to the socket until the engine closes it; keep-alives while the engine has not written.
void pump(int fd, Outbox& box, Job& job, int keepalive_ms, const std::function<void()>& keepalive) {
  for (;;) {
    std::string chunk;
    bool closed = false, tick = false;
    {
      std::unique_lock<std::mutex> l(box.mu);
      const bool ready = keepalive_ms > 0
                             ? box.cv.wait_for(l, std::chrono::milliseconds(keepalive_ms), [&] { return !box.buf.empty() || box.closed; })
                             : (box.cv.wait(l, [&] { return !box.buf.empty() || box.closed; }), true);
      chunk.swap(box.buf);
      closed = box.closed;
      tick = !ready && !box.wrote;
    }
    if (!chunk.empty() && !write_all(fd, chunk)) {
      job.cancelled->store(true);
      return;
    }
    if (closed) return;
    if (tick && keepalive) keepalive();
  }
}

void handle_generate(Server& S, int fd, const HttpRequest& hr, const std::string& route) {
  const bool responses = route == "responses", chat = route == "chat";
  auto job = std::make_shared<Job>();
  job->route = route;
  try {
    const json body = json::parse(hr.body);
    job->req = responses ? parse_responses_request(body, false, true) : parse_request(body, chat, false);
    Request& q = job->req;
    if (q.n != 1) throw std::invalid_argument("n > 1 is not supported on the Metal backend");
    if (q.logprobs) throw std::invalid_argument("logprobs are not supported on the Metal backend");
    if (!q.image_urls.empty()) throw std::invalid_argument("images are not supported on the Metal backend");
    q.repetition = resolve_repetition(q, S.repetition);
    job->prompt = q.prompt_ids.empty() ? S.tok->encode(q.prompt_text) : q.prompt_ids;
    for (int t : job->prompt)
      if (t < 0 || t >= S.tok->vocab_size())
        throw std::invalid_argument("prompt token " + std::to_string(t) + " is outside the vocabulary");
  } catch (const std::exception& e) {
    write_all(fd, json_response(400, error_body(e.what(), "invalid_request_error")));
    return;
  }
  const Request req = job->req;
  const std::string model = req.model.empty() ? S.model_name : req.model;
  const long created = std::time(nullptr);
  const int max_ctx = S.engine->max_context();
  auto box = std::make_shared<Outbox>();
  std::function<void()> keepalive;

  if (responses) {
    struct State {
      std::mutex mu;  // the engine's sink calls and the connection thread's keep-alive
      long seq = 0;
      std::shared_ptr<Outbox> box;
      bool stream = false;
      std::optional<OutputParser> parser;
      std::optional<ResponseItems> items;
      std::string id = new_id("resp_");
      void emit(const std::string& type, json payload, bool from_engine = true) {
        if (!stream) return;
        payload["type"] = type;
        payload["sequence_number"] = seq++;
        box->put("event: " + type + "\ndata: " + dump_json(payload) + "\n\n", from_engine);
      }
    };
    auto st = std::make_shared<State>();
    st->box = box;
    st->stream = req.stream;
    st->parser.emplace(req.reasoning, req.tools);
    st->items.emplace(req.custom_tools, [st](const std::string& type, json payload) { st->emit(type, std::move(payload)); });
    st->items->set_namespaces(req.tool_namespaces);
    auto response_obj = [st, created, model](const std::string& status) {
      return json{{"id", st->id}, {"object", "response"}, {"created_at", created}, {"status", status},
                  {"model", model}, {"output", st->items->output()}, {"tools", json::array()}};
    };
    auto fail = [=](const std::string& msg, const std::string& code) {
      std::lock_guard<std::mutex> l(st->mu);
      if (req.stream) {
        json failed = response_obj("failed");
        failed["error"] = {{"code", code}, {"message", msg}};
        st->emit("response.failed", {{"response", failed}});
      } else {
        box->put(json_response(code == kContextLengthExceeded ? 400 : 500, error_body(msg, error_type(code), code)));
      }
      box->close();
    };
    if (req.stream) box->put(head_of(200, "text/event-stream", -1), false);
    st->emit("response.created", {{"response", response_obj("in_progress")}}, false);
    if (exceeds_context(job->prompt.size(), max_ctx)) {
      fail(context_overflow_message(job->prompt.size(), max_ctx), kContextLengthExceeded);
    } else {
      job->sink.text = [st](const std::string& piece) {
        std::lock_guard<std::mutex> l(st->mu);
        st->items->deliver(st->parser->push(piece));
      };
      job->sink.error = fail;
      const bool repetition_completed = S.repetition_completed;
      job->sink.done = [=](const std::string& finish, int prompt_tokens, int completion_tokens, int cached,
                           int reasoning_tokens) {
        std::lock_guard<std::mutex> l(st->mu);
        st->items->deliver(st->parser->finish());
        st->items->close();
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
        if (req.stream) st->emit(rep_incomplete ? "response.incomplete" : "response.completed", {{"response", resp}});
        else box->put(json_response(200, resp));
        box->close();
      };
      if (req.stream) {
        const json in_progress = {{"response", response_obj("in_progress")}};
        keepalive = [st, in_progress] {
          std::lock_guard<std::mutex> l(st->mu);
          st->emit("response.in_progress", in_progress, false);
        };
      }
      S.engine->submit(job);
    }
  } else {
    if (exceeds_context(job->prompt.size(), max_ctx)) {  // refused before any header
      write_all(fd, json_response(400, context_overflow_error(job->prompt.size(), max_ctx)));
      return;
    }
    struct State {
      OutputParser parser;
      std::string content, reasoning;
      std::vector<ToolCall> calls;
      bool first = true;
      State(bool r, const json& t) : parser(r, t) {}
    };
    auto st = std::make_shared<State>(req.reasoning, req.tools);
    const std::string id = new_id(chat ? "chatcmpl-" : "cmpl-");
    auto chunk_of = [=](json delta, json finish) {
      json c = {{"id", id}, {"object", chat ? "chat.completion.chunk" : "text_completion"}, {"created", created},
                {"model", model}};
      json choice = {{"index", 0}};
      if (chat) choice["delta"] = std::move(delta);
      else choice["text"] = delta.is_string() ? delta : json("");
      choice["finish_reason"] = std::move(finish);
      c["choices"] = json::array({choice});
      return c;
    };
    auto sse = [box](const json& c) { box->put("data: " + dump_json(c) + "\n\n"); };
    auto deliver = [=](const OutputParser::Delta& d) {
      st->content += d.content;
      st->reasoning += d.reasoning;
      if (!req.stream) {
        st->calls.insert(st->calls.end(), d.tool_calls.begin(), d.tool_calls.end());
        return;
      }
      if (!chat) {
        if (!d.content.empty()) sse(chunk_of(json(d.content), nullptr));
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
      if (delta.size() > 0) sse(chunk_of(delta, nullptr));
    };
    job->sink.text = [=](const std::string& piece) {
      deliver(chat ? st->parser.push(piece) : OutputParser::Delta{{}, piece, {}});
    };
    job->sink.error = [=](const std::string& msg, const std::string& code) {
      const json error = error_body(msg, error_type(code), code);
      if (req.stream) box->put(stream_error_tail(error, chunk_of(json::object(), kStreamErrorFinish)));
      else box->put(json_response(code == kContextLengthExceeded ? 400 : 500, error));
      box->close();
    };
    job->sink.done = [=](const std::string& finish_in, int prompt_tokens, int completion_tokens, int cached,
                         int reasoning_tokens) {
      if (chat) deliver(st->parser.finish());
      const std::string finish = !st->calls.empty() && finish_in == "stop" ? "tool_calls" : finish_in;
      json usage = {{"prompt_tokens", prompt_tokens},
                    {"completion_tokens", completion_tokens},
                    {"total_tokens", prompt_tokens + completion_tokens},
                    {"prompt_tokens_details", {{"cached_tokens", cached}}}};
      if (chat) usage["completion_tokens_details"] = {{"reasoning_tokens", reasoning_tokens}};
      if (req.stream) {
        sse(chunk_of(json::object(), finish));
        if (req.include_usage)
          sse({{"id", id}, {"object", chat ? "chat.completion.chunk" : "text_completion"}, {"created", created},
               {"model", model}, {"choices", json::array()}, {"usage", usage}});
        box->put("data: [DONE]\n\n");
        box->close();
        return;
      }
      json choice = {{"index", 0}, {"finish_reason", finish}};
      if (chat) {
        json msg = {{"role", "assistant"},
                    {"content", st->content.empty() && !st->calls.empty() ? json() : json(st->content)}};
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
      box->put(json_response(200, {{"id", id}, {"object", chat ? "chat.completion" : "text_completion"},
                                   {"created", created}, {"model", model}, {"choices", json::array({choice})},
                                   {"usage", usage}}));
      box->close();
    };
    if (req.stream) {
      box->put(head_of(200, "text/event-stream", -1), false);
      keepalive = [box] { box->put(": keep-alive\n\n", false); };
    }
    S.engine->submit(job);
  }
  pump(fd, *box, *job, req.stream ? S.keepalive_ms : 0, keepalive);
}

void handle(Server& S, int fd) {
  HttpRequest hr;
  int status = 0;
  if (!read_request(fd, hr, status)) {
    if (status) write_all(fd, json_response(status, error_body(reason_of(status), "invalid_request_error")));
    return;
  }
  if (hr.path == "/health" || hr.path == "/healthz") {
    write_all(fd, head_of(200, "text/plain", 2) + "ok");
    return;
  }
  if (!S.api_key.empty()) {
    auto a = hr.headers.find("authorization");
    if (a == hr.headers.end() || a->second != "Bearer " + S.api_key) {
      write_all(fd, json_response(401, error_body("invalid API key", "invalid_request_error", "invalid_api_key")));
      return;
    }
  }
  const int max_ctx = S.engine->max_context();
  if (hr.method == "GET" && hr.path == "/v1/models") {
    json m = {{"id", S.model_name}, {"object", "model"}, {"created", S.started}, {"owned_by", "ragon-engine"},
              {"root", S.model_name}, {"max_model_len", max_ctx}};
    write_all(fd, json_response(200, {{"object", "list"}, {"data", json::array({m})}}));
    return;
  }
  if (hr.method == "GET" && hr.path == "/model_info") {
    write_all(fd, json_response(200, {{"model_path", S.model_name}, {"backend", "metal"}, {"max_model_len", max_ctx},
                                      {"context_length", max_ctx}}));
    return;
  }
  if (hr.method == "POST") {
    const auto ends = [&](const char* s) {
      const size_t n = std::strlen(s);
      return hr.path.size() >= n && hr.path.compare(hr.path.size() - n, n, s) == 0;
    };
    if (ends("/responses")) return handle_generate(S, fd, hr, "responses");
    if (ends("/chat/completions")) return handle_generate(S, fd, hr, "chat");
    if (hr.path == "/v1/completions") return handle_generate(S, fd, hr, "completions");
  }
  write_all(fd, json_response(404, error_body("no route " + hr.method + " " + hr.path, "invalid_request_error")));
}

constexpr int kSpecBlock = 8;  // the draft block on the M5 Max (lane 7 sweep)

int usage() {
  std::fprintf(stderr,
               "usage: ragon-serve --model DIR [--host 127.0.0.1] [--port 8000] [--served-model-name NAME] "
               "[--max-context N] [--chunk C] [--keepalive-seconds N] [--repetition-detection P,p,R,K|off] "
               "[--repetition-report incomplete|completed] [--api-key KEY] [--pretokenizer production|checkpoint] "
               "[--draft DIR [--spec-block N] [--sparse-head-dir DIR]]\n");
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  @autoreleasepool {
    std::string model_dir, host = "127.0.0.1", name, pretokenizer = "production", repetition_arg, report = "incomplete";
    std::string api_key = std::getenv("RAGON_API_KEY") ? std::getenv("RAGON_API_KEY") : "";
    std::string draft_dir, sparse_head_dir;
    int port = 8000, max_ctx = 65536, chunk = 2048, keepalive_s = 15, spec_block = 0;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      auto next = [&]() -> std::string {
        if (i + 1 >= argc) {
          std::fprintf(stderr, "%s needs a value\n", a.c_str());
          std::exit(2);
        }
        return argv[++i];
      };
      if (a == "--model") model_dir = next();
      else if (a == "--host") host = next();
      else if (a == "--port") port = std::atoi(next().c_str());
      else if (a == "--served-model-name") name = next();
      else if (a == "--max-context") max_ctx = std::atoi(next().c_str());
      else if (a == "--chunk") chunk = std::atoi(next().c_str());
      else if (a == "--keepalive-seconds") keepalive_s = std::atoi(next().c_str());
      else if (a == "--repetition-detection") repetition_arg = next();
      else if (a == "--repetition-report") report = next();
      else if (a == "--api-key") api_key = next();
      else if (a == "--pretokenizer") pretokenizer = next();
      else if (a == "--draft") draft_dir = next();
      else if (a == "--spec-block") spec_block = std::atoi(next().c_str());
      else if (a == "--sparse-head-dir") sparse_head_dir = next();
      else return usage();
    }
    if (model_dir.empty() || max_ctx < 2 || chunk < 1 || (report != "incomplete" && report != "completed") ||
        (pretokenizer != "production" && pretokenizer != "checkpoint"))
      return usage();
    while (model_dir.size() > 1 && model_dir.back() == '/') model_dir.pop_back();
    if (name.empty()) name = model_dir.substr(model_dir.find_last_of('/') + 1);
    if (const char* k = std::getenv("RAGON_KEEPALIVE_SECONDS")) keepalive_s = std::atoi(k);

    Server S;
    S.model_name = name;
    S.model_dir = model_dir;
    S.api_key = api_key;
    S.keepalive_ms = std::max(0, keepalive_s) * 1000;
    S.repetition_completed = report == "completed";
    if (repetition_arg == "off") {
      S.repetition = RepetitionDetection{0, 0, 0, 0};
    } else if (!repetition_arg.empty()) {
      RepetitionDetection r{};
      if (std::sscanf(repetition_arg.c_str(), "%d,%d,%d,%d", &r.max_pattern_size, &r.min_pattern_size,
                      &r.min_count, &r.max_run) != 4)
        return usage();
      validate(r);
      S.repetition = r;
    }

    Tokenizer tok(model_dir + "/tokenizer.json", pretokenizer == "production" ? ragon::kProductionPretokenizer : "");
    S.tok = &tok;
    Specials sp;
    sp.think_end = tok.token_id("</think>");
    sp.im_end = tok.token_id("<|im_end|>");
    if (const auto one = tok.encode("\n"); one.size() == 1) sp.newline = one[0];
    if (const auto two = tok.encode("\n\n"); two.size() == 1) sp.blank_line = two[0];

    Queue q;
    q.open();
    Model m;
    const double t0 = now_s();
    m.load(q, model_dir, max_ctx, chunk);
    const char* ie = std::getenv("RAGON_METAL_IGNORE_EOS");
    const bool ignore_eos = ie && std::atoi(ie) == 1;
    std::fprintf(stderr, "metal serve: %s loaded in %.1f s, max context %d, chunk %d%s\n", name.c_str(), now_s() - t0,
                 max_ctx, chunk, ignore_eos ? ", END TOKENS BANNED (RAGON_METAL_IGNORE_EOS=1)" : "");
    if (!draft_dir.empty()) {
      m.draft_load(draft_dir);
      if (spec_block <= 0) spec_block = kSpecBlock;
      spec_block = std::min(spec_block, m.draft_block());
      // The sparse head (a top-k candidate head over the target's LM head) is not ported to Metal yet: the
      // drafter scores the full vocabulary, and the log says so rather than claiming the production head.
      std::fprintf(stderr, "metal serve: drafter %s, block %d, drafter head: dense%s\n", draft_dir.c_str(), spec_block,
                   sparse_head_dir.empty() ? "" : " (--sparse-head-dir given; the sparse head is not used on Metal yet)");
    }
    Engine engine(m, tok, sp, max_ctx, chunk, ignore_eos, spec_block);
    S.engine = &engine;

    signal(SIGPIPE, SIG_IGN);
    const int ls = ::socket(AF_INET, SOCK_STREAM, 0);
    const int one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(uint16_t(port));
    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
      std::fprintf(stderr, "--host %s: not an IPv4 address\n", host.c_str());
      return 2;
    }
    if (::bind(ls, reinterpret_cast<sockaddr*>(&addr), sizeof addr) != 0 || ::listen(ls, 64) != 0) {
      std::perror("bind/listen");
      return 1;
    }
    std::fprintf(stderr, "metal serve: listening on %s:%d\n", host.c_str(), port);
    std::fflush(stderr);
    for (;;) {
      const int fd = ::accept(ls, nullptr, nullptr);
      if (fd < 0) continue;
      setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
      setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
      std::thread([&S, fd] {
        @autoreleasepool {
          try {
            handle(S, fd);
          } catch (const std::exception& e) {
            std::fprintf(stderr, "connection: %s\n", e.what());
          }
          ::shutdown(fd, SHUT_WR);
          ::close(fd);
        }
      }).detach();
    }
  }
}
