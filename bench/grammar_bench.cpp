// ragon-grammar-bench [TOKENIZER_JSON]: the CPU cost of grammar masks at the model's vocabulary (248,320 ids).
//
// Five workloads, each a grammar and a recorded output tokenized with the model's tokenizer:
//   codex_schema   a Codex-like nested JSON schema, the output inside nested objects and strings
//   tool_value     the tool-call grammar, the output inside a parameter VALUE
//   apply_patch    Codex's apply_patch Lark grammar, the output inside add_line
//   code_mode      Codex's exec grammar (Code Mode), the output inside SOURCE
//   schema_30      a flat schema with 30 properties
// Per workload: one-row fill median and p99 over every position of the output; a 16-row step (16 matchers at
// spread positions, one fill each) on 1, 2 and 4 threads; compile cold and cached; the compiled size.
// The tokenizer comes from argv[1], RAGON_TOKENIZER_JSON, or the model snapshot in the Hugging Face cache.
#include <glob.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "grammar/grammar.hpp"
#include "tokenizer/tokenizer.hpp"

using namespace ragon;
using namespace ragon::grammar;
using Clock = std::chrono::steady_clock;

namespace {

std::string tokenizer_path(int argc, char** argv) {
  if (argc > 1) return argv[1];
  if (const char* p = std::getenv("RAGON_TOKENIZER_JSON")) return p;
  const char* home = std::getenv("HOME");
  const std::string pattern = std::string(home ? home : "") +
                              "/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/*/tokenizer.json";
  glob_t g{};
  std::string out;
  if (glob(pattern.c_str(), 0, nullptr, &g) == 0 && g.gl_pathc > 0) out = g.gl_pathv[0];
  globfree(&g);
  return out;
}

double us_since(Clock::time_point t) { return std::chrono::duration<double, std::micro>(Clock::now() - t).count(); }

double pct(std::vector<double> v, double p) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[std::min(v.size() - 1, static_cast<std::size_t>(p * (v.size() - 1) + 0.5))];
}

const char* kApplyPatchLark =
    "start: begin_patch hunk+ end_patch\nbegin_patch: \"*** Begin Patch\" LF\nend_patch: \"*** End Patch\" LF?\n\n"
    "hunk: add_hunk | delete_hunk | update_hunk\nadd_hunk: \"*** Add File: \" filename LF add_line+\n"
    "delete_hunk: \"*** Delete File: \" filename LF\n"
    "update_hunk: \"*** Update File: \" filename LF change_move? change?\n\nfilename: /(.+)/\n"
    "add_line: \"+\" /(.*)/ LF -> line\n\nchange_move: \"*** Move to: \" filename LF\n"
    "change: (change_context | change_line)+ eof_line?\nchange_context: (\"@@\" | \"@@ \" /(.+)/) LF\n"
    "change_line: (\"+\" | \"-\" | \" \") /(.*)/ LF\neof_line: \"*** End of File\" LF\n\n%import common.LF\n";

const char* kExecLark =
    "start: pragma_source | plain_source\npragma_source: PRAGMA_LINE NEWLINE SOURCE\nplain_source: SOURCE\n\n"
    "PRAGMA_LINE: /[ \\t]*\\/\\/ @exec:[^\\r\\n]*/\nNEWLINE: /\\r?\\n/\nSOURCE: /[\\s\\S]+/\n";

nlohmann::ordered_json codex_schema() {
  return nlohmann::ordered_json::parse(R"({
    "type": "object",
    "properties": {
      "summary": {"type": "string"},
      "plan": {"type": "array", "items": {"type": "object", "properties": {
        "step": {"type": "string"}, "status": {"enum": ["pending", "in_progress", "completed"]}},
        "required": ["step", "status"]}},
      "commands": {"type": "array", "items": {"type": "object", "properties": {
        "argv": {"type": "array", "items": {"type": "string"}},
        "workdir": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "timeout_ms": {"type": "integer"},
        "env": {"type": "object", "additionalProperties": {"type": "string"}}},
        "required": ["argv", "workdir", "timeout_ms"]}},
      "findings": {"type": "array", "items": {"$ref": "#/$defs/finding"}},
      "confidence": {"type": "number"},
      "done": {"type": "boolean"}
    },
    "required": ["summary", "plan", "commands", "findings", "confidence", "done"],
    "$defs": {"finding": {"type": "object", "properties": {
      "file": {"type": "string"}, "line": {"type": "integer"},
      "severity": {"enum": ["low", "medium", "high"]}, "note": {"type": "string"}},
      "required": ["file", "line", "severity", "note"]}}
  })");
}

const char* kCodexOutput = R"({
  "summary": "The parser drops the last hunk when the patch has no trailing newline.",
  "plan": [
    {"step": "Reproduce with a patch without a final newline", "status": "completed"},
    {"step": "Fix the end_patch rule and add a regression test", "status": "in_progress"}
  ],
  "commands": [
    {"argv": ["cargo", "test", "-p", "apply-patch", "--", "--nocapture"], "workdir": "codex-rs", "timeout_ms": 120000, "env": {"RUST_LOG": "debug"}},
    {"argv": ["git", "diff", "--stat"], "workdir": null, "timeout_ms": 10000}
  ],
  "findings": [{"file": "codex-rs/apply-patch/src/parser.rs", "line": 214, "severity": "high", "note": "the loop exits before the last hunk is pushed"}],
  "confidence": 0.82,
  "done": false
})";

nlohmann::ordered_json schema_30() {
  nlohmann::ordered_json props = nlohmann::ordered_json::object();
  nlohmann::ordered_json req = nlohmann::ordered_json::array();
  for (int i = 0; i < 30; ++i) {
    const std::string k = "field_" + std::to_string(i);
    const char* t = i % 3 == 0 ? "string" : i % 3 == 1 ? "integer" : "boolean";
    props[k] = {{"type", t}};
    req.push_back(k);
  }
  return {{"type", "object"}, {"properties", props}, {"required", req}};
}

std::string schema_30_output() {
  std::string s = "{";
  for (int i = 0; i < 30; ++i) {
    s += std::string(i ? ", " : "") + "\"field_" + std::to_string(i) + "\": ";
    s += i % 3 == 0 ? "\"value number " + std::to_string(i) + "\"" : i % 3 == 1 ? std::to_string(i * 37) : "true";
  }
  return s + "}";
}

const char* kToolOutput =
    "<tool_call>\n<function=exec_command>\n<parameter=cmd>\n"
    "cd codex-rs && cargo test -p codex-core --test all -- suite::exec::sandbox_denies_network 2>&1 | tail -n 40\n"
    "</parameter>\n<parameter=workdir>\n/workspace/project\n</parameter>\n<parameter=timeout_ms>\n600000\n"
    "</parameter>\n</function>\n</tool_call>";

const char* kPatchOutput =
    "*** Begin Patch\n*** Add File: src/grammar/notes.md\n+# Notes\n+\n"
    "+The mask is filled once per row and step; the sampler reads it as int32 words.\n"
    "+Whitespace slots allow one newline and at most sixteen blanks.\n+    indented line with code: x = f(y) + 1;\n"
    "*** Update File: src/main.rs\n@@ fn main() {\n-    println!(\"hello\");\n+    println!(\"hello, world\");\n"
    "     let x = 1;\n*** End Patch\n";

const char* kExecOutput =
    "// @exec: {\"yield_time_ms\": 1000}\nconst files = await tools.shell({command: [\"rg\", \"-l\", \"TODO\", \"src\"]});\n"
    "for (const f of files.stdout.split(\"\\n\")) {\n  if (!f) continue;\n  const text = await tools.read_file({path: f});\n"
    "  console.log(f, text.split(\"\\n\").filter(l => l.includes(\"TODO\")).length);\n}\n";

struct Workload {
  std::string name;
  std::function<Grammar(Compiler&)> compile;
  std::string output;
};

// A small pool that runs one task per worker and waits for all: the 16-row step split over threads.
class Pool {
 public:
  explicit Pool(int n) : n_(n) {
    for (int i = 1; i < n; ++i) threads_.emplace_back([this, i] { loop(i); });
  }
  ~Pool() {
    {
      std::lock_guard<std::mutex> l(mu_);
      stop_ = true;
      ++gen_;
    }
    cv_.notify_all();
    for (auto& t : threads_) t.join();
  }
  void run(const std::function<void(int)>& f) {
    {
      std::lock_guard<std::mutex> l(mu_);
      task_ = &f;
      pending_ = n_ - 1;
      ++gen_;
    }
    cv_.notify_all();
    f(0);
    std::unique_lock<std::mutex> l(mu_);
    done_.wait(l, [&] { return pending_ == 0; });
  }

 private:
  void loop(int i) {
    uint64_t seen = 0;
    while (true) {
      const std::function<void(int)>* f;
      {
        std::unique_lock<std::mutex> l(mu_);
        cv_.wait(l, [&] { return gen_ != seen; });
        seen = gen_;
        if (stop_) return;
        f = task_;
      }
      (*f)(i);
      std::lock_guard<std::mutex> l(mu_);
      if (--pending_ == 0) done_.notify_one();
    }
  }
  int n_;
  std::vector<std::thread> threads_;
  std::mutex mu_;
  std::condition_variable cv_, done_;
  const std::function<void(int)>* task_ = nullptr;
  int pending_ = 0;
  uint64_t gen_ = 0;
  bool stop_ = false;
};

}  // namespace

int main(int argc, char** argv) {
  const std::string path = tokenizer_path(argc, argv);
  if (path.empty()) {
    std::fprintf(stderr, "no tokenizer.json (argv[1], RAGON_TOKENIZER_JSON or the Hugging Face cache)\n");
    return 2;
  }
  Tokenizer tok(path);
  Vocab vocab(tok);
  const int words = vocab.words();

  ToolCallSpec tools;
  tools.choice = ToolCallSpec::Choice::kAuto;
  tools.tools.push_back({Tool::Kind::kFunction, "exec_command", {"cmd", "workdir", "timeout_ms", "yield_time_ms"}});
  tools.tools.push_back({Tool::Kind::kFunction, "update_plan", {"explanation", "plan"}});
  tools.tools.push_back({Tool::Kind::kCustom, "apply_patch", {}, Tool::Format::kLark, kApplyPatchLark});

  std::vector<Workload> loads = {
      {"codex_schema", [](Compiler& c) { return c.json_schema(codex_schema(), "response_format.json_schema.schema"); },
       kCodexOutput},
      {"tool_value", [&](Compiler& c) { return c.tool_calls(tools); }, kToolOutput},
      {"apply_patch", [](Compiler& c) { return c.lark(kApplyPatchLark, "tools[0].format"); }, kPatchOutput},
      {"code_mode", [](Compiler& c) { return c.lark(kExecLark, "tools[0].format"); }, kExecOutput},
      {"schema_30", [](Compiler& c) { return c.json_schema(schema_30(), "response_format.json_schema.schema"); },
       schema_30_output()},
  };

  std::printf("vocab %d ids, %d words per row\n", vocab.size(), words);
  std::printf("%-13s %9s %9s %8s %8s %8s %9s %9s %9s %9s %6s\n", "workload", "cold_ms", "cached_us", "mem_KiB",
              "fill_p50", "fill_p99", "step16_1t", "step16_2t", "step16_4t", "s16_1t_p99", "tokens");
  std::vector<int32_t> buf(static_cast<std::size_t>(words) * 16);
  for (const Workload& w : loads) {
    Compiler compiler(vocab, std::size_t{256} << 20, 1);
    auto t0 = Clock::now();
    Grammar g = w.compile(compiler);
    const double cold_ms = us_since(t0) / 1000;
    t0 = Clock::now();
    Grammar g2 = w.compile(compiler);
    const double cached_us = us_since(t0);

    const std::vector<int> ids = tok.encode(w.output);
    // Walk the output: a fill (timed) before every token, as decoding does. Matchers at every position kept for
    // the 16-row step.
    std::vector<double> fills;
    std::vector<Matcher> at;
    for (int rep = 0; rep < 5; ++rep) {
      Matcher m(g);
      for (std::size_t i = 0; i < ids.size(); ++i) {
        t0 = Clock::now();
        m.fill(buf.data());
        fills.push_back(us_since(t0));
        if (rep == 0) at.push_back(m.fork());
        if (!m.accept(ids[i])) {
          std::fprintf(stderr, "%s: token %zu (id %d, '%s') refused\n", w.name.c_str(), i, ids[i],
                       tok.token_bytes(ids[i]).c_str());
          return 1;
        }
      }
    }
    // 16 rows at spread positions; each step fills all 16 and advances every row by its next token.
    std::vector<double> step[3];
    const int threads[3] = {1, 2, 4};
    for (int ti = 0; ti < 3; ++ti) {
      Pool pool(threads[ti]);
      std::vector<Matcher> rows;
      std::vector<std::size_t> pos;
      for (int r = 0; r < 16; ++r) {
        const std::size_t p = (at.size() - 1) * r / 16;
        rows.push_back(at[p].fork());
        pos.push_back(p);
      }
      for (int s = 0; s < 200; ++s) {
        const int n = threads[ti];
        std::function<void(int)> work = [&](int t) {
          for (int r = t; r < 16; r += n) rows[r].fill(buf.data() + static_cast<std::size_t>(r) * words);
        };
        t0 = Clock::now();
        pool.run(work);
        step[ti].push_back(us_since(t0));
        for (int r = 0; r < 16; ++r) {
          if (pos[r] + 1 >= ids.size()) {  // wrap: back to an earlier position
            pos[r] = (at.size() - 1) * r / 16 / 2;
            rows[r] = at[pos[r]].fork();
          } else {
            rows[r].accept(ids[pos[r]]);
            ++pos[r];
          }
        }
      }
    }
    std::printf("%-13s %9.1f %9.1f %8zu %8.1f %8.1f %9.1f %9.1f %9.1f %9.1f %6zu\n", w.name.c_str(), cold_ms,
                cached_us, g.memory_bytes() / 1024, pct(fills, 0.5), pct(fills, 0.99), pct(step[0], 0.5),
                pct(step[1], 0.5), pct(step[2], 0.5), pct(step[0], 0.99), ids.size());
  }
  std::printf("(fill and step16 in microseconds; step16 = 16 matchers, one fill each, split over 1/2/4 threads)\n");
  return 0;
}
