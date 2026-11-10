// ragon_force: the engine side of the agreement study (bench/agreement/run.sh), built by hand against any
// ragon-engine build's libraries so older builds can be scored too:
//   nvcc -std=c++20 -O2 -gencode arch=compute_121a,code=sm_121a -I$SRC/src ragon_force.cpp \
//        $SRC/build/libling_core.a $SRC/build/libling_tokenizer.a -lcuda -lpcre2-8 -licuuc -lpng16 -lwebp -o ragon-force
//   ragon-force tokenize MODEL < prompts.jsonl            -> {"ids": [...]} per line (prompt strings, JSON)
//   ragon-force score MODEL K < cases.jsonl               -> per case: engine top-K log-probs at each forced position
//        case: {"prompt": [ids], "cont": [ids]}; position j scores the token after prompt + cont[:j]
//   ragon-force gen MODEL MAXTOK < cases.jsonl            -> greedy continuation of {"prompt": [ids]}: {"ids", "text"}
//   ragon-force dump MODEL < cases.jsonl                  -> a cold prefill of each {"ids": [ids]} with RAGON_DUMP_FEATURES
//        set and a drafter loaded, so the engine appends every position's drafter features (DRAFTER_RETRAIN §3);
//        prints {"n": len, "ms": prefill ms} per case. bench/drafter/teacher_dump.py drives it.
// RAGON_FORCE_DRAFT=DIR loads that drafter (dump needs one: the features are the drafter's target layers);
// RAGON_FORCE_MAX_CONTEXT=N raises the context from 40000.
// RAGON_LORA=LIBRARY/NAME: score and generate with that adapter (LORA §3, one GPU slot) instead of the base.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/engine.hpp"
#include "tokenizer/tokenizer.hpp"

using json = nlohmann::json;

static json topk(const std::vector<float>& logits, int K, const std::vector<int>& extra) {
  const int V = static_cast<int>(logits.size());
  double mx = -1e30;
  for (float v : logits) mx = std::max(mx, double(v));
  double sum = 0;
  for (float v : logits) sum += std::exp(double(v) - mx);
  const double lse = mx + std::log(sum);
  std::vector<int> idx(V);
  std::iota(idx.begin(), idx.end(), 0);
  std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                    [&](int a, int b) { return logits[a] > logits[b] || (logits[a] == logits[b] && a < b); });
  json t = json::array();
  for (int i = 0; i < K; ++i) t.push_back({idx[i], double(logits[idx[i]]) - lse});
  json e = json::object();
  for (int id : extra)
    if (id >= 0 && id < V) e[std::to_string(id)] = double(logits[id]) - lse;
  return {{"top", t}, {"lp", e}};
}

int main(int argc, char** argv) {
  if (argc < 3) return 2;
  const std::string mode = argv[1], model = argv[2];
  ragon::Tokenizer tok(model + "/tokenizer.json");
  std::string line;
  if (mode == "tokenize") {
    while (std::getline(std::cin, line))
      if (!line.empty()) std::cout << json{{"ids", tok.encode(json::parse(line).get<std::string>())}}.dump() << "\n";
    return 0;
  }
  ragon::EngineOptions opts;
  opts.max_context = 40000;
  if (const char* m = std::getenv("RAGON_FORCE_MAX_CONTEXT")) opts.max_context = std::atoi(m);
  if (const char* d = std::getenv("RAGON_FORCE_DRAFT")) opts.draft_dir = d;
  const char* lora = std::getenv("RAGON_LORA");
  if (lora) opts.lora_slots = 1;
  ragon::Engine engine(model, opts);
  if (lora) {
    const std::filesystem::path p = std::filesystem::path(lora).lexically_normal();
    ragon::LoraLibrary lib(p.parent_path().string(), 1, engine.lora_shape(), engine.lora_draft_shape());
    const auto adapter = lib.load(p.filename().string(), p.string());
    engine.lora_upload(0, *adapter);
    engine.activate_session(0, "", false, ragon::lora_scope(0, adapter.get()), 0);
    std::fprintf(stderr, "ragon-force: adapter %s (%zu bytes)\n", adapter->name.c_str(), adapter->bytes());
  }
  const int im_end = tok.token_id("<|im_end|>");
  if (mode == "score") {
    const int K = std::stoi(argv[3]);
    while (std::getline(std::cin, line)) {
      if (line.empty()) continue;
      const json c = json::parse(line);
      const std::vector<int> prompt = c["prompt"], cont = c["cont"];
      // Production's top-k ids at each position (optional), so the engine's log-prob of each is reported.
      const json want = c.value("want", json::array());
      engine.reset();
      const std::vector<float>* logits = &engine.prefill(prompt);
      json out = json::array();
      for (size_t j = 0; j < cont.size(); ++j) {
        std::vector<int> extra = {cont[j]};
        if (j < want.size())
          for (const auto& w : want[j]) extra.push_back(w.get<int>());
        out.push_back(topk(*logits, K, extra));
        if (j + 1 < cont.size()) logits = &engine.step(cont[j]);
      }
      std::cout << json{{"pos", out}}.dump() << "\n" << std::flush;
    }
    return 0;
  }
  if (mode == "dump") {
    if (opts.draft_dir.empty() || !std::getenv("RAGON_DUMP_FEATURES")) {
      std::fprintf(stderr, "ragon-force dump: set RAGON_FORCE_DRAFT and RAGON_DUMP_FEATURES\n");
      return 2;
    }
    while (std::getline(std::cin, line)) {
      if (line.empty()) continue;
      const std::vector<int> ids = json::parse(line)["ids"];
      engine.reset();
      const auto t0 = std::chrono::steady_clock::now();
      engine.prefill(ids, nullptr, true);
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      std::cout << json{{"n", ids.size()}, {"ms", ms}}.dump() << "\n" << std::flush;
    }
    return 0;
  }
  if (mode == "gen") {
    const int maxtok = std::stoi(argv[3]);
    while (std::getline(std::cin, line)) {
      if (line.empty()) continue;
      const std::vector<int> prompt = json::parse(line)["prompt"];
      engine.reset();
      const std::vector<float>* logits = &engine.prefill(prompt);
      std::vector<int> out;
      for (int i = 0; i < maxtok; ++i) {
        const int t = static_cast<int>(std::max_element(logits->begin(), logits->end()) - logits->begin());
        out.push_back(t);
        if (t == im_end || t == engine.config().eos_token) break;
        logits = &engine.step(t);
      }
      std::cout << json{{"ids", out}, {"text", tok.decode(out)}}.dump() << "\n" << std::flush;
    }
    return 0;
  }
  return 2;
}
