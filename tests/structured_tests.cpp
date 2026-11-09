// ragon-structured-tests [TOKENIZER_JSON]: structured output stage 2's server side (serve/structured.hpp) against the
// model's vocabulary: a chat request's response_format and tool_choice "required" as parse_request reads them, the
// grammar request_grammar builds from them (and its 400s), and GrammarConstraint driving the masked host sampler to
// an answer that parses and validates. The tokenizer comes as in ragon-grammar-tests; without one, exit 77.
#include <glob.h>

#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/sampling.hpp"
#include "serve/structured.hpp"

using namespace ragon;
using namespace ragon::serve;
using ragon::grammar::Compiler;
using ragon::grammar::Vocab;

static int failures = 0;
#define EXPECT(cond)                                                                 \
  do {                                                                               \
    if (!(cond)) {                                                                   \
      ++failures;                                                                    \
      std::fprintf(stderr, "%s:%d: EXPECT(%s) failed\n", __FILE__, __LINE__, #cond); \
    }                                                                                \
  } while (0)

namespace {

const Tokenizer* g_tok = nullptr;
Compiler* g_comp = nullptr;
int g_words = 0, g_vocab_size = 0, g_im_end = -1;

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

const char* kSchema = R"({"type": "object", "properties": {"name": {"type": "string"}, "n": {"type": "integer"},
    "tags": {"type": "array", "items": {"type": "string"}}, "ok": {"type": "boolean"}},
    "required": ["name", "n", "tags", "ok"], "additionalProperties": false})";
const char* kTools = R"([{"type": "function", "function": {"name": "get_weather", "description": "Weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}])";

json chat_body() {
  return json{{"model", "m"}, {"messages", json::array({json{{"role", "user"}, {"content", "hi"}}})}};
}

// The 400 request_grammar's caller answers: "code|message", "" when it builds.
std::string refusal(const Request& r) {
  try {
    (void)request_grammar(*g_comp, r);
  } catch (const grammar::Error& e) {
    return e.code + "|" + e.what();
  } catch (const std::invalid_argument& e) {
    return std::string("invalid_request_error|") + e.what();
  }
  return "";
}

void test_parse() {
  json b = chat_body();
  Request r = parse_request(b, true);
  EXPECT(r.response_format.is_null() && !r.tool_choice_required);
  EXPECT(!request_grammar(*g_comp, r));
  b["response_format"] = {{"type", "text"}};
  EXPECT(parse_request(b, true).response_format.is_null());
  b["response_format"] = {{"type", "json_object"}};
  r = parse_request(b, true);
  EXPECT(r.response_format.value("type", "") == "json_object" && request_grammar(*g_comp, r));
  b["response_format"] = {{"type", "json_schema"}, {"json_schema", {{"name", "x"}, {"schema", json::parse(kSchema)}}}};
  r = parse_request(b, true);
  EXPECT(r.response_format.value("type", "") == "json_schema" && request_grammar(*g_comp, r));
  b.erase("response_format");
  b["tool_choice"] = "required";
  b["tools"] = json::parse(kTools);
  r = parse_request(b, true);
  EXPECT(r.tool_choice_required && request_grammar(*g_comp, r));
  b["tools"] = json::array();
  b.erase("tools");
  EXPECT(!request_grammar(*g_comp, parse_request(b, true)));  // required without tools: nothing to call
  b["tool_choice"] = "auto";
  EXPECT(!parse_request(b, true).tool_choice_required);
  // Completions never: the fields are chat only.
  Request c;
  c.chat = false;
  c.response_format = json{{"type", "json_object"}};
  EXPECT(!request_grammar(*g_comp, c));
}

void test_refusals() {
  json b = chat_body();
  b["response_format"] = {{"type", "json_schema"}, {"json_schema", {{"name", "x"}, {"schema", {{"type", "string"}, {"pattern", "a+"}}}}}};
  const std::string pat = refusal(parse_request(b, true));
  EXPECT(pat.rfind("unsupported_json_schema|response_format.json_schema.schema: the keyword 'pattern' at \"\"", 0) == 0);
  b["response_format"] = {{"type", "json_schema"}, {"json_schema", {{"name", "x"}}}};
  EXPECT(refusal(parse_request(b, true)) ==
         "invalid_request_error|response_format.json_schema.schema must be an object");
  b["response_format"] = {{"type", "json_schema"}, {"json_schema", {{"name", "x"}, {"schema", "nope"}}}};
  EXPECT(refusal(parse_request(b, true)).rfind("invalid_request_error|", 0) == 0);
}

// The answer `drive` samples under the request's grammar: random logits with a bonus on the tokens that close
// things, so the walk ends; at most `cap` tokens. Returns the text before the stop token ("" if it did not stop).
std::string drive(const grammar::Grammar& g, std::mt19937_64& rng, float temperature, int cap, bool* stopped) {
  GrammarConstraint con(g, g_words);
  std::vector<int32_t> mask(g_words);
  std::vector<float> logits(g_vocab_size);
  static std::vector<int> bonus;
  if (bonus.empty())
    for (const char* s : {"\"", "}", "]", "1", "true", ",", "\"}", "\n</parameter>", "</tool_call>", "\n"})
      for (int id : g_tok->encode(s)) bonus.push_back(id);
  std::normal_distribution<float> n(0.f, 1.f);
  SamplingParams p;
  p.temperature = temperature;
  std::vector<int> out;
  *stopped = false;
  for (int step = 0; step < cap; ++step) {
    for (float& x : logits) x = n(rng);
    for (int id : bonus) logits[id] += 6.f;
    logits[g_im_end] += 6.f;
    con.fill(mask.data());
    const int t = sample_logits_masked(logits, p, out, rng, mask.data());
    EXPECT(con.accept(t));
    if (t == g_im_end) {
      EXPECT(con.terminated());
      *stopped = true;
      break;
    }
    out.push_back(t);
  }
  // Rollback: back to the start, the first mask is the fresh matcher's.
  std::vector<int32_t> fresh(g_words), back(g_words);
  GrammarConstraint(g, g_words).fill(fresh.data());
  con.rollback(int(out.size()) + (*stopped ? 1 : 0));
  con.fill(back.data());
  EXPECT(fresh == back && !con.terminated());
  return g_tok->decode(out);
}

bool valid(const json& v) {
  if (!v.is_object() || v.size() != 4) return false;
  if (!v.contains("name") || !v["name"].is_string()) return false;
  if (!v.contains("n") || !v["n"].is_number_integer()) return false;
  if (!v.contains("ok") || !v["ok"].is_boolean()) return false;
  if (!v.contains("tags") || !v["tags"].is_array()) return false;
  for (const json& t : v["tags"])
    if (!t.is_string()) return false;
  return true;
}

void test_drive_schema() {
  json b = chat_body();
  b["response_format"] = {{"type", "json_schema"}, {"json_schema", {{"name", "x"}, {"schema", json::parse(kSchema)}}}};
  const auto g = request_grammar(*g_comp, parse_request(b, true));
  EXPECT(g.has_value());
  std::mt19937_64 rng(3);
  int good = 0;
  for (int trial = 0; trial < 40; ++trial) {
    bool stopped = false;
    const std::string text = drive(*g, rng, trial % 2 ? 1.f : 0.f, 400, &stopped);
    EXPECT(stopped);
    if (!stopped) continue;
    json v;
    try {
      v = json::parse(text);
    } catch (const std::exception&) {
      std::fprintf(stderr, "does not parse: %s\n", text.c_str());
    }
    EXPECT(valid(v));
    good += valid(v);
  }
  std::printf("  %d of 40 sampled answers parse and validate\n", good);
}

void test_drive_required() {
  json b = chat_body();
  b["tool_choice"] = "required";
  b["tools"] = json::parse(kTools);
  const auto g = request_grammar(*g_comp, parse_request(b, true));
  EXPECT(g.has_value());
  std::mt19937_64 rng(9);
  for (int trial = 0; trial < 20; ++trial) {
    bool stopped = false;
    const std::string text = drive(*g, rng, 1.f, 300, &stopped);
    size_t i = text.find_first_not_of('\n');
    const bool head = i != std::string::npos && text.compare(i, 35, "<tool_call>\n<function=get_weather>\n") == 0;  // s1: names, not `required`
    EXPECT(head);
    if (!head) std::fprintf(stderr, "starts: [%s]\n", text.substr(0, 80).c_str());
    if (stopped) EXPECT(text.size() >= 12 && text.compare(text.size() - 12, 12, "</tool_call>") == 0);
  }
  // Plain text is refused at the start.
  GrammarConstraint con(*g, g_words);
  EXPECT(!con.accept(g_tok->encode("Sure")[0]));
}

}  // namespace

int main(int argc, char** argv) {
  const std::string path = tokenizer_path(argc, argv);
  if (path.empty()) {
    std::printf("SKIP: no tokenizer.json (argv[1], RAGON_TOKENIZER_JSON or the Hugging Face cache)\n");
    return 77;
  }
  Tokenizer tok(path);
  Vocab vocab(tok);
  Compiler comp(vocab);
  g_tok = &tok;
  g_comp = &comp;
  g_words = vocab.words();
  g_vocab_size = vocab.size();
  g_im_end = tok.token_id("<|im_end|>");
  const std::pair<const char*, void (*)()> tests[] = {
      {"parse", test_parse},
      {"refusals", test_refusals},
      {"drive_schema", test_drive_schema},
      {"drive_required", test_drive_required},
  };
  for (const auto& [name, fn] : tests) {
    const int before = failures;
    try {
      fn();
    } catch (const std::exception& e) {
      ++failures;
      std::fprintf(stderr, "%s: exception: %s\n", name, e.what());
    }
    std::printf("%-16s %s\n", name, failures == before ? "ok" : "FAILED");
  }
  if (failures) {
    std::printf("%d failure(s)\n", failures);
    return 1;
  }
  std::printf("all structured tests passed\n");
  return 0;
}
