// ragon-structured-tests [TOKENIZER_JSON]: structured output's server side (serve/structured.hpp, stages 2 and 4)
// against the model's vocabulary: every route's response_format, text.format, tool_choice, parallel_tool_calls and
// custom-tool grammars as parse_request and parse_responses_request read them, the
// grammar request_grammar builds from them (and its 400s), and GrammarConstraint driving the masked host sampler to
// an answer that parses and validates. The tokenizer comes as in ragon-grammar-tests; without one, exit 77.
#include <glob.h>

#include <cstdio>
#include <cstdlib>
#include <optional>
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
  // Completions take response_format (stage 4), never tools.
  json cb = {{"model", "m"}, {"prompt", "Answer in JSON:"}};
  EXPECT(!request_grammar(*g_comp, parse_request(cb, false)));
  cb["response_format"] = {{"type", "text"}};
  EXPECT(!request_grammar(*g_comp, parse_request(cb, false)));
  cb["response_format"] = {{"type", "json_object"}};
  EXPECT(request_grammar(*g_comp, parse_request(cb, false)));
  cb["response_format"] = {{"type", "json_schema"}, {"json_schema", {{"name", "x"}, {"schema", json::parse(kSchema)}}}};
  EXPECT(request_grammar(*g_comp, parse_request(cb, false)));
  cb["response_format"]["json_schema"]["schema"] = {{"type", "text"}};
  EXPECT(refusal(parse_request(cb, false)) ==
         "invalid_json_schema|response_format.json_schema.schema: 'type' must be one of object, array, string, number, "
         "integer, boolean, null at \"\"");
  cb["response_format"] = {{"type", "json_schema"}};
  EXPECT(refusal(parse_request(cb, false)) == "invalid_request_error|response_format.json_schema.schema must be an object");
  cb["response_format"] = {{"type", "json_schema"}, {"json_schema", {{"name", "x"}, {"schema",
      {{"type", "object"}, {"properties", {{"a", {{"type", "string"}, {"pattern", "x"}}}}}}}}}};
  EXPECT(refusal(parse_request(cb, false)) ==
         "unsupported_json_schema|response_format.json_schema.schema: the keyword 'pattern' at /properties/a is not "
         "supported (supported: type, properties, required, additionalProperties, items, enum, const, anyOf, $ref, $defs)");
  cb["response_format"]["json_schema"]["schema"] = {{"type", "string"}, {"pattern", "a+"}};
  EXPECT(refusal(parse_request(cb, false)).rfind(
             "unsupported_json_schema|response_format.json_schema.schema: the keyword 'pattern' at \"\"", 0) == 0);
}

const char* kApplyPatchLark =
    "start: begin_patch hunk+ end_patch\nbegin_patch: \"*** Begin Patch\" LF\nend_patch: \"*** End Patch\" LF?\n\n"
    "hunk: add_hunk | delete_hunk | update_hunk\nadd_hunk: \"*** Add File: \" filename LF add_line+\n"
    "delete_hunk: \"*** Delete File: \" filename LF\n"
    "update_hunk: \"*** Update File: \" filename LF change_move? change?\n\nfilename: /(.+)/\n"
    "add_line: \"+\" /(.*)/ LF -> line\n\nchange_move: \"*** Move to: \" filename LF\n"
    "change: (change_context | change_line)+ eof_line?\nchange_context: (\"@@\" | \"@@ \" /(.+)/) LF\n"
    "change_line: (\"+\" | \"-\" | \" \") /(.*)/ LF\neof_line: \"*** End of File\" LF\n\n%import common.LF\n";
const char* kCallEnd = "\n</parameter>\n</function>\n</tool_call>";

// A Responses body as Codex sends one: a function tool, then the apply_patch custom tool (index 1).
json responses_body() {
  return json{{"model", "m"},
              {"input", "fix it"},
              {"tool_choice", "auto"},
              {"tools", json::array({json{{"type", "function"}, {"name", "shell"}, {"description", "Run a command"},
                                          {"parameters", {{"type", "object"},
                                                          {"properties", {{"command", {{"type", "string"}}}}}}}},
                                     json{{"type", "custom"}, {"name", "apply_patch"}, {"description", "Edit files"},
                                          {"format", {{"type", "grammar"}, {"syntax", "lark"},
                                                      {"definition", kApplyPatchLark}}}}})}};
}

// Whether the grammar accepts `text` and may stop after it.
bool complete(const grammar::Grammar& g, const std::string& text) {
  grammar::Matcher m(g);
  return m.accept_string(text) && m.can_stop();
}

// Stage 4: tool_choice named, parallel_tool_calls, the Responses route's text.format and custom-tool grammars.
void test_routes() {
  const std::string weather = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n"
                              "</tool_call>";
  // Chat, named: calls only, the name fixed; a name that is not offered is the stage 1 400.
  json b = chat_body();
  b["tools"] = json::parse(kTools);
  b["tools"].push_back({{"type", "function"}, {"function", {{"name", "get_time"}, {"parameters", {{"type", "object"}}}}}});
  EXPECT(!request_grammar(*g_comp, parse_request(b, true)));  // auto, function tools only: unconstrained
  b["tool_choice"] = {{"type", "function"}, {"function", {{"name", "get_weather"}}}};
  Request r = parse_request(b, true);
  EXPECT(r.tool_choice_named == "get_weather" && r.tools.size() == 1);
  std::optional<grammar::Grammar> g = request_grammar(*g_comp, r);
  EXPECT(g && complete(*g, weather) && !complete(*g, "Sure.\n" + weather));
  b["tool_choice"] = {{"type", "function"}, {"function", {{"name", "get_stock"}}}};
  EXPECT(refusal(parse_request(b, true)) == "invalid_tools|tool_choice: the function 'get_stock' is not in tools");
  // parallel_tool_calls false: one call. Under required no second call; under auto free text and at most one call.
  b["tool_choice"] = "required";
  g = request_grammar(*g_comp, parse_request(b, true));
  EXPECT(g && complete(*g, weather + "\n" + weather));
  b["parallel_tool_calls"] = false;
  g = request_grammar(*g_comp, parse_request(b, true));
  EXPECT(g && complete(*g, weather) && !complete(*g, weather + "\n" + weather));
  b["tool_choice"] = "auto";
  g = request_grammar(*g_comp, parse_request(b, true));
  EXPECT(g && complete(*g, "Let me look.\n\n" + weather) && complete(*g, "No call needed.") &&
         !complete(*g, weather + "\n" + weather));

  // Responses: a custom tool with a Lark format is enforced under auto; free text and function calls stay allowed.
  json rb = responses_body();
  r = parse_responses_request(rb);
  EXPECT(r.custom_formats.size() == 1 && r.custom_formats[0].index == 1 && r.custom_formats[0].syntax == "lark");
  g = request_grammar(*g_comp, r);
  const std::string patch = "*** Begin Patch\n*** Add File: a.txt\n+hello\n*** End Patch";
  const std::string patch_call = grammar::custom_tool_prefix("apply_patch") + patch + kCallEnd;
  const std::string shell_call = "<tool_call>\n<function=shell>\n<parameter=command>\nls -la\n</parameter>\n</function>\n"
                                 "</tool_call>";
  EXPECT(g && complete(*g, "I will add the file.\n\n" + patch_call) && complete(*g, shell_call) &&
         complete(*g, "Done.") && complete(*g, shell_call + "\n" + patch_call));
  EXPECT(g && !complete(*g, grammar::custom_tool_prefix("apply_patch") + "--- a/a.txt\n+++ b/a.txt\n" + kCallEnd));
  EXPECT(g && !complete(*g, grammar::custom_tool_prefix("apply_patch") + "*** Begin Patch\n" + kCallEnd));
  // Code Mode's `exec` beside apply_patch, as the launcher offers both: its grammar compiles and takes any source.
  json eb = responses_body();
  eb["tools"].push_back({{"type", "custom"}, {"name", "exec"}, {"description", "Run JavaScript"},
                         {"format", {{"type", "grammar"}, {"syntax", "lark"},
                                     {"definition", "start: pragma_source | plain_source\n"
                                                    "pragma_source: PRAGMA_LINE NEWLINE SOURCE\nplain_source: SOURCE\n\n"
                                                    "PRAGMA_LINE: /[ \\t]*\\/\\/ @exec:[^\\r\\n]*/\nNEWLINE: /\\r?\\n/\n"
                                                    "SOURCE: /[\\s\\S]+/\n"}}}});
  r = parse_responses_request(eb);
  EXPECT(r.custom_formats.size() == 2);
  g = request_grammar(*g_comp, r);
  const std::string exec_pre = grammar::custom_tool_prefix("exec");
  EXPECT(g && complete(*g, exec_pre + "const r = await tools.exec_command({cmd: \"ls\"});\ntext(r);" + kCallEnd) &&
         complete(*g, exec_pre + "// @exec: {\"yield_time_ms\": 1000}\ntext(\"hi\");" + kCallEnd) &&
         complete(*g, "Checking.\n\n" + exec_pre + "text(1)" + kCallEnd + "\n" + patch_call) &&
         !complete(*g, exec_pre + kCallEnd));
  // Named custom tool, in the Responses shape.
  rb["tool_choice"] = {{"type", "custom"}, {"name", "apply_patch"}};
  g = request_grammar(*g_comp, parse_responses_request(rb));
  EXPECT(g && complete(*g, patch_call) && !complete(*g, shell_call));
  rb["tool_choice"] = {{"type", "function"}, {"name", "shell"}};
  g = request_grammar(*g_comp, parse_responses_request(rb));
  EXPECT(g && complete(*g, shell_call) && !complete(*g, patch_call));
  rb["tool_choice"] = "required";
  rb["parallel_tool_calls"] = false;
  g = request_grammar(*g_comp, parse_responses_request(rb));
  EXPECT(g && complete(*g, patch_call) && !complete(*g, shell_call + "\n" + patch_call) && !complete(*g, "Done."));
  rb["tool_choice"] = "none";
  EXPECT(!request_grammar(*g_comp, parse_responses_request(rb)));
  // The 400s name the tool by its index in the body.
  rb = responses_body();
  rb["tools"][1]["format"]["definition"] = "start: \"a\"i\n";
  EXPECT(refusal(parse_responses_request(rb)).rfind("unsupported_grammar|tools[1].format: ", 0) == 0);
  rb["tools"][1]["format"]["syntax"] = "ebnf";
  EXPECT(refusal(parse_responses_request(rb)) ==
         "unsupported_grammar|tools[1].format: the syntax 'ebnf' is not supported (supported: lark, regex)");
  rb["tools"][1]["format"] = {{"type", "grammar"}, {"syntax", "regex"}, {"definition", "[a-z]+(?=x)"}};
  EXPECT(refusal(parse_responses_request(rb)).rfind("unsupported_grammar|tools[1].format: ", 0) == 0);
  rb["tools"][1]["format"] = {{"type", "grammar"}, {"syntax", "regex"}, {"definition", "[0-9]{3}"}};
  g = request_grammar(*g_comp, parse_responses_request(rb));
  EXPECT(g && complete(*g, grammar::custom_tool_prefix("apply_patch") + "123" + kCallEnd) &&
         !complete(*g, grammar::custom_tool_prefix("apply_patch") + "12a" + kCallEnd));
  rb["tools"][1]["format"] = {{"type", "grammar"}, {"syntax", "lark"}, {"definition", ""}};
  EXPECT(refusal(parse_responses_request(rb)) == "invalid_grammar|tools[1].format: the definition is missing or empty");
  rb["tools"][1]["format"]["definition"] = "start: missing_rule\n";
  EXPECT(refusal(parse_responses_request(rb)).rfind("invalid_grammar|", 0) == 0);
  // A custom tool in the text format asks for nothing: the request stays unconstrained.
  rb["tools"][1]["format"] = {{"type", "text"}};
  EXPECT(!request_grammar(*g_comp, parse_responses_request(rb)));
  rb["tools"][1].erase("format");
  EXPECT(!request_grammar(*g_comp, parse_responses_request(rb)));

  // text.format: json_schema (flat), json_object, text; its 400s name text.format.
  json tb = {{"model", "m"}, {"input", "hi"}};
  EXPECT(!request_grammar(*g_comp, parse_responses_request(tb)));
  tb["text"] = {{"format", {{"type", "text"}}}};
  EXPECT(!request_grammar(*g_comp, parse_responses_request(tb)));
  tb["text"]["format"] = {{"type", "json_schema"}, {"name", "x"}, {"strict", true}, {"schema", json::parse(kSchema)}};
  g = request_grammar(*g_comp, parse_responses_request(tb));
  EXPECT(g && complete(*g, R"({"name": "a", "n": 1, "tags": [], "ok": true})") && !complete(*g, "{}"));
  tb["text"]["format"] = {{"type", "json_object"}};
  g = request_grammar(*g_comp, parse_responses_request(tb));
  EXPECT(g && complete(*g, "{}") && !complete(*g, "hello"));
  tb["text"]["format"] = {{"type", "json_schema"}, {"name", "x"}, {"schema", {{"type", "string"}, {"pattern", "a+"}}}};
  EXPECT(refusal(parse_responses_request(tb)).rfind(
             "unsupported_json_schema|text.format.schema: the keyword 'pattern' at \"\" is not supported (supported: ", 0) == 0);
  tb["text"]["format"] = {{"type", "json_schema"}, {"name", "x"}};
  EXPECT(refusal(parse_responses_request(tb)) == "invalid_request_error|text.format.schema must be an object");
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
      {"routes", test_routes},
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
