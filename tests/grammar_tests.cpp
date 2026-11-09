// ragon-grammar-tests [TOKENIZER_JSON]: the grammar library (structured output, stage 1) against the model's
// vocabulary. The tokenizer comes from argv[1], RAGON_TOKENIZER_JSON, or the model snapshot in the Hugging Face
// cache; without one the test is skipped (exit 77).
#include <glob.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "grammar/grammar.hpp"
#include "tokenizer/chat_template.hpp"
#include "tokenizer/tokenizer.hpp"

using namespace ragon;
using namespace ragon::grammar;
using ojson = nlohmann::ordered_json;

static int failures = 0;
#define EXPECT(cond)                                                          \
  do {                                                                        \
    if (!(cond)) {                                                            \
      ++failures;                                                             \
      std::fprintf(stderr, "%s:%d: EXPECT(%s) failed\n", __FILE__, __LINE__, #cond); \
    }                                                                         \
  } while (0)
#define EXPECT_EQ(a, b)                                                                                 \
  do {                                                                                                  \
    const std::string ea_ = (a), eb_ = (b);                                                             \
    if (ea_ != eb_) {                                                                                   \
      ++failures;                                                                                       \
      std::fprintf(stderr, "%s:%d: %s\n  got:      %s\n  expected: %s\n", __FILE__, __LINE__, #a, ea_.c_str(), \
                   eb_.c_str());                                                                        \
    }                                                                                                   \
  } while (0)

namespace {

const std::string kSchemaParam = "response_format.json_schema.schema";
const std::string kSupported =
    " is not supported (supported: type, properties, required, additionalProperties, items, enum, const, anyOf, "
    "$ref, $defs)";

const Tokenizer* g_tok = nullptr;
const Vocab* g_vocab = nullptr;
Compiler* g_comp = nullptr;
int g_im_end = -1, g_eot = -1;

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

// The whole string as characters (bytes), then the grammar must be complete.
bool chars_ok(const Grammar& g, std::string_view s) {
  Matcher m(g);
  return m.accept_string(s) && m.can_stop();
}

// The string as the tokenizer encodes it, one token at a time, then the grammar must be complete.
bool ids_ok(const Grammar& g, const std::vector<int>& ids) {
  Matcher m(g);
  for (int id : ids)
    if (!m.accept(id)) return false;
  return m.can_stop();
}
bool tokens_ok(const Grammar& g, std::string_view s) { return ids_ok(g, g_tok->encode(s)); }

// Both levels must agree with `want`.
void check_doc(const Grammar& g, std::string_view s, bool want, int line) {
  const bool c = chars_ok(g, s), t = tokens_ok(g, s);
  if (c != want || t != want) {
    ++failures;
    std::fprintf(stderr, "grammar_tests.cpp:%d: %s (chars %d, tokens %d, want %d): %.*s\n", line,
                 want ? "refused" : "accepted", c, t, want, static_cast<int>(s.size()), s.data());
  }
}
#define ACCEPTS(g, s) check_doc((g), (s), true, __LINE__)
#define REJECTS(g, s) check_doc((g), (s), false, __LINE__)

// "code|message" of the Error `f` throws; "" when it does not throw.
std::string refusal(const std::function<void()>& f) {
  try {
    f();
  } catch (const Error& e) {
    return e.code + "|" + e.what();
  } catch (const std::exception& e) {
    return std::string("other|") + e.what();
  }
  return "";
}

Grammar schema(const char* text) { return g_comp->json_schema(ojson::parse(text), kSchemaParam); }

const char* kApplyPatchLark =
    "start: begin_patch hunk+ end_patch\nbegin_patch: \"*** Begin Patch\" LF\nend_patch: \"*** End Patch\" LF?\n\n"
    "hunk: add_hunk | delete_hunk | update_hunk\nadd_hunk: \"*** Add File: \" filename LF add_line+\n"
    "delete_hunk: \"*** Delete File: \" filename LF\n"
    "update_hunk: \"*** Update File: \" filename LF change_move? change?\n\nfilename: /(.+)/\n"
    "add_line: \"+\" /(.*)/ LF -> line\n\nchange_move: \"*** Move to: \" filename LF\n"
    "change: (change_context | change_line)+ eof_line?\nchange_context: (\"@@\" | \"@@ \" /(.+)/) LF\n"
    "change_line: (\"+\" | \"-\" | \" \") /(.*)/ LF\neof_line: \"*** End of File\" LF\n\n%import common.LF\n";

// codex-rs execute_spec.rs, the Code Mode exec tool's grammar.
const char* kExecLark =
    "start: pragma_source | plain_source\npragma_source: PRAGMA_LINE NEWLINE SOURCE\nplain_source: SOURCE\n\n"
    "PRAGMA_LINE: /[ \\t]*\\/\\/ @exec:[^\\r\\n]*/\nNEWLINE: /\\r?\\n/\nSOURCE: /[\\s\\S]+/\n";

const char* kPatchAdd = "*** Begin Patch\n*** Add File: docs/hello.txt\n+Hello world\n+\n+  indented\n*** End Patch\n";
const char* kPatchMove =
    "*** Begin Patch\n*** Update File: src/a.rs\n*** Move to: src/b.rs\n@@ fn main() {\n-    old();\n+    new();\n"
    "     keep();\n*** End of File\n*** End Patch";
const char* kPatchMixed =
    "*** Begin Patch\n*** Delete File: old.txt\n*** Update File: x.py\n@@\n-a = 1\n+a = 2\n@@ def f():\n"
    "     return a\n*** Add File: y.py\n+print(1)\n*** End Patch\n";

// ---------------------------------------------------------------------------------------------------------

void test_vocab() {
  EXPECT(g_vocab->size() == 248320);
  EXPECT(g_vocab->words() == 7760);
  EXPECT(g_im_end >= 0 && g_eot >= 0);
  EXPECT(g_vocab->stop_ids().size() == 2);
  EXPECT(!g_vocab->never_allowed(g_im_end));
  EXPECT(g_vocab->never_allowed(g_tok->token_id("<|im_start|>")));
  EXPECT(g_vocab->never_allowed(g_tok->token_id("<|vision_start|>")));
  EXPECT(!g_vocab->never_allowed(g_tok->token_id("<tool_call>")));
  EXPECT(!g_vocab->never_allowed(g_tok->token_id("</tool_call>")));
  EXPECT(!g_vocab->never_allowed(g_tok->token_id("<think>")));
  EXPECT(!g_vocab->never_allowed(g_tok->token_id("</think>")));
  for (int id = g_tok->vocab_size(); id < g_vocab->size(); ++id) EXPECT(g_vocab->never_allowed(id));
}

// §2.1: every whitelisted keyword compiles and constrains.
void test_schema_whitelist() {
  struct Row {
    const char* schema;
    const char* good;
    const char* bad;
  };
  const Row rows[] = {
      {R"({"type": "string"})", R"("hi")", "1"},
      {R"({"type": "number"})", "-1.5e3", R"("x")"},
      {R"({"type": "integer"})", "42", "1.5"},
      {R"({"type": "boolean"})", "true", "1"},
      {R"({"type": "null"})", "null", "0"},
      {R"({"type": ["string", "null"]})", "null", "1"},
      {R"({"type": "array", "items": {"type": "integer"}})", "[1, 2]", R"(["a"])"},
      {R"({"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]})", R"({"a": 1})", "{}"},
      // additionalProperties: false, absent (means false), a schema, true
      {R"({"type": "object", "properties": {"a": {"type": "integer"}}, "additionalProperties": false})",
       R"({"a": 1})", R"({"a": 1, "b": 2})"},
      {R"({"type": "object", "properties": {"a": {"type": "integer"}}})", R"({"a": 1})", R"({"a": 1, "b": 2})"},
      {R"({"type": "object", "additionalProperties": {"type": "integer"}})", R"({"x": 1, "y": 2})", R"({"x": "y"})"},
      {R"({"type": "object", "properties": {}, "additionalProperties": true})", R"({"x": [1, {"y": null}]})", "[]"},
      {R"({"enum": ["a", "b", 1]})", "1", R"("c")"},
      {R"({"const": "fixed"})", R"("fixed")", R"("other")"},
      {R"({"anyOf": [{"type": "integer"}, {"type": "string"}]})", R"("s")", "true"},
      {R"({"type": "object", "properties": {"n": {"$ref": "#/$defs/n"}}, "required": ["n"],
           "$defs": {"n": {"type": "integer"}}})",
       R"({"n": 3})", R"({"n": "x"})"},
      {R"({"type": "object", "properties": {"n": {"$ref": "#/definitions/n"}}, "required": ["n"],
           "definitions": {"n": {"type": "boolean"}}})",
       R"({"n": false})", R"({"n": 0})"},
      {R"({"type": "object", "properties": {"c": {"$ref": "#"}}})", R"({"c": {"c": {}}})", R"({"c": 1})"},
      {R"({"type": "string", "title": "t", "description": "d", "default": "x", "examples": ["e"],
           "$schema": "https://json-schema.org/draft/2020-12/schema", "$id": "urn:x", "$comment": "c",
           "format": "date-time"})",
       R"("not a date")", "1"},
  };
  for (const Row& r : rows) {
    std::string err = refusal([&] { (void)schema(r.schema); });
    if (!err.empty()) {
      ++failures;
      std::fprintf(stderr, "whitelist row refused: %s\n  %s\n", r.schema, err.c_str());
      continue;
    }
    Grammar g = schema(r.schema);
    ACCEPTS(g, r.good);
    REJECTS(g, r.bad);
  }
}

// §2.1: every keyword outside the whitelist is refused with its exact text and pointer.
void test_schema_refusals() {
  const char* refused[] = {"minLength",        "maxLength",        "pattern",           "minimum",
                           "maximum",          "exclusiveMinimum", "exclusiveMaximum",  "multipleOf",
                           "uniqueItems",      "prefixItems",
                           "contains",         "minContains",      "maxContains",       "oneOf",
                           "allOf",            "not",              "if",                "then",
                           "else",             "patternProperties", "propertyNames",    "minProperties",
                           "maxProperties",    "dependentRequired", "dependentSchemas", "unevaluatedProperties",
                           "unevaluatedItems", "contentEncoding",  "contentMediaType",  "$anchor",
                           "$dynamicRef",      "nullable"};
  for (const char* kw : refused) {
    ojson s = ojson::parse(R"({"type": "object", "properties": {"a/b~c": {"type": "string"}}})");
    s["properties"]["a/b~c"][kw] = 1;
    EXPECT_EQ(refusal([&] { (void)g_comp->json_schema(s, kSchemaParam); }),
              "unsupported_json_schema|" + kSchemaParam + ": the keyword '" + kw + "' at /properties/a~1b~0c" +
                  kSupported);
  }
  // At the root, the pointer is "", and /v1/responses names its own field.
  EXPECT_EQ(refusal([&] { (void)g_comp->json_schema(ojson::parse(R"({"oneOf": [{"type": "string"}]})"),
                                                    "text.format.schema"); }),
            "unsupported_json_schema|text.format.schema: the keyword 'oneOf' at \"\"" + kSupported);
  // Inside items, anyOf, $defs.
  EXPECT_EQ(refusal([&] { (void)schema(R"({"type": "array", "items": {"anyOf": [{"type": "string"},
                                           {"type": "integer", "minimum": 0}]}})"); }),
            "unsupported_json_schema|" + kSchemaParam + ": the keyword 'minimum' at /items/anyOf/1" + kSupported);
  EXPECT_EQ(refusal([&] { (void)schema(R"({"$defs": {"x": {"type": "string", "pattern": "a"}}, "$ref": "#/$defs/x"})"); }),
            "unsupported_json_schema|" + kSchemaParam + ": the keyword 'pattern' at /$defs/x" + kSupported);
  // The first refused keyword in document order.
  EXPECT_EQ(refusal([&] { (void)schema(R"({"maxLength": 3, "minLength": 1})"); }),
            "unsupported_json_schema|" + kSchemaParam + ": the keyword 'maxLength' at \"\"" + kSupported);
  // A remote reference is refused as $ref.
  EXPECT_EQ(refusal([&] { (void)schema(R"({"$ref": "https://example.com/s.json"})"); }),
            "unsupported_json_schema|" + kSchemaParam + ": the keyword '$ref' at \"\"" + kSupported);
  // Invalid schemas.
  EXPECT_EQ(refusal([&] { (void)schema(R"({"type": "object", "properties": {"a": {"$ref": "#/$defs/nope"}}})"); }),
            "invalid_json_schema|" + kSchemaParam + ": the reference '#/$defs/nope' does not resolve at /properties/a");
  EXPECT_EQ(refusal([&] { (void)schema(R"({"type": "date"})"); }),
            "invalid_json_schema|" + kSchemaParam +
                ": 'type' must be one of object, array, string, number, integer, boolean, null at \"\"");
  EXPECT_EQ(refusal([&] { (void)schema(R"({"enum": []})"); }),
            "invalid_json_schema|" + kSchemaParam + ": 'enum' is not a non-empty list at \"\"");
  EXPECT_EQ(refusal([&] { (void)schema("[1]"); }),
            "invalid_json_schema|" + kSchemaParam + ": the schema is not an object at \"\"");
  EXPECT_EQ(refusal([&] { (void)g_comp->json_schema_text("{\"type\": ", kSchemaParam); }),
            "invalid_json_schema|" + kSchemaParam + ": the schema is not valid JSON at \"\"");
  // Nesting deeper than 64 levels.
  std::string deep;
  for (int i = 0; i < 70; ++i) deep += R"({"type": "array", "items": )";
  deep += R"({"type": "string"})";
  for (int i = 0; i < 70; ++i) deep += "}";
  EXPECT(refusal([&] { (void)schema(deep.c_str()); }).rfind("invalid_json_schema|", 0) == 0);
}

const char* kDocSchema = R"({"type": "object", "properties": {
    "name": {"type": "string"},
    "tags": {"type": "array", "items": {"type": "string"}},
    "n": {"type": "integer"},
    "ratio": {"anyOf": [{"type": "number"}, {"type": "null"}]}},
  "required": ["name", "n"]})";

// JSON accept and reject sets, at character and at token level.
void test_json_sets() {
  Grammar g = schema(kDocSchema);
  ACCEPTS(g, R"({"name": "a", "n": 1})");
  ACCEPTS(g, R"({"name":"a","tags":["x","y"],"n":-3,"ratio":null})");
  ACCEPTS(g, "{\n  \"name\": \"h\\u00e9llo \\\"q\\\" \\n\",\n  \"tags\": [],\n  \"n\": 0,\n  \"ratio\": 2.5e-3\n}");
  ACCEPTS(g, "{\"name\": \"caf\xc3\xa9 \xe2\x9c\x93\", \"n\": 7}");
  REJECTS(g, R"({"name": "a"})");                    // n is required
  REJECTS(g, R"({"name": "a", "n": 1, "x": 1})");    // no additional properties
  REJECTS(g, R"({"name": "a", "n": "1"})");          // wrong type
  REJECTS(g, R"({"name": "a", "n": 1,})");           // trailing comma
  REJECTS(g, R"({'name': 'a', "n": 1})");            // single quotes
  REJECTS(g, R"({"name": "a", "n": 01})");           // leading zero
  REJECTS(g, R"({"name": "a", "n": NaN})");
  REJECTS(g, R"({"name": "a", "n": 1)");             // unterminated
  REJECTS(g, R"({"name": "a\x", "n": 1})");          // bad escape
  REJECTS(g, "{\"name\": \"a\nb\", \"n\": 1}");      // raw newline in a string
  REJECTS(g, R"({"name": "a", "n": 1} trailing)");

  // json_object mode: any JSON object, nothing else.
  Grammar o = g_comp->json_object();
  ACCEPTS(o, R"({})");
  ACCEPTS(o, R"({"a": [1, {"b": null}], "c": "d", "e": -0.5})");
  REJECTS(o, "[1]");
  REJECTS(o, R"("x")");
  REJECTS(o, R"({"a": })");
  REJECTS(o, "");
}

// Every whitespace slot: at most one newline, then at most 16 spaces or tabs; no carriage return.
void test_whitespace_bound() {
  Grammar o = g_comp->json_object();
  Grammar s = schema(R"({"type": "object", "properties": {"a": {"type": "array", "items": {"type": "integer"}}}})");
  for (const Grammar* g : {&o, &s}) {
    const std::string b16(16, ' '), b17(17, ' '), t16(16, '\t'), t17(17, '\t');
    ACCEPTS(*g, "{\"a\":" + b16 + "[1]}");
    ACCEPTS(*g, "{\"a\":\n" + b16 + "[1]}");
    ACCEPTS(*g, "{\"a\":\n" + t16 + "[1]}");
    ACCEPTS(*g, "{\n  \"a\": [\n    1,\n    2\n  ]\n}");
    REJECTS(*g, "{\"a\":" + b17 + "[1]}");
    REJECTS(*g, "{\"a\":\n" + b17 + "[1]}");
    REJECTS(*g, "{\"a\":\n" + t17 + "[1]}");
    REJECTS(*g, "{\"a\":\n\n[1]}");
    REJECTS(*g, "{\"a\":\r\n[1]}");
    REJECTS(*g, "{\"a\": \n[1]}");  // a blank before the newline
    REJECTS(*g, "{\"a\": [1,\n\n2]}");
  }
  // A model that loops on newlines is stopped at the second one.
  Matcher m(o);
  EXPECT(m.accept_string("{\n"));
  EXPECT(!m.accept_string("\n"));
}

// The Codex grammars: apply_patch (custom tool, Lark) and exec (Code Mode, Lark).
void test_codex_grammars() {
  Grammar patch = g_comp->lark(kApplyPatchLark, "tools[0].format");
  ACCEPTS(patch, kPatchAdd);
  ACCEPTS(patch, kPatchMove);  // Move to, End of File, End Patch without its LF
  ACCEPTS(patch, kPatchMixed);
  REJECTS(patch, "*** Begin Patch\n*** Add File: a\nno plus\n*** End Patch\n");
  REJECTS(patch, "*** Begin Patch\n*** End Patch\n");  // no hunk
  REJECTS(patch, "*** Begin Patch\n*** Frobnicate File: a\n*** End Patch\n");
  REJECTS(patch, "Begin Patch\n*** Add File: a\n+x\n*** End Patch\n");
  REJECTS(patch, "*** Begin Patch\n*** Add File: \n+x\n*** End Patch\n");  // empty file name
  REJECTS(patch, "*** Begin Patch\n*** Add File: a\n+x\n");                // no end
  REJECTS(patch, "*** Begin Patch\n*** Update File: a\n*** End of File\n*** End Patch\n");  // eof_line alone

  Grammar exec = g_comp->lark(kExecLark, "tools[0].format");
  ACCEPTS(exec, "// @exec: {\"yield_time_ms\": 500}\nconst r = await tools.shell({command: [\"ls\"]});\nconsole.log(r);\n");
  ACCEPTS(exec, "  // @exec: x\r\nprint(1)");
  ACCEPTS(exec, "const x = 1;\nconsole.log(x);");
  ACCEPTS(exec, "\n");
  REJECTS(exec, "");

  // The tool-call grammar with apply_patch as a custom tool: the body form the output parser reads.
  ToolCallSpec spec;
  spec.choice = ToolCallSpec::Choice::kRequired;
  spec.tools.push_back({Tool::Kind::kCustom, "apply_patch", {}, Tool::Format::kLark, kApplyPatchLark});
  spec.tools.push_back({Tool::Kind::kCustom, "exec", {}, Tool::Format::kLark, kExecLark});
  spec.tools.push_back({Tool::Kind::kCustom, "notes", {}, Tool::Format::kText, ""});
  Grammar calls = g_comp->tool_calls(spec);
  const std::string end = "\n</parameter>\n</function>\n</tool_call>";
  EXPECT_EQ(custom_tool_prefix("apply_patch"), "<tool_call>\n<function=apply_patch>\n<parameter=input>\n");
  ACCEPTS(calls, custom_tool_prefix("apply_patch") + kPatchAdd + end);
  ACCEPTS(calls, custom_tool_prefix("apply_patch") + kPatchMove + end);
  ACCEPTS(calls, custom_tool_prefix("exec") + "// @exec: {}\nconsole.log(1)" + end);
  ACCEPTS(calls, custom_tool_prefix("notes") + "anything at all\n<b>" + end);
  REJECTS(calls, custom_tool_prefix("apply_patch") + "*** Begin Patch\n+x\n*** End Patch\n" + end);
  // Under auto, the same call after free text, triggered on <tool_call>.
  spec.choice = ToolCallSpec::Choice::kAuto;
  Grammar auto_calls = g_comp->tool_calls(spec);
  ACCEPTS(auto_calls, "I will patch it.\n\n" + custom_tool_prefix("apply_patch") + kPatchAdd + end);
  REJECTS(auto_calls, "I will patch it.\n\n" + custom_tool_prefix("apply_patch") + "*** Begin Patch\n" + end);
}

// §2.3: the Lark subset and its refusals, with line and column.
void test_lark() {
  const std::string p = "tools[0].format";
  auto lark_err = [&](const char* text) { return refusal([&] { (void)g_comp->lark(text, p); }); };
  auto unsupported = [&](const std::string& what) { return "unsupported_grammar|" + p + ": " + what + " is not supported"; };
  EXPECT_EQ(lark_err("start: \"a\"i\n"), unsupported("a case-insensitive literal at line 1 column 8"));
  EXPECT_EQ(lark_err("start: /(?=a)b/\n"), unsupported("a regex lookaround at line 1 column 9"));
  EXPECT_EQ(lark_err("start: /(?<!a)b/\n"), unsupported("a regex lookaround at line 1 column 9"));
  EXPECT_EQ(lark_err("start: /(a)\\1/\n"), unsupported("a regex backreference at line 1 column 12"));
  EXPECT_EQ(lark_err("start: /a+?/\n"), unsupported("a lazy regex quantifier at line 1 column 11"));
  EXPECT_EQ(lark_err("start: /a/i\n"), unsupported("regex flags at line 1 column 8"));
  EXPECT_EQ(lark_err("start: /(?i)a/\n"), unsupported("regex flags at line 1 column 9"));
  EXPECT_EQ(lark_err("start: /^a$/\n"), unsupported("a regex anchor '^' at line 1 column 9"));
  EXPECT_EQ(lark_err("start: A\nA: \"a\"\n%ignore A\n"), unsupported("%ignore at line 3 column 1"));
  EXPECT_EQ(lark_err("%declare X\nstart: \"a\"\n"), unsupported("%declare at line 1 column 1"));
  EXPECT_EQ(lark_err("start: \"a\"\n%override start: \"b\"\n"), unsupported("%override at line 2 column 1"));
  EXPECT_EQ(lark_err("start.2: \"a\"\n"), unsupported("a priority at line 1 column 6"));
  EXPECT_EQ(lark_err("start: \"a\"\nA.2: \"b\"\n"), unsupported("a priority at line 2 column 2"));
  EXPECT_EQ(lark_err("start: pair{\"a\"}\n"), unsupported("a template at line 1 column 12"));
  EXPECT_EQ(lark_err("start: \"a\"~3\n"), unsupported("'~' repetition at line 1 column 11"));
  EXPECT_EQ(lark_err("start: \"a\"..\"z\"\n"), unsupported("a literal range at line 1 column 11"));
  EXPECT_EQ(lark_err("start: X\n%import common.CNAME\n"), unsupported("%import of 'common.CNAME' at line 2 column 16"));
  EXPECT_EQ(lark_err("start: X\n%import python.X\n"), unsupported("%import of 'python' at line 2 column 9"));
  EXPECT_EQ(lark_err("start: foo\n"), "invalid_grammar|" + p + ": the name 'foo' is not defined at line 1 column 8");
  EXPECT_EQ(lark_err("a: \"x\"\n"), "invalid_grammar|" + p + ": the grammar has no 'start' rule at line 1 column 1");
  EXPECT_EQ(lark_err("start: \"a\"\nstart: \"b\"\n"),
            "invalid_grammar|" + p + ": 'start' is defined twice at line 2 column 1");
  EXPECT_EQ(lark_err("start: (\"a\"\n"), "invalid_grammar|" + p + ": an unclosed '(' at line 1 column 8");
  EXPECT_EQ(lark_err("start: \"a\n"), "invalid_grammar|" + p + ": an unterminated string at line 1 column 8");
  EXPECT_EQ(lark_err("start: /[a-/\n"), "invalid_grammar|" + p + ": an unclosed '[' in the regex at line 1 column 9");

  // What the subset accepts.
  Grammar g = g_comp->lark("// a comment\n?start: greeting \" \" NAME [\"!\"]\n!greeting: \"hi\"\n    | \"hello\"\n"
                           "NAME: /[A-Z][a-z]{1,8}/\n", p);
  ACCEPTS(g, "hi Bob");
  ACCEPTS(g, "hello Alexandra!");
  REJECTS(g, "hey Bob");
  REJECTS(g, "hi bob");
  REJECTS(g, "hi Bobbbbbbbbb");
  Grammar c = g_comp->lark("start: ESCAPED_STRING WS_INLINE SIGNED_NUMBER NEWLINE INT WORD\n"
                           "%import common (ESCAPED_STRING, WS_INLINE, SIGNED_NUMBER, NEWLINE, INT, WORD)\n", p);
  ACCEPTS(c, "\"a \\\" b\" -1.5e3\r\n42abc");
  REJECTS(c, "\"a\" x\n1a");
  Grammar u = g_comp->lark("start: /[\\u00e9\\w-]+(?:\\.\\d+)?/ \"\\u2713\"\n", p);
  ACCEPTS(u, "\xc3\xa9t\xc3\xa9-1.25\xe2\x9c\x93");
  REJECTS(u, "\xc3\xa9t\xc3\xa9.\xe2\x9c\x93");

  // Regex formats compile directly.
  Grammar r = g_comp->regex("[a-z]{2,3}\\d(-[A-F0-9]+)*", p);
  ACCEPTS(r, "ab1");
  ACCEPTS(r, "abc2-FF-01");
  REJECTS(r, "a1");
  REJECTS(r, "abcd1");
  REJECTS(r, "ab1-ff");
  EXPECT_EQ(refusal([&] { (void)g_comp->regex("a(?=b)", p); }), unsupported("a regex lookaround at line 1 column 2"));
}

const ojson kTools = ojson::parse(R"([
  {"type": "function", "function": {"name": "exec_command", "parameters": {"type": "object",
    "properties": {"cmd": {"type": "string"}, "workdir": {"type": "string"}, "timeout_ms": {"type": "integer"}}}}},
  {"type": "function", "function": {"name": "update_plan", "parameters": {"type": "object",
    "properties": {"explanation": {"type": "string"}, "plan": {"type": "array"}}}}},
  {"type": "function", "function": {"name": "list_mcp_resources", "parameters": {"type": "object", "properties": {}}}}
])");

ToolCallSpec function_spec(ToolCallSpec::Choice choice) {
  ToolCallSpec spec;
  spec.choice = choice;
  for (const auto& t : kTools) {
    Tool tool;
    tool.name = t["function"]["name"];
    for (auto it = t["function"]["parameters"]["properties"].begin();
         it != t["function"]["parameters"]["properties"].end(); ++it)
      tool.params.push_back(it.key());
    spec.tools.push_back(tool);
  }
  return spec;
}

// What chat_template renders after </think> for an assistant turn with these calls (up to <|im_end|>).
std::string rendered_calls(const ojson& calls, const std::string& content = "") {
  ojson messages = ojson::array();
  messages.push_back({{"role", "user"}, {"content", "go"}});
  messages.push_back({{"role", "assistant"}, {"content", content}, {"tool_calls", calls}});
  ChatOptions opts;
  opts.add_generation_prompt = false;
  const std::string text = render_chat(messages, kTools, opts);
  const std::size_t a = text.rfind("</think>");
  const std::size_t b = text.rfind("<|im_end|>");
  if (a == std::string::npos || b == std::string::npos || b < a) return "";
  return text.substr(a + 8, b - a - 8);
}

// §2.2: the tool-call grammar against chat_template's rendering.
void test_tool_calls() {
  const ojson one = ojson::parse(R"([{"type": "function", "function": {"name": "exec_command",
      "arguments": {"cmd": "ls -la src | head -n 5", "workdir": "/tmp/x"}}}])");
  const ojson two = ojson::parse(R"([
      {"type": "function", "function": {"name": "update_plan",
        "arguments": "{\"explanation\": \"two steps\", \"plan\": [{\"step\": \"a\", \"status\": \"pending\"}]}"}},
      {"type": "function", "function": {"name": "exec_command", "arguments": {"cmd": "make", "timeout_ms": 60000}}},
      {"type": "function", "function": {"name": "list_mcp_resources", "arguments": {}}}])");
  const std::string r1 = rendered_calls(one), r2 = rendered_calls(two), r3 = rendered_calls(one, "Running it.");
  EXPECT(r1.rfind("\n\n<tool_call>\n<function=exec_command>\n<parameter=cmd>\nls -la", 0) == 0);
  EXPECT(r2.find("</tool_call>\n<tool_call>") != std::string::npos);
  EXPECT(r2.find("<parameter=timeout_ms>\n60000\n</parameter>") != std::string::npos);

  Grammar req = g_comp->tool_calls(function_spec(ToolCallSpec::Choice::kRequired));
  Grammar aut = g_comp->tool_calls(function_spec(ToolCallSpec::Choice::kAuto));
  ToolCallSpec named_spec = function_spec(ToolCallSpec::Choice::kNamed);
  named_spec.named = "exec_command";
  Grammar named = g_comp->tool_calls(named_spec);
  for (const Grammar* g : {&req, &aut}) {
    ACCEPTS(*g, r1);
    ACCEPTS(*g, r2);
  }
  ACCEPTS(named, r1);
  REJECTS(named, r2);  // update_plan is not the named function
  ACCEPTS(aut, r3);    // free text before the calls under auto
  REJECTS(req, r3);    // ... but not under required
  ACCEPTS(aut, "Just an answer, no call.");
  ACCEPTS(aut, "");
  REJECTS(req, "");

  const std::string call = "<tool_call>\n<function=exec_command>\n<parameter=cmd>\nls\n</parameter>\n</function>\n</tool_call>";
  ACCEPTS(req, call);
  ACCEPTS(req, "\n" + call);
  ACCEPTS(req, "\n\n" + call);
  REJECTS(req, "\n\n\n" + call);  // at most two newlines after </think>
  REJECTS(req, "Sure.\n" + call);  // leading text under required
  REJECTS(req, "<tool_call>\n<function=rm_rf>\n</function>\n</tool_call>");  // unknown function
  REJECTS(aut, "Ok.\n<tool_call>\n<function=rm_rf>\n</function>\n</tool_call>");
  REJECTS(req, "<tool_call>\n<function=exec_command>\n<parameter=sudo>\nyes\n</parameter>\n</function>\n</tool_call>");
  REJECTS(aut, "<tool_call>\n<function=exec_command>\n<parameter=sudo>\nyes\n</parameter>\n</function>\n</tool_call>");
  REJECTS(req, call + "\n\n" + call);  // calls are separated by one newline
  REJECTS(req, call + " trailing");

  // <tool_call> as the added token and as byte pieces.
  const int tc = g_tok->token_id("<tool_call>");
  EXPECT(tc >= 0);
  const std::vector<int> whole = g_tok->encode(call);
  EXPECT(!whole.empty() && whole[0] == tc);
  std::vector<int> pieces = g_tok->encode("<tool");
  for (int id : g_tok->encode("_call>")) pieces.push_back(id);
  for (int id : pieces) EXPECT(id != tc);
  for (int id : g_tok->encode(call.substr(std::string("<tool_call>").size()))) pieces.push_back(id);
  EXPECT(ids_ok(req, whole));
  EXPECT(ids_ok(req, pieces));
  EXPECT(ids_ok(aut, pieces));

  // json | calls: a response_format with tools under auto.
  ToolCallSpec js = function_spec(ToolCallSpec::Choice::kAuto);
  js.json_schema = ojson::parse(R"({"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]})");
  Grammar jc = g_comp->tool_calls(js);
  ACCEPTS(jc, R"({"answer": "42"})");
  ACCEPTS(jc, r1);
  ACCEPTS(jc, r2);
  REJECTS(jc, "The answer is 42.");
  REJECTS(jc, R"({"other": 1})");
  ToolCallSpec jo = function_spec(ToolCallSpec::Choice::kAuto);
  jo.json_object = true;
  Grammar jog = g_comp->tool_calls(jo);
  ACCEPTS(jog, R"({"x": [1]})");
  ACCEPTS(jog, call);
  REJECTS(jog, "plain text");
  // A schema refused inside the tool-call grammar keeps its own text.
  js.json_schema = ojson::parse(R"({"type": "string", "pattern": "a"})");
  EXPECT_EQ(refusal([&] { (void)g_comp->tool_calls(js); }),
            "unsupported_json_schema|" + kSchemaParam + ": the keyword 'pattern' at \"\"" + kSupported);
  // Bad tool lists.
  named_spec.named = "nope";
  EXPECT_EQ(refusal([&] { (void)g_comp->tool_calls(named_spec); }),
            "invalid_tools|tool_choice: the function 'nope' is not in tools");
  ToolCallSpec empty;
  EXPECT_EQ(refusal([&] { (void)g_comp->tool_calls(empty); }), "invalid_tools|tools: the list is empty");
}

// response_format with no tools (stage 2): at most two newlines, then the JSON answer; refusals keep the s1 texts.
void test_response_format() {
  const ojson sch = ojson::parse(R"({"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]})");
  Grammar rf = g_comp->response_format(sch, kSchemaParam);
  ACCEPTS(rf, R"({"answer": "42"})");
  ACCEPTS(rf, "\n" + std::string(R"({"answer": "42"})"));
  ACCEPTS(rf, "\n\n" + std::string(R"({"answer": "42"})"));
  REJECTS(rf, "\n\n\n" + std::string(R"({"answer": "42"})"));
  REJECTS(rf, R"({"other": 1})");
  REJECTS(rf, "The answer is 42.");
  REJECTS(rf, "<tool_call>\n<function=exec_command>\n</function>\n</tool_call>");
  Grammar ro = g_comp->response_format(std::nullopt, "response_format");
  ACCEPTS(ro, "\n\n{\"x\": [1, {\"y\": null}]}");
  REJECTS(ro, "[1]");
  REJECTS(ro, "plain");
  // Its own cache entry: the s1 json_schema grammar (no lead) is unchanged.
  Grammar js = g_comp->json_schema(sch, kSchemaParam);
  EXPECT(&js.impl() != &rf.impl());
  REJECTS(js, "\n\n" + std::string(R"({"answer": "42"})"));
  EXPECT(&g_comp->response_format(sch, kSchemaParam).impl() == &rf.impl());
  EXPECT_EQ(refusal([&] { (void)g_comp->response_format(ojson::parse(R"({"type": "string", "pattern": "a"})"), kSchemaParam); }),
            "unsupported_json_schema|" + kSchemaParam + ": the keyword 'pattern' at \"\"" + kSupported);
  // The stop token only once the answer is complete.
  Matcher m(rf);
  for (int id : g_tok->encode(R"({"answer": "4)")) EXPECT(m.accept(id));
  EXPECT(!m.can_stop() && !m.allowed(g_im_end));
  for (int id : g_tok->encode(R"(2"})")) EXPECT(m.accept(id));
  EXPECT(m.can_stop() && m.allowed(g_im_end) && m.accept(g_im_end) && m.terminated());
}

// Stage 3: minItems and maxItems bound an array (XGrammar's repetition); a bounded array stops at its bound: after
// the last item the grammar refuses a comma, takes the bracket, and then the stop token. Bad bounds are invalid.
void test_array_bounds() {
  Grammar g = schema(R"({"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 3})");
  ACCEPTS(g, "[1]");
  ACCEPTS(g, "[1, 2, 3]");
  REJECTS(g, "[]");
  REJECTS(g, "[1, 2, 3, 4]");
  Grammar none = schema(R"({"type": "array", "items": {"type": "string"}, "maxItems": 0})");
  ACCEPTS(none, "[]");
  REJECTS(none, R"(["a"])");
  Grammar obj = schema(R"({"type": "object", "properties": {"xs": {"type": "array", "items": {"type": "boolean"},
                           "maxItems": 2}}, "required": ["xs"]})");
  ACCEPTS(obj, R"({"xs": [true, false]})");
  REJECTS(obj, R"({"xs": [true, false, true]})");
  // The bounded array stops: token by token, the comma is refused after the third item.
  Grammar rf = g_comp->response_format(ojson::parse(R"({"type": "array", "items": {"type": "integer"}, "maxItems": 3})"),
                                       kSchemaParam);
  Matcher m(rf);
  for (int id : g_tok->encode("[1, 2, 3")) EXPECT(m.accept(id));
  for (const char* more : {",", ", "}) {
    const std::vector<int> ids = g_tok->encode(more);
    EXPECT(ids.size() == 1 && !m.allowed(ids[0]));
  }
  EXPECT(!m.can_stop() && !m.allowed(g_im_end));
  for (int id : g_tok->encode("]")) EXPECT(m.accept(id));
  EXPECT(m.can_stop() && m.allowed(g_im_end) && m.accept(g_im_end) && m.terminated());
  // Bad bounds.
  EXPECT_EQ(refusal([&] { (void)schema(R"({"type": "array", "minItems": -1})"); }),
            "invalid_json_schema|" + kSchemaParam + ": 'minItems' is not an integer from 0 to 2147483647 at \"\"");
  EXPECT_EQ(refusal([&] { (void)schema(R"({"type": "array", "maxItems": 1.5})"); }),
            "invalid_json_schema|" + kSchemaParam + ": 'maxItems' is not an integer from 0 to 2147483647 at \"\"");
  EXPECT_EQ(refusal([&] { (void)schema(R"({"type": "array", "maxItems": 4294967296})"); }),
            "invalid_json_schema|" + kSchemaParam + ": 'maxItems' is not an integer from 0 to 2147483647 at \"\"");
  EXPECT_EQ(refusal([&] { (void)schema(R"({"type": "object", "properties": {"a": {"minItems": 3, "maxItems": 2}}})"); }),
            "invalid_json_schema|" + kSchemaParam + ": 'minItems' is greater than 'maxItems' at /properties/a");
}

// The stop token is allowed exactly where the grammar is complete; padding and control tokens never are.
void test_stop_and_padding() {
  Grammar g = schema(kDocSchema);
  const std::string doc = R"({"name": "a b c", "tags": ["x"], "n": 12})";
  const std::vector<int> ids = g_tok->encode(doc);
  Matcher m(g);
  for (std::size_t i = 0; i < ids.size(); ++i) {
    EXPECT(!m.allowed(g_im_end));
    EXPECT(!m.allowed(g_eot));
    EXPECT(!m.can_stop());
    EXPECT(m.accept(ids[i]));
  }
  EXPECT(m.allowed(g_im_end));
  EXPECT(m.allowed(g_eot));
  EXPECT(!m.terminated());
  EXPECT(m.accept(g_im_end));
  EXPECT(m.terminated());

  // apply_patch: complete after "*** End Patch", with or without its LF, not before.
  Grammar patch = g_comp->lark(kApplyPatchLark, "tools[0].format");
  Matcher p(patch);
  const std::string text = kPatchAdd;
  for (std::size_t i = 0; i < text.size(); ++i) {
    const bool complete = i + 1 == text.size() || i == text.size() - 1;  // after "*** End Patch" and after its LF
    (void)complete;
    EXPECT(p.accept_string(text.substr(i, 1)));
    const bool want = i + 2 >= text.size();
    if (p.can_stop() != want) {
      ++failures;
      std::fprintf(stderr, "apply_patch: can_stop %d at byte %zu, want %d\n", p.can_stop(), i, want);
    }
  }

  // Padding and control tokens in every state of several grammars, including free text where nearly every
  // token is allowed.
  std::vector<int32_t> mask(g_vocab->words());
  auto bit = [&](int id) { return (mask[id >> 5] >> (id & 31)) & 1; };
  Grammar aut = g_comp->tool_calls(function_spec(ToolCallSpec::Choice::kAuto));
  int states = 0, allowed_free = 0;
  for (const Grammar* gg : {&g, &aut, &patch}) {
    Matcher x(*gg);
    x.fill(mask.data());
    for (int id = 0; id < g_vocab->size(); ++id) {
      if (g_vocab->never_allowed(id) && bit(id)) {
        ++failures;
        std::fprintf(stderr, "never-allowed id %d allowed\n", id);
        break;
      }
    }
    if (gg == &aut)
      for (int id = 0; id < g_vocab->size(); ++id) allowed_free += bit(id);
    ++states;
  }
  EXPECT(states == 3);
  EXPECT(allowed_free > 240000);  // free text: everything but control tokens and padding
  int never = 0;
  for (int id = 0; id < g_vocab->size(); ++id) never += g_vocab->never_allowed(id);
  EXPECT(never >= 248320 - 248077);
  EXPECT(allowed_free == g_vocab->size() - never);
}

// The mask equals a brute-force try-accept of every id, on sampled states of several grammars.
void test_mask_brute_force() {
  struct Walk {
    Grammar g;
    std::string text;
  };
  ToolCallSpec spec = function_spec(ToolCallSpec::Choice::kAuto);
  spec.tools.push_back({Tool::Kind::kCustom, "apply_patch", {}, Tool::Format::kLark, kApplyPatchLark});
  std::vector<Walk> walks = {
      {schema(kDocSchema), "{\n  \"name\": \"caf\xc3\xa9\",\n  \"tags\": [\"a\", \"b\"],\n  \"n\": -12,\n  \"ratio\": 0.5\n}"},
      {g_comp->tool_calls(spec), rendered_calls(ojson::parse(R"([{"type": "function", "function": {"name":
           "exec_command", "arguments": {"cmd": "ls", "workdir": "/"}}}])"), "Let me look.")},
      {g_comp->tool_calls(spec), "Patching.\n\n" + custom_tool_prefix("apply_patch") + kPatchMixed +
                                     "\n</parameter>\n</function>\n</tool_call>"},
      {g_comp->lark(kExecLark, "tools[0].format"), "// @exec: {}\nfor (const x of [1, 2]) console.log(x);\n"},
      {g_comp->json_object(), "{\"k\": [true, null, {\"z\": \"\\u00e9\"}], \"m\": 1e5}"},
  };
  const int per_walk = 12;
  int sampled = 0, mismatches = 0;
  std::vector<int32_t> mask(g_vocab->words());
  const int threads = 4;
  for (const Walk& w : walks) {
    const std::vector<int> ids = g_tok->encode(w.text);
    EXPECT(ids_ok(w.g, ids));
    Matcher m(w.g);
    for (std::size_t i = 0; i <= ids.size(); ++i) {
      const bool sample = i % std::max<std::size_t>(1, (ids.size() + per_walk - 1) / per_walk) == 0 || i == ids.size();
      if (sample) {
        m.fill(mask.data());
        std::atomic<int> bad{0};
        std::vector<std::thread> pool;
        for (int t = 0; t < threads; ++t) {
          pool.emplace_back([&, t] {
            Matcher f = m.fork();
            for (int id = t; id < g_vocab->size(); id += threads) {
              const bool want = (mask[id >> 5] >> (id & 31)) & 1;
              const bool got = f.accept(id);
              if (got) f.rollback(1);
              if (got != want && bad.fetch_add(1) < 3)
                std::fprintf(stderr, "mask mismatch at token %zu of a walk: id %d mask %d accept %d\n", i, id, want, got);
            }
          });
        }
        for (auto& th : pool) th.join();
        mismatches += bad.load();
        ++sampled;
      }
      if (i < ids.size()) EXPECT(m.accept(ids[i]));
    }
  }
  std::printf("brute force: %d states, %d mismatches\n", sampled, mismatches);
  EXPECT(sampled >= 50);
  EXPECT(mismatches == 0);
}

// Matcher operations: rollback, fork, reset.
void test_matcher_ops() {
  Grammar g = schema(kDocSchema);
  const std::vector<int> ids = g_tok->encode(R"({"name": "x", "n": 5})");
  Matcher m(g);
  for (int id : ids) EXPECT(m.accept(id));
  EXPECT(m.can_stop());
  m.rollback(2);
  EXPECT(!m.can_stop());
  Matcher f = m.fork();
  EXPECT(f.accept(ids[ids.size() - 2]) && f.accept(ids.back()) && f.can_stop());
  EXPECT(!m.can_stop());
  EXPECT(!m.accept(g_tok->token_id("<|im_start|>")));
  m.reset();
  EXPECT(!m.accept(ids.back()) || ids.back() == ids.front());
  std::vector<int32_t> mask(g_vocab->words());
  EXPECT(m.fill(mask.data()));  // the start of a JSON object masks almost everything
}

// The compiled-grammar cache: hits on the canonical text, LRU eviction by bytes.
void test_cache() {
  Compiler c(*g_vocab);
  Grammar a = c.json_schema(ojson::parse(R"({"type": "object", "properties": {"a": {"type": "string"}}})"), kSchemaParam);
  Grammar b = c.json_schema(
      ojson::parse(R"({"type": "object", "title": "T", "properties": {"a": {"type": "string", "description": "d"}}})"),
      kSchemaParam);
  EXPECT(&a.impl() == &b.impl());  // annotations are not part of the key
  Compiler::Stats s = c.stats();
  EXPECT(s.hits == 1 && s.misses == 1 && s.entries == 1);
  EXPECT(s.bytes == a.memory_bytes() && s.bytes > 0);
  Grammar o = c.json_object();  // another kind, another entry
  EXPECT(&o.impl() != &a.impl());
  EXPECT(c.stats().entries == 2);
  Grammar l1 = c.lark(kExecLark, "tools[0].format");
  Grammar l2 = c.lark(kExecLark, "tools[1].format");  // the param is not part of the key
  EXPECT(&l1.impl() == &l2.impl());

  Compiler small(*g_vocab, 1);  // a budget below any grammar: keeps only the newest entry
  Grammar x = small.json_object();
  Grammar y = small.lark(kExecLark, "tools[0].format");
  s = small.stats();
  EXPECT(s.entries == 1 && s.evictions == 1);
  Grammar x2 = small.json_object();  // evicted: compiled again, the old grammar stays valid for its matchers
  EXPECT(small.stats().misses == 3);
  EXPECT(chars_ok(x, "{}") && chars_ok(x2, "{}"));
  small.clear_cache();
  EXPECT(small.stats().entries == 0 && small.stats().bytes == 0);
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
  g_vocab = &vocab;
  g_comp = &comp;
  g_im_end = tok.token_id("<|im_end|>");
  g_eot = tok.token_id("<|endoftext|>");

  const std::pair<const char*, void (*)()> tests[] = {
      {"vocab", test_vocab},
      {"schema_whitelist", test_schema_whitelist},
      {"schema_refusals", test_schema_refusals},
      {"json_sets", test_json_sets},
      {"whitespace_bound", test_whitespace_bound},
      {"codex_grammars", test_codex_grammars},
      {"lark", test_lark},
      {"tool_calls", test_tool_calls},
      {"response_format", test_response_format},
      {"array_bounds", test_array_bounds},
      {"stop_and_padding", test_stop_and_padding},
      {"matcher_ops", test_matcher_ops},
      {"cache", test_cache},
      {"mask_brute_force", test_mask_brute_force},
  };
  for (const auto& [name, fn] : tests) {
    const int before = failures;
    try {
      fn();
    } catch (const std::exception& e) {
      ++failures;
      std::fprintf(stderr, "%s: exception: %s\n", name, e.what());
    }
    std::printf("%-18s %s\n", name, failures == before ? "ok" : "FAILED");
  }
  if (failures) {
    std::printf("%d failure(s)\n", failures);
    return 1;
  }
  std::printf("all grammar tests passed\n");
  return 0;
}
