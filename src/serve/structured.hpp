// Structured output, stage 2 (SPECULATION "Structured output, stage 2"): a chat request's `response_format` and
// `tool_choice: "required"` as one grammar, and that grammar as the engine's TokenConstraint. ragon-serve uses it only
// with RAGON_STRUCTURED=1 in a build with RAGON_GRAMMAR; without either, the fields stay ignored (API §3).
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include "core/constraint.hpp"
#include "grammar/grammar.hpp"
#include "serve/api.hpp"

namespace ragon::serve {

// One request's matcher: built at admission (begin_decode), dropped at the request's end.
class GrammarConstraint final : public TokenConstraint {
 public:
  explicit GrammarConstraint(const grammar::Grammar& g, int words) : m_(g), words_(words) {}
  int words() const override { return words_; }
  void fill(int32_t* mask) override { m_.fill(mask); }
  bool accept(int id) override { return m_.accept(id); }
  void rollback(int n) override { m_.rollback(n); }
  bool terminated() const override { return m_.terminated(); }

 private:
  grammar::Matcher m_;
  int words_;
};

// The offered function tools (canonical_tools' shape) as the tool-call grammar sees them: the name and the declared
// parameter names.
inline grammar::ToolCallSpec tool_spec(const json& tools, grammar::ToolCallSpec::Choice choice) {
  grammar::ToolCallSpec s;
  s.choice = choice;
  for (const json& t : tools) {
    if (!t.is_object() || !t.contains("function") || !t["function"].is_object()) continue;
    const json& f = t["function"];
    grammar::Tool tool;
    tool.name = f.contains("name") && f["name"].is_string() ? f["name"].get<std::string>() : std::string();
    if (f.contains("parameters") && f["parameters"].is_object() && f["parameters"].contains("properties") &&
        f["parameters"]["properties"].is_object())
      for (const auto& [k, v] : f["parameters"]["properties"].items()) tool.params.push_back(k);
    s.tools.push_back(std::move(tool));
  }
  return s;
}

// The request's grammar, or nothing when it asks for none. `tool_choice: "required"` (with tools) is calls only; a
// response_format is the JSON answer, or with tools under `auto` the JSON answer or calls. Throws grammar::Error
// (a 400 with its code and the stage 1 text) or std::invalid_argument (a 400).
inline std::optional<grammar::Grammar> request_grammar(grammar::Compiler& c, const Request& r) {
  if (!r.chat) return std::nullopt;
  if (r.tool_choice_required && !r.tools.empty())
    return c.tool_calls(tool_spec(r.tools, grammar::ToolCallSpec::Choice::kRequired));
  if (r.response_format.is_null()) return std::nullopt;
  std::optional<json> schema;
  if (r.response_format.value("type", "") == "json_schema") {
    const json& rf = r.response_format;
    if (!rf.contains("json_schema") || !rf["json_schema"].is_object() || !rf["json_schema"].contains("schema") ||
        !rf["json_schema"]["schema"].is_object())
      throw std::invalid_argument("response_format.json_schema.schema must be an object");
    schema = rf["json_schema"]["schema"];
  }
  if (!r.tools.empty()) {
    grammar::ToolCallSpec s = tool_spec(r.tools, grammar::ToolCallSpec::Choice::kAuto);
    if (schema) s.json_schema = *schema;
    else s.json_object = true;
    return c.tool_calls(s);
  }
  return c.response_format(schema, schema ? "response_format.json_schema.schema" : "response_format");
}

}  // namespace ragon::serve
