// Structured output (SPECULATION "Structured output", stages 2 and 4): a request's `response_format` (`text.format`),
// `tool_choice`, `parallel_tool_calls` and custom-tool grammars as one grammar, and that grammar as the engine's
// TokenConstraint. ragon-serve uses it in a build with RAGON_GRAMMAR unless RAGON_STRUCTURED=0; without it the
// fields are read and not enforced (API §3, §4).
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

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

// The offered tools (canonical_tools' shape) as the tool-call grammar sees them: the name and the declared parameter
// names; a custom tool whose `format` is a grammar (Request::custom_formats) is its custom form, its input
// constrained by that grammar. Throws grammar::Error for a grammar format of another syntax or without a definition.
inline grammar::ToolCallSpec tool_spec(const json& tools, grammar::ToolCallSpec::Choice choice,
                                       const std::vector<Request::CustomFormat>& formats = {}) {
  grammar::ToolCallSpec s;
  s.choice = choice;
  for (const json& t : tools) {
    if (!t.is_object() || !t.contains("function") || !t["function"].is_object()) continue;
    const json& f = t["function"];
    grammar::Tool tool;
    tool.name = f.contains("name") && f["name"].is_string() ? f["name"].get<std::string>() : std::string();
    const Request::CustomFormat* custom = nullptr;
    for (const Request::CustomFormat& c : formats)
      if (c.name == tool.name) custom = &c;
    if (custom) {
      const std::string where = "tools[" + std::to_string(custom->index) + "].format";
      if (custom->syntax == "lark") tool.format = grammar::Tool::Format::kLark;
      else if (custom->syntax == "regex") tool.format = grammar::Tool::Format::kRegex;
      else
        throw grammar::Error("unsupported_grammar", where + ": the syntax '" + custom->syntax +
                                                        "' is not supported (supported: lark, regex)");
      if (custom->definition.empty())
        throw grammar::Error("invalid_grammar", where + ": the definition is missing or empty");
      tool.kind = grammar::Tool::Kind::kCustom;
      tool.definition = custom->definition;
      tool.index = custom->index;
    } else if (f.contains("parameters") && f["parameters"].is_object() && f["parameters"].contains("properties") &&
               f["parameters"]["properties"].is_object()) {
      for (const auto& [k, v] : f["parameters"]["properties"].items()) tool.params.push_back(k);
    }
    s.tools.push_back(std::move(tool));
  }
  return s;
}

// The request's grammar, or nothing when it asks for none (API §3, §4):
//   - tool_choice "required" or named (with tools): calls only, the name fixed when named;
//   - response_format: the JSON answer, or with tools under auto the JSON answer or calls;
//   - under auto, a custom tool with a grammar format, or parallel_tool_calls false: free text, and from
//     "<tool_call>" on a call of an offered tool (a custom tool's input in its grammar); one call when not parallel;
//   - anything else: nothing, so the request runs exactly as without the knob.
// Completions take only response_format. Throws grammar::Error (a 400 with its code and the stage 1 text) or
// std::invalid_argument (a 400).
inline std::optional<grammar::Grammar> request_grammar(grammar::Compiler& c, const Request& r) {
  using Choice = grammar::ToolCallSpec::Choice;
  const bool tools = r.chat && !r.tools.empty();
  if (r.chat && !r.tool_choice_named.empty() && r.tools.empty())
    throw grammar::Error("invalid_tools", "tool_choice: the function '" + r.tool_choice_named + "' is not in tools");
  if (tools && (r.tool_choice_required || !r.tool_choice_named.empty())) {
    grammar::ToolCallSpec s = tool_spec(r.tools, r.tool_choice_required ? Choice::kRequired : Choice::kNamed,
                                        r.custom_formats);
    s.named = r.tool_choice_named;
    s.parallel = r.parallel_tool_calls;
    return c.tool_calls(s);
  }
  std::optional<json> schema;
  const std::string& param = r.response_format_param;
  if (!r.response_format.is_null() && r.response_format.value("type", "") == "json_schema") {
    const json& rf = r.response_format;
    if (!rf.contains("json_schema") || !rf["json_schema"].is_object() || !rf["json_schema"].contains("schema") ||
        !rf["json_schema"]["schema"].is_object())
      throw std::invalid_argument(param + " must be an object");
    schema = rf["json_schema"]["schema"];
  }
  if (tools && (!r.response_format.is_null() || !r.custom_formats.empty() || !r.parallel_tool_calls)) {
    grammar::ToolCallSpec s = tool_spec(r.tools, Choice::kAuto, r.custom_formats);
    if (schema) s.json_schema = *schema;
    else if (!r.response_format.is_null()) s.json_object = true;
    s.json_param = param;
    s.parallel = r.parallel_tool_calls;
    return c.tool_calls(s);
  }
  if (r.response_format.is_null()) return std::nullopt;
  // json_object names the field itself in a 400: "response_format" or "text.format".
  const std::string object_param = param == "text.format.schema" ? "text.format" : "response_format";
  return c.response_format(schema, schema ? param : object_param);
}

}  // namespace ragon::serve
