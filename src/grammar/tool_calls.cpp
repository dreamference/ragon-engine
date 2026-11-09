// The tool-call grammar (§2.2): the calls exactly as chat_template renders them, as an XGrammar structural tag.
//
//   <tool_call>\n<function=NAME>\n
//   <parameter=KEY>\nVALUE\n</parameter>\n          (per argument, declared keys only)
//   </function>\n</tool_call>
//
// Calls are separated by one "\n"; with spec.parallel false there is one call and then the stop. Under required
// and named the payload is at most two newlines (after </think>) and then one or more calls; under auto, free text
// with calls triggered by "<tool_call>". With a response_format the payload is `json | calls` (vllm#39929).
#include "grammar/grammar.hpp"

namespace ragon::grammar {

std::string custom_tool_prefix(const std::string& name) {
  return "<tool_call>\n<function=" + name + ">\n<parameter=input>\n";
}

namespace detail {

std::string tool_calls_tag(const ToolCallSpec& spec);

namespace {

using ojson = nlohmann::ordered_json;

constexpr const char* kCallEnd = "</function>\n</tool_call>";
constexpr const char* kLead = "root ::= \"\" | \"\\n\" | \"\\n\\n\"\n";  // at most 2 newlines after </think>

void check_name(const std::string& name, const std::string& what) {
  if (name.empty()) throw Error("invalid_tools", "tools: " + what + " has an empty name");
  for (unsigned char c : name)
    if (c < 0x20 || c == '<' || c == '>' || c == 0x7F)
      throw Error("invalid_tools", "tools: the name '" + name + "' has a character a tool name cannot have");
}

ojson function_tag(const Tool& t) {
  ojson params = ojson::array();
  for (const std::string& p : t.params) {
    check_name(p, "a parameter of '" + t.name + "'");
    params.push_back({{"type", "tag"},
                      {"begin", "<parameter=" + p + ">\n"},
                      {"content", {{"type", "any_text"}}},
                      {"end", "\n</parameter>\n"}});
  }
  ojson content;
  if (params.empty()) content = {{"type", "const_string"}, {"value", ""}};
  else if (params.size() == 1) content = {{"type", "star"}, {"content", params[0]}};
  else content = {{"type", "star"}, {"content", {{"type", "or"}, {"elements", params}}}};
  return {{"type", "tag"},
          {"begin", "<tool_call>\n<function=" + t.name + ">\n"},
          {"content", content},
          {"end", kCallEnd}};
}

ojson custom_tag(const Tool& t, std::size_t index) {
  const std::string param = "tools[" + std::to_string(t.index >= 0 ? static_cast<std::size_t>(t.index) : index) + "].format";
  ojson content;
  switch (t.format) {
    case Tool::Format::kText:
      content = {{"type", "any_text"}};
      break;
    case Tool::Format::kLark:
      content = {{"type", "grammar"}, {"grammar", lark_to_ebnf(t.definition, param)}};
      break;
    case Tool::Format::kRegex:
      content = {{"type", "grammar"}, {"grammar", "root ::= " + regex_to_ebnf(t.definition, param) + "\n"}};
      break;
  }
  return {{"type", "tag"},
          {"begin", custom_tool_prefix(t.name)},
          {"content", content},
          {"end", std::string("\n</parameter>\n") + kCallEnd}};
}

}  // namespace

std::string tool_calls_tag(const ToolCallSpec& spec) {
  if (spec.tools.empty()) throw Error("invalid_tools", "tools: the list is empty");
  ojson tags = ojson::array();
  bool found = false;
  for (std::size_t i = 0; i < spec.tools.size(); ++i) {
    const Tool& t = spec.tools[i];
    check_name(t.name, "tools[" + std::to_string(i) + "]");
    if (spec.choice == ToolCallSpec::Choice::kNamed && t.name != spec.named) continue;
    found = true;
    tags.push_back(t.kind == Tool::Kind::kCustom ? custom_tag(t, i) : function_tag(t));
  }
  if (spec.choice == ToolCallSpec::Choice::kNamed && !found)
    throw Error("invalid_tools", "tool_choice: the function '" + spec.named + "' is not in tools");

  ojson format;
  if (spec.choice == ToolCallSpec::Choice::kAuto) {
    ojson calls = {{"type", "triggered_tags"},
                   {"triggers", ojson::array({"<tool_call>"})},
                   {"tags", tags},
                   {"at_least_one", false},
                   {"stop_after_first", !spec.parallel}};
    if (spec.json_schema || spec.json_object) {
      // json | calls: a JSON answer, or tool calls (one or more, after at most two newlines).
      const std::string json = spec.json_schema ? schema_ebnf(canonical_schema(*spec.json_schema, spec.json_param), true)
                                                : schema_ebnf(ojson{{"type", "object"}}, false);
      ojson only_calls = {{"type", "tags_with_separator"},
                          {"tags", tags},
                          {"separator", "\n"},
                          {"at_least_one", true},
                          {"stop_after_first", !spec.parallel}};
      format = {{"type", "sequence"},
                {"elements",
                 ojson::array({ojson{{"type", "grammar"}, {"grammar", kLead}},
                               ojson{{"type", "or"},
                                     {"elements", ojson::array({ojson{{"type", "grammar"}, {"grammar", json}},
                                                                only_calls})}}})}};
    } else {
      format = std::move(calls);
    }
  } else {
    // required and named: calls only; a response_format does not apply (the answer is a call).
    format = {{"type", "sequence"},
              {"elements",
               ojson::array({ojson{{"type", "grammar"}, {"grammar", kLead}},
                             ojson{{"type", "tags_with_separator"},
                                   {"tags", tags},
                                   {"separator", "\n"},
                                   {"at_least_one", true},
                                   {"stop_after_first", !spec.parallel}}})}};
  }
  return ojson{{"type", "structural_tag"}, {"format", format}}.dump();
}

std::string response_format_tag(const std::optional<ojson>& schema, const std::string& param) {
  const std::string json = schema ? schema_ebnf(canonical_schema(*schema, param), true)
                                  : schema_ebnf(ojson{{"type", "object"}}, false);
  const ojson format = {{"type", "sequence"},
                        {"elements", ojson::array({ojson{{"type", "grammar"}, {"grammar", kLead}},
                                                   ojson{{"type", "grammar"}, {"grammar", json}}})}};
  return ojson{{"type", "structural_tag"}, {"format", format}}.dump();
}

}  // namespace detail
}  // namespace ragon::grammar
