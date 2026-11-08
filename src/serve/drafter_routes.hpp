// Which resident drafter serves a request (MULTI_DRAFTER §2.1, §2.2, §5.2, §5.3). Pure: no engine, no clock, no other
// request, so a seeded request replayed with the same hint picks the same drafter (§2.4). The hint only ever chooses
// what speculation proposes; the tokens stay what they would be without it.
#pragma once

#include <nlohmann/json.hpp>

#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace ragon::serve {

inline const char* const kGeneralDrafter = "general";

// `--drafter ID=DIR`. The id goes into a response header and a metric label, so it is kept to their safe characters.
struct DrafterArg {
  std::string id, dir;
};

inline DrafterArg parse_drafter_arg(const std::string& arg) {
  const size_t eq = arg.find('=');
  if (eq == std::string::npos || eq == 0 || eq + 1 == arg.size())
    throw std::invalid_argument("--drafter wants ID=DIR, got '" + arg + "'");
  DrafterArg out{arg.substr(0, eq), arg.substr(eq + 1)};
  for (const char c : out.id)
    if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-' || c == '_'))
      throw std::invalid_argument("a drafter id is lower-case letters, digits, '-' and '_', got '" + out.id + "'");
  return out;
}

inline bool known_workload(const std::string& w) {
  return w == "work" || w == "ask" || w == "phone" || w == "swe-bench" || w == "night" || w == "general";
}

// The request's hint (§2.1), empty when it carries none we know: `"ragon": {"workload": "ask"}`, or Codex's per-turn
// metadata, whose `x-codex-turn-metadata` value is a JSON string with the client's own keys at its top level
// (`ragon_workload`). The documented form wins. An unknown value and anything malformed is ignored here; the parser
// refuses one case before this, a `ragon.workload` that is no string (API §8).
inline std::string declared_workload(const nlohmann::ordered_json& body) {
  using json = nlohmann::ordered_json;
  auto known = [](const json& object, const char* key) -> std::string {
    if (!object.is_object() || !object.contains(key) || !object[key].is_string()) return "";
    const std::string w = object[key].get<std::string>();
    return known_workload(w) ? w : "";
  };
  if (!body.is_object()) return "";
  if (body.contains("ragon")) {
    const std::string w = known(body["ragon"], "workload");
    if (!w.empty()) return w;
  }
  if (body.contains("client_metadata") && body["client_metadata"].is_object() &&
      body["client_metadata"].contains("x-codex-turn-metadata") &&
      body["client_metadata"]["x-codex-turn-metadata"].is_string()) {
    const json blob = json::parse(body["client_metadata"]["x-codex-turn-metadata"].get<std::string>(), nullptr, false);
    if (!blob.is_discarded()) return known(blob, "ragon_workload");
  }
  return "";
}

// The last input item's kind, which §2.2's rule reads.
enum class LastItem { kNone, kUser, kToolResult, kOther };

inline LastItem last_item_of(const nlohmann::ordered_json& messages) {
  if (!messages.is_array() || messages.empty()) return LastItem::kNone;
  const auto& last = messages.back();
  if (!last.is_object() || !last.contains("role") || !last["role"].is_string()) return LastItem::kOther;
  const std::string role = last["role"].get<std::string>();
  return role == "tool" ? LastItem::kToolResult : role == "user" ? LastItem::kUser : LastItem::kOther;
}

// Whether the offered tools (function tools, flat or under `function`) include a shell or patch tool (§2.2).
inline bool tools_include_shell(const nlohmann::ordered_json& tools) {
  if (!tools.is_array()) return false;
  for (const auto& t : tools) {
    if (!t.is_object()) continue;
    const auto& f = t.contains("function") && t["function"].is_object() ? t["function"] : t;
    const std::string name = f.contains("name") && f["name"].is_string() ? f["name"].get<std::string>() : "";
    if (name == "exec_command" || name == "shell" || name == "shell_command" || name == "local_shell" ||
        name == "apply_patch")
      return true;
  }
  return false;
}

struct DrafterChoice {
  std::string id;
  const char* how;  // `declared`, `inferred` or `default` (§2.5)
};

// The workload-to-drafter table (--drafter-routes, §5.2), for example
// {"work":"work","swe-bench":"work","night":"work","ask":"ask","phone":"ask","*":"general"}. With no table every
// request is served by `general`.
class DrafterRoutes {
 public:
  // Throws std::invalid_argument on anything but an object of strings, or a key that names no workload.
  static DrafterRoutes parse(const std::string& text) {
    using json = nlohmann::ordered_json;
    const json j = json::parse(text, nullptr, false);
    if (j.is_discarded() || !j.is_object()) throw std::invalid_argument("the drafter routes must be a JSON object");
    DrafterRoutes out;
    for (auto it = j.begin(); it != j.end(); ++it) {
      if (!it.value().is_string())
        throw std::invalid_argument("the drafter route for '" + it.key() + "' must name a drafter id");
      if (it.key() != "*" && !known_workload(it.key()))
        throw std::invalid_argument("the drafter routes name an unknown workload '" + it.key() + "'");
      out.routes_[it.key()] = it.value().get<std::string>();
    }
    return out;
  }

  // The drafter ids the table names, for the check at start that each one is loaded.
  std::vector<std::string> targets() const {
    std::vector<std::string> out;
    for (const auto& [workload, id] : routes_)
      if (std::find(out.begin(), out.end(), id) == out.end()) out.push_back(id);
    return out;
  }

  // The drafter a workload goes to: its route, else `*`, else `general`.
  std::string target(const std::string& workload) const {
    auto it = routes_.find(workload);
    if (it == routes_.end()) it = routes_.find("*");
    return it == routes_.end() ? kGeneralDrafter : it->second;
  }

  // §2.1's table, then §2.2's rule for a request with no hint (or the hint `general`, which the table sends to the
  // rule too). `ask` is inferred only once a route names it: until then it is not a drafter that has passed its gate.
  DrafterChoice choose(const std::string& hint, bool has_tools, bool shell_tool, LastItem last) const {
    if (routes_.empty()) return {kGeneralDrafter, "default"};
    if (!hint.empty() && hint != "general") return {target(hint), "declared"};
    const bool work = has_tools && (last == LastItem::kToolResult || shell_tool);
    if (work && routes_.count("work")) return {routes_.at("work"), "inferred"};
    if (!has_tools && routes_.count("ask")) return {routes_.at("ask"), "inferred"};
    return {target("general"), "default"};
  }

 private:
  std::map<std::string, std::string> routes_;
};

}  // namespace ragon::serve
