// The drafter selection (serve/drafter_routes.hpp, MULTI_DRAFTER §2.1, §2.2, §5.2, §5.3); host only.
#include <cstdio>
#include <stdexcept>
#include <string>

#include "serve/drafter_routes.hpp"

using ragon::serve::declared_workload;
using ragon::serve::DrafterChoice;
using ragon::serve::DrafterRoutes;
using ragon::serve::last_item_of;
using ragon::serve::LastItem;
using ragon::serve::parse_drafter_arg;
using ragon::serve::tools_include_shell;
using json = nlohmann::ordered_json;

static int failures = 0;

static void check(bool ok, const char* what) {
  if (!ok) {
    std::printf("FAILED: %s\n", what);
    ++failures;
  }
}

static bool refuses_arg(const std::string& arg) {
  try {
    parse_drafter_arg(arg);
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

static bool refuses_routes(const std::string& text) {
  try {
    DrafterRoutes::parse(text);
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

static bool is(const DrafterChoice& c, const std::string& id, const std::string& how) {
  return c.id == id && how == c.how;
}

int main() {
  // --drafter ID=DIR
  check(parse_drafter_arg("work=/some/dir=x").id == "work", "the id ends at the first '='");
  check(parse_drafter_arg("work=/some/dir=x").dir == "/some/dir=x", "the directory keeps the rest");
  for (const char* bad : {"work", "=dir", "work=", "Work=dir", "a b=dir", "wo\"rk=dir", "wo\nrk=dir"})
    check(refuses_arg(bad), bad);

  // The hint's carriers.
  const json blob = json{{"turn_id", "t"}, {"ragon_workload", "ask"}};
  const json meta = {{"x-codex-turn-metadata", blob.dump()}};
  check(declared_workload(json{{"ragon", {{"workload", "work"}}}}) == "work", "the documented form");
  check(declared_workload(json{{"client_metadata", meta}}) == "ask", "Codex's per-turn metadata");
  check(declared_workload(json{{"ragon", {{"workload", "work"}}}, {"client_metadata", meta}}) == "work",
        "the documented form wins");
  check(declared_workload(json{{"ragon", {{"workload", "nonsense"}}}, {"client_metadata", meta}}) == "ask",
        "an unknown value is as if absent");
  check(declared_workload(json::object()).empty(), "no hint");
  check(declared_workload(json{{"ragon", "work"}}).empty(), "a ragon that is no object");
  check(declared_workload(json{{"ragon", {{"workload", 3}}}}).empty(), "a workload that is no string");
  check(declared_workload(json{{"client_metadata", {{"x-codex-turn-metadata", "{not json"}}}}).empty(),
        "a metadata blob that is no JSON");
  check(declared_workload(json{{"client_metadata", {{"x-codex-turn-metadata", 3}}}}).empty(), "a blob that is no string");
  check(declared_workload(json{{"client_metadata", "x"}}).empty(), "metadata that is no object");

  // What the rule reads.
  check(last_item_of(json::array()) == LastItem::kNone, "no messages");
  check(last_item_of(json::parse(R"([{"role":"user"},{"role":"tool"}])")) == LastItem::kToolResult, "a tool result");
  check(last_item_of(json::parse(R"([{"role":"tool"},{"role":"user"}])")) == LastItem::kUser, "a user message");
  check(last_item_of(json::parse(R"([{"role":"assistant"}])")) == LastItem::kOther, "an assistant message");
  check(tools_include_shell(json::parse(R"([{"type":"function","function":{"name":"exec_command"}}])")), "exec_command");
  check(tools_include_shell(json::parse(R"([{"name":"view"},{"name":"apply_patch"}])")), "apply_patch, flat");
  check(!tools_include_shell(json::parse(R"([{"type":"function","function":{"name":"web_search"}}])")), "no shell");

  // The table of the spec.
  const DrafterRoutes full = DrafterRoutes::parse(
      R"({"work":"work","swe-bench":"work","night":"work","ask":"ask","phone":"ask","*":"general"})");
  check(is(full.choose("swe-bench", false, false, LastItem::kNone), "work", "declared"), "swe-bench pins work");
  check(is(full.choose("phone", true, true, LastItem::kToolResult), "ask", "declared"), "a hint beats the rule");
  check(is(full.choose("", true, false, LastItem::kToolResult), "work", "inferred"), "tools and a tool result");
  check(is(full.choose("", true, true, LastItem::kUser), "work", "inferred"), "a shell tool and a user message");
  check(is(full.choose("", true, false, LastItem::kUser), "general", "default"), "other tools and a user message");
  check(is(full.choose("", false, false, LastItem::kUser), "ask", "inferred"), "no tools");
  check(is(full.choose("general", true, true, LastItem::kUser), "work", "inferred"), "the hint general goes to the rule");

  // `ask` not routed yet: no tools stays on general.
  const DrafterRoutes no_ask = DrafterRoutes::parse(R"({"work":"work","*":"general"})");
  check(is(no_ask.choose("", false, false, LastItem::kUser), "general", "default"), "no ask drafter yet");
  check(is(no_ask.choose("ask", false, false, LastItem::kUser), "general", "declared"), "declared, routed by *");

  // Everything to general, §4.4's regression arm.
  const DrafterRoutes all = DrafterRoutes::parse(R"({"*":"general"})");
  check(is(all.choose("", true, true, LastItem::kToolResult), "general", "default"), "only *: the rule has no route");
  check(is(all.choose("work", true, true, LastItem::kToolResult), "general", "declared"), "only *: a hint");
  check(all.targets() == std::vector<std::string>{"general"}, "the table's drafters");

  // No table at all: as with one drafter.
  const DrafterRoutes none;
  check(is(none.choose("work", true, true, LastItem::kToolResult), "general", "default"), "no table");
  check(none.targets().empty(), "no table names no drafter");

  for (const char* bad : {"[]", "{not json", R"({"work":3})", R"({"nonsense":"work"})"}) check(refuses_routes(bad), bad);

  if (failures) return 1;
  std::printf("drafter routes: ok\n");
  return 0;
}
