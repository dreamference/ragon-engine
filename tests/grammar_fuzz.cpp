// libFuzzer target for the grammar library: schema JSON, Lark text, regex patterns and custom-tool grammars
// through compile, then a short walk of a matcher. Built and run by scripts/fuzz_grammar.sh.
//
// The first input byte picks the front end; the rest is the text. A grammar::Error (a 400) is an expected
// outcome; any other exception, a crash or a sanitizer report is a finding.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "grammar/grammar.hpp"

using namespace ragon::grammar;

namespace {

// A small synthetic vocabulary: every byte, a few multi-byte pieces, two control ids and a stop id.
const Vocab& vocab() {
  static const Vocab v = [] {
    std::vector<std::string> bytes;
    for (int b = 0; b < 256; ++b) bytes.emplace_back(1, static_cast<char>(b));
    for (const char* s : {"<tool_call>", "</tool_call>", "<function=", "<parameter=", "</parameter>", "</function>",
                          "{\"", "\": ", "\", \"", "\"}", "\n", "  ", "true", "null", "*** ", "Begin Patch",
                          "End Patch", "\xc3\xa9", "\xe2\x9c\x93"})
      bytes.emplace_back(s);
    bytes.emplace_back("");  // a control token: never allowed
    bytes.emplace_back("");
    bytes.emplace_back("<|im_end|>");
    const int stop = static_cast<int>(bytes.size()) - 1;
    return Vocab(bytes, {stop}, static_cast<int>(bytes.size()) + 5);  // 5 padding ids
  }();
  return v;
}

void walk(const Grammar& g, const std::string& text) {
  Matcher m(g);
  std::vector<int32_t> mask(vocab().words());
  m.fill(mask.data());
  std::size_t n = 0;
  for (unsigned char c : text) {
    if (++n > 64) break;
    if (!m.accept(c)) break;
    if (n % 8 == 0) m.fill(mask.data());
  }
  if (n > 2) m.rollback(1);
  (void)m.can_stop();
  m.fill(mask.data());
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size < 1 || size > 4096) return 0;
  static Compiler comp(vocab(), std::size_t{8} << 20);
  const std::string text(reinterpret_cast<const char*>(data) + 1, size - 1);
  try {
    switch (data[0] % 5) {
      case 0: walk(comp.json_schema_text(text, "response_format.json_schema.schema"), text); break;
      case 1: walk(comp.lark(text, "tools[0].format"), text); break;
      case 2: walk(comp.regex(text, "tools[0].format"), text); break;
      case 3: {
        ToolCallSpec spec;
        spec.choice = data[0] & 8 ? ToolCallSpec::Choice::kRequired : ToolCallSpec::Choice::kAuto;
        spec.tools.push_back({Tool::Kind::kFunction, "f", {"a", "b"}, Tool::Format::kText, ""});
        spec.tools.push_back({Tool::Kind::kCustom, "g", {}, data[0] & 16 ? Tool::Format::kRegex : Tool::Format::kLark,
                              text});
        walk(comp.tool_calls(spec), text);
        break;
      }
      case 4: {
        ToolCallSpec spec;
        spec.choice = ToolCallSpec::Choice::kAuto;
        spec.tools.push_back({Tool::Kind::kFunction, "f", {"a"}, Tool::Format::kText, ""});
        spec.json_schema = nlohmann::ordered_json::parse(text, nullptr, false);
        if (spec.json_schema->is_discarded()) return 0;
        walk(comp.tool_calls(spec), text);
        break;
      }
    }
  } catch (const Error&) {
    // a 400: expected
  } catch (const std::exception& e) {
    std::fprintf(stderr, "finding: an exception that is not a grammar::Error: %s\n", e.what());
    std::abort();
  }
  return 0;
}
