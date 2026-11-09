// JSON schema, the stage 1 subset (§2.1): checks the keywords, strips annotations, and turns the schema into a
// JSON grammar whose whitespace slots are bounded.
#include <cstdint>
#include <regex>
#include <set>

#include <xgrammar/xgrammar.h>

#include "grammar/grammar.hpp"

namespace ragon::grammar::detail {

namespace {

using ojson = nlohmann::ordered_json;

constexpr int kMaxDepth = 64;  // schema nesting; deeper schemas are refused (and keep the walk off the stack)
constexpr std::uint64_t kMaxItemsBound = 2147483647;  // minItems, maxItems: XGrammar's repetition bound (int32)

const std::set<std::string> kAnnotations = {"title",   "description", "default",  "examples",
                                            "$schema", "$id",         "$comment", "format"};
const std::set<std::string> kTypes = {"object", "array", "string", "number", "integer", "boolean", "null"};

// RFC 6901: '~' -> "~0", '/' -> "~1".
std::string escape_token(const std::string& key) {
  std::string out;
  for (char c : key) {
    if (c == '~') out += "~0";
    else if (c == '/') out += "~1";
    else out += c;
  }
  return out;
}

std::string show_ptr(const std::string& ptr) { return ptr.empty() ? "\"\"" : ptr; }

struct Walker {
  const ojson& root;
  const std::string& param;

  [[noreturn]] void unsupported(const std::string& kw, const std::string& ptr) const {
    throw Error("unsupported_json_schema",
                param + ": the keyword '" + kw + "' at " + show_ptr(ptr) +
                    " is not supported (supported: type, properties, required, additionalProperties, items, "
                    "enum, const, anyOf, $ref, $defs)");
  }
  [[noreturn]] void invalid(const std::string& reason, const std::string& ptr) const {
    throw Error("invalid_json_schema", param + ": " + reason + " at " + show_ptr(ptr));
  }

  // Resolves a local reference "#/a/b" against the root; nullptr when it does not resolve.
  const ojson* resolve(const std::string& ref) const {
    if (ref == "#") return &root;
    if (ref.size() < 2 || ref[0] != '#' || ref[1] != '/') return nullptr;
    const ojson* cur = &root;
    std::size_t pos = 2;
    while (true) {
      std::size_t next = ref.find('/', pos);
      std::string tok = ref.substr(pos, next == std::string::npos ? std::string::npos : next - pos);
      std::string key;
      for (std::size_t i = 0; i < tok.size(); ++i) {
        if (tok[i] == '~' && i + 1 < tok.size() && (tok[i + 1] == '0' || tok[i + 1] == '1')) {
          key += tok[i + 1] == '0' ? '~' : '/';
          ++i;
        } else {
          key += tok[i];
        }
      }
      if (cur->is_object()) {
        auto it = cur->find(key);
        if (it == cur->end()) return nullptr;
        cur = &*it;
      } else if (cur->is_array()) {
        if (key.empty() || key.find_first_not_of("0123456789") != std::string::npos || key.size() > 9) return nullptr;
        std::size_t i = std::stoul(key);
        if (i >= cur->size()) return nullptr;
        cur = &(*cur)[i];
      } else {
        return nullptr;
      }
      if (next == std::string::npos) return cur;
      pos = next + 1;
    }
  }

  ojson schema(const ojson& node, const std::string& ptr, int depth) const {
    if (depth > kMaxDepth) invalid("the schema nests deeper than " + std::to_string(kMaxDepth) + " levels", ptr);
    if (node.is_boolean()) return node;
    if (!node.is_object()) invalid("the schema is not an object", ptr);
    ojson out = ojson::object();
    for (auto it = node.begin(); it != node.end(); ++it) {
      const std::string& kw = it.key();
      const ojson& v = it.value();
      const std::string here = ptr + "/" + escape_token(kw);
      if (kAnnotations.count(kw)) continue;
      if (kw == "type") {
        auto check = [&](const ojson& t) {
          if (!t.is_string() || !kTypes.count(t.get<std::string>()))
            invalid("'type' must be one of object, array, string, number, integer, boolean, null", ptr);
        };
        if (v.is_array()) {
          if (v.empty()) invalid("'type' is an empty list", ptr);
          for (const auto& t : v) check(t);
        } else {
          check(v);
        }
        out[kw] = v;
      } else if (kw == "properties") {
        if (!v.is_object()) invalid("'properties' is not an object", ptr);
        ojson props = ojson::object();
        for (auto p = v.begin(); p != v.end(); ++p)
          props[p.key()] = schema(p.value(), here + "/" + escape_token(p.key()), depth + 1);
        out[kw] = std::move(props);
      } else if (kw == "required") {
        if (!v.is_array()) invalid("'required' is not a list", ptr);
        for (const auto& r : v)
          if (!r.is_string()) invalid("'required' lists a name that is not a string", ptr);
        out[kw] = v;
      } else if (kw == "additionalProperties") {
        if (v.is_boolean()) out[kw] = v;
        else if (v.is_object()) out[kw] = schema(v, here, depth + 1);
        else invalid("'additionalProperties' is neither a boolean nor a schema", ptr);
      } else if (kw == "items") {
        if (!v.is_object() && !v.is_boolean()) invalid("'items' is not one schema", ptr);
        out[kw] = schema(v, here, depth + 1);
      } else if (kw == "minItems" || kw == "maxItems") {  // an array's bounds, compiled by XGrammar as a repetition
        if (!v.is_number_integer() || v.get<std::int64_t>() < 0 || v.get<std::uint64_t>() > kMaxItemsBound)
          invalid("'" + kw + "' is not an integer from 0 to " + std::to_string(kMaxItemsBound), ptr);
        out[kw] = v;
      } else if (kw == "enum") {
        if (!v.is_array() || v.empty()) invalid("'enum' is not a non-empty list", ptr);
        out[kw] = v;
      } else if (kw == "const") {
        out[kw] = v;
      } else if (kw == "anyOf") {
        if (!v.is_array() || v.empty()) invalid("'anyOf' is not a non-empty list", ptr);
        ojson alts = ojson::array();
        for (std::size_t i = 0; i < v.size(); ++i) alts.push_back(schema(v[i], here + "/" + std::to_string(i), depth + 1));
        out[kw] = std::move(alts);
      } else if (kw == "$ref") {
        if (!v.is_string()) invalid("'$ref' is not a string", ptr);
        const std::string ref = v.get<std::string>();
        if (ref.empty() || ref[0] != '#') unsupported(kw, ptr);  // a non-local reference
        // XGrammar resolves references into $defs and definitions (and the root); others are refused here.
        if (ref != "#" && ref.rfind("#/$defs/", 0) != 0 && ref.rfind("#/definitions/", 0) != 0)
          invalid("the reference '" + ref + "' does not point into $defs or definitions", ptr);
        if (!resolve(ref)) invalid("the reference '" + ref + "' does not resolve", ptr);
        out[kw] = v;
      } else if (kw == "$defs" || kw == "definitions") {
        if (!v.is_object()) invalid("'" + kw + "' is not an object", ptr);
        ojson defs = ojson::object();
        for (auto d = v.begin(); d != v.end(); ++d)
          defs[d.key()] = schema(d.value(), here + "/" + escape_token(d.key()), depth + 1);
        out[kw] = std::move(defs);
      } else {
        unsupported(kw, ptr);
      }
    }
    if (out.contains("minItems") && out.contains("maxItems") &&
        out["minItems"].get<std::uint64_t>() > out["maxItems"].get<std::uint64_t>())
      invalid("'minItems' is greater than 'maxItems'", ptr);
    return out;
  }
};

}  // namespace

ojson canonical_schema(const ojson& schema, const std::string& param) {
  Walker w{schema, param};
  if (!schema.is_object()) w.invalid("the schema is not an object", "");
  // A $ref target is checked where it is defined ($defs, definitions); a $ref elsewhere is refused above.
  return w.schema(schema, "", 0);
}

std::string ws_rules(const std::string& name) {
  // At most one newline, then at most kMaxWsBlanks blanks. Blanks before the newline (trailing spaces) are not
  // allowed: a rule that counted blanks on both sides of the newline made every compiled schema ~9x larger
  // (63 MB against 6.9 MB for the bench's Codex schema) and its compile ~5x slower.
  return name + " ::= \"\\n\"? [ \\t]{0," + std::to_string(kMaxWsBlanks) + "}\n";
}

std::string schema_ebnf(const ojson& canonical, bool strict) {
  // XGrammar writes every whitespace slot as whitespace{0, N} with N the library's total-count limit; a sentinel
  // N that appears nowhere else lets us swap each slot for the bounded rule.
  constexpr int kSentinel = 7919;
  const auto g = xgrammar::Grammar::FromJSONSchema(canonical.dump(), /*any_whitespace=*/true, std::nullopt,
                                                   std::nullopt, strict, kSentinel);
  std::string text = g.ToString();
  static const std::regex slot(R"(whitespace(_[0-9]+)?\{0, 7919\})");
  text = std::regex_replace(text, slot, "ragon_ws");
  if (text.find("{0, 7919}") != std::string::npos)
    throw Error("invalid_json_schema", "internal: a whitespace slot of an unexpected shape");
  return text + ws_rules("ragon_ws");
}

}  // namespace ragon::grammar::detail
