// Structured output, stage 1 (BACKLOG §19.17): grammars compiled against the vocabulary and token masks, on the
// CPU. The engine side (where the mask is applied, the API fields) is stage 2; this library only depends on the
// tokenizer and on the XGrammar prebuilt (scripts/build_xgrammar.sh), which stays behind this header.
//
// Masks are rows of int32 words, bit (id % 32) of word (id / 32) set when the token is allowed: the layout
// XGrammar fills and the sampler will read. 248,320 ids make 7,760 words (31,040 bytes) per row.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "tokenizer/tokenizer.hpp"

namespace ragon::grammar {

inline constexpr int kVocabSize = 248320;  // the model's logits width; ids past the tokenizer are padding
inline constexpr int mask_words(int vocab_size) { return (vocab_size + 31) / 32; }

// Every JSON whitespace slot allows at most one newline and then at most 16 spaces or tabs (vllm#38696:
// unbounded whitespace lets a model loop on newlines until max_tokens). No carriage return, no blank before
// the newline.
inline constexpr int kMaxWsBlanks = 16;

// A request the library refuses: the HTTP layer answers 400 invalid_request_error with `code` and `what()`.
// Codes: unsupported_json_schema, invalid_json_schema, unsupported_grammar, invalid_grammar, invalid_tools.
struct Error : std::runtime_error {
  Error(std::string c, const std::string& message) : std::runtime_error(message), code(std::move(c)) {}
  std::string code;
};

// The vocabulary as the grammar engine sees it: the raw bytes of every id, the stop ids <|im_end|> and
// <|endoftext|>. Control tokens (<|im_start|>, vision and audio markers) and padding are never allowed;
// <tool_call>, </tool_call>, <think> and </think> are ordinary byte strings, so the model may write them
// either as the added token or as byte pieces.
class Vocab {
 public:
  explicit Vocab(const Tokenizer& tok, int vocab_size = kVocabSize);
  // A synthetic vocabulary (the fuzz target): an empty string marks a never-allowed id.
  Vocab(const std::vector<std::string>& token_bytes, const std::vector<int>& stop_ids, int vocab_size);
  ~Vocab();
  int size() const { return size_; }
  int words() const { return mask_words(size_); }
  const std::vector<int>& stop_ids() const { return stop_ids_; }
  bool never_allowed(int id) const;  // control tokens and padding
  struct Impl;
  const Impl& impl() const { return *impl_; }

 private:
  int size_;
  std::vector<int> stop_ids_;
  std::vector<bool> never_;
  std::unique_ptr<Impl> impl_;
};

// A grammar compiled against the vocabulary; immutable and shared by every matcher (and the cache).
class Grammar {
 public:
  struct Impl;
  explicit Grammar(std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}
  std::size_t memory_bytes() const;
  const std::string& ebnf() const;  // the grammar text handed to the compiler (diagnostics, tests)
  const Impl& impl() const { return *impl_; }
  const std::shared_ptr<const Impl>& shared() const { return impl_; }

 private:
  std::shared_ptr<const Impl> impl_;
};

// One request's position in its grammar.
class Matcher {
 public:
  explicit Matcher(const Grammar& g);
  ~Matcher();
  Matcher(Matcher&&) noexcept;
  Matcher& operator=(Matcher&&) noexcept;

  bool accept(int id);                   // false (state unchanged) when the grammar refuses the token
  bool accept_string(std::string_view s);  // bytes, not tokens (tests)
  void rollback(int n = 1);              // undo the last n accepted tokens
  // Fills `words` (mask_words(vocab) int32 words) with the allowed set. Returns false when every token is
  // allowed (the row needs no masking); the buffer is filled either way.
  bool fill(int32_t* words);
  bool allowed(int id);                  // one id against the current state's mask
  bool terminated() const;               // the stop token was accepted
  bool can_stop();                       // the grammar is complete: a stop token is allowed now
  Matcher fork() const;                  // an independent copy at the same state
  void reset();

  struct Impl;
  Impl& impl() { return *impl_; }

 private:
  explicit Matcher(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

// Tools for the tool-call grammar (§2.2, §2.3). A function tool lists its declared parameter names; a custom
// tool has the single parameter `input` whose value follows `format`.
struct Tool {
  enum class Kind { kFunction, kCustom } kind = Kind::kFunction;
  std::string name;
  std::vector<std::string> params;        // kFunction: the declared property names, in schema order
  enum class Format { kText, kLark, kRegex } format = Format::kText;  // kCustom only
  std::string definition;                 // kLark: the grammar; kRegex: the pattern
};

struct ToolCallSpec {
  std::vector<Tool> tools;
  enum class Choice { kAuto, kRequired, kNamed } choice = Choice::kAuto;
  std::string named;                      // kNamed: the function tool_choice names
  // response_format with tools offered (vllm#39929): the payload is `json | calls`. Set at most one.
  std::optional<nlohmann::ordered_json> json_schema;
  bool json_object = false;
  std::string json_param = "response_format.json_schema.schema";  // names the schema in a 400
};

// The exact prefix the output parser reads before a custom tool's input; the structural tag triggers on it.
std::string custom_tool_prefix(const std::string& name);

class Compiler {
 public:
  // cache_bytes: the LRU budget for compiled grammars, counted with Grammar::memory_bytes().
  explicit Compiler(const Vocab& vocab, std::size_t cache_bytes = std::size_t{256} << 20, int threads = 1);
  ~Compiler();

  // `param` names the schema in a 400 (response_format.json_schema.schema, text.format.schema).
  Grammar json_schema(const nlohmann::ordered_json& schema, const std::string& param);
  Grammar json_schema_text(std::string_view schema_text, const std::string& param);
  Grammar json_object();
  // `param` names the grammar in a 400 (tools[<i>].format).
  Grammar lark(std::string_view text, const std::string& param);
  Grammar regex(std::string_view pattern, const std::string& param);
  Grammar tool_calls(const ToolCallSpec& spec);
  // response_format with no tools offered (structured output stage 2): at most two newlines, then the JSON
  // answer. `schema` set: response_format json_schema (`param` names it in a 400); unset: json_object.
  Grammar response_format(const std::optional<nlohmann::ordered_json>& schema, const std::string& param);

  struct Stats {
    std::size_t hits = 0, misses = 0, evictions = 0, bytes = 0, entries = 0;
  };
  Stats stats() const;
  void clear_cache();

  struct Impl;

 private:
  std::unique_ptr<Impl> impl_;
};

// Pieces exposed for tests and the fuzz target (no compile against the vocabulary).
namespace detail {
// Checks the §2.1 subset and returns the schema without annotations; throws Error.
nlohmann::ordered_json canonical_schema(const nlohmann::ordered_json& schema, const std::string& param);
// The JSON grammar (EBNF, root rule `root`) for a canonical schema with the bounded whitespace rule.
std::string schema_ebnf(const nlohmann::ordered_json& canonical, bool strict);
// The Lark subset (§2.3) to EBNF with root rule `root`; throws Error (unsupported_grammar, invalid_grammar).
std::string lark_to_ebnf(std::string_view lark, const std::string& param);
// A regex of the §2.3 subset as an EBNF expression; throws Error with line 1 and the column.
std::string regex_to_ebnf(std::string_view pattern, const std::string& param);
// The bounded whitespace rule, named `name`, and its helper rules.
std::string ws_rules(const std::string& name);
// The structural tag Compiler::response_format compiles (the newline lead, then the JSON grammar).
std::string response_format_tag(const std::optional<nlohmann::ordered_json>& schema, const std::string& param);
// Parses EBNF (syntax only); throws Error invalid_grammar. Used by the fuzz target.
void check_ebnf(const std::string& ebnf);
}  // namespace detail

}  // namespace ragon::grammar
