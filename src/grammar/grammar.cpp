// The vocabulary bridge, the matcher and the compiled-grammar cache (see grammar.hpp).
#include "grammar/grammar.hpp"

#include <algorithm>
#include <list>
#include <unordered_map>

#include <dlpack/dlpack.h>
#include <xgrammar/xgrammar.h>

namespace ragon::grammar {

// ---------------------------------------------------------------------------------------------------------
// Vocab

struct Vocab::Impl {
  explicit Impl(xgrammar::TokenizerInfo i) : info(std::move(i)) {}
  xgrammar::TokenizerInfo info;
};

namespace {
// Added tokens the tokenizer marks special=False but that are still text the model writes; everything else the
// tokenizer marks special is a control token and never allowed under a grammar.
bool is_text_token(const std::string& bytes) {
  return bytes == "<tool_call>" || bytes == "</tool_call>" || bytes == "<think>" || bytes == "</think>";
}
}  // namespace

Vocab::Vocab(const Tokenizer& tok, int vocab_size) : size_(vocab_size) {
  const int n = tok.vocab_size();
  if (n > vocab_size) throw std::invalid_argument("grammar::Vocab: the tokenizer has more ids than vocab_size");
  for (const char* s : {"<|im_end|>", "<|endoftext|>"}) {
    const int id = tok.token_id(s);
    if (id < 0) throw std::invalid_argument(std::string("grammar::Vocab: no ") + s + " in the tokenizer");
    stop_ids_.push_back(id);
  }
  never_.assign(vocab_size, false);
  std::vector<std::string> encoded(n);
  for (int id = 0; id < n; ++id) {
    std::string bytes = tok.token_bytes(id);
    const bool stop = std::find(stop_ids_.begin(), stop_ids_.end(), id) != stop_ids_.end();
    // XGrammar treats an empty string as a special token, never allowed. Only control tokens go that way.
    if (!stop && (bytes.empty() || (tok.is_special(id) && !is_text_token(bytes)))) {
      never_[id] = true;
      bytes.clear();
    }
    encoded[id] = std::move(bytes);
  }
  for (int id = n; id < vocab_size; ++id) never_[id] = true;  // padding: XGrammar marks these special too
  impl_ = std::make_unique<Impl>(xgrammar::TokenizerInfo(
      encoded, xgrammar::VocabType::RAW, vocab_size,
      std::vector<int32_t>(stop_ids_.begin(), stop_ids_.end())));
}

Vocab::Vocab(const std::vector<std::string>& token_bytes, const std::vector<int>& stop_ids, int vocab_size)
    : size_(vocab_size), stop_ids_(stop_ids) {
  never_.assign(vocab_size, false);
  std::vector<std::string> encoded = token_bytes;
  for (int id = 0; id < vocab_size; ++id)
    if (id >= static_cast<int>(encoded.size()) || encoded[id].empty()) never_[id] = true;
  for (int id : stop_ids_) never_[id] = false;
  impl_ = std::make_unique<Impl>(xgrammar::TokenizerInfo(
      encoded, xgrammar::VocabType::RAW, vocab_size, std::vector<int32_t>(stop_ids_.begin(), stop_ids_.end())));
}

Vocab::~Vocab() = default;

bool Vocab::never_allowed(int id) const { return id < 0 || id >= size_ || never_[id]; }

// ---------------------------------------------------------------------------------------------------------
// Grammar

struct Grammar::Impl {
  Impl(xgrammar::CompiledGrammar c, std::string text, int vocab_size, std::vector<int> stops)
      : compiled(std::move(c)), ebnf(std::move(text)), bytes(compiled.MemorySizeBytes()), vocab(vocab_size),
        stop_ids(std::move(stops)), special(vocab_size, false) {
    for (int id : compiled.GetTokenizerInfo().GetSpecialTokenIds())
      if (id >= 0 && id < vocab_size) special[id] = true;
    for (int id : stop_ids) special[id] = false;
  }
  xgrammar::CompiledGrammar compiled;
  std::string ebnf;
  std::size_t bytes;
  int vocab;
  std::vector<int> stop_ids;
  std::vector<bool> special;  // control tokens and padding: refused without asking XGrammar (which logs each one)
};

std::size_t Grammar::memory_bytes() const { return impl_->bytes; }
const std::string& Grammar::ebnf() const { return impl_->ebnf; }

// ---------------------------------------------------------------------------------------------------------
// Matcher

struct Matcher::Impl {
  Impl(std::shared_ptr<const Grammar::Impl> g, xgrammar::GrammarMatcher m)
      : grammar(std::move(g)), matcher(std::move(m)), mask(mask_words(grammar->vocab)) {}
  std::shared_ptr<const Grammar::Impl> grammar;
  xgrammar::GrammarMatcher matcher;
  std::vector<int32_t> mask;  // the current state's mask, for allowed() and can_stop()
  bool mask_valid = false;

  void fill_into(int32_t* words) {
    int64_t shape[1] = {static_cast<int64_t>(mask.size())};
    DLTensor t{};
    t.data = words;
    t.device = DLDevice{kDLCPU, 0};
    t.ndim = 1;
    t.dtype = DLDataType{kDLInt, 32, 1};
    t.shape = shape;
    t.strides = nullptr;
    t.byte_offset = 0;
    last_fill_masks = matcher.FillNextTokenBitmask(&t, 0);
  }
  const std::vector<int32_t>& current() {
    if (!mask_valid) {
      fill_into(mask.data());
      mask_valid = true;
    }
    return mask;
  }
  bool last_fill_masks = true;
};

Matcher::Matcher(const Grammar& g)
    : impl_(std::make_unique<Impl>(g.shared(), xgrammar::GrammarMatcher(g.impl().compiled))) {}

Matcher::Matcher(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Matcher::~Matcher() = default;
Matcher::Matcher(Matcher&&) noexcept = default;
Matcher& Matcher::operator=(Matcher&&) noexcept = default;

bool Matcher::accept(int id) {
  if (id < 0 || id >= impl_->grammar->vocab || impl_->grammar->special[id]) return false;
  const bool ok = impl_->matcher.AcceptToken(id);
  if (ok) impl_->mask_valid = false;
  return ok;
}

bool Matcher::accept_string(std::string_view s) {
  const bool ok = impl_->matcher.AcceptString(std::string(s));
  if (ok) impl_->mask_valid = false;
  return ok;
}

void Matcher::rollback(int n) {
  if (n <= 0) return;
  impl_->matcher.Rollback(n);
  impl_->mask_valid = false;
}

bool Matcher::fill(int32_t* words) {
  impl_->fill_into(words);
  return impl_->last_fill_masks;
}

bool Matcher::allowed(int id) {
  if (id < 0 || id >= impl_->grammar->vocab) return false;
  const auto& m = impl_->current();
  return (static_cast<uint32_t>(m[id >> 5]) >> (id & 31)) & 1u;
}

bool Matcher::terminated() const { return impl_->matcher.IsTerminated(); }

bool Matcher::can_stop() {
  if (terminated()) return false;
  for (int id : impl_->grammar->stop_ids)
    if (allowed(id)) return true;
  return false;
}

Matcher Matcher::fork() const {
  auto copy = std::make_unique<Impl>(impl_->grammar, impl_->matcher.Fork());
  copy->mask = impl_->mask;
  copy->mask_valid = impl_->mask_valid;
  return Matcher(std::move(copy));
}

void Matcher::reset() {
  impl_->matcher.Reset();
  impl_->mask_valid = false;
}

// ---------------------------------------------------------------------------------------------------------
// Compiler and the LRU cache of compiled grammars

struct Compiler::Impl {
  Impl(const Vocab& v, std::size_t budget, int threads)
      : vocab(v), compiler(v.impl().info, threads, /*cache_enabled=*/false), budget(budget) {}

  const Vocab& vocab;
  xgrammar::GrammarCompiler compiler;
  std::size_t budget;

  struct Entry {
    std::string key;
    std::shared_ptr<const Grammar::Impl> grammar;
  };
  mutable std::mutex mu;
  std::list<Entry> lru;  // front = most recent
  std::unordered_map<std::string, std::list<Entry>::iterator> index;  // hashed by the full key
  Stats stats;

  std::shared_ptr<const Grammar::Impl> find(const std::string& key) {
    std::lock_guard<std::mutex> lock(mu);
    auto it = index.find(key);
    if (it == index.end()) {
      ++stats.misses;
      return nullptr;
    }
    ++stats.hits;
    lru.splice(lru.begin(), lru, it->second);
    return it->second->grammar;
  }

  void insert(const std::string& key, std::shared_ptr<const Grammar::Impl> g) {
    std::lock_guard<std::mutex> lock(mu);
    if (index.count(key)) return;
    lru.push_front(Entry{key, g});
    index[key] = lru.begin();
    stats.bytes += g->bytes;
    while (stats.bytes > budget && lru.size() > 1) {
      auto& last = lru.back();
      stats.bytes -= last.grammar->bytes;
      index.erase(last.key);
      lru.pop_back();
      ++stats.evictions;
    }
    stats.entries = lru.size();
  }

  // kind and the whitespace bound are part of the key; the text is canonical for its kind.
  template <class Build>
  Grammar get(char kind, const std::string& text, Build&& build) {
    std::string key;
    key.reserve(text.size() + 8);
    key += kind;
    key += '\x1f';
    key += std::to_string(kMaxWsBlanks);
    key += '\x1f';
    key += text;
    if (auto g = find(key)) return Grammar(g);
    std::pair<xgrammar::CompiledGrammar, std::string> built = build();
    auto g = std::make_shared<const Grammar::Impl>(std::move(built.first), std::move(built.second),
                                                    vocab.size(), vocab.stop_ids());
    insert(key, g);
    return Grammar(g);
  }
};

Compiler::Compiler(const Vocab& vocab, std::size_t cache_bytes, int threads)
    : impl_(std::make_unique<Impl>(vocab, cache_bytes, threads)) {}
Compiler::~Compiler() = default;

Compiler::Stats Compiler::stats() const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  return impl_->stats;
}

void Compiler::clear_cache() {
  std::lock_guard<std::mutex> lock(impl_->mu);
  impl_->lru.clear();
  impl_->index.clear();
  impl_->stats.bytes = 0;
  impl_->stats.entries = 0;
}

namespace {
std::string first_line(const char* what) {
  std::string s(what);
  // XGrammar prefixes "[hh:mm:ss] /path/file.cc:line: "; keep the message only.
  if (auto p = s.find(".cc:"); p != std::string::npos) {
    auto q = s.find(": ", p + 4);
    if (q != std::string::npos) s = s.substr(q + 2);
  }
  if (auto nl = s.find('\n'); nl != std::string::npos) s.resize(nl);
  return s;
}

xgrammar::CompiledGrammar compile_ebnf(xgrammar::GrammarCompiler& c, const std::string& ebnf,
                                       const std::string& code, const std::string& param) {
  try {
    return c.CompileGrammar(xgrammar::Grammar::FromEBNF(ebnf, "root"));
  } catch (const std::exception& e) {
    throw Error(code, param + ": " + first_line(e.what()));
  }
}
}  // namespace

Grammar Compiler::json_schema(const nlohmann::ordered_json& schema, const std::string& param) {
  const nlohmann::ordered_json canon = detail::canonical_schema(schema, param);
  return impl_->get('j', canon.dump(), [&] {
    std::string ebnf;
    try {
      ebnf = detail::schema_ebnf(canon, true);
    } catch (const Error&) {
      throw;
    } catch (const std::exception& e) {
      throw Error("invalid_json_schema", param + ": " + first_line(e.what()) + " at \"\"");
    }
    auto cg = compile_ebnf(impl_->compiler, ebnf, "invalid_json_schema", param);
    return std::make_pair(std::move(cg), std::move(ebnf));
  });
}

Grammar Compiler::json_schema_text(std::string_view text, const std::string& param) {
  nlohmann::ordered_json schema = nlohmann::ordered_json::parse(text, nullptr, false);
  if (schema.is_discarded()) throw Error("invalid_json_schema", param + ": the schema is not valid JSON at \"\"");
  return json_schema(schema, param);
}

Grammar Compiler::json_object() {
  return impl_->get('o', "", [&] {
    std::string ebnf = detail::schema_ebnf(nlohmann::ordered_json{{"type", "object"}}, false);
    auto cg = compile_ebnf(impl_->compiler, ebnf, "invalid_json_schema", "response_format");
    return std::make_pair(std::move(cg), std::move(ebnf));
  });
}

Grammar Compiler::lark(std::string_view text, const std::string& param) {
  return impl_->get('l', std::string(text), [&] {
    std::string ebnf = detail::lark_to_ebnf(text, param);
    auto cg = compile_ebnf(impl_->compiler, ebnf, "invalid_grammar", param);
    return std::make_pair(std::move(cg), std::move(ebnf));
  });
}

Grammar Compiler::regex(std::string_view pattern, const std::string& param) {
  return impl_->get('r', std::string(pattern), [&] {
    std::string ebnf = "root ::= " + detail::regex_to_ebnf(pattern, param) + "\n";
    auto cg = compile_ebnf(impl_->compiler, ebnf, "invalid_grammar", param);
    return std::make_pair(std::move(cg), std::move(ebnf));
  });
}

namespace detail {
std::string tool_calls_tag(const ToolCallSpec& spec);  // tool_calls.cpp
}

Grammar Compiler::response_format(const std::optional<nlohmann::ordered_json>& schema, const std::string& param) {
  const std::string tag = detail::response_format_tag(schema, param);  // the schema checks throw a 400 first
  return impl_->get('r', tag, [&] {
    try {
      return std::make_pair(impl_->compiler.CompileStructuralTag(tag), tag);
    } catch (const std::exception& e) {
      throw Error("invalid_json_schema", param + ": " + first_line(e.what()) + " at \"\"");
    }
  });
}

Grammar Compiler::tool_calls(const ToolCallSpec& spec) {
  const std::string tag = detail::tool_calls_tag(spec);
  return impl_->get('t', tag, [&] {
    try {
      return std::make_pair(impl_->compiler.CompileStructuralTag(tag), tag);
    } catch (const std::exception& e) {
      throw Error("invalid_tools", "tools: " + first_line(e.what()));
    }
  });
}

void detail::check_ebnf(const std::string& ebnf) {
  try {
    (void)xgrammar::Grammar::FromEBNF(ebnf, "root");
  } catch (const std::exception& e) {
    throw Error("invalid_grammar", first_line(e.what()));
  }
}

}  // namespace ragon::grammar
