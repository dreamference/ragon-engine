// The Lark subset of §2.3 and its regexes, translated to XGrammar's EBNF. Our own translator, not XGrammar's,
// so the refusals carry our codes and a line and column.
//
// Lark lexes terminals greedily; this grammar is character-level. The two accept the same language for the
// grammars Codex sends (apply_patch, exec); the mask needs the language, not the parse tree.
#include <algorithm>
#include <cstdint>
#include <map>
#include <set>

#include "grammar/grammar.hpp"

namespace ragon::grammar::detail {

namespace {

constexpr int kMaxNesting = 64;       // groups inside groups (regex and rule expressions)
constexpr int kMaxRepeat = 1000;      // {m,n} bounds
constexpr uint32_t kMaxCp = 0x10FFFF;

struct Cp {
  uint32_t c;
  int line, col;
};

struct Fail {
  const std::string& param;
  [[noreturn]] void unsupported(const std::string& what, int line, int col) const {
    throw Error("unsupported_grammar", param + ": " + what + " at line " + std::to_string(line) + " column " +
                                           std::to_string(col) + " is not supported");
  }
  [[noreturn]] void invalid(const std::string& what, int line, int col) const {
    throw Error("invalid_grammar",
                param + ": " + what + " at line " + std::to_string(line) + " column " + std::to_string(col));
  }
};

// UTF-8 to code points with their line and column (1-based, in code points).
std::vector<Cp> decode(std::string_view s, const Fail& fail) {
  std::vector<Cp> out;
  out.reserve(s.size() + 1);
  int line = 1, col = 1;
  for (std::size_t i = 0; i < s.size();) {
    const unsigned char b = s[i];
    uint32_t c;
    int n;
    if (b < 0x80) c = b, n = 1;
    else if ((b & 0xE0) == 0xC0) c = b & 0x1F, n = 2;
    else if ((b & 0xF0) == 0xE0) c = b & 0x0F, n = 3;
    else if ((b & 0xF8) == 0xF0) c = b & 0x07, n = 4;
    else fail.invalid("a byte that is not UTF-8", line, col);
    if (i + n > s.size()) fail.invalid("a byte that is not UTF-8", line, col);
    for (int k = 1; k < n; ++k) {
      const unsigned char cb = s[i + k];
      if ((cb & 0xC0) != 0x80) fail.invalid("a byte that is not UTF-8", line, col);
      c = (c << 6) | (cb & 0x3F);
    }
    if (c > kMaxCp || (c >= 0xD800 && c <= 0xDFFF) || (n == 2 && c < 0x80) || (n == 3 && c < 0x800) ||
        (n == 4 && c < 0x10000))
      fail.invalid("a byte that is not UTF-8", line, col);
    out.push_back({c, line, col});
    if (c == '\n') ++line, col = 1;
    else ++col;
    i += n;
  }
  out.push_back({0xFFFFFFFFu, line, col});  // end marker
  return out;
}

constexpr uint32_t kEnd = 0xFFFFFFFFu;

std::string hex(uint32_t c, int digits) {
  static const char* d = "0123456789ABCDEF";
  std::string s(digits, '0');
  for (int i = digits - 1; i >= 0; --i, c >>= 4) s[i] = d[c & 15];
  return s;
}

// One code point inside an EBNF string literal or character class.
std::string ebnf_char(uint32_t c, bool in_class) {
  if (c >= 0x20 && c < 0x7F) {
    const char ch = static_cast<char>(c);
    if (ch == '\\' || ch == '"' || (in_class && (ch == ']' || ch == '-' || ch == '^' || ch == '[')))
      return std::string("\\") + ch;
    return std::string(1, ch);
  }
  if (c <= 0xFFFF) return "\\u" + hex(c, 4);
  return "\\U" + hex(c, 8);
}

std::string ebnf_literal(const std::vector<uint32_t>& cps) {
  std::string out = "\"";
  for (uint32_t c : cps) out += ebnf_char(c, false);
  return out + "\"";
}

// ---------------------------------------------------------------------------------------------------------
// Character sets as sorted, disjoint code-point intervals.

using Ranges = std::vector<std::pair<uint32_t, uint32_t>>;

Ranges normalize(Ranges r) {
  std::sort(r.begin(), r.end());
  Ranges out;
  for (auto [a, b] : r) {
    if (!out.empty() && a <= out.back().second + 1) out.back().second = std::max(out.back().second, b);
    else out.push_back({a, b});
  }
  return out;
}

Ranges complement(const Ranges& in) {
  Ranges n = normalize(in), out;
  uint32_t next = 0;
  for (auto [a, b] : n) {
    if (a > next) out.push_back({next, a - 1});
    next = b + 1;
  }
  if (next <= kMaxCp) out.push_back({next, kMaxCp});
  return out;
}

std::string ebnf_class(const Ranges& in) {
  Ranges r = normalize(in);
  std::string out = "[";
  for (auto [a, b] : r) {
    out += ebnf_char(a, true);
    if (b != a) out += "-" + ebnf_char(b, true);
  }
  return out + "]";
}

Ranges digit() { return {{'0', '9'}}; }
Ranges word() { return {{'0', '9'}, {'A', 'Z'}, {'a', 'z'}, {'_', '_'}}; }
Ranges space() { return {{'\t', '\r'}, {' ', ' '}}; }  // \t \n \v \f \r and space

// ---------------------------------------------------------------------------------------------------------
// Regex subset: classes, \s \S \d \D \w \W \r \n \t \f \v, escaped punctuation, '.', groups ( ) and (?: ),
// alternation, ? * + {m} {m,} {m,n}. Lookaround, backreferences, lazy and possessive quantifiers, anchors and
// flags are refused.

class RegexParser {
 public:
  RegexParser(const std::vector<Cp>& cps, std::size_t begin, std::size_t end, const Fail& fail)
      : s_(cps), pos_(begin), end_(end), fail_(fail) {}

  std::string parse() {
    std::string out = alternation(0);
    if (pos_ < end_) fail_.invalid("an unbalanced ')' in the regex", s_[pos_].line, s_[pos_].col);
    return out;
  }

 private:
  const std::vector<Cp>& s_;
  std::size_t pos_, end_;
  const Fail& fail_;

  uint32_t peek(std::size_t k = 0) const { return pos_ + k < end_ ? s_[pos_ + k].c : kEnd; }
  const Cp& here() const { return s_[std::min(pos_, end_)]; }

  std::string alternation(int depth) {
    if (depth > kMaxNesting) fail_.unsupported("regex groups nested deeper than 64", here().line, here().col);
    std::vector<std::string> alts{concat(depth)};
    while (peek() == '|') {
      ++pos_;
      alts.push_back(concat(depth));
    }
    if (alts.size() == 1) return alts[0];
    std::string out = "(";
    for (std::size_t i = 0; i < alts.size(); ++i) out += (i ? " | " : "") + alts[i];
    return out + ")";
  }

  std::string concat(int depth) {
    std::vector<std::string> parts;
    while (pos_ < end_ && peek() != '|' && peek() != ')') parts.push_back(piece(depth));
    if (parts.empty()) return "\"\"";
    std::string out;
    for (std::size_t i = 0; i < parts.size(); ++i) out += (i ? " " : "") + parts[i];
    return parts.size() == 1 ? out : "(" + out + ")";
  }

  // {m}, {m,}, {m,n}; false (nothing consumed) when the brace is a literal, as in Python.
  bool counted(int& lo, int& hi) {
    std::size_t p = pos_ + 1;
    auto number = [&](int& v) {
      std::size_t start = p;
      long long x = 0;
      while (p < end_ && s_[p].c >= '0' && s_[p].c <= '9') {
        x = x * 10 + (s_[p].c - '0');
        if (x > kMaxRepeat) x = kMaxRepeat + 1;
        ++p;
      }
      v = static_cast<int>(x);
      return p > start;
    };
    if (!number(lo)) return false;
    hi = lo;
    if (p < end_ && s_[p].c == ',') {
      ++p;
      if (!number(hi)) hi = -1;
    }
    if (p >= end_ || s_[p].c != '}') return false;
    const Cp& at = here();
    if (lo > kMaxRepeat || hi > kMaxRepeat) fail_.unsupported("a regex repetition above 1000", at.line, at.col);
    if (hi >= 0 && hi < lo) fail_.invalid("a regex repetition {m,n} with n < m", at.line, at.col);
    pos_ = p + 1;
    return true;
  }

  std::string piece(int depth) {
    const Cp at = here();
    std::string a = atom(depth);
    bool quantified = false;
    while (true) {
      const uint32_t q = peek();
      std::string suffix;
      int lo = 0, hi = 0;
      if (q == '*' || q == '+' || q == '?') {
        suffix = std::string(1, static_cast<char>(q));
        ++pos_;
      } else if (q == '{' && counted(lo, hi)) {
        suffix = "{" + std::to_string(lo) + (hi == lo ? "" : "," + (hi < 0 ? std::string() : std::to_string(hi))) + "}";
      } else {
        break;
      }
      if (quantified) fail_.unsupported("a repeated regex quantifier", at.line, at.col);
      quantified = true;
      if (peek() == '?') fail_.unsupported("a lazy regex quantifier", here().line, here().col);
      if (peek() == '+') fail_.unsupported("a possessive regex quantifier", here().line, here().col);
      a = "(" + a + ")" + suffix;
    }
    return a;
  }

  // An escape after '\' at pos_ (pos_ on the escaped char); returns the set, or a single code point in `one`.
  Ranges escape(bool in_class, uint32_t& one) {
    const Cp at = s_[pos_ - 1];
    const uint32_t e = peek();
    if (e == kEnd) fail_.invalid("a regex ending in '\\'", at.line, at.col);
    ++pos_;
    one = kEnd;
    switch (e) {
      case 'd': return digit();
      case 'D': return complement(digit());
      case 'w': return word();
      case 'W': return complement(word());
      case 's': return space();
      case 'S': return complement(space());
      case 'n': one = '\n'; return {};
      case 'r': one = '\r'; return {};
      case 't': one = '\t'; return {};
      case 'f': one = '\f'; return {};
      case 'v': one = '\v'; return {};
      case 'x':
      case 'u':
      case 'U': {
        const int digits = e == 'x' ? 2 : e == 'u' ? 4 : 8;
        uint32_t v = 0;
        for (int i = 0; i < digits; ++i) {
          const uint32_t h = peek();
          int d = h >= '0' && h <= '9' ? int(h - '0') : h >= 'a' && h <= 'f' ? int(h - 'a' + 10)
                                     : h >= 'A' && h <= 'F' ? int(h - 'A' + 10) : -1;
          if (d < 0) fail_.invalid("a bad regex escape '\\" + std::string(1, char(e)) + "'", at.line, at.col);
          v = v * 16 + d;
          ++pos_;
        }
        if (v > kMaxCp || (v >= 0xD800 && v <= 0xDFFF)) fail_.invalid("a regex escape outside Unicode", at.line, at.col);
        one = v;
        return {};
      }
      default:
        break;
    }
    if (e < 0x80 && !std::isalnum(static_cast<int>(e))) {  // escaped punctuation: the character itself
      one = e;
      return {};
    }
    (void)in_class;
    std::string shown = e < 0x80 ? std::string(1, static_cast<char>(e)) : "U+" + hex(e, 4);
    fail_.unsupported("the regex escape '\\" + shown + "'", at.line, at.col);
  }

  std::string atom(int depth) {
    const Cp at = here();
    const uint32_t c = peek();
    if (c == '(') {
      ++pos_;
      if (peek() == '?') {
        if (peek(1) == ':') {
          pos_ += 2;
        } else {
          const uint32_t k = peek(1);
          if (k == '=' || k == '!' || (k == '<' && (peek(2) == '=' || peek(2) == '!')))
            fail_.unsupported("a regex lookaround", at.line, at.col);
          if (k == 'P' && peek(2) == '=') fail_.unsupported("a regex backreference", at.line, at.col);
          if (k == 'P' || k == '<') fail_.unsupported("a named regex group", at.line, at.col);
          fail_.unsupported("regex flags", at.line, at.col);
        }
      }
      std::string inner = alternation(depth + 1);
      if (peek() != ')') fail_.invalid("an unclosed '(' in the regex", at.line, at.col);
      ++pos_;
      return "(" + inner + ")";
    }
    if (c == '[') return char_class(at);
    if (c == '.') {
      ++pos_;
      return ebnf_class(complement({{'\n', '\n'}}));
    }
    if (c == '^' || c == '$') fail_.unsupported("a regex anchor '" + std::string(1, char(c)) + "'", at.line, at.col);
    if (c == '*' || c == '+' || c == '?') fail_.invalid("a regex quantifier with nothing to repeat", at.line, at.col);
    if (c == '\\') {
      ++pos_;
      if (peek() >= '1' && peek() <= '9') fail_.unsupported("a regex backreference", at.line, at.col);
      uint32_t one;
      Ranges r = escape(false, one);
      if (one != kEnd) return ebnf_literal({one});
      return ebnf_class(r);
    }
    ++pos_;
    return ebnf_literal({c});
  }

  std::string char_class(const Cp& at) {
    ++pos_;  // '['
    bool negate = false;
    if (peek() == '^') negate = true, ++pos_;
    Ranges set;
    bool first = true;
    while (true) {
      uint32_t c = peek();
      if (c == kEnd) fail_.invalid("an unclosed '[' in the regex", at.line, at.col);
      if (c == ']' && !first) {
        ++pos_;
        break;
      }
      first = false;
      uint32_t lo;
      if (c == '\\') {
        ++pos_;
        Ranges r = escape(true, lo);
        if (lo == kEnd) {
          set.insert(set.end(), r.begin(), r.end());
          continue;
        }
      } else {
        if (c == '[' && peek(1) == ':') fail_.unsupported("a POSIX class in a regex", here().line, here().col);
        lo = c;
        ++pos_;
      }
      if (peek() == '-' && peek(1) != ']' && peek(1) != kEnd) {
        const Cp dash = here();
        ++pos_;
        uint32_t hi = peek();
        if (hi == '\\') {
          ++pos_;
          Ranges r = escape(true, hi);
          if (hi == kEnd) fail_.invalid("a regex range ending in a class", dash.line, dash.col);
        } else {
          ++pos_;
        }
        if (hi < lo) fail_.invalid("a regex range out of order", dash.line, dash.col);
        set.push_back({lo, hi});
      } else {
        set.push_back({lo, lo});
      }
    }
    if (negate) set = complement(set);
    set = normalize(set);
    if (set.empty()) fail_.invalid("a regex class that matches nothing", at.line, at.col);
    return ebnf_class(set);
  }
};

// ---------------------------------------------------------------------------------------------------------
// Lark

const std::map<std::string, std::pair<std::string, std::vector<std::string>>>& common_table() {
  // Lark's common.lark, written out as EBNF (the terminals §2.3 lists).
  static const std::map<std::string, std::pair<std::string, std::vector<std::string>>> t = {
      {"LF", {"\"\\n\"", {}}},
      {"CR", {"\"\\r\"", {}}},
      {"NEWLINE", {"(\"\\r\"? \"\\n\")+", {}}},
      {"WS", {"[ \\t\\u000C\\r\\n]+", {}}},
      {"WS_INLINE", {"[ \\t]+", {}}},
      {"DIGIT", {"[0-9]", {}}},
      {"INT", {"[0-9]+", {}}},
      {"LETTER", {"[A-Za-z]", {}}},
      {"WORD", {"[A-Za-z]+", {}}},
      {"ESCAPED_STRING", {"\"\\\"\" ([^\"\\\\\\n] | \"\\\\\" [^\\n])* \"\\\"\"", {}}},
      {"SIGNED_NUMBER",
       {"[+\\-]? ([0-9]+ [eE] [+\\-]? [0-9]+ | ([0-9]+ \".\" [0-9]* | \".\" [0-9]+) ([eE] [+\\-]? [0-9]+)? | [0-9]+)",
        {}}},
  };
  return t;
}

enum class Tok { kName, kString, kRegex, kOp, kDirective, kNumber, kNewline, kEnd };

struct Token {
  Tok kind;
  std::string text;              // name, operator, directive name, number
  std::vector<uint32_t> value;   // a string literal's code points
  std::size_t rx_begin = 0, rx_end = 0;  // a regex body, as indices into the code points
  std::string flags;             // after a string ("i") or a regex
  int line, col;
};

class Lexer {
 public:
  Lexer(const std::vector<Cp>& s, const Fail& fail) : s_(s), fail_(fail) {}

  std::vector<Token> run() {
    std::vector<Token> out;
    while (true) {
      skip_blank();
      const Cp at = s_[pos_];
      const uint32_t c = at.c;
      if (c == kEnd) {
        out.push_back({Tok::kEnd, "", {}, 0, 0, "", at.line, at.col});
        return out;
      }
      if (c == '\n') {
        ++pos_;
        out.push_back({Tok::kNewline, "", {}, 0, 0, "", at.line, at.col});
      } else if (is_name_start(c)) {
        std::string name;
        while (is_name_char(s_[pos_].c)) name += static_cast<char>(s_[pos_++].c);
        out.push_back({Tok::kName, name, {}, 0, 0, "", at.line, at.col});
      } else if (c >= '0' && c <= '9') {
        std::string num;
        while (s_[pos_].c >= '0' && s_[pos_].c <= '9') num += static_cast<char>(s_[pos_++].c);
        out.push_back({Tok::kNumber, num, {}, 0, 0, "", at.line, at.col});
      } else if (c == '"') {
        out.push_back(string_literal());
      } else if (c == '/') {
        out.push_back(regex_literal());
      } else if (c == '%') {
        ++pos_;
        std::string name;
        while (is_name_char(s_[pos_].c)) name += static_cast<char>(s_[pos_++].c);
        out.push_back({Tok::kDirective, name, {}, 0, 0, "", at.line, at.col});
      } else if (c == '-' && s_[pos_ + 1].c == '>') {
        pos_ += 2;
        out.push_back({Tok::kOp, "->", {}, 0, 0, "", at.line, at.col});
      } else if (c == '.' && s_[pos_ + 1].c == '.') {
        pos_ += 2;
        out.push_back({Tok::kOp, "..", {}, 0, 0, "", at.line, at.col});
      } else if (std::string_view(":|()[]?!*+~.,{}").find(static_cast<char>(c < 0x80 ? c : 0)) != std::string_view::npos &&
                 c != 0) {
        ++pos_;
        out.push_back({Tok::kOp, std::string(1, static_cast<char>(c)), {}, 0, 0, "", at.line, at.col});
      } else if (c == '\'') {
        fail_.unsupported("a single-quoted literal", at.line, at.col);
      } else {
        fail_.invalid("an unexpected character", at.line, at.col);
      }
    }
  }

 private:
  const std::vector<Cp>& s_;
  const Fail& fail_;
  std::size_t pos_ = 0;

  static bool is_name_start(uint32_t c) { return c == '_' || (c < 0x80 && std::isalpha(static_cast<int>(c))); }
  static bool is_name_char(uint32_t c) { return c == '_' || (c < 0x80 && std::isalnum(static_cast<int>(c))); }

  void skip_blank() {
    while (true) {
      const uint32_t c = s_[pos_].c;
      if (c == ' ' || c == '\t' || c == '\r') {
        ++pos_;
      } else if (c == '/' && s_[pos_ + 1].c == '/') {
        while (s_[pos_].c != '\n' && s_[pos_].c != kEnd) ++pos_;
      } else if (c == '#') {
        while (s_[pos_].c != '\n' && s_[pos_].c != kEnd) ++pos_;
      } else {
        return;
      }
    }
  }

  Token string_literal() {
    const Cp at = s_[pos_++];
    Token t{Tok::kString, "", {}, 0, 0, "", at.line, at.col};
    while (true) {
      const uint32_t c = s_[pos_].c;
      if (c == kEnd || c == '\n') fail_.invalid("an unterminated string", at.line, at.col);
      ++pos_;
      if (c == '"') break;
      if (c != '\\') {
        t.value.push_back(c);
        continue;
      }
      const uint32_t e = s_[pos_].c;
      if (e == kEnd) fail_.invalid("an unterminated string", at.line, at.col);
      ++pos_;
      switch (e) {
        case 'n': t.value.push_back('\n'); break;
        case 't': t.value.push_back('\t'); break;
        case 'r': t.value.push_back('\r'); break;
        case 'f': t.value.push_back('\f'); break;
        case 'v': t.value.push_back('\v'); break;
        case '0': t.value.push_back(0); break;
        case '\\': t.value.push_back('\\'); break;
        case '"': t.value.push_back('"'); break;
        case '\'': t.value.push_back('\''); break;
        case 'x':
        case 'u':
        case 'U': {
          const int digits = e == 'x' ? 2 : e == 'u' ? 4 : 8;
          uint32_t v = 0;
          for (int i = 0; i < digits; ++i) {
            const uint32_t h = s_[pos_].c;
            int d = h >= '0' && h <= '9' ? int(h - '0') : h >= 'a' && h <= 'f' ? int(h - 'a' + 10)
                                       : h >= 'A' && h <= 'F' ? int(h - 'A' + 10) : -1;
            if (d < 0) fail_.invalid("a bad string escape", at.line, at.col);
            v = v * 16 + d;
            ++pos_;
          }
          if (v > kMaxCp || (v >= 0xD800 && v <= 0xDFFF)) fail_.invalid("a string escape outside Unicode", at.line, at.col);
          t.value.push_back(v);
          break;
        }
        default:  // Python keeps an unknown escape as written
          t.value.push_back('\\');
          t.value.push_back(e);
      }
    }
    while (is_name_char(s_[pos_].c)) t.flags += static_cast<char>(s_[pos_++].c);
    return t;
  }

  Token regex_literal() {
    const Cp at = s_[pos_++];
    Token t{Tok::kRegex, "", {}, pos_, 0, "", at.line, at.col};
    while (true) {
      const uint32_t c = s_[pos_].c;
      if (c == kEnd || c == '\n') fail_.invalid("an unterminated regex", at.line, at.col);
      if (c == '\\') {
        if (s_[pos_ + 1].c == kEnd || s_[pos_ + 1].c == '\n') fail_.invalid("an unterminated regex", at.line, at.col);
        pos_ += 2;
        continue;
      }
      if (c == '/') break;
      ++pos_;
    }
    t.rx_end = pos_++;
    if (t.rx_end == t.rx_begin) fail_.invalid("an empty regex", at.line, at.col);
    while (is_name_char(s_[pos_].c)) t.flags += static_cast<char>(s_[pos_++].c);
    return t;
  }
};

class LarkParser {
 public:
  LarkParser(const std::vector<Cp>& cps, std::vector<Token> toks, const Fail& fail)
      : cps_(cps), t_(std::move(toks)), fail_(fail) {}

  std::string run() {
    while (true) {
      skip_newlines();
      const Token& tok = t_[i_];
      if (tok.kind == Tok::kEnd) break;
      if (tok.kind == Tok::kDirective) directive();
      else definition();
    }
    // Every reference is defined.
    for (const auto& [name, where] : refs_) {
      if (defined_.count(name)) continue;
      if (imported_.count(name)) continue;
      fail_.invalid("the name '" + name + "' is not defined", where.first, where.second);
    }
    if (!defined_.count("start")) fail_.invalid("the grammar has no 'start' rule", 1, 1);
    std::string out = "root ::= r_start\n" + body_;
    for (const auto& name : imported_) out += emit_name(name) + " ::= " + common_table().at(name).first + "\n";
    return out;
  }

 private:
  const std::vector<Cp>& cps_;
  std::vector<Token> t_;
  const Fail& fail_;
  std::size_t i_ = 0;
  std::string body_;
  std::map<std::string, std::pair<int, int>> defined_;
  std::map<std::string, std::pair<int, int>> refs_;
  std::set<std::string> imported_;

  static bool is_terminal(const std::string& name) {
    for (char c : name)
      if (c != '_') return std::isupper(static_cast<unsigned char>(c));
    return false;
  }
  static std::string emit_name(const std::string& name) { return (is_terminal(name) ? "t_" : "r_") + name; }

  const Token& peek() const { return t_[i_]; }
  bool is_op(const char* op) const { return peek().kind == Tok::kOp && peek().text == op; }
  void skip_newlines() {
    while (peek().kind == Tok::kNewline) ++i_;
  }
  // A definition continues onto the next line when that line starts with '|'.
  void continue_lines() {
    std::size_t j = i_;
    while (t_[j].kind == Tok::kNewline) ++j;
    if (j != i_ && t_[j].kind == Tok::kOp && t_[j].text == "|") i_ = j;
  }

  void directive() {
    const Token d = t_[i_++];
    if (d.text != "import") fail_.unsupported("%" + d.text, d.line, d.col);
    const Token mod = t_[i_];
    if (mod.kind != Tok::kName) fail_.invalid("a malformed %import", d.line, d.col);
    if (mod.text != "common") fail_.unsupported("%import of '" + mod.text + "'", mod.line, mod.col);
    ++i_;
    std::vector<Token> names;
    if (is_op(".")) {
      ++i_;
      if (peek().kind != Tok::kName) fail_.invalid("a malformed %import", d.line, d.col);
      names.push_back(t_[i_++]);
      if (is_op("->")) fail_.unsupported("a renamed %import", peek().line, peek().col);
    } else if (is_op("(")) {
      ++i_;
      while (peek().kind == Tok::kName) {
        names.push_back(t_[i_++]);
        if (is_op(",")) ++i_;
      }
      if (!is_op(")")) fail_.invalid("a malformed %import", d.line, d.col);
      ++i_;
    } else {
      fail_.invalid("a malformed %import", d.line, d.col);
    }
    for (const Token& n : names) {
      if (!common_table().count(n.text)) fail_.unsupported("%import of 'common." + n.text + "'", n.line, n.col);
      if (defined_.count(n.text)) fail_.invalid("'" + n.text + "' is defined twice", n.line, n.col);
      imported_.insert(n.text);
    }
    end_of_line();
  }

  void end_of_line() {
    if (peek().kind != Tok::kNewline && peek().kind != Tok::kEnd)
      fail_.invalid("unexpected text", peek().line, peek().col);
  }

  void definition() {
    if (is_op("?") || is_op("!")) ++i_;  // shape only
    const Token name = peek();
    if (name.kind != Tok::kName) fail_.invalid("expected a rule or terminal name", name.line, name.col);
    ++i_;
    if (is_op(".")) fail_.unsupported("a priority", peek().line, peek().col);
    if (is_op("{")) fail_.unsupported("a template", peek().line, peek().col);
    if (!is_op(":")) fail_.invalid("expected ':' after '" + name.text + "'", peek().line, peek().col);
    ++i_;
    if (defined_.count(name.text) || imported_.count(name.text))
      fail_.invalid("'" + name.text + "' is defined twice", name.line, name.col);
    defined_[name.text] = {name.line, name.col};
    std::string rhs = expansions(0);
    end_of_line();
    body_ += emit_name(name.text) + " ::= " + rhs + "\n";
  }

  std::string expansions(int depth) {
    if (depth > kMaxNesting) fail_.unsupported("expressions nested deeper than 64", peek().line, peek().col);
    std::vector<std::string> alts;
    while (true) {
      alts.push_back(alternative(depth));
      if (depth == 0) continue_lines();
      if (!is_op("|")) break;
      ++i_;
      if (depth == 0) continue_lines();
    }
    if (alts.size() == 1) return alts[0];
    std::string out = "(";
    for (std::size_t k = 0; k < alts.size(); ++k) out += (k ? " | " : "") + alts[k];
    return out + ")";
  }

  std::string alternative(int depth) {
    std::vector<std::string> items;
    while (true) {
      const Token& tok = peek();
      if (tok.kind == Tok::kNewline || tok.kind == Tok::kEnd) break;
      if (tok.kind == Tok::kOp && (tok.text == "|" || tok.text == ")" || tok.text == "]" || tok.text == "->")) break;
      items.push_back(item(depth));
    }
    if (is_op("->")) {  // an alias names the tree node; the language is unchanged
      ++i_;
      if (peek().kind != Tok::kName) fail_.invalid("expected a name after '->'", peek().line, peek().col);
      ++i_;
    }
    if (items.empty()) return "\"\"";
    std::string out;
    for (std::size_t k = 0; k < items.size(); ++k) out += (k ? " " : "") + items[k];
    return items.size() == 1 ? out : "(" + out + ")";
  }

  std::string item(int depth) {
    std::string a = atom(depth);
    const Token& q = peek();
    if (q.kind == Tok::kOp && (q.text == "?" || q.text == "*" || q.text == "+")) {
      ++i_;
      return "(" + a + ")" + q.text;
    }
    if (q.kind == Tok::kOp && q.text == "~") fail_.unsupported("'~' repetition", q.line, q.col);
    return a;
  }

  std::string atom(int depth) {
    const Token tok = t_[i_];
    if (tok.kind == Tok::kOp && (tok.text == "(" || tok.text == "[")) {
      ++i_;
      std::string inner = expansions(depth + 1);
      const char* close = tok.text == "(" ? ")" : "]";
      if (!is_op(close)) fail_.invalid(std::string("an unclosed '") + tok.text + "'", tok.line, tok.col);
      ++i_;
      return tok.text == "(" ? "(" + inner + ")" : "(" + inner + ")?";
    }
    if (tok.kind == Tok::kString) {
      ++i_;
      if (!tok.flags.empty()) {
        if (tok.flags == "i") fail_.unsupported("a case-insensitive literal", tok.line, tok.col);
        fail_.invalid("unknown literal flags '" + tok.flags + "'", tok.line, tok.col);
      }
      if (is_op("..")) fail_.unsupported("a literal range", peek().line, peek().col);
      return tok.value.empty() ? "\"\"" : ebnf_literal(tok.value);
    }
    if (tok.kind == Tok::kRegex) {
      ++i_;
      if (!tok.flags.empty()) fail_.unsupported("regex flags", tok.line, tok.col);
      RegexParser rp(cps_, tok.rx_begin, tok.rx_end, fail_);
      return "(" + rp.parse() + ")";
    }
    if (tok.kind == Tok::kName) {
      ++i_;
      if (is_op("{")) fail_.unsupported("a template", peek().line, peek().col);
      if (!refs_.count(tok.text)) refs_[tok.text] = {tok.line, tok.col};
      return emit_name(tok.text);
    }
    if (tok.kind == Tok::kOp && tok.text == "~") fail_.unsupported("'~' repetition", tok.line, tok.col);
    if (tok.kind == Tok::kDirective) fail_.invalid("a directive inside a rule", tok.line, tok.col);
    fail_.invalid("unexpected '" + tok.text + "'", tok.line, tok.col);
  }
};

}  // namespace

std::string lark_to_ebnf(std::string_view lark, const std::string& param) {
  const Fail fail{param};
  const std::vector<Cp> cps = decode(lark, fail);
  Lexer lexer(cps, fail);
  LarkParser parser(cps, lexer.run(), fail);
  return parser.run();
}

std::string regex_to_ebnf(std::string_view pattern, const std::string& param) {
  const Fail fail{param};
  const std::vector<Cp> cps = decode(pattern, fail);
  if (cps.size() == 1) return "\"\"";
  RegexParser rp(cps, 0, cps.size() - 1, fail);
  return "(" + rp.parse() + ")";
}

}  // namespace ragon::grammar::detail
