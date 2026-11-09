// A token constraint on one request's output (structured output, SPECULATION "Structured output, stage 2"): the
// engine asks it for the allowed ids before each sampled token and tells it the tokens it commits. The server builds
// one from the request's grammar (a grammar::Matcher) at admission and drops it at the end of the request; the core
// sees only this interface, so it builds without XGrammar.
//
// Masks are int32 words, bit id%32 of word id/32 set when id is allowed (core/sampling.hpp mask_allows).
#pragma once

#include <cstdint>
#include <memory>

namespace ragon {

class TokenConstraint {
 public:
  virtual ~TokenConstraint() = default;
  // Words per mask row (vocab / 32, rounded up).
  virtual int words() const = 0;
  // Writes the allowed ids after the tokens accepted so far into mask[0, words()).
  virtual void fill(int32_t* mask) = 0;
  // Advances by one token; false (state unchanged) when the grammar refuses it.
  virtual bool accept(int id) = 0;
  // Undoes the last n accepted tokens.
  virtual void rollback(int n) = 0;
  // The grammar reached its end: nothing but the stop token may follow.
  virtual bool terminated() const = 0;
};

}  // namespace ragon
