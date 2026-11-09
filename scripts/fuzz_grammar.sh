#!/usr/bin/env bash
# Builds ragon-grammar-fuzz (tests/grammar_fuzz.cpp, libFuzzer) with RAGON_SANITIZE=ON (ASan and UBSan on the
# host code) and runs it.
#
#   scripts/fuzz_grammar.sh [SECONDS]      (default 600)
#
# CLANGXX names a clang++ with libFuzzer (default clang++-18); extra cmake arguments (a CMAKE_PREFIX_PATH for
# nlohmann_json, RAGON_XGRAMMAR_PREFIX) go in FUZZ_CMAKE_ARGS. XGrammar itself is not instrumented; the
# sanitizers cover the library and the glue. Build and corpus go to build-fuzz/. A UBSan report or a sanitizer
# abort fails the run.
set -euo pipefail
cd "$(dirname "$0")/.."
SECONDS_TO_RUN=${1:-600}
OUT=build-fuzz
mkdir -p "$OUT/corpus" "$OUT/findings"
# shellcheck disable=SC2086
nice -n 19 cmake -S . -B "$OUT" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_COMPILER="${CLANGXX:-clang++-18}" \
  -DRAGON_GRAMMAR=ON -DRAGON_SANITIZE=ON ${FUZZ_CMAKE_ARGS:-} > "$OUT/configure.log"
nice -n 19 cmake --build "$OUT" -j8 --target ragon-grammar-fuzz > "$OUT/build.log"

# Seeds: one per front end (the first byte picks it).
seed() { printf '%b' "$2" > "$OUT/corpus/$1"; }
seed schema '\x00{"type": "object", "properties": {"a": {"type": "array", "items": {"$ref": "#/$defs/x"}}}, "$defs": {"x": {"anyOf": [{"type": "string"}, {"enum": [1, null]}]}}, "required": ["a"]}'
seed lark '\x01start: begin_patch hunk+ end_patch\nbegin_patch: "*** Begin Patch" LF\nend_patch: "*** End Patch" LF?\nhunk: "+" /(.*)/ LF -> line\n%%import common.LF\n'
seed exec '\x01start: pragma_source | plain_source\npragma_source: PRAGMA_LINE NEWLINE SOURCE\nplain_source: SOURCE\nPRAGMA_LINE: /[ \\t]*\\/\\/ @exec:[^\\r\\n]*/\nNEWLINE: /\\r?\\n/\nSOURCE: /[\\s\\S]+/\n'
seed regex '\x02[a-z]{2,3}\\d(-[A-F0-9]+)*|(?:\\w+\\s?)+\\.'
seed tools '\x03?start: "x" NAME\nNAME: /[A-Z][a-z]{1,8}/\n'
seed json_calls '\x04{"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}'

UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 ASAN_OPTIONS=detect_leaks=1:allocator_may_return_null=1 \
  nice -n 19 "$OUT/ragon-grammar-fuzz" "$OUT/corpus" -max_total_time="$SECONDS_TO_RUN" -max_len=4096 \
  -rss_limit_mb=4096 -timeout=30 -artifact_prefix="$OUT/findings/" -print_final_stats=1
