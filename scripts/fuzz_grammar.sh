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

# Seeds: one per front end (the first byte picks it; 5 and 6 are the route bodies, chat with bit 3 set).
seed() { printf '%b' "$2" > "$OUT/corpus/$1"; }
seed schema '\x00{"type": "object", "properties": {"a": {"type": "array", "items": {"$ref": "#/$defs/x"}}}, "$defs": {"x": {"anyOf": [{"type": "string"}, {"enum": [1, null]}]}}, "required": ["a"]}'
seed lark '\x01start: begin_patch hunk+ end_patch\nbegin_patch: "*** Begin Patch" LF\nend_patch: "*** End Patch" LF?\nhunk: "+" /(.*)/ LF -> line\n%%import common.LF\n'
seed exec '\x01start: pragma_source | plain_source\npragma_source: PRAGMA_LINE NEWLINE SOURCE\nplain_source: SOURCE\nPRAGMA_LINE: /[ \\t]*\\/\\/ @exec:[^\\r\\n]*/\nNEWLINE: /\\r?\\n/\nSOURCE: /[\\s\\S]+/\n'
seed regex '\x02[a-z]{2,3}\\d(-[A-F0-9]+)*|(?:\\w+\\s?)+\\.'
seed tools '\x03?start: "x" NAME\nNAME: /[A-Z][a-z]{1,8}/\n'
seed json_calls '\x04{"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}'
seed chat '\x0c{"messages": [{"role": "user", "content": "hi"}], "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {"a": {"type": "string"}}}}}], "tool_choice": {"type": "function", "function": {"name": "f"}}, "parallel_tool_calls": false, "response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object"}}}}'
seed completions '\x05{"prompt": "Q:", "response_format": {"type": "json_object"}}'
seed responses '\x06{"input": "hi", "tools": [{"type": "function", "name": "f", "parameters": {"type": "object"}}, {"type": "custom", "name": "apply_patch", "format": {"type": "grammar", "syntax": "lark", "definition": "start: \\"*** Begin Patch\\" LF /(.+)/ LF\\n%%import common.LF\\n"}}], "tool_choice": "required", "text": {"format": {"type": "json_schema", "name": "x", "schema": {"type": "object"}}}}'

UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 ASAN_OPTIONS=detect_leaks=1:allocator_may_return_null=1 \
  nice -n 19 "$OUT/ragon-grammar-fuzz" "$OUT/corpus" -max_total_time="$SECONDS_TO_RUN" -max_len=4096 \
  -rss_limit_mb=4096 -timeout=30 -artifact_prefix="$OUT/findings/" -print_final_stats=1
