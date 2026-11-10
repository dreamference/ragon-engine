#!/usr/bin/env bash
# Rung 2 of a drafter A/B (DRAFTER_RETRAIN §5) for the drafter in DRAFT_DIR, output in OUT_DIR:
#   1. correctness with this drafter (ragon-run): greedy speculative output equals plain decoding (--spec-check),
#      CUDA graphs change nothing (--graph-check), two batched sessions equal two single ones (--batch-check);
#   2. acceptance by context: the agentlong set greedy (16 turns, contexts 28-62K) through ragon-serve at production's
#      flags, tokens per step per context bucket (rung2.py turns); then again with the chain alone (--tree 0) and
#      RAGON_DUMP_STEPS, the acceptance at each draft depth (rung2.py steps);
#   3. the agreement gate through the same server: greedy completions of the 54 gate prompts and the 24 agent
#      prompts' first responses against production's tokens (rung2.py agree).
# Run it for the served drafter and for the candidate on the same build; the candidate passes if no bucket's
# tokens/step is lower, every check passes and its agreement texts equal the served drafter's (DRAFTER_RETRAIN §5).
#
# Usage: rung2.sh DRAFT_DIR OUT_DIR
# Environment: SERVE, RUN (ragon-run), MODEL, SPARSE_HEAD_DIR (or "none"), PROMPTS (ragon-run's prompts file),
#   AGR_DIR (cases.jsonl, cases-long.jsonl, prod-gen400-long.jsonl), AGENTLONG_BODIES, PORT (8300), PIDS_FILE,
#   REF_AGREE (optional: the served drafter's agree.jsonl from this script on the same build; then rung2.py same
#   compares the texts and a difference fails the run).
set -u
DRAFT=${1:?drafter dir}; OUT=${2:?out dir}
: "${SERVE:?}" "${RUN:?}" "${MODEL:?}" "${SPARSE_HEAD_DIR:?}" "${PROMPTS:?}" "${AGR_DIR:?}" "${AGENTLONG_BODIES:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
PORT=${PORT:-8300}
NAME=RadixArk/Qwen3.8-27B-NVFP4
mkdir -p "$OUT"
head_args=(--draft-head full); want=full
if [ "$SPARSE_HEAD_DIR" != none ]; then head_args=(--sparse-head-dir "$SPARSE_HEAD_DIR"); want=sparse; fi
rc=0
C=(--model "$MODEL" --draft "$DRAFT" --prompts-file "$PROMPTS" --max-tokens 400)
"$RUN" "${C[@]}" --spec-check --graphs --pdl > "$OUT/spec-check.log" 2>&1 || rc=1
"$RUN" "${C[@]}" --spec --graph-check > "$OUT/graph-check.log" 2>&1 || rc=1
# RAGON_TREE_ROWS=0: since the 2026-11-10 integration a one-stream step verifies 24 tree rows and a batched one 16, so
# a sequence alone and in a batch take different steps (tokens and state stay equal); 0 gives both the same tree.
RAGON_TREE_ROWS=0 "$RUN" "${C[@]}" --batch-check --sessions 2 --seed 7 --graphs --pdl --kv-pages > "$OUT/batch-check.log" 2>&1 || rc=1
for c in spec graph batch; do echo "$c-check: $(tail -n 1 "$OUT/$c-check.log" | cut -c1-200)"; done

# serve DRAFTER-ENV-AND-FLAGS...: a ragon-serve on PORT at production's flags plus the arguments; waits for health and
# checks the head line.
serve() {
  env "$@" "$SERVE" --model "$MODEL" --draft "$DRAFT" "${head_args[@]}" --port "$PORT" --served-model-name "$NAME" \
    --max-context 65536 --draft-block 12 --graphs --pdl --sessions 2 "${extra[@]}" > "$OUT/serve-$tag.log" 2>&1 8>&- \
    < /dev/null &
  srv=$!
  [ -n "${PIDS_FILE:-}" ] && echo $srv >> "$PIDS_FILE"
  for _ in $(seq 1 120); do curl -sf -m 2 "localhost:$PORT/health" > /dev/null && break; sleep 3; done
  # the full head prints no head line (only a sparse head that failed to load says "drafter head: full")
  if { [ $want = sparse ] && ! grep -q "drafter head: sparse" "$OUT/serve-$tag.log"; } || \
     { [ $want = full ] && grep -q "drafter head: sparse" "$OUT/serve-$tag.log"; }; then
    echo "rung2: serve log does not show the $want head" >&2; kill $srv; tail --pid=$srv -f /dev/null; exit 3
  fi
}
agentlong() {
  python3 -I "$HERE/../agentlong/run.py" --url "http://127.0.0.1:$PORT" --mode greedy --label "rung2-$tag" \
    --out "$OUT/agentlong-$tag.jsonl" --bodies "$AGENTLONG_BODIES" --manifest "$HERE/../agentlong/manifest.json" 8>&- \
    > "$OUT/agentlong-$tag.log" 2>&1 || rc=1
}

# 2a + 3: production's flags (tree verification): tokens per step per context bucket, and the agreement gate.
tag=prod; extra=(); serve RAGON_RUNG2=1
agentlong
python3 -I "$HERE/rung2.py" turns "$OUT/agentlong-prod.jsonl" > "$OUT/turns.json" || rc=1
python3 -I "$HERE/rung2.py" agree "http://127.0.0.1:$PORT" "$AGR_DIR" "$MODEL/tokenizer.json" "$OUT/agree.jsonl" \
  > "$OUT/agree.json" 2> "$OUT/agree.err" || rc=1
curl -s -m 10 "localhost:$PORT/metrics" > "$OUT/metrics-prod"
kill $srv; tail --pid=$srv -f /dev/null

# 2b: the chain alone (--tree 0; RAGON_DUMP_STEPS records chain steps only): acceptance at each draft depth.
rm -f "$OUT/steps.bin"
tag=chain; extra=(--tree 0); serve RAGON_DUMP_STEPS="$OUT/steps.bin"
agentlong
kill $srv; tail --pid=$srv -f /dev/null
python3 -I "$HERE/rung2.py" steps "$OUT/steps.bin" > "$OUT/steps.json" || rc=1
echo "tokens/step (tree, production flags): $(cat "$OUT/turns.json")"
echo "chain acceptance: $(cat "$OUT/steps.json")"
echo "agreement: $(cat "$OUT/agree.json")"
if [ -n "${REF_AGREE:-}" ]; then
  python3 -I "$HERE/rung2.py" same "$REF_AGREE" "$OUT/agree.jsonl" > "$OUT/same.json" || rc=1
  echo "agreement texts against the served drafter: $(cat "$OUT/same.json")"
fi
exit $rc
