#!/usr/bin/env bash
# Rung 2 of a drafter A/B (DRAFTER_RETRAIN §5) for the drafter in DRAFT_DIR, output in OUT_DIR:
#   1. correctness with this drafter (ragon-run): greedy speculative output equals plain decoding (--spec-check),
#      CUDA graphs change nothing (--graph-check), two batched sessions equal two single ones (--batch-check);
#   2. per-position acceptance: the agentlong set greedy (16 turns, positions to 64K) through ragon-serve with
#      RAGON_DUMP_STEPS, analysed by rung2.py steps (context buckets, acceptance per draft depth);
#   3. the agreement gate through the same server: greedy completions of the 54 gate prompts and the 24 agent
#      prompts' first responses against production's tokens (rung2.py agree).
# Run it for the served drafter and for the candidate; the candidate passes if no bucket's tokens/step is lower,
# every check passes and the gate's exact counts are not lower (DRAFTER_RETRAIN §5).
#
# Usage: rung2.sh DRAFT_DIR OUT_DIR
# Environment: SERVE, RUN (ragon-run), MODEL, SPARSE_HEAD_DIR (or "none"), PROMPTS (ragon-run's prompts file),
#   AGR_DIR (cases.jsonl, cases-long.jsonl, prod-gen400-long.jsonl), AGENTLONG_BODIES, PORT (8300), PIDS_FILE.
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
"$RUN" "${C[@]}" --batch-check --sessions 2 --seed 7 --graphs --pdl --kv-pages > "$OUT/batch-check.log" 2>&1 || rc=1
for c in spec graph batch; do echo "$c-check: $(tail -n 1 "$OUT/$c-check.log" | cut -c1-200)"; done

rm -f "$OUT/steps.bin"
RAGON_DUMP_STEPS=$OUT/steps.bin setsid "$SERVE" --model "$MODEL" --draft "$DRAFT" "${head_args[@]}" --port "$PORT" \
  --served-model-name "$NAME" --max-context 65536 --draft-block 12 --graphs --pdl --sessions 2 \
  > "$OUT/serve.log" 2>&1 8>&- < /dev/null &
srv=$!
[ -n "${PIDS_FILE:-}" ] && echo $srv >> "$PIDS_FILE"
for _ in $(seq 1 120); do curl -sf -m 2 "localhost:$PORT/health" > /dev/null && break; sleep 3; done
if ! grep -q "drafter head: $want" "$OUT/serve.log"; then
  echo "rung2: serve log lacks 'drafter head: $want'" >&2; kill $srv; tail --pid=$srv -f /dev/null; exit 3
fi
python3 -I "$HERE/../agentlong/run.py" --url "http://127.0.0.1:$PORT" --mode greedy --label rung2 \
  --out "$OUT/agentlong.jsonl" --bodies "$AGENTLONG_BODIES" --manifest "$HERE/../agentlong/manifest.json" 8>&- \
  > "$OUT/agentlong.log" 2>&1 || rc=1
upto=$(stat -c %s "$OUT/steps.bin" 2>/dev/null || echo 0)
python3 -I "$HERE/rung2.py" agree "http://127.0.0.1:$PORT" "$AGR_DIR" "$MODEL/tokenizer.json" "$OUT/agree.jsonl" \
  > "$OUT/agree.json" 2> "$OUT/agree.err" || rc=1
curl -s -m 10 "localhost:$PORT/metrics" > "$OUT/metrics"
kill $srv; tail --pid=$srv -f /dev/null
python3 -I "$HERE/rung2.py" steps "$OUT/steps.bin" --upto "$upto" > "$OUT/steps.json" || rc=1
echo "acceptance: $(cat "$OUT/steps.json")"
echo "agreement: $(cat "$OUT/agree.json")"
exit $rc
