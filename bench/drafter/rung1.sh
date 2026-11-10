#!/usr/bin/env bash
# Rung 1 of a drafter A/B (DRAFTER_RETRAIN §5): the seeded agent-session replay at one stream against a fresh
# ragon-serve that loads the drafter under test. Prints, per seed, tokens per step (generated tokens / verify calls,
# the accepted-per-step figure the spec tracks), ms per step and decode tok/s, all from the server's /metrics.
#
# Usage: rung1.sh DRAFT_DIR OUT_DIR [SEED ...]          (default seeds 1013 2026)
# Environment (no defaults on purpose; the script carries no machine paths):
#   SERVE            the ragon-serve binary
#   MODEL            the target checkpoint (the served RadixArk/Qwen3.8-27B-NVFP4 snapshot)
#   SPARSE_HEAD_DIR  the sparse drafter-head snapshot (dreamference/qwen3.8-27b-sparse-head) as production serves
#                    it; set it to "none" to measure with the full head (--draft-head full; without it ragon-serve
#                    loads <DRAFT_DIR>/sparse-head when that exists) (the 2026-11-10 baseline numbers were taken
#                    that way). With a directory, the serve log must say "drafter head: sparse" or the run aborts.
#   REPLAY_DIR       directory holding tools-capture.json, tools-index.json, set12.txt and the rollouts it names
#   REPLAY_PY        bench/replay/replay.py of this checkout (default: next to this script)
#   PORT             default 8300
# The GPU must be free (production stopped): this script starts and stops its own server per seed.
set -u
DRAFT=${1:?drafter dir}; OUT=${2:?out dir}; shift 2
SEEDS=${*:-1013 2026}
: "${SERVE:?}" "${MODEL:?}" "${SPARSE_HEAD_DIR:?}" "${REPLAY_DIR:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
REPLAY_PY=${REPLAY_PY:-$HERE/../replay/replay.py}
PORT=${PORT:-8300}
NAME=RadixArk/Qwen3.8-27B-NVFP4
mkdir -p "$OUT"
head_args=(--draft-head full); want=full
if [ "$SPARSE_HEAD_DIR" != none ]; then head_args=(--sparse-head-dir "$SPARSE_HEAD_DIR"); want=sparse; fi

metric() { awk -v k="$1" '$1 ~ "^"k"[{ ]" { s += $NF } END { printf "%.6f", s }' "$2"; }

rc=0
for seed in $SEEDS; do
  label=r$seed-$want
  setsid "$SERVE" --model "$MODEL" --draft "$DRAFT" "${head_args[@]}" --port "$PORT" --served-model-name "$NAME" \
    --max-context 65536 --draft-block 12 --graphs --pdl --sessions 2 > "$OUT/$label.serve.log" 2>&1 8>&- < /dev/null &
  srv=$!
  [ -n "${PIDS_FILE:-}" ] && echo $srv >> "$PIDS_FILE"
  for _ in $(seq 1 120); do curl -sf -m 2 "localhost:$PORT/health" > /dev/null && break; sleep 3; done
  # the full head prints no head line (only a sparse head that failed to load says "drafter head: full")
  if { [ $want = sparse ] && ! grep -q "drafter head: sparse" "$OUT/$label.serve.log"; } || \
     { [ $want = full ] && grep -q "drafter head: sparse" "$OUT/$label.serve.log"; }; then
    echo "rung1: $label: serve log does not show the $want head" >&2; kill $srv; tail --pid=$srv -f /dev/null; exit 3
  fi
  (cd "$REPLAY_DIR" && python3 -I "$REPLAY_PY" --url "http://127.0.0.1:$PORT" --model "$NAME" \
     --tools tools-capture.json --tools-index tools-index.json --sessions-file set12.txt --window 12 \
     --concurrency 1 --sampling-seed "$seed" --custom-tools keep --label "$label" --out "$OUT/$label.jsonl" 8>&-) \
     > "$OUT/$label.replay.log" 2>&1 || rc=1
  curl -s -m 10 "localhost:$PORT/metrics" > "$OUT/$label.metrics"
  kill $srv; tail --pid=$srv -f /dev/null
  m=$OUT/$label.metrics
  gen=$(metric sglang:generation_tokens_total "$m"); calls=$(metric sglang:spec_verify_calls_total "$m")
  e2e=$(metric sglang:e2e_request_latency_seconds_sum "$m"); ttft=$(metric sglang:time_to_first_token_seconds_sum "$m")
  reqs=$(metric sglang:e2e_request_latency_seconds_count "$m")
  python3 -I -c 'import json,sys
g,c,e,t,r=map(float,sys.argv[2:7]); d=e-t
print(json.dumps({"label":sys.argv[1],"tok_per_step":round(g/c,3) if c else None,"ms_per_step":round(1e3*d/c,1) if c else None,
 "decode_tok_s":round((g-r)/d,2) if d>0 else None,"generated":int(g),"verify_calls":int(c),"requests":int(r)}))' \
    "$label" "$gen" "$calls" "$e2e" "$ttft" "$reqs" | tee -a "$OUT/rung1.jsonl"
done
exit $rc
