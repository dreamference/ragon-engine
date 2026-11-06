#!/bin/bash
# measure.sh: the several-drafters measurement in one command (MULTI_DRAFTER phase 0): every arm on the workload sets of
# workloads.py, under the GPU lock, then the tables of by_workload.py.
#
# Usage: measure.sh OUTDIR [ARM...]      run it with setsid nohup ... < /dev/null, on a machine with the agentlong set
#   ARM is LABEL:DRAFTER:STREAMS:MODE[:EXTRA]
#     DRAFTER  rtn (the file production loads), ss (its ScaleSearch re-quantization), none, or a directory
#     STREAMS  1 runs all the sets through run.py; N > 1 runs each workload's set through n_streams.py, N sessions at once
#     MODE     greedy or seedN
#     EXTRA    ragon-serve flags, commas for spaces; they follow the defaults and so override them
#   No ARM: the matrix in DEFAULT_ARMS.
# BUILD names the engine checkout (default: this one); PROD_STOP and PROD_START the commands that free the GPU and bring
# back what was running, run only if a model server was up at the start, and at the end even if an arm fails. An arm
# with STREAMS > 1 waits for MEM_FREE_GB (default 105) of available memory and is skipped if it never comes. For a short
# pass: ONLY=edit_py1,chat0 (run.py's turns) and WORKLOADS="prose" (n_streams.py's sets). N sessions use --warm except on
# prose: cold 30K prefills would keep long sets from ever decoding together (batched share 0.01 when tried).
set -u
PATH=$HOME/.local/bin:$PATH
O=$1; shift; mkdir -p "$O"
HERE=$(cd "$(dirname "$0")" && pwd)
BUILD=${BUILD:-$(cd "$HERE/../.." && pwd)}
M=$(ls -d ~/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/*)
RTN=$(ls -d ~/.cache/huggingface/hub/models--maurienne-ai--Qwen3.8-27B-DFlash2-NVFP4-RTNcal/snapshots/*)
SS=$HOME/.cache/dreamference/drafters/dflash2-nvfp4-scalesearch
PROD_STOP=${PROD_STOP:-ragon-admin server stop}
PROD_START=${PROD_START:-ragon-admin server start --engine sglang}
ONE_FLAGS="--max-context 65536 --graphs --pdl"
MANY_FLAGS="--max-context 196608 --no-kv-pages --fair-pairing 0 --graphs --pdl"
DEFAULT_ARMS=(rtn:rtn:1:greedy ss:ss:1:greedy
  block8:rtn:1:greedy:--draft-block,8 block16:rtn:1:greedy:--draft-block,16
  tree0:rtn:1:greedy:--tree,0 lookup0:rtn:1:greedy:--lookup,0 lookup2:rtn:1:greedy:--lookup,2
  rtn-seed1013:rtn:1:seed1013 rtn-again:rtn:1:greedy
  rtn-4:rtn:4:greedy ss-4:ss:4:greedy)
[ $# -gt 0 ] || set -- "${DEFAULT_ARMS[@]}"

python3 "$HERE/workloads.py" --out "$O/set" > "$O/set.routes" || exit 1

SRV=
PROD_UP=
finish() {
  [ -n "$SRV" ] && kill "$SRV" 2>/dev/null
  if [ -n "$PROD_UP" ] && [ "${KEEP_PROD_DOWN:-0}" != 1 ]; then
    $PROD_START > "$O/prod-start.log" 2>&1 8>&- < /dev/null
    echo "production started, exit $? ($(date -u +%T))"
  fi
  echo "=== lock released $(date -u +%T) UTC, held $(( ($(date -u +%s) - T_LOCK) / 60 )) min"
  echo "$(date -u -d @"$T_LOCK" +%F\ %T)-$(date -u +%T) UTC drafter/measure.sh $O" >> ~/.ragon-gpu-holds.log
}

exec 8> "$HOME"/.ragon-gpu.lock; flock 8
T_LOCK=$(date -u +%s); echo "=== lock taken $(date -u +%T) UTC"
trap finish EXIT
if curl -sf -m 3 localhost:8000/health > /dev/null || [ -n "$(docker ps -q -f name=^dreamference-vllm-8000$)" ]; then
  PROD_UP=1
  $PROD_STOP > "$O/prod-stop.log" 2>&1 8>&- < /dev/null || { echo "could not stop production"; exit 1; }
fi

for arm in "$@"; do
  IFS=: read -r label drafter streams mode extra <<< "$arm"
  case $drafter in rtn) D=$RTN ;; ss) D=$SS ;; none) D= ;; *) D=$drafter ;; esac
  if [ "$streams" -gt 1 ]; then
    flags=$MANY_FLAGS; flags+=" --sessions $streams --batch $streams"
    for i in $(seq 1 60); do
      [ "$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)" -ge "${MEM_FREE_GB:-105}" ] && break
      sleep 10
    done
    [ "$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)" -ge "${MEM_FREE_GB:-105}" ] || { echo "$label: skipped, too little free memory"; continue; }
  else
    flags=$ONE_FLAGS
  fi
  draft_flags=(); [ -n "$D" ] && draft_flags=(--draft "$D" --draft-block 12)
  # shellcheck disable=SC2086  # the flags are lists of words
  setsid "$BUILD"/build/ragon-serve --model "$M" "${draft_flags[@]}" --port 8300 --served-model-name RadixArk/Qwen3.8-27B-NVFP4 \
    $flags ${extra//,/ } > "$O/$label.serve.log" 2>&1 8>&- < /dev/null &
  SRV=$!
  for i in $(seq 1 120); do curl -sf -m 2 localhost:8300/health > /dev/null && break; sleep 3; done
  if [ "$streams" = 1 ]; then
    python3 "$HERE/../agentlong/run.py" --url http://localhost:8300 --mode "$mode" --label "$label" --out "$O/$label.jsonl" \
      --bodies "$O/set/all.jsonl" --manifest "$O/set/all.json" ${ONLY:+--only "$ONLY"} > "$O/$label.log" 2>&1 8>&-
  else
    for wl in ${WORKLOADS:-code-long code-short read-agent prose}; do
      warm=--warm; [ "$wl" = prose ] && warm=
      python3 "$HERE/../agentlong/n_streams.py" $warm --url http://localhost:8300 --streams "$streams" --mode "$mode" --label "$label-$wl" \
        --out "$O/$label.$wl.streams.jsonl" --bodies "$O/set/$wl.jsonl" --manifest "$O/set/$wl.json" > "$O/$label.$wl.log" 2>&1 8>&-
    done
  fi
  curl -s -m 10 localhost:8300/spec_stats > "$O/$label.spec.json" 2>/dev/null
  kill "$SRV"; wait "$SRV" 2>/dev/null; SRV=
  echo "$label done ($(date -u +%T))"
done
python3 "$HERE/by_workload.py" "$O" > "$O/tables.md" 2>&1
echo "tables: $O/tables.md"
