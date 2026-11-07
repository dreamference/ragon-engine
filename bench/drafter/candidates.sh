#!/bin/bash
# candidates.sh: drafters ragon-serve cannot load yet, in a scratch SGLang on production's image, for tokens per step only:
# each arm replays the workload sets of measure.sh through run.py. SGLang's timing does not predict the engine's, so read
# the arms against the DFlash2 arm of the same run and only for the speculation rate.
#
# Usage: candidates.sh OUTDIR IMAGE [ARM...]      run it with setsid nohup ... < /dev/null
#   ARM is LABEL:ALGORITHM:DRAFTS:DIRECTORY[:QUANTIZATION]   DIRECTORY as the container sees it (~/.cache is /root/.cache)
#   DRAFTS is the block size; for DSPARK that is gamma, and SGLang wants gamma + 1 as its draft count
# ONLY names the turns; the sets come from measure.sh's OUTDIR/set, or are built here. PROD_STOP and PROD_START as in
# measure.sh. A config.json naming remote code (auto_map) is refused: copy the directory and strip it, never run it.
set -u
PATH=$HOME/.local/bin:$PATH
O=$1; IMAGE=$2; shift 2; mkdir -p "$O"
HERE=$(cd "$(dirname "$0")" && pwd)
PROD_STOP=${PROD_STOP:-ragon-admin server stop}
PROD_START=${PROD_START:-ragon-admin server start --engine sglang}
[ -f "$O/set/all.jsonl" ] || python3 "$HERE/workloads.py" --out "$O/set" > "$O/set.routes" || exit 1

for arm in "$@"; do
  IFS=: read -r label algo drafts dir quant <<< "$arm"
  host_dir=${dir/#\/root\/.cache/$HOME/.cache}
  if grep -q '"auto_map"' "$host_dir/config.json" 2> /dev/null; then echo "$label: config names remote code, refused"; exit 1; fi
done

PROD_UP=
NAME=drafter-candidate
finish() {
  docker rm -f "$NAME" > /dev/null 2>&1
  if [ -n "$PROD_UP" ] && [ "${KEEP_PROD_DOWN:-0}" != 1 ]; then
    $PROD_START > "$O/prod-start.log" 2>&1 8>&- < /dev/null
    echo "production started, exit $? ($(date -u +%T))"
  fi
  echo "=== lock released $(date -u +%T) UTC, held $(( ($(date -u +%s) - T_LOCK) / 60 )) min"
  echo "$(date -u -d @"$T_LOCK" +%F\ %T)-$(date -u +%T) UTC drafter/candidates.sh $O" >> ~/.ragon-gpu-holds.log
}

exec 8> "$HOME"/.ragon-gpu.lock; flock 8
T_LOCK=$(date -u +%s); echo "=== lock taken $(date -u +%T) UTC"
trap finish EXIT
if curl -sf -m 3 localhost:8000/health > /dev/null || [ -n "$(docker ps -q -f name=^dreamference-vllm-8000$)" ]; then
  PROD_UP=1
  $PROD_STOP > "$O/prod-stop.log" 2>&1 8>&- < /dev/null || { echo "could not stop production"; exit 1; }
fi

for arm in "$@"; do
  IFS=: read -r label algo drafts dir quant <<< "$arm"
  q=(--drop speculative-draft-model-quantization)
  [ -n "$quant" ] && q=(--set speculative-draft-model-quantization "$quant")
  n=$drafts
  if [ "$algo" = DSPARK ]; then q+=(--set speculative-dspark-block-size "$drafts"); n=$((drafts + 1)); fi
  python3 "$HERE/../sglang/serve.py" "$NAME" --image "$IMAGE" --port 30000 --drop cuda-graph-max-bs --set cuda-graph-max-bs-decode 8 \
    --set speculative-algorithm "$algo" \
    --set speculative-num-draft-tokens "$n" --set speculative-draft-model-path "$dir" "${q[@]}" --log-requests > "$O/$label.cmd" 8>&-
  up=
  for i in $(seq 1 240); do
    curl -sf -m 3 localhost:30000/v1/models > /dev/null && { up=1; break; }
    [ -n "$(docker ps -q -f name=^$NAME$)" ] || break
    sleep 5
  done
  docker logs "$NAME" > "$O/$label.serverlog" 2>&1
  if [ -z "$up" ]; then echo "$label: server did not come up, see $label.serverlog"; docker rm -f "$NAME" > /dev/null 2>&1; continue; fi
  python3 "$HERE/../agentlong/run.py" --url http://localhost:30000 --mode greedy --label "$label" --out "$O/$label.jsonl" \
    --bodies "$O/set/all.jsonl" --manifest "$O/set/all.json" ${ONLY:+--only "$ONLY"} > "$O/$label.log" 2>&1 8>&-
  docker logs "$NAME" > "$O/$label.serverlog" 2>&1
  docker rm -f "$NAME" > /dev/null 2>&1
  echo "$label done ($(date -u +%T))"
done
python3 "$HERE/by_workload.py" "$O" "${BASE:-sg-dflash2}" > "$O/tables.md" 2>&1
echo "tables: $O/tables.md"
