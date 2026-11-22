#!/bin/bash
# arms.sh ARM OUTDIR [run.py args...]: start one server arm in docker on $PORT, run the fork bench against it, stop it.
# The caller owns the GPU (lock taken, any other server on the GPU stopped) and restores the box afterwards.
#
# ARM: sglang      the SGLang production launch, plus --enable-cache-report (usage then reports cached tokens)
#      flat        ragon-serve production recipe: flat KV (--no-kv-pages), sessions/batch $FLAT_SESSIONS
#      paged       the same with paged KV
#      fanout      paged KV with room for every fork: --sessions $FANOUT_SESSIONS --batch $FANOUT_BATCH
#      sglang-n, fanout-n   the same servers driven with one request of n=k (run.py --arm n), no retention step
# env: SGLANG_IMAGE RAGON_IMAGE (pinned digests), HF_CACHE DF_CACHE (host dirs mounted at /root/.cache/huggingface
#      and /root/.cache/dreamference), MODEL DRAFT SPARSE CHAT_TEMPLATE (paths inside the container),
#      SGLANG_MOUNTS (extra -v args of the production launch), PORT (18100)
set -u
ARM=$1; OUT=$2; shift 2
PORT=${PORT:-18100}; FLAT_SESSIONS=${FLAT_SESSIONS:-4}; FANOUT_SESSIONS=${FANOUT_SESSIONS:-10}; FANOUT_BATCH=${FANOUT_BATCH:-8}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
NAME=forkbench-${ARM%-n}
RUNARGS=(--label "${ARM%-n}" --url "http://127.0.0.1:$PORT" --out "$OUT/fork.jsonl")
case $ARM in *-n) RUNARGS+=(--arm n --idle 0) ;; esac
DOCKER=(docker run -d --name "$NAME" --gpus all --network host --shm-size 64m
        -v "$HF_CACHE:/root/.cache/huggingface" -v "$DF_CACHE:/root/.cache/dreamference" -e HF_HUB_OFFLINE=1)
RAGON=(ragon-serve --model "$MODEL" --served-model-name RadixArk/Qwen3.8-27B-NVFP4 --host 127.0.0.1 --port "$PORT"
       --max-context 196608 --draft "$DRAFT" --draft-block 12 --fair-pairing 0 --vision on --sparse-head-dir "$SPARSE"
       --graphs --pdl)
case ${ARM%-n} in
  sglang) # shellcheck disable=SC2086
    "${DOCKER[@]}" ${SGLANG_MOUNTS:-} -e TORCHINDUCTOR_CACHE_DIR=/root/.cache/dreamference/sglang/inductor \
      --entrypoint /opt/nvidia/nvidia_entrypoint.sh "$SGLANG_IMAGE" python3 -m sglang.launch_server \
      --model-path "$MODEL" --trust-remote-code --served-model-name RadixArk/Qwen3.8-27B-NVFP4 --host 127.0.0.1 \
      --port "$PORT" --tp-size 1 --mem-fraction-static 0.5 --context-length 262144 --tool-call-parser qwen3_coder \
      --reasoning-parser qwen3 --speculative-algorithm DFLASH --speculative-draft-model-path "$DRAFT" \
      --speculative-num-draft-tokens 12 --speculative-draft-model-quantization modelopt_fp4 \
      --chat-template "$CHAT_TEMPLATE" --attention-backend flashinfer --sampling-backend pytorch \
      --chunked-prefill-size 8192 --disable-prefill-cuda-graph --cuda-graph-max-bs-decode 8 \
      --disable-flashinfer-autotune --mamba-radix-cache-strategy extra_buffer --mamba-ssm-dtype bfloat16 \
      --max-mamba-cache-size 96 --max-running-requests 8 --enable-torch-compile --torch-compile-max-bs 4 \
      --num-continuous-decode-steps 2 --sleep-on-idle --enable-metrics --enable-cache-report > /dev/null ;;
  flat)   "${DOCKER[@]}" "$RAGON_IMAGE" "${RAGON[@]}" --sessions "$FLAT_SESSIONS" --batch "$FLAT_SESSIONS" --no-kv-pages > /dev/null ;;
  paged)  "${DOCKER[@]}" "$RAGON_IMAGE" "${RAGON[@]}" --sessions "$FLAT_SESSIONS" --batch "$FLAT_SESSIONS" --kv-pages > /dev/null ;;
  fanout) "${DOCKER[@]}" "$RAGON_IMAGE" "${RAGON[@]}" --sessions "$FANOUT_SESSIONS" --batch "$FANOUT_BATCH" --kv-pages > /dev/null ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac || { echo "$(date -u +%T) $ARM: docker run failed"; exit 1; }
down() { docker stop -t 30 "$NAME" > /dev/null 2>&1; docker logs "$NAME" > "$OUT/$ARM.server.log" 2>&1; docker rm "$NAME" > /dev/null 2>&1; }
trap down EXIT
T0=$(date +%s)
until curl -sf -m 3 "http://127.0.0.1:$PORT/v1/models" > /dev/null; do
  [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" = true ] || { echo "$(date -u +%T) $ARM: server exited"; docker logs "$NAME" 2>&1 | tail -5; exit 1; }
  [ $(( $(date +%s) - T0 )) -gt 1500 ] && { echo "$(date -u +%T) $ARM: not up after 1500 s"; exit 1; }
  sleep 5
done
echo "$(date -u +%T) $ARM up after $(( $(date +%s) - T0 )) s; $(docker logs "$NAME" 2>&1 | grep -m1 -oE 'drafter head: [a-z]+|kv pages[^,;]*' | tr '\n' ' ')"
python3 "$HERE/run.py" "${RUNARGS[@]}" "$@" > "$OUT/$ARM.run.log" 2>&1
echo "$(date -u +%T) $ARM bench exit $?; $(grep -c . "$OUT/fork.jsonl" 2>/dev/null) rows"
case $ARM in *-n) ;; *)  # exactness probe: the same 8k history and 4 instructions on every arm; compare text_sha
  python3 "$HERE/run.py" --label "$ARM-exact" --url "http://127.0.0.1:$PORT" --out "$OUT/exact.jsonl" --fixed-nonce \
    --sizes 8000 --forks 4 --reps 1 --idle 0 --no-warmup > "$OUT/$ARM.exact.log" 2>&1
  echo "$(date -u +%T) $ARM exact probe exit $?" ;; esac
