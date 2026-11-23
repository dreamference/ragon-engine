#!/bin/bash
# arms.sh ARM OUTDIR [run.py args...]: start one server arm in docker on $PORT, run the fork bench against it, stop it.
# The caller owns the GPU (lock taken, any other server on the GPU stopped) and restores the box afterwards.
#
# ARM: sglang      the SGLang production launch, plus --enable-cache-report (usage then reports cached tokens)
#      flat        ragon-serve production recipe: flat KV (--no-kv-pages), sessions/batch $FLAT_SESSIONS
#      paged       the same with paged KV
#      fanout      paged KV with room for every fork: --sessions $FANOUT_SESSIONS --batch $FANOUT_BATCH
#      sglang-n, fanout-n   the same servers driven with one request of n=k (run.py --arm n), no retention step
#      <ragon arm>-t        that arm's server, and instead of the bench the engine's server tests against it:
#                  $PYTEST ("python3 -m pytest") $TESTS_DIR/test_regressions.py test_fanout.py, log $OUT/<arm>-t.pytest.log
#      ckl         fix arm: paged, sessions/batch $FIX_SESSIONS, --checkpoint-limit $CKPT_LIMIT (prefix checkpoints and the
#                  branch point kept past 32768); needs $RAGON_BIN, a ragon-serve with that flag, mounted over the image's
#                  (exact probe at 8000 and 100000)
#      ckb         ckl plus --branch-checkpoint-min $BRANCH_MIN (a fork's resumed snap joins the store for its
#                  siblings when it is that far past the furthest checkpoint, not 2048), 8k and 32k only; $RAGON_BIN
#                  must have that flag (eng/checkpoint-limit b706d0c)
#      ckb-sp0     ckb with --side-prefill 0 (a sibling's suffix prefill alternates on the main stream with all SMs
#                  instead of the 40-SM side stream), 8k and 32k
#      ckb-nf      ckb with --decode-floor off (no floor holds or cut side chunks), 8k and 32k
#      (CKB_SIZES overrides the sizes of ckb, ckb-sp0 and ckb-nf, e.g. 8000,32000,100000)
#      ck24        fix arm: paged, sessions/batch $FIX_SESSIONS, --prefix-checkpoints 24 (the image's binary), 8k and 32k only
#      paged-x100  no bench: the paged arm's exact probe at 100000, the baseline for ckl's
# env: SGLANG_IMAGE RAGON_IMAGE (pinned digests), HF_CACHE DF_CACHE (host dirs mounted at /root/.cache/huggingface
#      and /root/.cache/dreamference), MODEL DRAFT SPARSE CHAT_TEMPLATE (paths inside the container),
#      SGLANG_MOUNTS (extra -v args of the production launch), PORT (18100), RAGON_BIN (host path of a ragon-serve
#      binary run in the image instead of its own; the ckl arm needs it), EXACT_ONLY=1 (no bench, the exact probe only)
set -u
ARM=$1; OUT=$2; shift 2
TAG=$ARM; TESTS=0
case $ARM in *-t) TESTS=1; ARM=${ARM%-t}
  [ -d "${TESTS_DIR:-}" ] || { echo "$(date -u +%T) $TAG: TESTS_DIR is not a directory"; exit 2; } ;; esac
PORT=${PORT:-18100}; FLAT_SESSIONS=${FLAT_SESSIONS:-4}; FANOUT_SESSIONS=${FANOUT_SESSIONS:-10}; FANOUT_BATCH=${FANOUT_BATCH:-8}
FIX_SESSIONS=${FIX_SESSIONS:-8}; CKPT_LIMIT=${CKPT_LIMIT:-196608}; BRANCH_MIN=${BRANCH_MIN:-64}; EXACT_SIZES=8000; BENCH=1
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
NAME=forkbench-${ARM%-n}
RUNARGS=(--label "${ARM%-n}" --url "http://127.0.0.1:$PORT" --out "$OUT/fork.jsonl")
case $ARM in *-n) RUNARGS+=(--arm n --idle 0) ;; esac
case $ARM in
  ckl) EXACT_SIZES=8000,100000
       [ -x "${RAGON_BIN:-}" ] || { echo "$(date -u +%T) $ARM: RAGON_BIN is not an executable"; exit 2; } ;;
  ckb|ckb-sp0|ckb-nf) RUNARGS+=(--sizes "${CKB_SIZES:-8000,32000}")
       [ -x "${RAGON_BIN:-}" ] || { echo "$(date -u +%T) $ARM: RAGON_BIN is not an executable"; exit 2; } ;;
  ck24) RUNARGS+=(--sizes 8000,32000) ;;
  paged-x100) EXACT_SIZES=100000; BENCH=0 ;;
esac
DOCKER=(docker run -d --name "$NAME" --gpus all --network host --shm-size 64m
        -v "$HF_CACHE:/root/.cache/huggingface" -v "$DF_CACHE:/root/.cache/dreamference" -e HF_HUB_OFFLINE=1)
[ -n "${RAGON_BIN:-}" ] && case $ARM in ckl|ckb*) true ;; *) false ;; esac && DOCKER+=(-v "$RAGON_BIN:/usr/local/bin/ragon-serve:ro")
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
  ckl)    "${DOCKER[@]}" "$RAGON_IMAGE" "${RAGON[@]}" --sessions "$FIX_SESSIONS" --batch "$FIX_SESSIONS" --kv-pages \
            --checkpoint-limit "$CKPT_LIMIT" > /dev/null ;;
  ckb)    "${DOCKER[@]}" "$RAGON_IMAGE" "${RAGON[@]}" --sessions "$FIX_SESSIONS" --batch "$FIX_SESSIONS" --kv-pages \
            --checkpoint-limit "$CKPT_LIMIT" --branch-checkpoint-min "$BRANCH_MIN" > /dev/null ;;
  ckb-sp0) "${DOCKER[@]}" "$RAGON_IMAGE" "${RAGON[@]}" --sessions "$FIX_SESSIONS" --batch "$FIX_SESSIONS" --kv-pages \
            --checkpoint-limit "$CKPT_LIMIT" --branch-checkpoint-min "$BRANCH_MIN" --side-prefill 0 > /dev/null ;;
  ckb-nf) "${DOCKER[@]}" "$RAGON_IMAGE" "${RAGON[@]}" --sessions "$FIX_SESSIONS" --batch "$FIX_SESSIONS" --kv-pages \
            --checkpoint-limit "$CKPT_LIMIT" --branch-checkpoint-min "$BRANCH_MIN" --decode-floor off > /dev/null ;;
  ck24)   "${DOCKER[@]}" "$RAGON_IMAGE" "${RAGON[@]}" --sessions "$FIX_SESSIONS" --batch "$FIX_SESSIONS" --kv-pages \
            --prefix-checkpoints 24 > /dev/null ;;
  paged-x100) "${DOCKER[@]}" "$RAGON_IMAGE" "${RAGON[@]}" --sessions "$FLAT_SESSIONS" --batch "$FLAT_SESSIONS" --kv-pages > /dev/null ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac || { echo "$(date -u +%T) $ARM: docker run failed"; exit 1; }
down() { docker stop -t 30 "$NAME" > /dev/null 2>&1; docker logs "$NAME" > "$OUT/$TAG.server.log" 2>&1; docker rm "$NAME" > /dev/null 2>&1; }
trap down EXIT
T0=$(date +%s)
until curl -sf -m 3 "http://127.0.0.1:$PORT/v1/models" > /dev/null; do
  [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" = true ] || { echo "$(date -u +%T) $ARM: server exited"; docker logs "$NAME" 2>&1 | tail -5; exit 1; }
  [ $(( $(date +%s) - T0 )) -gt 1500 ] && { echo "$(date -u +%T) $ARM: not up after 1500 s"; exit 1; }
  sleep 5
done
echo "$(date -u +%T) $ARM up after $(( $(date +%s) - T0 )) s; $(docker logs "$NAME" 2>&1 | grep -m4 -oE 'drafter head: [a-z]+|prefix checkpoints: .*|branch checkpoints: .*|kv pages[^,;]*' | tr '\n' ' ')"
if [ $TESTS = 1 ]; then
  RAGON_SERVE_URL="http://127.0.0.1:$PORT" ${PYTEST:-python3 -m pytest} -q -rfE "$TESTS_DIR/test_regressions.py" \
    "$TESTS_DIR/test_fanout.py" > "$OUT/$TAG.pytest.log" 2>&1
  echo "$(date -u +%T) $TAG pytest exit $?; $(tail -1 "$OUT/$TAG.pytest.log")"; exit 0
fi
[ "${EXACT_ONLY:-0}" = 1 ] && BENCH=0  # re-run of an arm whose window ended before its exact probe
if [ $BENCH = 1 ]; then
  python3 "$HERE/run.py" "${RUNARGS[@]}" "$@" > "$OUT/$ARM.run.log" 2>&1
  echo "$(date -u +%T) $ARM bench exit $?; $(grep -c . "$OUT/fork.jsonl" 2>/dev/null) rows"
fi
case $ARM in *-n) ;; *)  # exactness probe: the same 8k history and 4 instructions on every arm; compare text_sha
  python3 "$HERE/run.py" --label "$ARM-exact" --url "http://127.0.0.1:$PORT" --out "$OUT/exact.jsonl" --fixed-nonce \
    --sizes "$EXACT_SIZES" --forks 4 --reps 1 --idle 0 --no-warmup > "$OUT/$ARM.exact.log" 2>&1
  echo "$(date -u +%T) $ARM exact probe exit $?" ;; esac
