#!/usr/bin/env bash
# One command for the FP4 decode crossover (bench/micro/fp4_decode_bench.cu), under the hosts' GPU lock convention:
# flock ~/.ragon-gpu.lock, one line per event in ~/.ragon-gpu-holds.log. It stops nothing: a resident model server may
# stay up if idle (it holds memory, not the bus), and the script refuses a busy GPU or a running SWE-bench unit.
# Usage: bench/micro/fp4_decode.sh OUT_DIR [ROWS] [RUNS=2]   (from a tree with build/ragon-fp4-decode-bench)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:?out dir}
ROWS=${2:-1,8,16,32,64,128,256}
RUNS=${3:-2}
BIN=${BIN:-$HERE/../../build/ragon-fp4-decode-bench}
WHO=fp4-decode-bench
LOG=~/.ragon-gpu-holds.log
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
util() { nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' '; }
[ -x "$BIN" ] || { echo "no $BIN: ninja -C build ragon-fp4-decode-bench" >&2; exit 2; }
mkdir -p "$OUT"
echo "$(stamp) $WHO waiting for the lock (FP4 decode crossover; nothing stopped)" >> "$LOG"
exec 9> ~/.ragon-gpu.lock
flock 9
T0=$(date +%s)
release() { echo "$(stamp) $WHO lock released, held $((($(date +%s) - T0) / 60)) min" >> "$LOG"; }
trap release EXIT
echo "$(stamp) $WHO lock taken" >> "$LOG"
if systemctl --user list-units --no-legend 'ling-swe-*' 'ragon-swe-*' 2>/dev/null | grep -q .; then
  echo "a SWE-bench unit runs: not measuring" >&2
  exit 3
fi
for i in 1 2 3 4 5; do
  [ "$(util)" = 0 ] || { echo "GPU busy ($(util) %): not measuring" >&2; exit 3; }
  sleep 1
done
for r in $(seq 1 "$RUNS"); do
  { echo "# run $r started $(stamp), GPU $(util) %"; nvidia-smi --query-gpu=memory.used,clocks.sm,clocks.mem,power.draw --format=csv,noheader; } >> "$OUT/window.txt"
  timeout 1800 "$BIN" "$ROWS" > "$OUT/run$r.csv" 2> "$OUT/run$r.err"
  echo "# run $r exit $? at $(stamp), GPU $(util) %" >> "$OUT/window.txt"
done
grep -h '^#' "$OUT/run$RUNS.csv"
