#!/usr/bin/env bash
# The adapters' bar (LORA §6) in one command, in a GPU window (window.sh): the same long agent workload on the base and on
# the zero adapter, greedy continuations of the fixture prompts on the base and on rand-split, and bar.py's verdict.
#
#     GPU_LOCK=... bench/lora/bar.sh [OUT_DIR]
#
# The reference comes from ref.steps (a scratch SGLang), once per reference directory: LORA_REF, default ~/lora-ref.
# The numbers stay in OUT_DIR, on the machine.
set -u
here=$(cd "$(dirname "$0")" && pwd)
out=${1:-$HOME/lora-bar/$(date -u +%Y%m%d-%H%M)}
mkdir -p "$out"
exec "$here/window.sh" lora-bar "the adapters' bar" "$here/bar.steps" "$out"
