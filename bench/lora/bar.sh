#!/usr/bin/env bash
# The adapters' bar (LORA §6) in one command, in two GPU windows (window.sh; one would pass its 40 minutes): the same long
# agent workload on the base and on the zero adapter (bar.steps), then the fixture prompts on rand-strong, with and without it
# (agree.steps), and bar.py's verdict.
#
#     GPU_LOCK=... bench/lora/bar.sh [OUT_DIR]
#
# The reference comes from ref.steps (a scratch SGLang), once per reference directory: LORA_REF, default ~/lora-ref-strong.
# The numbers stay in OUT_DIR, on the machine.
set -u
here=$(cd "$(dirname "$0")" && pwd)
out=${1:-$HOME/lora-bar/$(date -u +%Y%m%d-%H%M)}
mkdir -p "$out"
"$here/window.sh" lora-bar "the adapters' bar: speed" "$here/bar.steps" "$out" || exit
"$here/window.sh" lora-bar "the adapters' bar: agreement" "$here/agree.steps" "$out" || exit
python3 "$here/bar.py" "$out" "${LORA_REF:-$HOME/lora-ref-strong}" | tee "$out/verdict.log"
exit "${PIPESTATUS[0]}"
