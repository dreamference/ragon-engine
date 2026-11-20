#!/bin/bash
# The section 6 speed recipe on the Metal backend (no drafter): greedy 400 tokens with EOS banned, fresh state per
# prompt, wrapped in caffeinate -i (run with no desktop session).
#   tools/metal/run_recipe.sh BUILD_DIR MODEL_DIR ctx32k IDS_FILE OUT_PREFIX     (--ctx 32768)
#   tools/metal/run_recipe.sh BUILD_DIR MODEL_DIR agentlong IDS_FILE OUT_PREFIX  (16 turns, --ctx 62720)
# CHUNK (default 2048) sets the prefill chunk. Writes OUT_PREFIX.{log,err,jsonl}; agentlong prints mean and p10.
set -u
build=$1 model=$2 kind=$3 ids=$4 out=$5
case $kind in
  ctx32k) ctx=32768 ;;
  agentlong) ctx=62720 ;;
  *) echo "kind is ctx32k or agentlong" >&2; exit 2 ;;
esac
caffeinate -i "$build/ragon-metal-run" --model "$model" --prompts-ids "$ids" --max-tokens 400 --ban-eos \
  --ctx $ctx --chunk "${CHUNK:-2048}" --greedy-out "$out.jsonl" > "$out.log" 2> "$out.err"
rc=$?
echo "rc=$rc" >> "$out.log"
python3 - "$out.jsonl" <<'PY'
import json, sys
r = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
d = sorted(x["decode_tok_s"] for x in r)
p = [x["prompt_tokens"] / x["prefill_s"] for x in r]
i = 0.1 * (len(d) - 1)  # p10 with linear interpolation (numpy's default)
p10 = d[int(i)] + (d[min(int(i) + 1, len(d) - 1)] - d[int(i)]) * (i - int(i))
print(f"{len(r)} prompts: decode mean {sum(d)/len(d):.2f} p10 {p10:.2f} tok/s, prefill mean {sum(p)/len(p):.1f} tok/s")
PY
exit $rc
