# Session fork bench

An agent splits one context into several agents that continue from it. `run.py` measures that through
`/v1/chat/completions` on any OpenAI-compatible server: a fresh history of 8k, 32k or 100k tokens, one short turn,
then 2, 4 or 8 concurrent continuations with a different instruction each, and the same session forked again
after an idle period (retention). Per fork: time to first token, prompt and cached tokens from usage, decode
tok/s; per case: total tok/s while every fork runs, wall clock, counter deltas and the peak of the memory gauges.
`--arm n` sends one request with `n` choices instead (the engine's no-copy fork path, same instruction).

    bench/fork/arms.sh ARM OUTDIR       # sglang | flat | paged | fanout | sglang-n | fanout-n; env in its header
    bench/fork/table.py OUTDIR/fork.jsonl --base sglang

`arms.sh` also runs an exactness probe per arm (`--fixed-nonce`, 8k, 4 forks, greedy) into `exact.jsonl`: the
`text_sha` of each fork must match across the ragon arms.
