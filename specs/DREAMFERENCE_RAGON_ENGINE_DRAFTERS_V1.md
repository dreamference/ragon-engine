# ragon-engine: Several drafters, version 1

Part of the ragon-engine specification; the index is [README.md](./README.md). Written on 2026-11-06 for the owner's decision of the same day: the first version is a drafter chosen by workload (code, prose, long context), two or three draft models shipped with the engine, selected per request by heuristics. This note holds the measurement that the decision asked for. It builds on [MULTI_DRAFTER](./DREAMFERENCE_RAGON_ENGINE_MULTI_DRAFTER.md), whose names (`general`, `work`, `ask`) and bars it uses. Nothing was trained and no engine code changed.

**The answer: ship three drafters, `general`, `work` and `ask`, and two of them have to be trained first.** No trained drafter other than today's beats it on any workload, so a second model loaded today gains nothing. A long-context drafter is not worth having, because length does not move the rate. Routing today's settings per request gives 1.2%.

## 1. How it was measured

One command, on one machine, under the GPU lock:

```bash
bench/drafter/measure.sh OUTDIR            # the engine arms, then the tables of by_workload.py
bench/drafter/candidates.sh OUTDIR IMAGE ARM...   # drafters the engine cannot load, in a scratch SGLang
```

`bench/drafter/workloads.py` builds 52 turns next to the agentlong set: code-long (11: agentlong's edit turns, 25-64K tokens), code-short (8: small bugs to fix, about 12K tokens of context), prose (20: answers without tools: chat, phone, thinking on, non-English) and read-long (13: agentlong's turns without an edit and new long reads over this repository's specs and sources). Every arm replays them through `bench/agentlong/run.py`, N sessions through `n_streams.py`. Ratios are of sums over the turns, with a bootstrap 95% interval over turns.

The new turns are authored fixtures, not sampled traffic. They say how the classes differ; they do not say how often each class occurs.

## 2. Today's drafter by workload

DFlash2 NVFP4 with 12 drafts and the default tree, greedy, one session.

| Workload | Turns | Tokens/step | Accepted/step | tok/s |
|---|---:|---:|---:|---:|
| code-long | 11 | 6.62 | 5.66 | 64.5 |
| code-short | 8 | 6.48 | 5.56 | 68.5 |
| prose | 20 | 4.32 | 3.35 | 48.1 |
| read-long | 13 | 4.87 | 3.89 | 47.5 |

Greedy output was identical in every engine arm, and the same arm run again drifted by 0.2-0.3% (x0.997-0.998).

By kind of turn, tokens per step, greedy and then sampled with a per-turn seed:

| Kind | Greedy | Sampled |
|---|---:|---:|
| short chat | 4.14 | 3.67 |
| phone | 4.55 | 4.23 |
| thinking on | 5.26 | 4.66 |
| non-English | 3.60 | 3.47 |
| long document | 4.22 | 3.78 |
| long repository read | 5.38 | 4.40 |
| long agent turn | 6.73 | 5.60 |
| single-file edit | 6.48 | 5.47 |

Two things follow.

- **Length does not move the rate; the kind does.** A long document runs as a short chat does (4.22 and 4.14), a long agent turn as a short edit does (6.73 and 6.48). A drafter for long context would have nothing to specialise in.
- **The `ask` gate of MULTI_DRAFTER's phase 0 (§6) passes on this proxy.** `asklong` has 4.31 tokens per step over 24 turns and `agentlong` 6.73 over 16: 36.0% fewer [25.0-45.5], against a bar of 15%. So `ask` continues. This is that one gate, not all of phase 0: `workheld` and `structset` are still not built.

The per-workload `read-long` figure mixes the two kinds and should not be read as a class of its own.

N sessions, 4 at once, warm except on prose:

| Workload | Accepted/seq step | tok/s per request | Batched share |
|---|---:|---:|---:|
| code-long | 5.42 | 21.7 | 0.31 |
| code-short | 5.31 | 29.8 | 0.50 |
| read (agent) | 5.73 | 13.4 | 0.31 |
| prose | 2.56 | 19.9 | 0.99 |

Only prose truly batches: long turns of different output lengths rarely overlap, so the other rows are not batching figures. Prose reaches 79.1 tok/s in aggregate while all four decode. Its accepted drafts fall from 3.35 to 2.56 per step because batched steps drop the tree (`speculate.cpp`, non-batched steps only): with the tree off, one session gives 2.55.

## 3. What choosing per request gives today

The engine's settings per workload, as tok/s against those settings (x1.000), with the 95% interval:

| Arm | code-long | code-short | prose | read-long |
|---|---:|---:|---:|---:|
| again (drift) | 0.997 | 0.998 | 0.998 | 0.997 |
| block 16 | 1.016 [.991-1.048] | 1.018 [.970-1.052] | 0.993 [.987-1.000] | 0.996 [.984-1.014] |
| block 8 | 0.903 [.849-.954] | 0.949 [.883-1.067] | 0.967 [.948-.982] | 0.959 [.905-.994] |
| tree off | 0.895 [.880-.916] | 0.930 [.865-.992] | 0.831 [.814-.848] | 0.838 [.813-.881] |
| lookup off | 0.995 | 0.997 | 0.996 | 0.996 |
| lookup 2 (min 8) | 1.100 [.994-1.207] | 1.015 [.987-1.062] | 0.978 [.973-.981] | 0.981 [.968-1.005] |
| sampled | 0.915 [.765-1.050] | 0.842 [.732-1.070] | 0.905 [.860-.948] | 0.821 [.714-.901] |

Choosing among the six greedy arms over the 52 turns, leave-one-out, so a class's own noise does not count:

| Choice | tok/s | x always today's |
|---|---:|---:|
| Always today's settings | 50.67 | 1.000 |
| By declared workload | 51.29 | 1.012 |
| By request shape (§5) | 51.28 | 1.012 |
| Oracle per turn | 52.08 | 1.028 |

That gain would need `--draft-block`, `--tree` and `--lookup`, server-wide today, to become per-session settings. 1.2% does not pay for that on its own, and it misses the +5% bar of MULTI_DRAFTER §4.3.

The one setting worth keeping an eye on is lookup 2 on agent edits, where its steps accept 7.0 tokens each. It carries a cost of about 2% because it turns captured graphs off (`speculate.cpp`), which is why it reads x0.978 on prose at unchanged tokens per step. Recovering that cost could lift code-long's interval above 1, which today it just misses. By turn it is mixed, some 30% faster and some 12% slower. Other minimum matches do not help: min 16 is the only code-long interval above 1 (x1.101 [1.006-1.199]) but costs code-short 6%; min 4 is worse everywhere.

ScaleSearch and RTN are one drafter in two quantizations: x1.000-1.006 on every workload, inside the drift, and choosing between them by workload gives x0.999.

## 4. Candidates for a second or third drafter

No other trained drafter loads in the engine ([BIG_DRAFTER](./DREAMFERENCE_RAGON_ENGINE_BIG_DRAFTER.md) §2): DSpark needs loader and kernel changes (more taps, its own head). So the two public DSpark checkpoints ran in a scratch SGLang on production's image, over 20 of the 52 turns, for tokens per step only. Read them only against the DFlash2 arm of the same run: SGLang's rate is not the engine's.

| Tokens/step | code-long | code-short | prose | read-long |
|---|---:|---:|---:|---:|
| DFlash2, 12 drafts | 5.79 | 4.20 | 3.87 | 3.70 |
| DSpark Agentic, gamma 7 | 3.43 (x.59) | 3.08 (x.73) | 2.26 (x.58) | 2.66 (x.72) |
| DSpark RedHat, gamma 8 | 4.45 (x.77) | 2.95 (x.70) | 3.08 (x.80) | 3.61 (x.98) |

- **Neither wins on any workload**, so a choice among the three is always DFlash2 (x1.000, oracle included). Their smaller block sizes explain little of it: block 8 costs DFlash2 only 3-10% on the engine.
- **Their greedy outputs differ from DFlash2's**, on 18 and 15 of the 20 turns. There is no repeat of the SGLang arms, so this run cannot tell numerical noise from an inexact draft path. Either way §4.3's exactness bar is unproven for them.
- Their tok/s are not reported: the scratch server's logs carry `torch.compile` failures, and SGLang's timing does not predict the engine's anyway.
- The RedHat checkpoint's config has no `model_type` and names remote code, so it ran under a local config with the same layer sizes, tap list and block size.

They are also the larger files, 2.6 GB and 3.8 GB in BF16 against 1.5 GB in NVFP4. Their rates do not justify the loader work.

**Memory is not the constraint.** The 27B takes about 57 GB and a drafter 1.61 GB resident plus 42 MB per session. The lowest available memory measured at 4 sessions was 28 GB; two more drafters take about 3.6 GB.

## 5. The classifier

A sketch in `bench/drafter/route_rule.py`, costing microseconds on the request's JSON and needing no tokenizer. Features: length of the input, share of lines that look like code, share of input items that are tool calls or their outputs, whether tools are offered, whether one is a shell or patch tool, and the kind of the last item.

The rule that uses them is MULTI_DRAFTER §2.2's, which needs only three of them:

| Request | Drafter |
|---|---|
| No tools | `ask` |
| Tools, and the last item is a tool result or a shell or patch tool is offered | `work` |
| Other tools | `general` |

Section 2 says length and code share add nothing to that. A misroute costs speed only, never output.

It routed all 52 turns as they were labelled, and that proves little: the fixtures were written with these shapes. **The §2.2 limit of 10% misroutes has not been tested on real traffic.** The declared hint from `ragon` remains the primary signal; the rule is the fallback for clients that send none.

## 6. Recommendation

1. Three drafters: `general` (today's file, unchanged), `work` and `ask`, as MULTI_DRAFTER has it, without a long-context drafter.
2. Nothing new ships until a drafter is trained and passes §4.3: `work` first, as MULTI_DRAFTER §6 orders it, about two weeks of nights per drafter. Today's numbers leave no gain to ship in the meantime.
3. How much training buys is not measured here. Prose at 4.3 tokens per step against 6.7 is room, not a promise: if `ask` reached the agent rate it would be about 55% more tokens per step on prose, and nothing here says it will.

Two findings that are not a choice of drafter, and may be worth more than any setting here, as separate tasks:

- **The tree in batched steps.** Batched prose accepts 2.56 per step against 3.35 for one session with the tree. If the tree gave batches what it gives one session, that is 31% more drafts on the one load that truly batches; this is inferred, not measured.
- **Lookup 2 with captured graphs on** (§3), then measured again as a default for `work` sessions.
