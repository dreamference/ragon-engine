# ragon-engine: Retraining the drafter on one RTX 5090

Status: prepared 2026-11-10 (job eng-drafter-retrain-prep); the retrain starts the day the 5090 PC is up.

The owner's decisions (item 12, Q69-Q71) are the frame. The retrain:

- runs on the 5090 (one RTX 5090, 32 GB, Windows with WSL2);
- produces **one** drafter for all workloads;
- targets the **served** `RadixArk/Qwen3.8-27B-NVFP4`: the features are the NVFP4 target's own, captured by ragon-engine, not Minima's and not a BF16 target's;
- is public, recipe included.

The data source is decided later; both options stay open (§2).

## 0. Summary

| | |
|---|---|
| What is trained | the served DFlash2 drafter's exact shape (5 sliding-window layers, hidden 5,120, I 17,408, taps 5/19/33/47/61, frozen 248K-row head), **warm-started** from the stock BF16 `incoai/Qwen3.8-27B-DFlash2` |
| Trainer | SpecForge (MIT) DFlash2, offline mode, on the 5090 under WSL2 |
| Features | captured by ragon-engine on the GB10 nodes (`RAGON_DUMP_FEATURES` through `ragon-force dump`), converted to SpecForge's offline `.ckpt` records, shipped to the 5090 in chunks |
| Per step | 8,192 tokens, BF16 weights, 8-bit AdamW, accumulation 4 (32K tokens per update) |
| Speed (estimate) | about 6,500 tokens/s on the 5090: **~1 h per 20M-token epoch of trained windows**, see §4.4 |
| Back to served form | `bench/drafter/requant.py` (RTN in ModelOpt's layout, reproduces the served RTNcal file byte for byte) then `bench/quant/scalesearch.py` |
| Eval | rung 1 `bench/drafter/rung1.sh` (seeded replay, one stream), rung 2 `bench/drafter/rung2.sh` (exactness, per-position acceptance, agreement gate) |
| Ship bar | rung 1 tokens/step at least the served drafter's on both seeds, no rung-2 bucket lower, every check passing, agreement exact counts not lower (§5.3) |

## 1. What changes and what stays

- **Architecture and config: unchanged.** The engine's `DraftConfig::from_file` accepts the served shape, and ragon-serve's kernels, graphs and tree (`--tree 7`, chain 8) need nothing new. `block_size` stays the stock config's 8 (the engine drafts 12 at serve time either way, as today).
- **The LM head is frozen** (it is the target's), so the sparse-head asset (`dreamference/qwen3.8-27b-sparse-head`), a reduced vocabulary over that head, stays valid and ships unchanged. Every measurement serves with it (`--sparse-head-dir`, log line "drafter head: sparse").
- **The format the engine loads: unchanged.** NVFP4 projections with `weight_scale`, `weight_scale_2` and `input_scale`; BF16 for fc, the norms, the convolution and the selector. §4.6 makes it.

## 2. Data: the two options

Both options give the target's own tokens: the drafter learns the distribution the served target writes. Hold-outs are by **task**:

- the 9 instance ids of the replay set `set12` (rung 1) are never trained on;
- the agreement gate's prompts (54 short, 24 agent) carry no instance ids and are not in either corpus.

### 2.1 Option (a): replays of our own agent sessions

`teacher_dump.py sessions` turns codex rollouts (`rollout-*.jsonl`) into cases:

- one case per compaction segment;
- the segment's last request rendered through `ragon-template --responses` (effort none, the request's own tools);
- the supervised spans are the assistant turns (from after `<|im_start|>assistant\n` through `<|im_end|>`);
- everything else (system, tools, user, tool output) is context only.

Only our own sessions on public tasks: no user data. The owner's sessions are never used (MULTI_DRAFTER §3.1).

What exists today (2026-11-10):

| | |
|---|---|
| Rollouts on the four GB10 nodes | 1,971 (2.09 GB of JSON), 408 distinct instance ids |
| Measured on one node's 539 rollouts | 1,051 cases, 43.7 M tokens, 10.3 M supervised (23.6%) |
| Extrapolated to all 1,971 | 160 M tokens, 37.7 M supervised |
| Features, raw (51,204 B per token) | 8.2 TB; after windowing (§3.2) 8.9 (stored/raw 1.08 on the shard) TB |

**Conflict with MULTI_DRAFTER §3.1.** Every one of these rollouts comes from a SWE-bench **Verified** night: the run manifests name `SWE-bench/SWE-bench_Verified`. §3.1 rules Verified nights out as training data, because a drafter tuned on Verified would make the night's tok/s a measurement of the training set. The task hold-out protects rung 1 (set12), but not the nights.

The §3.1-clean version of option (a) generates fresh sessions on **SWE-smith** (MIT) or **SWE-rebench** (CC-BY-4.0). The same `sessions` command reads their rollouts.

- Cost: about 180K written tokens per stream-hour (MULTI_DRAFTER §3.3).
- 20M written tokens is about 111 stream-hours (28 h on one node at 4 streams): one night on the four GB10 nodes at 4 streams each, or 3-4 nights on one node.

### 2.2 Option (b): public code and instruction prompts

`teacher_dump.py public`:

- takes the prompt field of a public corpus;
- renders it as one user turn (effort none);
- has the served target answer it through ragon-serve (`/v1/completions` on the rendered ids, `--max-tokens 2048`, a fixed seed per row);
- supervises the answer.

Only prompts are taken from the corpus; every supervised token is the target's.

| Corpus | Licence | Rows | Measured on the shard (§3.3) |
|---|---|---|---|
| `ise-uiuc/Magicoder-OSS-Instruct-75K` (prompts written by GPT-3.5; only the prompts are used) | MIT | 75,197, prompts ~1,230 chars | 816 tokens per row, 530 (65%) supervised |
| `nvidia/OpenCodeInstruct` (for scale) | CC-BY-4.0 | ~5M | not run |
| `bigcode/self-oss-instruct-sc2-exec-filter-50k` | ODC-BY | 50,661 | not run |

Size of all of Magicoder:

- 75K × 816 ≈ 61.4 M tokens, 39.9 M supervised;
- features 3.1 (one window per row, nothing to cut) TB;
- generation at the measured 73.7 generated tokens/s per stream.

(b) is short and single-turn. It does not teach the long tool-heavy contexts where the served drafter does most of its work: agent turns at 25-64K, which (a) covers. The usual recipe mixes them: (a) for the agent distribution, (b) for breadth. This is the owner's decision.

## 3. Teacher features

### 3.1 Capture: ragon-engine, the served target, on the GB10 nodes

`RAGON_DUMP_FEATURES=FILE` (src/core/engine.cpp, `Engine::chunk_end`) appends, for every prefill chunk the drafter's capture ran on:

- an int32 header: magic `0x54414546`, pos0, n, nfeat, hidden;
- the n token ids (int32);
- the `cap_` rows as BF16 [n][5 × 5,120].

`cap_` holds the five tapped layers' hidden states, exactly what the drafter's fc reads at serve time. So the training features are the NVFP4 target's own, bit for bit what the served drafter sees.

`ragon-force dump` (bench/agreement/ragon_force.cpp) feeds it:

- one case per input line, `{"ids": [...]}`;
- it resets the engine and prefills the case;
- `RAGON_FORCE_MAX_CONTEXT` sizes the KV pool;
- `RAGON_FORCE_DRAFT` loads the drafter so that the capture runs.

`teacher_dump.py dump` drives it and records per case tokens, prefill time and bytes.

### 3.2 Convert: SpecForge's offline records

`teacher_dump.py convert` (in any image with torch, e.g. production's SGLang image on the CPU):

- reads the raw file;
- checks that every case's records are contiguous and carry the case's ids;
- cuts each case into windows of `max_length` 8,192 tokens with an overlap of 2,048 (the drafter's sliding window). The overlap's loss mask is zeroed, so a supervised token is trained once, with up to 2,048 tokens of real context features before it;
- skips windows with no two consecutive supervised tokens (SpecForge needs two), and NaN windows;
- writes SpecForge's offline record per window to `rows_{g}-{g+2000}/data_{i}.ckpt`, a `torch.save` dict:
  - `input_ids` int64 [T];
  - `loss_mask` int64 [T];
  - `hidden_states` BF16 [1, T, 25,600].

Windowing saves storage only where long stretches have no supervised token: a window with no two consecutive supervised tokens is not written. In agent sessions the supervised turns are spread through the whole session, so nearly every window is kept and the overlap adds ~8% (the (a) shard below: 265,356 stored tokens for 244,876 captured).

### 3.3 Measured on one GB10 node, one GPU window (2026-11-10)

| Shard | Cases | Tokens | Supervised | Prefill tok/s | Raw bytes/token | Windows | .ckpt bytes/row (= per token) |
|---|---|---|---|---|---|---|---|
| (a) 6 sessions from 6 tasks | 6 | 244,876 | 56,483 | 1,870 | 51,204 | 34 (265,356 stored tokens) | 51,216 |
| (b) 24 Magicoder prompts | 24 | 19,591 | 12,731 | 2,236 | 51,204 | 24 | 51,219 |

(b) generation (one stream, served drafter): 12,710 tokens for 24 prompts in 172 s, 73.7 tok/s; all 75K rows ≈ 150 stream-hours, ≈ 9 h on 4 nodes × 4 streams if the streams scale (unmeasured).

### 3.4 Storage and shipping

At about 51 KB per stored token, the features cannot all sit on disk at once. They are produced and consumed in **chunks of about 1M stored tokens (~51 GB)**:

1. the GB10 nodes dump and convert a chunk;
2. it is copied to the 5090 PC (~7 min at 1 GbE, ~1 min at 10 GbE);
3. the trainer reads it;
4. it is deleted after the epoch has used it.

Disk needed on the 5090 PC: two chunks (~100 GB) plus checkpoints. Two epochs dump twice, or keep the chunks on the store drive between epochs (1 TB holds ~20M stored tokens).

Capturing on the 5090 itself is possible only if ragon-engine builds for sm_120 under WSL2 (untested). The target (21 GB) and the trainer (~27 GB) do not fit the 32 GB together, so it would alternate: dump a chunk, then train on it. The checklist (§7) tries it once; the plan does not depend on it.

## 4. The recipe on one 5090

### 4.1 Trainer

- SpecForge `specforge train`, offline colocated DFlash2 (`training.strategy: dflash`, `data.hidden_states_path` = the chunk directory).
- Draft config: the stock `qwen3.8-27b-dflash2.json` (above), unchanged.
- Warm start (`model.draft_checkpoint_path`) from the stock BF16 `incoai/Qwen3.8-27B-DFlash2`, the checkpoint the served NVFP4 file was rounded from (§4.6 checks this).
- Pin the SpecForge commit in the run's notes on day one. The trainer's flags change quickly; this spec names config fields, not CLI flags.

### 4.2 Memory: why 8-bit AdamW

The GB10 benchmark (`bench/drafter/train_step.py`, BIG_DRAFTER §6): this drafter with full-vocabulary CE, BF16 AdamW and per-layer checkpointing peaks at **30 GB at 8,192 tokens per step**.

- The 5090 has 32 GB, of which WSL2 and the Windows desktop leave about 30-31.
- 8-bit AdamW states (torchao `AdamW8bit` or bitsandbytes `PagedAdamW8bit`) take the ~1.7B trainable parameters' optimizer state from 6.9 GB to 3.4 GB: **~26-27 GB peak**.
- No FP32 master weights: +14 GB, does not fit.
- Fallback if the first hour's peak exceeds 29 GB: `max_length` 4,096 with accumulation 8 (the same 32K tokens per update), or chunked CE over the vocabulary.

### 4.3 Hyper-parameters (starting point)

| | |
|---|---|
| Precision | BF16 weights and activations, 8-bit AdamW (β 0.9/0.95, wd 0.01), grad clip 1.0 |
| Sequence | `max_length` 8,192, `batch_size` 1, accumulation 4 → 32,768 tokens per update |
| LR | 1e-4 peak (SpecForge's 5e-4 is for from-scratch; a warm start needs less), 2% warm-up, cosine to 1e-5 |
| Loss | SpecForge DFlash2 defaults: hard-target CE, `loss_decay_gamma` 7, selector warm-up/ramp as shipped |
| Volume | pilot: 2M supervised tokens; full: 20M supervised (MULTI_DRAFTER §3.3), 2 epochs |
| Steps | stored tokens ÷ 32,768 per epoch: e.g. 40M stored → 1,220 updates per epoch |

### 4.4 Hours

The rate is an estimate until the first hour on the 5090 measures it:

- the GB10 trains 3,438 tokens/s at 95 TFLOPS of BF16 GEMM;
- the 5090's dense BF16 tensor rate (~209 TFLOPS) is ~2.2×;
- SpecForge's isolated H200 rank (6.66 samples/s × ~4.5K tokens, 989 TFLOPS) scales to ~6,300.

So plan on **about 6,500 tokens/s**.

| | Stored tokens per epoch | Hours per epoch | 2 epochs |
|---|---|---|---|
| Pilot (2M supervised) | 9.2M (at (a)'s ratio) | 0.4 | 0.8 |
| (a) clean, 20M supervised | 92M (4.7 TB) | 3.9 | 7.8 |
| (b) Magicoder 75K | 61.4 M | 2.6 | 5.2 |

The GB10 side runs in parallel:

- feature capture at 1,870 tokens/s per node;
- generation for (a)-clean or (b);
- capture paces the 5090 at 3.5 (all four GB10 nodes keep one 5090 busy) nodes per trainer.

### 4.5 Checkpoints and the eval plan

- **Save** every 250 updates (~8M tokens, ~21 min) and at each epoch end. Keep the last 3 and the best by held-out loss.
- **Held-out loss.** One converted chunk of held-out tasks (never trained), evaluated at each save: loss and SpecForge's per-position acceptance. This is a cheap trend, not the bar.
- **Candidates.** The best two saves of each epoch go through §4.6 to NVFP4 and then rung 1 and rung 2 on a GB10 node in a GPU window. Each candidate costs about 2 × 25 min (rung 1, two seeds) + ~40 min (rung 2).

### 4.6 Back to the served format

1. `specforge export --to hf` gives the BF16 HF directory.
2. `bench/drafter/requant.py BF16_DIR SERVED_RTN_DIR OUT_DIR`:
   - RTN in ModelOpt's layout: the global scale = amax / (6 · 448) shared across the fused groups (q, k, v; gate, up), E4M3 block scales per 16, E2M1 values;
   - the served activation scales are kept;
   - checked on 2026-11-10: from the stock BF16 checkpoint it reproduces the served RTNcal file byte for byte (all 35 projections; the order amax / (6 · global) matters at E4M3 ties).
3. `bench/quant/scalesearch.py BF16_DIR OUT_DIR SS_DIR`: the per-block scale search the served file had. Checked: the chain from the stock BF16 checkpoint gives the served `qwen3.8-27b-dflash2-nvfp4-scalesearch` exactly, 186 of 186 tensors byte-identical.

## 5. Evaluation

### 5.1 Rung 1: the seeded replay A/B

`bench/drafter/rung1.sh DRAFT_DIR OUT_DIR [SEED…]`:

- a fresh ragon-serve with the drafter under test, at production's flags plus `--sessions 2`;
- the seeded agent-session replay `set12` at one stream, window 12, seeds 1013 and 2026.

Per seed it reports from /metrics:

- tokens per step (generated tokens ÷ verify calls);
- ms per step;
- decode tok/s.

Measured with the served drafter (one GB10 node, 2026-11-10):

| Head | Seed | Tokens/step | ms/step | Decode tok/s |
|---|---|---|---|---|
| sparse (production) | 1013 | 5.80 | 105.3 | 54.72 |
| sparse (production) | 2026 | 5.66 | 105.0 | 53.55 |
| full | 1013 | 5.79 | 106.6 | 53.96 |
| full | 2026 | 5.69 | 106.2 | 53.25 |
| full, earlier baseline (w4) | 1013 / 2026 | 5.570 / 5.471 | 99.1 / 99.3 | 55.81 / 54.74 |

On this build the two heads accept the same (5.80 vs 5.79, 5.66 vs 5.69: within a seed's noise), and the sparse head is about 1.2 ms per step cheaper (+0.8 and +0.3 decode tok/s). The earlier full-head baseline is a different build: since the 2026-11-10 integration a one-stream step verifies 24 tree rows instead of 16, which raises tokens per step by about 0.2 and ms per step by about 7. A candidate is therefore compared only with a served-drafter run on the same build, in the same window. The retrained drafter keeps the sparse head unchanged (§1: the LM head is frozen) and is measured with it.

### 5.2 Rung 2

LOWBIT §7's rung 2 measures the target's divergence. A new drafter leaves the target unchanged: greedy output is the target's own whatever was drafted. So rung 2 for a drafter is its analog, `bench/drafter/rung2.sh DRAFT_DIR OUT_DIR`, in three parts.

1. **Exactness with this drafter** (ragon-run): `--spec-check` (greedy speculative output equals plain decoding), `--graph-check`, `--batch-check` (two batched sessions equal two single ones; run with `RAGON_TREE_ROWS=0`, because since the 2026-11-10 integration a one-stream step verifies 24 tree rows and a batched step 16, so the steps differ while the tokens stay equal).
2. **Acceptance by context.** The agentlong set greedy (16 turns, contexts 28-62K) through ragon-serve, twice:
   - at production's flags (tree verification): `rung2.py turns` gives tokens per step per context bucket (<8K, 8-25K, 25-40K, 40-64K, ≥64K) from each turn's server counters;
   - with the chain alone (`--tree 0`) and `RAGON_DUMP_STEPS`, which records chain steps only (with the tree it writes nothing): `rung2.py steps` gives the acceptance at each draft depth per bucket, the signal a retrain moves directly.

   Greedy makes both drafters' runs the same token sequence, so the buckets compare like for like.
3. **The agreement gate through the candidate.** Greedy completions of the gate's 54 prompts (48 tokens) and 24 agent prompts (400 tokens), from the production-flags server. With an exact verifier a drafter cannot change greedy output, so the criterion is `rung2.py same` (run by rung2.sh when `REF_AGREE` names the served drafter's `agree.jsonl`): the candidate's texts equal the served drafter's text for text, on the same build. The counts against production's tokens (SGLang's) are printed as context. They are not exact even for the served drafter, because ragon-engine and SGLang round differently; they must only be equal for both drafters. A text that differs between the two drafters is an engine bug the candidate exposed, not a quality loss.

Measured with the served drafter, sparse head (W2, one GB10 node, 2026-11-10):

- checks: `--spec-check` and `--graph-check` pass; `--batch-check` passes under `RAGON_TREE_ROWS=0` (W3; at the default 24/16 rows it reports a difference in steps, not in tokens);
- tokens per step at production's flags: 25-40K 6.94 (8 turns), 40-64K 7.20 (8 turns), all 7.11 (16 turns, 5,287 tokens; 63.7 decode tok/s, 111 ms per step);
- agreement against production's tokens: short 20/54 exact (matched prefix 0.492), long 5/24 (0.198). These are the build's numbers a candidate must equal, with `same` reporting 0 texts differing;
- the per-depth chain acceptance was not measured in W2 (W2 ran the earlier script, whose step dump stayed empty under the tree); the first candidate A/B measures both drafters with the script as committed.

### 5.3 Acceptance target

A retrained drafter ships if it does all of the following:

- **rung 1:** tokens per step **at least the served drafter's** on both seeds at one stream, served with the sparse head, in the same window as a served-drafter run. The decode tok/s must not drop (ms per step may rise only with acceptance);
- **rung 2:** every check passes, no context bucket's tokens per step is lower than the served drafter's, and `rung2.py same` finds no agreement text that differs from the served drafter's on the same build (no loss on the agreement gate);
- **the agentlong rule of BIG_DRAFTER §2:** at least +5% tokens/s held out, before it replaces production. A candidate that only ties ships nothing.

## 6. What runs where

| Step | Where | Why |
|---|---|---|
| Rollouts or generation (option a-clean, b) | GB10 nodes, ragon-serve | the served target is NVFP4 on GB10 |
| Case rendering (`sessions`, `public`'s tokenize) | any GB10 node, CPU | needs ragon-template, ragon-force |
| Feature capture (`dump`) | GB10 nodes, GPU windows or nights | the served target's features, bit for bit |
| Convert to .ckpt | GB10 node, CPU, SGLang image | torch only |
| Training, held-out loss | **5090, WSL2** | the only step that needs a big BF16 GPU |
| Export, requant, ScaleSearch | 5090 WSL2 or a GB10 CPU | torch only |
| Rung 1, rung 2 | GB10 node, GPU window | the served engine, against the served drafter |

## 7. 5090 hand-off checklist (in order)

1. **WSL2 up.** Ubuntu 24.04, NVIDIA driver with sm_120 support, `nvidia-smi` inside WSL2 shows the 5090 and about 31 GB free. Put the data on a WSL2 ext4 disk, not /mnt/c (9P is ~10× slower).
2. **Python env.** torch with CUDA 12.8+ (sm_120 kernels), SpecForge at the pinned commit, torchao or bitsandbytes, safetensors. Record the versions.
3. **Copy from the GB10 side:**
   - the stock BF16 `incoai/Qwen3.8-27B-DFlash2`;
   - the served RTNcal and ScaleSearch drafters (for §4.6);
   - SpecForge's `qwen3.8-27b-dflash2.json`;
   - the target's `tokenizer.json`;
   - one converted pilot chunk;
   - one held-out chunk.
4. **Measure the step.** `bench/drafter/train_step.py` (5 layers, 8,192 tokens) with 8-bit AdamW: tokens/s and peak memory. Under 29 GB keeps 8,192; otherwise 4,096 × 8 (§4.2). Write the numbers into §4.4.
5. **Smoke-train** 50 updates on the pilot chunk from the warm start: the loss starts at the stock drafter's level (it is the served drafter, unquantized) and falls.
6. **Round-trip check.** Export the step-0 checkpoint, `requant.py` + `scalesearch.py`: it must reproduce the served ScaleSearch file (it is the stock weights). Then rung 1 on a GB10 node gives the served numbers: this proves the export path.
7. **Pilot**: 2M supervised tokens, 1 epoch. Rung 1 on the best save. Go on only if it is not below the served drafter.
8. **Full run**: the chosen data, 2 epochs, chunks streamed from the GB10 nodes (§3.4). Checkpoints and eval as in §4.5.
9. **Optional, once:** build ragon-engine for sm_120 under WSL2 and compare `ragon-force dump` rows with a GB10 dump of the same case. If they match to BF16 rounding, the 5090 can capture its own chunks between training phases.
10. **Ship:** the candidate passing §5.3 goes through the publish path as the served drafter did (HF asset `dreamference/qwen3.8-27b-dflash2-nvfp4-*`, public with this recipe).

## 8. Files

- `src/core/engine.cpp`: `RAGON_DUMP_FEATURES` (§3.1).
- `bench/agreement/ragon_force.cpp`: `dump` mode, `RAGON_FORCE_DRAFT`, `RAGON_FORCE_MAX_CONTEXT`.
- `bench/drafter/teacher_dump.py`: `sessions`, `public`, `dump`, `convert`, `stats`.
- `bench/drafter/rung1.sh`, `bench/drafter/rung2.sh`, `bench/drafter/rung2.py`: §5.
- `bench/drafter/requant.py` (with `bench/quant/scalesearch.py`): §4.6.
- `bench/drafter/train_step.py`: the step benchmark (BIG_DRAFTER §6).
