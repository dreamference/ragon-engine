# ragon-engine specification

> - **Subject:** Spec: Model-Specific Inference for Qwen3.8-27B on DGX Spark
> - **Target hardware:** NVIDIA GB10 (Blackwell SM121, 128 GB unified memory), one node
> - **Served model:** Qwen3.8-27B NVFP4 with the DFlash2 drafter, the model Ragon serves today with SGLang
> - **Original:** Nov 1, 2026 · revised the same day against Ragon's measured workload; split into this directory on 2026-11-01

This directory holds the specification, split into focused documents. This page is the index.

**Where the monolith went.** Until 2026-11-01 the whole specification was one file, `SPEC.md`, at the repository root, with nineteen numbered sections. It was split here by section, and **every document keeps the original section numbers** (`## 16. …`, `### 16.10 …`), because the reports under `reports/` cite them as `S§n` and `SPEC §n`, the backlog's ranking tags are built on them, and four source comments name them. A citation of `§n` resolves through the map below. Only measured results move from the reports into these documents.

A citation of `reports/<name>` names a working note that is not in this repository.

## Documents

| Document | What it covers |
|---|---|
| [DREAMFERENCE_RAGON_ENGINE_OVERVIEW.md](./DREAMFERENCE_RAGON_ENGINE_OVERVIEW.md) | §1 summary, §2 the measured agent workload, §3 goals and non-goals, §4 the GB10, §6 what production does, §7 the performance targets |
| [DREAMFERENCE_RAGON_ENGINE_MODEL.md](./DREAMFERENCE_RAGON_ENGINE_MODEL.md) | §5 Qwen3.8-27B's shapes, the real precision map, bytes per pass, the details the kernels must honour |
| [DREAMFERENCE_RAGON_ENGINE_ARCHITECTURE.md](./DREAMFERENCE_RAGON_ENGINE_ARCHITECTURE.md) | §8 the offline compiler and the resident runtime, §9 the five shape-specialized kernels per layer |
| [DREAMFERENCE_RAGON_ENGINE_SPECULATION.md](./DREAMFERENCE_RAGON_ENGINE_SPECULATION.md) | §10 the DFlash2 drafter, the context-lookup branch, tree verification, exactness |
| [DREAMFERENCE_RAGON_ENGINE_SESSIONS.md](./DREAMFERENCE_RAGON_ENGINE_SESSIONS.md) | §11 resident sessions, shared-prefix checkpoints, chunked prefill and batches |
| [DREAMFERENCE_RAGON_ENGINE_INTEGRATION.md](./DREAMFERENCE_RAGON_ENGINE_INTEGRATION.md) | §12 how Ragon selects, launches and talks to `ragon-serve`; the memory budget; the fallback to SGLang |
| [DREAMFERENCE_RAGON_ENGINE_CLI.md](./DREAMFERENCE_RAGON_ENGINE_CLI.md) | *Written from the code on 2026-11-01:* the command lines of `ragon-serve`, `ragon-run`, `ragon-tokenize` and `ragon-template`, their defaults, and the `RAGON_*` environment variables |
| [DREAMFERENCE_RAGON_ENGINE_API.md](./DREAMFERENCE_RAGON_ENGINE_API.md) | *Written from the code on 2026-11-01:* `ragon-serve`'s HTTP API: the routes, the request fields, the streamed and unstreamed shapes, the error contract, what is ignored |
| [DREAMFERENCE_RAGON_ENGINE_BATCHING.md](./DREAMFERENCE_RAGON_ENGINE_BATCHING.md) | *Measured on 2026-11-04:* several conversations at once against SGLang (1-8 agent streams, cold and warm), where a batched step's time and a session's memory go, the target for the default (four conversations each moving), the design and the phases (admission control, scheduling, the fused prefill-and-decode pass, wider batches, paged KV) |
| [DREAMFERENCE_RAGON_ENGINE_VALIDATION.md](./DREAMFERENCE_RAGON_ENGINE_VALIDATION.md) | §13 exactness, quality and the benchmark; §18 known pitfalls, each mapped to a regression test |
| [DREAMFERENCE_RAGON_ENGINE_MILESTONES.md](./DREAMFERENCE_RAGON_ENGINE_MILESTONES.md) | §14 the milestones with the standing status, the agreement gate and the exactness policy; §15 risks and open questions |
| [DREAMFERENCE_RAGON_ENGINE_RESEARCH.md](./DREAMFERENCE_RAGON_ENGINE_RESEARCH.md) | §16 the low-level speed-ups from the literature, measured or estimated; §17 sources |
| [DREAMFERENCE_RAGON_ENGINE_BACKLOG.md](./DREAMFERENCE_RAGON_ENGINE_BACKLOG.md) | §19 what users of SGLang and vLLM asked for and what the four clients send, ranked, plus the maintainer's requests |
| [DREAMFERENCE_RAGON_ENGINE_LOWBIT.md](./DREAMFERENCE_RAGON_ENGINE_LOWBIT.md) | *A study of 2026-11-04, no change to the shipped model:* a faster, non-exact target with fewer bits per weight (3-bit, MXFP4, FP6, all-NVFP4 projections), 2:4 sparsity or an FP8 KV cache; bytes and projected step per candidate, the literature on quality, the drafter, the quality protocol, the GB10 microbenchmarks, and the recommendation |
| [DREAMFERENCE_RAGON_ENGINE_BIG_DRAFTER.md](./DREAMFERENCE_RAGON_ENGINE_BIG_DRAFTER.md) | *Written on 2026-11-04 at the owner's request:* would a bigger drafter pay? No bigger public drafter for this target; the drafter's cost per step at 1-4× measured in the engine with dummy weights (NVFP4 and BF16); acceptance against size from the literature; the break-even; the training cost on one GB10 and rented; no-go |
| [DREAMFERENCE_RAGON_ENGINE_MULTI_DRAFTER.md](./DREAMFERENCE_RAGON_ENGINE_MULTI_DRAFTER.md) | *The owner's decision of 2026-11-04, a specification; nothing trained or built:* Ragon's next major release ships several drafters, one per kind of work (`general`, `work`, `ask`); which specialisations pay and the evidence; how the engine picks one per request from the declared workload, what a switch costs (the drafter ring, per-session binding), exactness and its tests; training data from our own target on public tasks, volume and cost on one GB10 or rented; the evaluation sets and the bar to ship; release assets with SHA-256 pins and the registry fields; phases with gates; open questions |
| [DREAMFERENCE_RAGON_ENGINE_DRAFTERS_V1.md](./DREAMFERENCE_RAGON_ENGINE_DRAFTERS_V1.md) | *A measurement of 2026-11-06, no engine change:* today's drafter by workload, what choosing per request gives, the DSpark candidates, the classifier sketch: three drafters, two of them still to be trained, none for long context |
| [DREAMFERENCE_RAGON_ENGINE_VISION.md](./DREAMFERENCE_RAGON_ENGINE_VISION.md) | §20 images: the checkpoint's vision tower, what production does with an image, the engine's four phases, memory and time on GB10, exactness, the API as Ragon's clients send images, images and the prefix caches, the test plan (new on 2026-11-04) |
| [DREAMFERENCE_RAGON_ENGINE_METAL.md](./DREAMFERENCE_RAGON_ENGINE_METAL.md) | *The owner's decisions of 2026-11-04, a specification for the next major release; nothing built:* a native Metal port for Apple silicon (Mac mini M5 Pro first): exactness within the Metal build and across CUDA and Metal, the component map (shared, mirrored, replaced), the weight format, unified memory and the memory plan per Mac size, command-buffer pipelining, simdgroup matrices and the M5's neural accelerators, performance by method (decode from bandwidth, prefill from compute, the verify's row budget), build, CI on a self-hosted Mac, packaging and the egress audit, milestones with gates, open questions |
| [DREAMFERENCE_RAGON_ENGINE_AMD.md](./DREAMFERENCE_RAGON_ENGINE_AMD.md) | *Research, 2026-10-26/27; the first milestone passed, no engine code ported; the postponement to 2.0 is superseded by the Windows node specification (2026-11-06):* a port of the engine to AMD Strix Halo (`gfx1151`) on Windows: the hardware ceiling, what is CUDA- or Blackwell-specific, formats, the plan. Probes and results in `bench/amd/`. |
| [DREAMFERENCE_RAGON_ENGINE_AMD_INVENTORY.md](./DREAMFERENCE_RAGON_ENGINE_AMD_INVENTORY.md) | *Inventory, 2026-11-06:* one row per source file of the engine (src, tests, tools) classified rewritten, hipify, host, replaced or unchanged for the AMD port, the CUDA constructs without a HIP twin by file and line, and the W0 to W3 weeks restated. |

## Section map

| Section of the former `SPEC.md` | Now in |
|---|---|
| §1 | [DREAMFERENCE_RAGON_ENGINE_OVERVIEW.md](./DREAMFERENCE_RAGON_ENGINE_OVERVIEW.md) |
| §2 | [DREAMFERENCE_RAGON_ENGINE_OVERVIEW.md](./DREAMFERENCE_RAGON_ENGINE_OVERVIEW.md) |
| §3 | [DREAMFERENCE_RAGON_ENGINE_OVERVIEW.md](./DREAMFERENCE_RAGON_ENGINE_OVERVIEW.md) |
| §4 | [DREAMFERENCE_RAGON_ENGINE_OVERVIEW.md](./DREAMFERENCE_RAGON_ENGINE_OVERVIEW.md) |
| §5 | [DREAMFERENCE_RAGON_ENGINE_MODEL.md](./DREAMFERENCE_RAGON_ENGINE_MODEL.md) |
| §6 | [DREAMFERENCE_RAGON_ENGINE_OVERVIEW.md](./DREAMFERENCE_RAGON_ENGINE_OVERVIEW.md) |
| §7 | [DREAMFERENCE_RAGON_ENGINE_OVERVIEW.md](./DREAMFERENCE_RAGON_ENGINE_OVERVIEW.md) |
| §8 | [DREAMFERENCE_RAGON_ENGINE_ARCHITECTURE.md](./DREAMFERENCE_RAGON_ENGINE_ARCHITECTURE.md) |
| §9 | [DREAMFERENCE_RAGON_ENGINE_ARCHITECTURE.md](./DREAMFERENCE_RAGON_ENGINE_ARCHITECTURE.md) |
| §10 | [DREAMFERENCE_RAGON_ENGINE_SPECULATION.md](./DREAMFERENCE_RAGON_ENGINE_SPECULATION.md) |
| §11 | [DREAMFERENCE_RAGON_ENGINE_SESSIONS.md](./DREAMFERENCE_RAGON_ENGINE_SESSIONS.md) |
| §12 | [DREAMFERENCE_RAGON_ENGINE_INTEGRATION.md](./DREAMFERENCE_RAGON_ENGINE_INTEGRATION.md) |
| §13 | [DREAMFERENCE_RAGON_ENGINE_VALIDATION.md](./DREAMFERENCE_RAGON_ENGINE_VALIDATION.md) |
| §14 | [DREAMFERENCE_RAGON_ENGINE_MILESTONES.md](./DREAMFERENCE_RAGON_ENGINE_MILESTONES.md) |
| §15 | [DREAMFERENCE_RAGON_ENGINE_MILESTONES.md](./DREAMFERENCE_RAGON_ENGINE_MILESTONES.md) |
| §16 | [DREAMFERENCE_RAGON_ENGINE_RESEARCH.md](./DREAMFERENCE_RAGON_ENGINE_RESEARCH.md) |
| §17 | [DREAMFERENCE_RAGON_ENGINE_RESEARCH.md](./DREAMFERENCE_RAGON_ENGINE_RESEARCH.md) |
| §18 | [DREAMFERENCE_RAGON_ENGINE_VALIDATION.md](./DREAMFERENCE_RAGON_ENGINE_VALIDATION.md) |
| §19 | [DREAMFERENCE_RAGON_ENGINE_BACKLOG.md](./DREAMFERENCE_RAGON_ENGINE_BACKLOG.md) |
| §20 (new, 2026-11-04) | [DREAMFERENCE_RAGON_ENGINE_VISION.md](./DREAMFERENCE_RAGON_ENGINE_VISION.md) |

§20 is the first section written after the split; it has no counterpart in the former file. The documents without a section number have no counterpart in the original file: `DREAMFERENCE_RAGON_ENGINE_CLI.md` and `DREAMFERENCE_RAGON_ENGINE_API.md`, written on 2026-11-01 from the code, two studies of 2026-11-04, `DREAMFERENCE_RAGON_ENGINE_LOWBIT.md` and `DREAMFERENCE_RAGON_ENGINE_BIG_DRAFTER.md`, and the multi-drafter specification of the same day, `DREAMFERENCE_RAGON_ENGINE_MULTI_DRAFTER.md`, and the Metal port's specification of 2026-11-04, `DREAMFERENCE_RAGON_ENGINE_METAL.md`.
