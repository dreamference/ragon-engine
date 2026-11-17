# ragon-engine: one backend interface for the kernels (CUDA, HIP, Metal)

Part of the ragon-engine specification; the index is [README.md](./README.md). Written on 2026-11-13 from the owner's decisions Q46, Q203 and Q204 (2026-11-12): the Windows node's native port starts now with HIP, the Mac node (Metal) is in the 1.0 gate, and both build on **one** backend interface (CUDA, HIP, Metal) with no behaviour change for CUDA. Priorities: nodes before engine speed work.

It builds on [ARCHITECTURE §8–9](./DREAMFERENCE_RAGON_ENGINE_ARCHITECTURE.md) (the compiler, the runtime, the kernels), [METAL](./DREAMFERENCE_RAGON_ENGINE_METAL.md), [AMD](./DREAMFERENCE_RAGON_ENGINE_AMD.md) and [AMD_INVENTORY](./DREAMFERENCE_RAGON_ENGINE_AMD_INVENTORY.md). The code it describes is `src/backend/` and the three interface headers `src/core/kernels.cuh`, `src/core/lora_kernels.cuh` and `src/core/vision.hpp`.

## 0. Summary

- **What.** Every kernel the serve layer launches (attention including the shared and cascade phases, GDN/DeltaNet, the FFN GEMMs, the LM head and the sparse head, the drafter, KV pages, the top-K behind the samplers and the grammar mask, LoRA, the vision tower) is a host function in namespace `ragon::kernels` declared in one of three interface headers. Those declarations name no vendor type: a stream is `gpu::Stream`, a 16-bit float `gpu::bf16` or `gpu::half`. A backend supplies those types (§3), its capabilities (§4) and the definitions of every entry point (§2).
- **How it is chosen.** At compile time: one CMake flag `-DRAGON_BACKEND=cuda|hip|metal` (default `cuda`), one backend per binary, no virtual call (§1).
- **CUDA.** The first implementation is today's `.cu` files, unchanged. The types are aliases of the CUDA ones, so every object file the compiler emits is the same code as before the interface (§6: identical PTX, host code and kernel code; only the debug line tables differ).
- **HIP and Metal.** `RAGON_BACKEND=hip` builds the kernels ported so far (topk.cu first, bit-identical to CUDA, §7); `=metal` is reserved and refused with a message until its row lands (§8). Each one implements §2 against its own types; the serve layer does not change.
- **The runtime (tier 2).** The host files (engine.cpp, speculate.cpp, engine_fanout.cpp, engine_lora.cpp, drafter.cpp, serve/image_input.cpp) call the runtime through `src/backend/runtime.hpp` (streams, events, graphs, allocation, copies, §5.2); the green contexts stay CUDA-only behind `gpu::kGreenContexts`. The CUDA objects are unchanged (§6); the HIP build compiles the host files (§7).

## 1. The interface is a compile-time backend namespace, not an abstract class

Two designs were possible: an abstract `Backend` class with a virtual function per entry point, or one set of declarations whose definitions and types come from the backend chosen when the engine is built. The engine takes the second.

| Question | Abstract class (runtime) | Compile-time namespace (chosen) |
|---|---|---|
| Is there ever a runtime choice? | It exists to make one | METAL §1 rules out a runtime choice of backend; one binary serves one vendor's GPU (a GB10 node, a Strix Halo node, a Mac), the installers ship one binary per platform |
| Native handles | A stream, an event, a `__nv_bfloat16*` must be type-erased (`void*`) and cast back in every implementation, or the interface becomes a template anyway | `gpu::Stream` is `cudaStream_t` under CUDA, `hipStream_t` under HIP, an `id<MTLCommandBuffer>` wrapper under Metal: the native type, checked by the compiler |
| CUDA identity | A virtual call per launch and new host code around every call site: the host objects change, identity has to be measured | The aliases are the same types: same mangled names, same overloads, same code. Identity holds by construction and is checked by comparing objects (§6) |
| Graph capture | An indirect call in the captured region is fine, but the vtable has to live as long as every graph | Nothing new to capture |
| A missing entry point in a new backend | A runtime abort or a stub | A link error naming the function, the day the backend is built |
| Cost of the HIP port | — | AMD_INVENTORY classifies most `.cu` files as hipify: the same source compiled for the other vendor, which is exactly a compile-time backend |

A virtual layer could be added above this one later (for a test double, say) without touching the backends; the reverse is not true.

## 2. The entry points

Every entry point is a host function in `ragon::kernels` (`ragon::vision` for the vision tower). Unless the row says otherwise, each one:

- **enqueues** work on the stream `s` it is given and returns without waiting; it does not allocate, free or synchronize (the exceptions are named);
- takes **device pointers** (`p` below) whose layout is the one in its header comment, and host scalars or small host structs whose fields are copied into the launch parameters, so the caller may change them after the call returns;
- is **graph-capturable**: no host read of a device value, no allocation inside a capture (the `*_prepare` and `prepare_kernels` calls are made once, outside any capture, for that reason);
- is **row-invariant** where the rows path needs it (ARCHITECTURE §9): a row's result depends only on that row's inputs, never on how many rows were batched with it; the speculative and batched exactness rules rest on this;
- is **deterministic**: the same inputs give the same bits on the same build, run after run.

Those five properties are the contract a backend implements. The header comment of each function is its full contract (shapes, layouts, rounding); the tables below give the one-line contract, the file that defines it today and the serve-layer files that call it. `—` means no caller outside the backend's own files: an internal helper or a test entry, kept in the interface because the tests call it.

### 2.1 Configuration and probes (kernels.cuh)

| Entry point | Contract | Callers |
|---|---|---|
| `set_pdl`, `pdl_enabled` | Dependent launch on or off for the rows path's kernels; a hint, the results are the same either way (a backend without it, §4, ignores it) | engine.cpp / — |
| `prepare_kernels` | Sets every kernel attribute (dynamic shared memory) once, outside capture | engine.cpp |
| `set_kv_layout`, `kv_head_stride`, `kv_pos_stride` | The KV caches' strides (elements): head h, position p at h·head_stride + p·pos_stride | engine.cpp, speculate.cpp / — |
| `set_attention_bulk`, `attention_bulk_enabled`, `set_attention_prefetch`, `attention_prefetch_tiles` | attention_rows' tile loads (bulk copies, L2 prefetch distance); hints, bit-identical results | engine.cpp / — |
| `attention_chunk` | The key range size attention_rows splits a context into (the shared and cascade phases and the fanout size their scratch and cuts by it) | engine.cpp, engine_fanout.cpp |
| `set_draft_attention_keys` | How a warp of draft_attention walks the context (1, 2, 4 or 8 keys a load); bit-identical | — |
| `l2_prefetch_weights` | Touches about `bytes` of a tiled weight blob into L2 (the megakernel probe); no output | engine.cpp |
| `attention_rows_scratch_floats`, `topk_scratch_floats`, `sparse_head_scratch_bytes` | Scratch sizes the caller allocates for the matching entry point (host arithmetic) | engine.cpp, engine_fanout.cpp, speculate.cpp |

### 2.2 Weights (kernels.cuh; model.cu at load)

| Entry point | Contract | Callers |
|---|---|---|
| `tiled_bytes_nvfp4`, `tiled_bytes_fp8`, `tiled_bytes_nvfp4_coded` | Bytes of the tiled layout (16-row tile × 128-value chunk contiguous) | model.cu |
| `retile_nvfp4`, `retile_fp8`, `retile_nvfp4_coded` | Checkpoint layout → tiled layout; values and scales bit for bit, only the order changes (METAL §3.2 changes the order for 32-lane simdgroups) | model.cu |
| `scale_code_table` | The code of every scale byte and the table, or false over 64 distinct scales (host) | model.cu |
| `dequant_tiled_nvfp4`, `dequant_tiled_fp8`, `dequant_nvfp4`, `dequant_fp8`, `to_bf16` | Weights → BF16 row-major [N][K] | lora.cpp, engine.cpp / — |

### 2.3 Decode and verify GEMMs: FFN, projections, LM head (kernels.cuh)

| Entry point | Contract | Callers |
|---|---|---|
| `stream_gemm_nvfp4`, `stream_gemm_fp8` | y[M][N] = scale · x · Wᵀ over tiled NVFP4 / FP8 weights, FP16 input with per-row power-of-two scales; the rows path's GEMM for the projections, the FFN and the LM head | engine.cpp, speculate.cpp |
| `stream_gemm_multi` | Several stream GEMMs over the same input in one launch, each bit for bit its own call | engine.cpp |
| `gemv_nvfp4`, `gemv_fp8`, `gemv_bf16` | The plain GEMV forms (reference and tests) | — |
| `bf16_rows`, `bf16_rows_bf16in` | y = x · Wᵀ for narrow BF16 matrices, up to 64 rows, row-invariant (the drafter's fc takes the BF16 input form) | engine.cpp, speculate.cpp |
| `gemm_bf16` | y[M][N] = x · Wᵀ, BF16 in, FP32 out, tensor cores | engine.cpp, speculate.cpp, vision.cu |
| `to_half_rows`, `rmsnorm_half`, `bf16_to_half_rows` | FP32 / BF16 rows → FP16 with a power-of-two scale per row (`xinv` undoes it); rmsnorm_half also writes the norm | engine.cpp, speculate.cpp |
| `mk_tail_rows`, `gdn_persistent`, `gdn_ffn_persistent`, `ffn_persistent`, `attn_persistent`, `attn_persistent_prepare` | The megakernel stages: one persistent kernel per layer (or layer and FFN) that is bit for bit the graph path's kernels; `false` means the shape is not covered and nothing was launched, so the caller takes the graph path. A backend without persistent kernels (§4) returns false | engine.cpp |

### 2.4 Prefill GEMMs and their producers (kernels.cuh)

| Entry point | Contract | Callers |
|---|---|---|
| `quant_act_nvfp4`, `silu_mul_quant_nvfp4`, `quant_act_fp8`, `quant_act_fp8_bf16in` | Activations → NVFP4 / FP8 for the block-scaled prefill GEMM, saturated | engine.cpp |
| `rmsnorm_quant`, `rmsnorm_quant_bf16`, `gated_rmsnorm_fp8`, `gated_rmsnorm_fp8_bf16`, `sigmoid_mul_fp8` | A norm or gate fused with the quantization (the FP32 tensor is not written) | engine.cpp |
| `prefill_gemm_nvfp4`, `prefill_gemm_fp8`, `prefill_gemm_fp8_bf16out` | y[M][0..N) = α · xq · Wᵀ (or +=), tiled weights, any M | engine.cpp |
| `prefill_swiglu_nvfp4` | Gate and up projections in one GEMM, straight into the down projection's quantized input | engine.cpp |
| `narrow_bf16`, `narrow_bf16_bf16in` | Two BF16 [48][K] projections for any M, row-invariant | engine.cpp |

### 2.5 Full attention, shared and cascade phases, KV pages (kernels.cuh)

| Entry point | Contract | Callers |
|---|---|---|
| `attn_prepare`, `attn_prepare_seqs`, `attn_persistent_prepare` | Q/K norms, partial RoPE (or M-RoPE rows for image chunks), the rows' K and V into the caches (flat or paged) at their positions; `_seqs`: each sequence at its own position | engine.cpp |
| `attention_rows` | Causal attention of M query rows over a session's cache (flat or paged), sigmoid gate applied by the caller; split-K with the scratch from attention_rows_scratch_floats | engine.cpp |
| `attention_rows_seqs` | attention_rows over every row of a batch, each sequence over its own cache | engine.cpp |
| `attention_rows_cascade` | The cascade (BATCHING §5.3): the keys below a group's shared cut computed once for the group, the rest per sequence; bit for bit attention_rows_seqs | engine.cpp |
| `kv_pages_copy` | KV of positions [from, to) between a session's pages and a flat buffer; a copy, bits unchanged | engine.cpp |
| `kv_table_set` | table[idx[i]] = val[i] from launch parameters (never a host buffer), on the stream, before the pass that reads it | engine.cpp |
| `kv_seal`, `kv_unseal` | The FP8 mirror of pages and their sealed flags (LOWBIT §8 M2); the flag is written after the mirror | engine.cpp |
| `kv_absmax` | The largest \|K\| and \|V\| of a position range (calibration only) | engine.cpp |
| `tree_gather` | After a tree verify's accept: the accepted rows moved to rows 0..n-1 in every listed buffer (flat or paged), in place | speculate.cpp |

### 2.6 Gated DeltaNet (kernels.cuh)

| Entry point | Contract | Callers |
|---|---|---|
| `gdn_conv`, `gdn_gating`, `gdn_recurrent` | Decode: causal depthwise conv (kernel 4) with state, the gates, the delta rule with q/k L2 norm; state [HV][Dk][Dv] | engine.cpp, speculate.cpp |
| `gdn_conv_seqs`, `gdn_recurrent_seqs` | The verify over a batch, each sequence from its own state (read only) | engine.cpp |
| `gdn_recurrent_fused_seqs` | The same with the previous step's commit replayed first | engine.cpp |
| `gdn_conv_commit_seqs`, `gdn_recurrent_commit_seqs` | The joint commit: the verify that also writes each sequence's accepted state | engine.cpp, engine_fanout.cpp |
| `gdn_conv_tree`, `gdn_recurrent_tree` | The verify over a token tree (ancestors' rows), state unwritten | engine.cpp |
| `gdn_conv_prefill`, `gdn_recurrent_prefill`, `gdn_conv_prefill_bf16`, `gdn_recurrent_prefill_bf16` | Prefill: parallel conv, chunked recurrence (32-token chunks at absolute positions), the same math | engine.cpp |
| `gated_rmsnorm` | out = rmsnorm(x) · w · silu(z) | engine.cpp |

### 2.7 Elementwise and embedding (kernels.cuh)

| Entry point | Contract | Callers |
|---|---|---|
| `embed` | Token ids → rows of the embedding table | engine.cpp, speculate.cpp |
| `rmsnorm` | out[r] = x[r] · rsqrt(mean(x[r]²) + eps) · (1 + w or w) | engine.cpp, speculate.cpp |
| `add_inplace`, `silu_mul`, `sigmoid_mul`, `copy_rows_bf16` | The obvious elementwise operations | engine.cpp, speculate.cpp |

### 2.8 Drafter and sparse head (kernels.cuh)

| Entry point | Contract | Callers |
|---|---|---|
| `draft_qk_rope`, `draft_qk_rope_seqs` | Per-head RMSNorm and NeoX RoPE at each row's position, in place | speculate.cpp |
| `draft_store_kv`, `draft_store_kv_seqs` | Rows' K/V into the BF16 rings at their positions | speculate.cpp |
| `draft_attention`, `draft_attention_seqs` | The block's queries over the cached window and the block | speculate.cpp |
| `grouped_conv` | DFlash2's two-tap grouped dynamic convolution over a block | speculate.cpp |
| `selector_lattice` | The candidate selector's 16 × 16 transition scores | speculate.cpp |
| `draft_head_rows` | The drafter's LM head over n sequences: the sparse head over each sequence's mask or the full head; values and ids bit for bit the n = 1 call. **Allocates nothing but may issue device-to-device copies** into its scratch | speculate.cpp |
| `stream_gemm_gather_nvfp4`, `sparse_head_select`, `sparse_head_ids` | The sparse head's parts (draft_head_rows calls them) | — |
| `sparse_head_mark` | mask[ids[i]] = 1 (a session's context vocabulary) | speculate.cpp |

### 2.9 Samplers and the grammar hook (kernels.cuh on the device, host code above it)

| Entry point | Contract | Callers |
|---|---|---|
| `topk_rows` | Top-K (K ≤ 64) of each row of x [rows][V], values descending, ties to the lower index | speculate.cpp, engine_fanout.cpp |
| `topk_rows_masked` | The same under a token mask per row: mask [rows][words] int32, bit id%32 of word id/32 set means allowed. **This is the grammar's device hook**: the grammar (src/grammar, src/core/constraint.hpp, src/serve/structured.hpp) builds the mask on the host and nothing else of it touches the GPU | speculate.cpp |

The samplers themselves (`sampling.hpp`: sample_logits, sample_logits_masked, sample_topk, mask_allows, the finite checks) run on the host over the top-K the device returns, and are backend-independent (§2.12).

### 2.10 LoRA (lora_kernels.cuh)

| Entry point | Contract | Callers |
|---|---|---|
| `lora_shrink`, `lora_expand` | t = x · Aᵀ per slot, y += scale · B · t per segment (the rows path) | engine.cpp, engine_lora.cpp |
| `lora_shrink_prefill`, `lora_gemm_parts` | The prefill path's shrink on tensor cores; the segment split by output (host) | engine_lora.cpp |
| `prefill_gemm_fp8_lora` | prefill_gemm_fp8 with the adapter's B · t folded in | engine.cpp |

The PTX helpers below `#ifdef __CUDACC__` in lora_kernels.cuh are CUDA device code, not interface.

### 2.11 Vision tower (vision.hpp, namespace ragon::vision)

| Entry point | Contract | Callers |
|---|---|---|
| `layernorm`, `rotary`, `attention`, `bias_act`, `add_pos` | The tower's kernels (bidirectional attention, 2-D rotary, interpolated position embedding) | vision.cu (and the vision tests) |
| `embed_rows` | An image's rows of the language model's input | engine.cpp, vision.cu |

`VisionTower` in the same header is host code over these and `gemm_bf16`.

### 2.12 Backend-independent code

These parts compute on the host or only arrange data; every backend shares them unchanged: `sampling.hpp` (the samplers and finite checks), the grammar (`src/grammar/grammar.hpp`, `src/core/constraint.hpp`, `src/serve/structured.hpp`), `tree.cpp` (the draft tree), `accept.cpp`, `lookup.cpp`, `prefix_kv.cpp` (the prefix store's index), `kv_pool.cpp` (page accounting), `safetensors.cpp`, and the host halves of vision.hpp (`vision_positions`, `torch_linspace`). A backend that needs one of them on the device adds an entry point; it does not fork them.

### 2.13 Device helpers shared by the kernels

`kernels.cuh` also carries small device functions every attention and GDN kernel uses (`pos_value`, `rope_delta`, `seq_of`, `seq_pos`, `seq_rdelta`). They compile where the backend compiles device code from C++ (`RAGON_GPU_DEVICE_CODE`: CUDA and HIP); Metal reimplements them in MSL with the same arithmetic (METAL §3).

## 3. The types

`src/backend/backend.hpp` includes the chosen backend's `types.hpp`, which defines in namespace `ragon::gpu`:

| Type | Meaning | CUDA | HIP | Metal |
|---|---|---|---|---|
| `gpu::Stream` | An ordered queue of device work | `cudaStream_t` | `hipStream_t` | a handle to the engine's command-buffer queue (§8) |
| `gpu::bf16` | BFloat16 storage, round to nearest even | `__nv_bfloat16` | `__hip_bfloat16` | `bfloat` (MSL), `uint16_t` storage on the host |
| `gpu::half` | IEEE binary16 storage | `__half` | `__half` | `half` (MSL), `uint16_t` storage on the host |
| `gpu::kWarpMask` | The full-warp mask of the `*_sync` warp intrinsics (a constant, not a type) | `0xffffffffu` (`unsigned`) | `0xffffffffull` (`unsigned long long`: HIP's warp-sync functions static_assert a 64-bit mask; wave32 only, §7) | — (simdgroup functions take no mask) |

They are aliases, never wrappers: an alias leaves the mangled name, the overload set and the code unchanged, which is what keeps CUDA byte-identical (§6). A device pointer is a raw pointer (`gpu::bf16*`, `float*`); on CUDA and HIP it is a device address, on Metal see §8. The KV, tree and batch descriptor structs (`DevPos`, `TreeDesc`, `GatherBuf`, `SparseHeadBufs`, `MkAttnParams`, the `*Seqs` structs) are plain structs of such pointers and integers and are part of the interface.

### 3.1 Device helpers and libraries

What the kernels' `.cu` files spell differently between vendors, beyond the types, goes through two headers; no other vendor name appears in a ported file.

- `src/backend/device.hpp` includes the chosen backend's `device.hpp`, which defines in `ragon::gpu`: `bf16x2` (two bfloat16, low first) and `floats_to_bf16x2(lo, hi)` (each rounded to nearest even); `e4m3_to_float`, `e4m3x2_to_float2` (OCP FP8 E4M3, exact) and `e4m3_from_float` (host and device, round to nearest even, saturating at ±448); `mul_rn` (a product never fused into an FMA); `prefetch_l2` (a hint; HIP drops it); and the runtime calls the kernels' host halves make (`Error`, `kSuccess`, `last_error`, `error_string`, `memset_async`, `copy_d2d_async`, `sm_count`, `blocks_per_sm`). On CUDA each is a one-line inline forward to the spelling the kernels used before, so the CUDA objects keep their code (§6); on HIP they are the HIP spellings, or a few lines where HIP has none (`__floats2bfloat162_rn`, the `__nv_cvt_fp8*` conversions).
- `src/backend/cub.hpp` makes `gpu::cub` a namespace alias for CUB on CUDA and hipCUB on HIP (`sparse_head.cu`'s flagged select).
- Tests reach device memory through `tests/gpu_test.hpp` (`ragon::test`: alloc and free, host-device copies, zero, a sync check, and the host's exact E4M3 decode), so one test source builds for either backend.

## 4. Capabilities

`types.hpp` also states, as `constexpr bool` in `ragon::gpu`, what the backend's hardware path offers, so a caller (or a test) can know without a vendor check. CUDA sets all of them; nothing branches on them yet, they record the contract the HIP and Metal rows must meet.

| Flag | What it means | Where it matters | Without it |
|---|---|---|---|
| `kName` | `"cuda"`, `"hip"` or `"metal"` | logs, `--version` | — |
| `kDependentLaunch` | Programmatic dependent launch | `set_pdl` | the hint is ignored; results identical |
| `kGraphs` | Captured graphs of a decode or verify step | engine.cpp's graph path | the step is launched kernel by kernel (same results, more launch time) |
| `kPersistentKernels` | The megakernel stages (§2.3) | `*_persistent` | they return false; the graph path runs |
| `kGreenContexts` | SM partitions for the prefill/decode split | engine.cpp | one context, priority streams |
| `kBulkCopy` | TMA bulk copies in attention_rows | `set_attention_bulk` | the plain loads (bit-identical) |

## 5. The CUDA implementation

### 5.1 Tier 1: the kernels (done)

- `src/backend/backend.hpp` picks the backend by macro: no macro is CUDA, `RAGON_BACKEND_HIP` is HIP (§7), `RAGON_BACKEND_METAL` stops with `#error` until its row lands.
- `src/backend/cuda/types.hpp` includes `cuda_bf16.h`, `cuda_fp16.h` and `cuda_runtime.h`, defines the aliases and capabilities, and `RAGON_GPU_DEVICE_CODE` when nvcc compiles device code.
- The three interface headers include `backend/backend.hpp` instead of the CUDA headers and spell `gpu::Stream`, `gpu::bf16`, `gpu::half`. The include block keeps the same number of lines (comment lines fill it), so every later line keeps its number and `-lineinfo` points where it pointed.
- The kernel files ported to HIP (§7) include `backend/device.hpp` (and `backend/cub.hpp`) in place of the CUDA headers, again keeping the include block's line count, and spell the helpers of §3.1 instead of the CUDA intrinsics HIP lacks. Their CUDA objects are unchanged by the port (§6 at eng/hip-backend against its base 1ecaf82: 58 of 59 objects identical; the 59th is `tests/kernel_tests.cu`, moved onto `gpu_test.hpp` on purpose).
- The definitions stay in today's files (`src/core/*.cu`), unmoved: about ten branches in flight edit them, and a move would turn each of their merges into a conflict. Moving them to `src/backend/cuda/` is a later, mechanical commit (a rename with no content change), best made when few branches are open.
- CMake: `RAGON_BACKEND` (cache string, `cuda` by default, `hip` builds the ported kernels (§7), `metal` refused with a message). With `cuda` the CUDA language is enabled exactly as before; every compile line is unchanged.

**Merging a branch that predates the interface.** Conflicts can only arise in the three interface headers. Apply to the branch's new lines there: `cudaStream_t` → `gpu::Stream`, `__nv_bfloat16` → `gpu::bf16`, `__half` → `gpu::half`. The `.cu` files and the host files may keep the CUDA spellings: under CUDA they are the same types.

### 5.2 Tier 2: the runtime (done for the host files)

The host files called the CUDA runtime directly. The census at main 47277016 (calls, by file): engine.cpp 118, speculate.cpp 66, engine_fanout.cpp 14, engine_lora.cpp 8, image_input.cpp 5, drafter.cpp 4, about 60 distinct functions:

- **copies and memsets**: `cudaMemcpyAsync` 49, `cudaMemcpy` 15, `cudaMemset` 13, `cudaMemsetAsync`;
- **synchronization**: `cudaStreamSynchronize` 32, `cudaEventRecord` 12, `cudaStreamWaitEvent` 6, `cudaEventSynchronize` 5, `cudaEventQuery`, `cudaDeviceSynchronize`, `cudaEventElapsedTime`;
- **objects**: `cudaEventCreate(WithFlags)`, `cudaEventDestroy`, `cudaStreamCreateWithFlags`, `cudaStreamCreateWithPriority`, `cudaStreamDestroy`, `cudaDeviceGetStreamPriorityRange`;
- **memory**: `cudaMalloc`, `cudaFree`, `cudaHostAlloc`, `cudaFreeHost`, `cudaMallocFromPoolAsync`, `cudaFreeAsync`, `cudaMemPoolCreate`, `cudaMemPoolSetAttribute`, `cudaMemPoolDestroy`, `cudaMemGetInfo`;
- **graphs**: `cudaStreamBeginCapture`, `cudaStreamEndCapture`, `cudaStreamIsCapturing`, `cudaGraphInstantiate`, `cudaGraphLaunch`, `cudaGraphExecDestroy` 11, `cudaGraphDestroy`;
- **green contexts** (driver API): `cuDeviceGet`, `cuDeviceGetDevResource`, `cuGreenCtxCreate`, `cuGreenCtxStreamCreate`, `cuGreenCtxDestroy`;
- **errors and device**: `cudaGetErrorString`, `cudaGetLastError`, `cudaGetDevice`.

The host structs in `model.hpp`, `drafter.hpp` and `engine.hpp` also spell the CUDA types. A few kernel entry points use the runtime internally (`cudaFuncSetAttribute` in the `prepare` calls, `cudaDeviceGetAttribute` for the SM count, a cached `cudaMalloc` in stream_gemm.cu's split-K buffer, `cudaMemsetAsync` and `cudaMemcpyAsync` in sparse_head.cu); those are the backend's own business.

The facade (eng/hip-w1): `src/backend/runtime.hpp` includes `backend/cuda/runtime.hpp` or `backend/hip/runtime.hpp` and gives `ragon::gpu` the runtime's types (`Error`, `Event`, `Graph`, `GraphExec`, `MemPool`, the copy kinds, capture modes, pool attributes), its constants (`kSuccess`, `kH2D`, `kEventDisableTiming`, `kStreamNonBlocking`, `kHostAllocDefault`, `kGraphNodePriority`, ...) and its calls (`memcpy_async`, `memcpy_2d_async`, `memset_async`, `event_record`, `stream_wait_event`, `capture_begin/end`, `graph_instantiate`, `pool_alloc_async`, `host_alloc`, ...). `backend/trace.hpp` holds the NVTX ranges (`range_push/pop`, no-ops on HIP). The host files, and the host structs in `model.hpp`, `drafter.hpp` and `engine.hpp`, spell only `gpu::` names.

- **CUDA: function references, not wrappers.** Each call is `inline constexpr Error (&memcpy_async)(...) = cudaMemcpyAsync;`, so a call through the facade is the same direct call. One-line inline wrappers were tried first and are not neutral: GCC numbered the SSA values differently, which swapped the operands of four compares in engine.cpp.o and speculate.cpp.o (same behaviour, different bytes). With references every host object is byte-identical (§6). Only `device_alloc` (CUDA's `T**` template overload of `cudaMalloc`) and `copy_d2d_async` stay inline functions.
- **HIP** maps every call one to one (`hipHostMalloc`/`hipHostFree` for the pinned host memory, `hipGraphInstantiateWithFlags` for the instantiation with `hipGraphInstantiateFlagUseNodePriority`); TheRock 10.0.0's `hip_runtime_api.h` has a twin for each. Two things have none and are kept out of the facade: the green contexts (driver API; on HIP `make_side_stream` creates a plain non-blocking stream and says so, §4 "one context, priority streams") and nvJPEG (`decode_jpeg_gpu` returns false on HIP, so JPEGs take the CPU libjpeg path every other format already takes).

## 6. How "no behaviour change" is proved

1. **Static identity.** Base (main 47277016) and branch are built clean at the same path with the same flags. The compile commands are identical (59 translation units). Every C++ object is byte-identical. Every CUDA object has identical PTX, identical host `.text` and identical kernel code sections; it differs only in the line tables (`.debug_line`, `.nv.merc.debug_line`: the CUDA headers are now reached through `backend/cuda/types.hpp`), in the `.nv_fatbin` that contains them, and in nvcc's process-numbered temporary symbol in `.strtab`, which differs between two builds of the same tree anyway. The linked binaries differ only in `.note.gnu.build-id`, `.nv_fatbin` and `.strtab`.
2. **Dynamic identity.** On a GB10: ctest; the exactness suite (`ragon-run --spec-check`, `--prefix-check`, `--batch-check` with and without graphs, dependent launch and pages, `--fanout-check`, `--graph-check`); greedy identity on agentlong (16 of 16 identical to base); one-stream and four-stream throughput within the run-to-run spread (A-B-B-A, two runs each). The results of this job are in its report.

A later change to the interface keeps the same proof: an alias or a declaration change must leave the CUDA code sections identical; a real code change is measured like any engine change.

## 7. HIP (Windows node, Q203)

The HIP row adds `src/backend/hip/types.hpp` (`hipStream_t`, `__hip_bfloat16`, `__half` from `hip_bf16.h` and `hip_fp16.h`; capabilities: graphs yes, dependent launch no, green contexts no, bulk copy no, persistent kernels per AMD §5), lifts the `#error` in backend.hpp and the CMake refusal, and compiles the `.cu` files as HIP (AMD_INVENTORY: hipify for most, rewritten for the tensor-core and PTX kernels). The entry points and their contracts are §2 unchanged; the exactness rules of AMD apply within the HIP build.

**State (eng/hip-backend, 2026-11-14).** The row is in place and W0's kernels are ported (`topk.cu`, `kernels.cu`, `draft_kernels.cu`, `sparse_head.cu`, with `elem.cuh` and `attention_body.cuh`):

- `hip/types.hpp` as above, with `kPersistentKernels` false (the Windows driver reports no cooperative launch) and `gpu::kWarpMask` 64-bit (§3). Only wave32 targets are supported (gfx1151; the test checks `warpSize == 32`): a wave64 target would need the masks and the warp-level reductions widened.
- `launch.cuh` includes `backend/backend.hpp`; under HIP `launch_kernel_pdl` and `launch_kernel` are a plain `<<<>>>` launch plus `hipGetLastError` (no dependent launch, `kDependentLaunch` false). The CUDA templates are unchanged.
- `topk.cu` compiles from the same source under both backends (`gpu::kWarpMask` for the three masks, `gpu::Stream` for the stream parameters); no hipified copy is kept. hipify-clang alone does not get there: it keeps the 32-bit masks, which HIP refuses, and leaves `cudaLaunchKernelEx` and the programmatic-serialization attribute in launch.cuh unconverted.
- CMake: `RAGON_BACKEND=hip` enables the HIP language (`CMAKE_HIP_ARCHITECTURES` gfx1151 by default) and builds only the ported kernels (`ragon_kernels_hip`) and their tests (`ragon-topk-tests`, `ragon-kernel-tests`, `ragon-w0-tests`); the rest of the tree follows file by file. On Windows the CXX and HIP compilers are both TheRock's `clang++` (GNU-like driver) inside a VS 2022 developer environment: with `clang-cl` as CXX, CMake 3.31 compiles HIP sources with the clang-cl rule, refuses the MSVC runtime selection for HIP and puts `/machine:x64` in the link line.
- Exactness (A-gate 2): `tests/topk_tests.cu` draws its inputs from `std::mt19937` with integer arithmetic only (the same on every standard library), covers plain, masked and special values (±inf, ±0, subnormals, ties) at K 1, 16, 20 and 64, checks every output bit for bit against a host reference, and hashes all outputs. The HIP build on gfx1151 (TheRock 10.0.0) gives the same bytes as the CUDA build on a GB10 (FNV-1a `bc18c88dc89f0d68`, checked in the test), across runs.
- W1 stays CUDA-only for now, behind `#if !defined(RAGON_BACKEND_HIP)` with the reason on the line: the sparse head's GEMMs on `stream_gemm` (`sparse_head.cu`). Their entry points are not defined in the HIP build, so a caller links only when the W1 port lands. The prefill kernels (`prefill_gdn.cu`, `prefill_gemm.cu`), the megakernel and LoRA are not ported yet either. `model.cu` holds no kernels: its device allocations and copies belong to the runtime tier (§5.2).
- The sparse head's approximate scores (`sparse_head.cu`) use `mma.sync` on CUDA. HIP has no `mma.sync`, so the HIP stage computes the same products on the vector units under the determinism rule: the CUDA kernel's warp layout, each lane's 64 products summed in index order, the row's four lanes joined by a fixed butterfly, no split-K and no atomics. Its bits are not the tensor-core stage's (another order of the same products; the scores are an approximation either way), but a host loop in the same order reproduces them (`w0_tests`' HIP-only `sparse_head_approx_hip_m3` and `_m20` compare the stage's keys with that loop bit for bit). The selection after it (a histogram with integer atomics, which are exact in any order, then a flagged select) is the same source on both backends.
- Exactness of W0 (`tests/w0_tests.cu`, 64 checks on both backends and 2 more on HIP): every W0 kernel (embedding, the norms, the elementwise kernels, the conversions and dequantizers, the bf16 GEMVs, the DeltaNet conv, gating and recurrence in chain, sequence and tree form, attention's prepare, the KV page moves, the drafter's non-GEMM kernels, the sparse head's selection) against an FP32/FP64 host reference, plus the batched and tree forms against the single-sequence kernels bit for bit. Every output is hashed and dumped, so two runs, or the two backends, compare byte for byte. The inputs come from raw `std::mt19937` words only. On gfx1151 every check passes, and two runs give identical bytes. Kernels that only move, convert or add (the conversions, the dequantizers, embed, add, the KV moves, `tree_gather`, the cache stores, the selection) are exact against the host, so their bytes must match CUDA's. Kernels that use the fast transcendentals (`__expf`, `rsqrtf`, `sincosf`, `powf`, `log1pf`: the norms, SiLU and sigmoid, gating, the recurrence's normalization, the rotations, the drafter's softmax) are held to the tolerance of the host reference across backends: the two vendors' approximations of these functions differ in the last bits. Within one backend they stay deterministic. Measured at eng/hip-backend (gfx1151 against a GB10, 2026-11-14): 36 of the 63 shared outputs are identical, the GEMVs and the fixed-order FP32 reductions among them; the 27 that differ all go through those functions, by 2 to 3 ulp in the elementwise kernels and at most 3e-6 absolute anywhere.
- The exactness rule for the HIP backend (coordinator, 2026-11-14): every kernel gives identical bytes on every run of the same backend; kernels that move, convert, add or reduce in a fixed order give CUDA's bytes; kernels that go through the hardware `exp`, `rsqrt`, `sin`/`cos`, `pow` or `log1p` are held to the FP32/FP64 host reference's tolerance across backends. The 27 outputs of `w0_tests` that differ between gfx1151 and a GB10, with the largest gap seen (ulp of FP32; the recurrence's ulp counts are large only on values near zero, so its absolute gap is given too): `rmsnorm_gemma_1024` 3; `silu_mul` 3; `sigmoid_mul` 2; `gated_rmsnorm` 3; `gdn_conv`, `gdn_conv_seqs`, `gdn_conv_commit_seqs`, `gdn_conv_tree` 2; `gdn_gating_g`, `gdn_gating_beta` 2; `gdn_recurrent`, `gdn_recurrent_seqs`, `gdn_recurrent_commit_seqs` 1504 (4.2e-9 absolute); `gdn_recurrent_state` 16384 (6.0e-8); `gdn_recurrent_commit_seqs_state` 347849 (3.0e-8); `gdn_recurrent_tree`, `gdn_recurrent_tree_vs_ref` 7024 (3.7e-9); `attn_prepare_q` 8 and `attn_prepare_seqs_q` 15 (2.4e-7); `draft_qk_rope`, `draft_qk_rope_seqs` 309 (3.05e-6); `draft_attention_keys0`, `_keys1`, `_keys2`, `_keys4`, `_keys8`, `draft_attention_seqs` 337 (2.4e-7).
- Host files (eng/hip-w1, §5.2): `ragon_host_hip` (CMake OBJECT library, `RAGON_HIP_HOST`, default on) compiles engine.cpp, speculate.cpp, engine_fanout.cpp, engine_lora.cpp, drafter.cpp and serve/image_input.cpp against HIP; they are not linked yet (their kernels' files are still CUDA-only). It needs nlohmann_json, libjpeg, libpng and libwebp (vcpkg x64-windows on the Windows node; list vcpkg before TheRock in `CMAKE_PREFIX_PATH`, whose own nlohmann_json config names a missing natvis file). Open for the Windows lanes: `long` is 32 bits there (speculate.cpp's `1L << 40` overflows), and HIP marks `hipError_t` `[[nodiscard]]` where CUDA does not (the teardown calls that drop it warn).
- The stream GEMM (eng/hip-w1, W1): `src/core/stream_gemm_hip.cu` is compiled in place of `stream_gemm.cu`, a file of its own because the CUDA kernels are built around `mma.sync`, `ldmatrix`, the coded scales and the persistent kernels. It is the gfx1151 gate's kernel (bench/amd/vg.hip, AMD §6.5): E2M1 and E4M3 decoded to FP16 in software (exact), WMMA 16x16x16 FP16 into FP32, one wave per 16 weight rows walking its K range in a fixed order, the activation rows staged in LDS (rows past M as zeros), no persistent barrier. M above 16 runs as 16-row passes. A weight the loader marks (`stream_gemm_split_k`, CUDA ignores it) splits K: the drafter's two conv projections, which run alone with 80 tiles (d_kproj row: 70 to 164 GB/s). The split count S comes from the marked weight's shape (`splits(N, K)`: none from 256 tiles up or when K is not whole 256-K chunks; else the fewest equal whole-chunk parts reaching 320 waves) and is the same in every launch of that blob, alone or grouped. Grouped weights are not marked: in a group whose wide target leaves a tail (q|k|v), split k and v no longer fill that tail and the group ran about 5% slower (a_qkv 214 vs 226 GB/s). Each split stores its FP32 partial tile and takes a ticket from a per-tile counter; the last one adds the S partials in split order and writes the row, so the sum does not depend on which split finishes last, and no second launch is needed. The partials and counters are made at the mark (load time, outside any capture) in four stream slots, each bound to the first stream that launches the blob through it, so the main, side, commit and fork streams may run one blob at once; a fifth stream throws. Retiling a blob forgets its mark. A row's bytes are therefore the same at every M from 1 to 64, grouped (`stream_gemm_multi`, up to 4 targets) or alone, and on every run; `tests/stream_gemm_hip_tests.cu` checks this against an FP64 host reference with stream_tests' tolerance. The HIP tiled layout differs from CUDA's (the sizes are the same): a tile is 16 rows by 64 K, tiles are numbered K-major (`(k / 64) * (N / 16) + n / 16`); NVFP4 is all tiles' codes (512 bytes, row r at r * 32) and then all tiles' scales (64 bytes, row r at r * 4); FP8 is 1024 bytes per tile (row r at r * 64). A HIP port of the prefill GEMMs or the dequantizers reads this layout. The retile only moves bytes, so the blob is the same on every load. There are no coded scales on HIP: `scale_code_table` returns false, so model.cu uploads the raw blob, and the coded entry points throw. Not ported: `stream_gemm_gather_nvfp4`, the drafter's sparse-head GEMM (its caller in `sparse_head.cu` stays CUDA-only), and the persistent kernels. `l2_prefetch_weights` is a no-op.
- The rows path's attention (eng/hip-w1, W1): `src/core/attention_body_hip.cuh` holds the HIP `attention_mma_body` and `attention_combine_body`, included at the end of `attention_body.cuh` in place of the CUDA ones (which are built around `ldmatrix`, `mma.sync`, `mbarrier` and TMA), so every kernel in `kernels.cu` that instantiates them (attention_rows, the seqs, the shared and cascade phases, the tree verify) builds on HIP unchanged. The block is 48 query rows (3 waves; `kAttnRows` is 48 on HIP, 96 on CUDA) over 32-key tiles in LDS (Q and K row-major with a padded 264-element stride, V transposed), WMMA 16x16x16 BF16 into FP32 with the roles swapped (S^T = K Q^T, O^T += V^T P^T), so a lane owns one query row's column and a row's arithmetic never sees another row: keys past a row's last visible key score -inf (p = 0). gfx1151's WMMA is not exact for zero products (a p = 0 times a negative v moves an FP32 sum one ulp), so a masked key's V must be zero too, not only its p: a row that ends inside a tile is computed in its own pass with the tile's V columns past its last key zeroed in LDS (rows in descending position, the zeros only growing), as a decode sees them (the tile is zero-filled from the context's end); each pass adds V^T P^T into a fresh accumulator and only the rows it owns fold it into o (o = o * alpha + t), the other rows keep their o bit for bit. So a row's bytes are the same at every M (decode, a 24-row verify, a prefill slice). The context splits into the same fixed 4096-key ranges as CUDA (`attention_chunk`), each range's partial (m, l, o) goes to scratch and the combine adds the ranges in range order, so the result does not depend on scheduling. The FP8 KV cache (LOWBIT §8) uses the CUDA arithmetic: `kv_q8_encode2` is e4m3 satfinite of `mul_rn(v, inv)` (`gpu::e4m3_from_float` is written out: HIP's software conversion on gfx1151 encodes magnitudes far below 2^-9 as 0x01 where CUDA gives 0) and `kv_q8_decode2` is the code's exact value (its bits moved into an FP16, times 2^8) times the scale, rounded to BF16; a tile every row of the block reads as R(v) is decoded from the sealed E4M3 mirror, otherwise rounded in place, the same values either way. `kv_absmax`, `kv_seal` and `kv_unseal` are the CUDA kernels. `tests/attention_rows_tests.cu` (both backends) checks kv_seal's codes for all 65536 BF16 patterns at six scales against the CUDA build's (FNV hashes), the row invariance and the run-to-run bytes for BF16, FP8 rounded and FP8 from the mirror, and all three formats' rows against an FP64 reference (FP8: the keys of the units below the row's own as R(v)) within 1%; its inputs come from `std::mt19937`'s raw words (Box-Muller), the same data on both standard libraries; `--bench` times one layer at 25.7K and 32K keys. `paged_tests`, `tree_tests` and `kvq8_tests` are ported to `tests/gpu_test.hpp` and run on both backends. A tile is one round trip to memory: a thread's K rows and V key quads are fixed, each step's address is a uniform page-run base plus the thread's 32-bit offset (64-bit offsets hoisted out of the loop were spilled, and their scratch reloads serialized the loads), and all of a tile's K loads are issued before its LDS stores, then V's; a sealed FP8 tile goes through the same compute call site as a rounded one (a second site made the compiler spill the O accumulators). At 25.7K keys one layer takes 0.57 ms at M = 1 and 1.06 ms at M = 16 in BF16, 0.91 and 1.91 ms in FP8 (`--bench`); the range length is not a lever on HIP any more (512 to 4096 keys time the same), so the 4096 of CUDA stays. There is no bulk-copy or L2-prefetch path on HIP (`set_attention_bulk` and the prefetch are bit-identical hints), and no shared-memory retile (`prepare_kernels` is a no-op).
- Kernel preconditions the tests keep: `rmsnorm` below H = 1024 launches H threads and `block_sum` shuffles whole warps, so H is a multiple of 32 (every model width is).

## 8. Metal (Mac node, Q204, in the 1.0 gate)

The Metal row adds `src/backend/metal/types.hpp` and implements §2 in Objective-C++ host functions that encode MSL kernels (METAL §3), with these interface choices, **assumptions until the Metal row confirms them**:

- `gpu::Stream` is a handle to an engine-owned queue that collects encoders into a command buffer; "enqueue on s" means encode into it, and a synchronizing call commits it.
- A device pointer is the `gpuAddress` of a buffer in one resident `MTLHeap` (argument buffers with GPU addresses, Metal 3), so the pointer arithmetic of the descriptors (`GatherBuf`, page tables, `*Seqs` structs) carries over unchanged.
- Graphs are indirect command buffers or none (`kGraphs` false at first); dependent launch, green contexts, bulk copy and persistent kernels are false.
- `gpu::bf16` and `gpu::half` are 16-bit storage types on the host side.

METAL §1 still holds: no runtime choice of backend and no third-party runtime; the "no device abstraction layer" of its first draft is superseded by this interface, which is a compile-time one.
