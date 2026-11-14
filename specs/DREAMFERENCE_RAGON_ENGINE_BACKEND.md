# ragon-engine: one backend interface for the kernels (CUDA, HIP, Metal)

Part of the ragon-engine specification; the index is [README.md](./README.md). Written on 2026-11-13 from the owner's decisions Q46, Q203 and Q204 (2026-11-12): the Windows node's native port starts now with HIP, the Mac node (Metal) is in the 1.0 gate, and both build on **one** backend interface (CUDA, HIP, Metal) with no behaviour change for CUDA. Priorities: nodes before engine speed work.

It builds on [ARCHITECTURE §8–9](./DREAMFERENCE_RAGON_ENGINE_ARCHITECTURE.md) (the compiler, the runtime, the kernels), [METAL](./DREAMFERENCE_RAGON_ENGINE_METAL.md), [AMD](./DREAMFERENCE_RAGON_ENGINE_AMD.md) and [AMD_INVENTORY](./DREAMFERENCE_RAGON_ENGINE_AMD_INVENTORY.md). The code it describes is `src/backend/` and the three interface headers `src/core/kernels.cuh`, `src/core/lora_kernels.cuh` and `src/core/vision.hpp`.

## 0. Summary

- **What.** Every kernel the serve layer launches (attention including the shared and cascade phases, GDN/DeltaNet, the FFN GEMMs, the LM head and the sparse head, the drafter, KV pages, the top-K behind the samplers and the grammar mask, LoRA, the vision tower) is a host function in namespace `ragon::kernels` declared in one of three interface headers. Those declarations name no vendor type: a stream is `gpu::Stream`, a 16-bit float `gpu::bf16` or `gpu::half`. A backend supplies those types (§3), its capabilities (§4) and the definitions of every entry point (§2).
- **How it is chosen.** At compile time: one CMake flag `-DRAGON_BACKEND=cuda|hip|metal` (default `cuda`), one backend per binary, no virtual call (§1).
- **CUDA.** The first implementation is today's `.cu` files, unchanged. The types are aliases of the CUDA ones, so every object file the compiler emits is the same code as before the interface (§6: identical PTX, host code and kernel code; only the debug line tables differ).
- **HIP and Metal.** `RAGON_BACKEND=hip` and `=metal` are reserved and refused with a message until their rows land (§7, §8). Each one implements §2 against its own types; the serve layer does not change.
- **Not yet behind the interface.** The runtime layer (streams, events, graphs, allocation, copies, green contexts) is still called directly from the host files; its census and the plan to move it are in §5.2 (tier 2).

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

They are aliases, never wrappers: an alias leaves the mangled name, the overload set and the code unchanged, which is what keeps CUDA byte-identical (§6). A device pointer is a raw pointer (`gpu::bf16*`, `float*`); on CUDA and HIP it is a device address, on Metal see §8. The KV, tree and batch descriptor structs (`DevPos`, `TreeDesc`, `GatherBuf`, `SparseHeadBufs`, `MkAttnParams`, the `*Seqs` structs) are plain structs of such pointers and integers and are part of the interface.

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

- `src/backend/backend.hpp` picks the backend by macro: no macro is CUDA, `RAGON_BACKEND_HIP` and `RAGON_BACKEND_METAL` stop with `#error` until their rows land.
- `src/backend/cuda/types.hpp` includes `cuda_bf16.h`, `cuda_fp16.h` and `cuda_runtime.h`, defines the aliases and capabilities, and `RAGON_GPU_DEVICE_CODE` when nvcc compiles device code.
- The three interface headers include `backend/backend.hpp` instead of the CUDA headers and spell `gpu::Stream`, `gpu::bf16`, `gpu::half`. The include block keeps the same number of lines (comment lines fill it), so every later line keeps its number and `-lineinfo` points where it pointed.
- The definitions stay in today's files (`src/core/*.cu`), unmoved: about ten branches in flight edit them, and a move would turn each of their merges into a conflict. Moving them to `src/backend/cuda/` is a later, mechanical commit (a rename with no content change), best made when few branches are open.
- CMake: `RAGON_BACKEND` (cache string, `cuda` by default, `hip` and `metal` refused with a message). With `cuda` the CUDA language is enabled exactly as before; every compile line is unchanged.

**Merging a branch that predates the interface.** Conflicts can only arise in the three interface headers. Apply to the branch's new lines there: `cudaStream_t` → `gpu::Stream`, `__nv_bfloat16` → `gpu::bf16`, `__half` → `gpu::half`. The `.cu` files and the host files may keep the CUDA spellings: under CUDA they are the same types.

### 5.2 Tier 2: the runtime (spec only, next)

The host files still call the CUDA runtime directly. The census at main 47277016 (calls, by file): engine.cpp 118, speculate.cpp 66, engine_fanout.cpp 14, engine_lora.cpp 8, image_input.cpp 5, drafter.cpp 4, about 60 distinct functions:

- **copies and memsets**: `cudaMemcpyAsync` 49, `cudaMemcpy` 15, `cudaMemset` 13, `cudaMemsetAsync`;
- **synchronization**: `cudaStreamSynchronize` 32, `cudaEventRecord` 12, `cudaStreamWaitEvent` 6, `cudaEventSynchronize` 5, `cudaEventQuery`, `cudaDeviceSynchronize`, `cudaEventElapsedTime`;
- **objects**: `cudaEventCreate(WithFlags)`, `cudaEventDestroy`, `cudaStreamCreateWithFlags`, `cudaStreamCreateWithPriority`, `cudaStreamDestroy`, `cudaDeviceGetStreamPriorityRange`;
- **memory**: `cudaMalloc`, `cudaFree`, `cudaHostAlloc`, `cudaFreeHost`, `cudaMallocFromPoolAsync`, `cudaFreeAsync`, `cudaMemPoolCreate`, `cudaMemPoolSetAttribute`, `cudaMemPoolDestroy`, `cudaMemGetInfo`;
- **graphs**: `cudaStreamBeginCapture`, `cudaStreamEndCapture`, `cudaStreamIsCapturing`, `cudaGraphInstantiate`, `cudaGraphLaunch`, `cudaGraphExecDestroy` 11, `cudaGraphDestroy`;
- **green contexts** (driver API): `cuDeviceGet`, `cuDeviceGetDevResource`, `cuGreenCtxCreate`, `cuGreenCtxStreamCreate`, `cuGreenCtxDestroy`;
- **errors and device**: `cudaGetErrorString`, `cudaGetLastError`, `cudaGetDevice`.

The host structs in `model.hpp`, `drafter.hpp` and `engine.hpp` also spell the CUDA types. A few kernel entry points use the runtime internally (`cudaFuncSetAttribute` in the `prepare` calls, `cudaDeviceGetAttribute` for the SM count, a cached `cudaMalloc` in stream_gemm.cu's split-K buffer, `cudaMemsetAsync` and `cudaMemcpyAsync` in sparse_head.cu); those are the backend's own business.

The plan: `src/backend/runtime.hpp` declares `ragon::gpu` functions with the same shapes (`memcpy_async(dst, src, bytes, Kind, Stream)`, `Event`, `record`, `wait`, `Graph`, `capture_begin/end`, `Pool`), defined inline in `backend/cuda/runtime.hpp` as one-line forwards, so the CUDA objects stay identical by the same argument as §6. The host files move to them a file at a time (engine_lora.cpp, image_input.cpp and drafter.cpp first; engine.cpp last, after the in-flight branches that touch it have landed). HIP maps them one to one (`hip*` has a twin for every call above but the green contexts, AMD §2); Metal maps them to command buffers, `MTLSharedEvent` and heaps (§8). The HIP row needs tier 2 for its host files; it can start on tier 1 (the kernels, hipified) at once.

## 6. How "no behaviour change" is proved

1. **Static identity.** Base (main 47277016) and branch are built clean at the same path with the same flags. The compile commands are identical (59 translation units). Every C++ object is byte-identical. Every CUDA object has identical PTX, identical host `.text` and identical kernel code sections; it differs only in the line tables (`.debug_line`, `.nv.merc.debug_line`: the CUDA headers are now reached through `backend/cuda/types.hpp`), in the `.nv_fatbin` that contains them, and in nvcc's process-numbered temporary symbol in `.strtab`, which differs between two builds of the same tree anyway. The linked binaries differ only in `.note.gnu.build-id`, `.nv_fatbin` and `.strtab`.
2. **Dynamic identity.** On a GB10: ctest; the exactness suite (`ragon-run --spec-check`, `--prefix-check`, `--batch-check` with and without graphs, dependent launch and pages, `--fanout-check`, `--graph-check`); greedy identity on agentlong (16 of 16 identical to base); one-stream and four-stream throughput within the run-to-run spread (A-B-B-A, two runs each). The results of this job are in its report.

A later change to the interface keeps the same proof: an alias or a declaration change must leave the CUDA code sections identical; a real code change is measured like any engine change.

## 7. HIP (Windows node, Q203)

The HIP row adds `src/backend/hip/types.hpp` (`hipStream_t`, `__hip_bfloat16`, `__half` from `hip_bf16.h` and `hip_fp16.h`; capabilities: graphs yes, dependent launch no, green contexts no, bulk copy no, persistent kernels per AMD §5), lifts the `#error` in backend.hpp and the CMake refusal, and compiles the `.cu` files as HIP (AMD_INVENTORY: hipify for most, rewritten for the tensor-core and PTX kernels). The entry points and their contracts are §2 unchanged; the exactness rules of AMD apply within the HIP build.

## 8. Metal (Mac node, Q204, in the 1.0 gate)

The Metal row adds `src/backend/metal/types.hpp` and implements §2 in Objective-C++ host functions that encode MSL kernels (METAL §3), with these interface choices, **assumptions until the Metal row confirms them**:

- `gpu::Stream` is a handle to an engine-owned queue that collects encoders into a command buffer; "enqueue on s" means encode into it, and a synchronizing call commits it.
- A device pointer is the `gpuAddress` of a buffer in one resident `MTLHeap` (argument buffers with GPU addresses, Metal 3), so the pointer arithmetic of the descriptors (`GatherBuf`, page tables, `*Seqs` structs) carries over unchanged.
- Graphs are indirect command buffers or none (`kGraphs` false at first); dependent launch, green contexts, bulk copy and persistent kernels are false.
- `gpu::bf16` and `gpu::half` are 16-bit storage types on the host side.

METAL §1 still holds: no runtime choice of backend and no third-party runtime; the "no device abstraction layer" of its first draft is superseded by this interface, which is a compile-time one.
