# ragon-engine on AMD: source inventory for the Windows port

Part of the ragon-engine specification; the index is [README.md](./README.md). Companion to [DREAMFERENCE_RAGON_ENGINE_AMD.md](./DREAMFERENCE_RAGON_ENGINE_AMD.md), whose §2 classified the engine at `c5ec4ec`. This inventory restates it for the tree of 2026-11-06 (base `e828f75`), which gained vision, paged KV, `kv_pool` and batches. Line counts are `wc -l` of the tracked files; the totals below add up to the same figure.

**Classes.** *rewritten*: device code that uses PTX, `mma.sync`, TMA, `cp.async`, `ldmatrix` or programmatic launch and is written again on WMMA and plain loads. *hipify*: device code that `hipify-clang` converts and a build checks. *host, adapted*: host code that calls the CUDA runtime or driver and moves to HIP calls or loses a feature. *replaced*: code written again for Windows. *unchanged*: host C++ with no platform content. *follows the kernels*: tests of device code, adapted with the kernel they test.

## Totals

| Tree | rewritten | hipify | host, adapted | replaced | unchanged | follows the kernels | Lines |
|---|---|---|---|---|---|---|---|
| `src/` | 4,876 | 2,890 | 5,562 | 2,012 | 4,247 | 0 | 19,587 |
| `tests/` | 0 | 0 | 1,117 | 0 | 3,892 | 3,579 | 8,588 |
| `tools/` | 0 | 0 | 1,164 | 0 | 111 | 0 | 1,275 |
| all | 4,876 | 2,890 | 7,843 | 2,012 | 8,250 | 3,579 | 29,450 |

`src/` is 19,587 lines. Of it, 4,876 are rewritten (the study said about 4,600 for the narrower table) and 2,890 hipify. The 4,876 include `launch.cuh`'s wrappers, which the study counted with the host code.

## Constructs without a HIP twin, by file and line

| Construct | Where | In the port |
|---|---|---|
| Programmatic dependent launch: `griddepcontrol.wait`/`.launch`, `cudaLaunchAttributeProgrammaticStreamSerialization` | `src/core/launch.cuh`:8, 18, 34, 42, 64; used by `kernels.cu`, `kernels.cuh`, `stream_gemm.cu` | Plain launches; more gap between kernels (study §2) |
| Green contexts: `cuGreenCtxCreate`, `cuDevSmResourceSplitByCount`, `cuGreenCtxStreamCreate`, `cuGreenCtxDestroy` | `src/core/engine.cpp`:51-63; `tests/stream_tests.cu`:957-983 | Dropped; one stream |
| TMA: `CUtensorMap`, `cuTensorMapEncodeTiled`, `cp.async.bulk.tensor`, `mbarrier` | `src/core/gemm_bf16.cu`:69-112, 178-210; `src/core/attention_body.cuh`:200-251, 333-335 | Plain loads and LDS; the barrier becomes `__syncthreads` |
| `cp.async.bulk.prefetch` and `prefetch.global.L2` | `src/core/stream_gemm.cu`:831, 835, 904, 907; `src/core/attention_body.cuh`:149, 151; `src/core/kernels.cu`:217 | Dropped or plain loads |
| Block-scaled FP4 and FP8 MMA: `kind::mxf4nvf4`, `kind::f8f6f4` | `src/core/prefill_gemm.cu`:5, 6, 78, 87 | WMMA W4A16/W8A16 or W8A8 (study §2) |
| `cvt.rn.f16x2.e2m1x2`, `cvt.rn.f16x2.e4m3x2` | `src/core/stream_gemm.cu`:50, 58 | Software decode with the `v_perm_b32` table (probe `bench/amd/vg.hip`) |
| `ld.acquire.gpu`, `red.release.gpu`, `%globaltimer` (persistent-block barrier) | `src/core/stream_gemm.cu`:854-863 | Only for the megakernel stages, which a first port leaves off |
| `ldmatrix` (all forms) | `src/core/attention_body.cuh`:414, 426; `gemm_bf16.cu`:34, 57; `prefill_gdn.cu`:88-115; `prefill_gemm.cu`:12, 65; `vision.cu`:152 | LDS loads into WMMA fragments |
| `L1::no_allocate` loads | `src/core/stream_gemm.cu`:119, 137, 162 | Plain loads with a cache hint where gfx11 has one |
| `cudaGraphInstantiateFlagUseNodePriority` | `src/core/speculate.cpp`:490 | Not supported by `hipGraph`; dropped |
| `cudaMemPoolCreate`/`SetAttribute`/`Destroy` | `src/core/engine.cpp`:149-155, 535; `engine.hpp`:747 | `hipMemPool*`, or `hipMallocAsync`; sized from the DXGI budget |
| NVTX ranges | `src/core/engine.cpp`:4, 23-24; `src/core/speculate.cpp`:48, 60-61 | Dropped or roctx |
| proxygen, folly, `SIGINT`/`SIGTERM` | `src/serve/main.cpp` (48 uses) | Replaced by a small HTTP server on Windows |
| `open`/`mmap` | `src/core/safetensors.cpp` | `MapViewOfFile` |
| `cub::` in `sparse_head.cu` | `src/core/sparse_head.cu` | hipcub (header-only) |

## One row per file

| File | Lines | Class | Notes |
|---|---|---|---|
| `src/core/accept.cpp` | 45 | unchanged | host C++ with no platform content |
| `src/core/accept.hpp` | 25 | unchanged | host C++ with no platform content |
| `src/core/attention_body.cuh` | 609 | rewritten | `ldmatrix` (414, 426), `cp.async.bulk.shared` (246, 251), `mbarrier` init/arrive/wait (200-239), `fence.proxy.async` (335), `cp.async.bulk.prefetch` (149, 151) |
| `src/core/draft_guard.hpp` | 84 | unchanged | host C++ with no platform content |
| `src/core/draft_kernels.cu` | 549 | hipify | `__shfl_*_sync` |
| `src/core/drafter.cpp` | 162 | host, adapted | CUDA runtime calls |
| `src/core/drafter.hpp` | 89 | unchanged | host C++ with no platform content |
| `src/core/elem.cuh` | 85 | hipify | element helpers |
| `src/core/engine.cpp` | 2,375 | host, adapted | green contexts `cuGreenCtxCreate`/`cuDevSmResourceSplitByCount`/`cuGreenCtxStreamCreate`/`cuGreenCtxDestroy` (51-63) dropped; `cudaMemPoolCreate` (149-155) and destroy (535) to `hipMemPool*` or `hipMallocAsync`; NVTX (4, 23-24) dropped or replaced by roctx |
| `src/core/engine.hpp` | 1,007 | host, adapted | `cudaMemPool_t` (747), graph and green-context members |
| `src/core/gemm_bf16.cu` | 229 | rewritten | TMA: `CUtensorMap`, `cuTensorMapEncodeTiled` (178-210), `cp.async.bulk.tensor` (85), `mbarrier` (69-112), `ldmatrix` (34, 57) |
| `src/core/kernels.cu` | 963 | hipify | `prefetch.global.L2` (217) and one `cp.async` have no twin and become plain loads; FP8 types via hip headers; programmatic launch calls dropped |
| `src/core/kernels.cuh` | 580 | hipify | declarations and launch wrappers |
| `src/core/kv_pool.cpp` | 185 | unchanged | host C++ with no platform content |
| `src/core/kv_pool.hpp` | 85 | unchanged | host C++ with no platform content |
| `src/core/launch.cuh` | 98 | rewritten | `griddepcontrol.wait`/`.launch` (8, 18, 34, 42) and `cudaLaunchAttributeProgrammaticStreamSerialization` (64): programmatic dependent launch has no HIP twin; the wrappers become plain launches |
| `src/core/lookup.cpp` | 26 | unchanged | host C++ with no platform content |
| `src/core/lookup.hpp` | 20 | unchanged | host C++ with no platform content |
| `src/core/model.cu` | 287 | hipify | FP8 and FP4 types |
| `src/core/model.hpp` | 142 | unchanged | host C++ with no platform content |
| `src/core/prefill_gdn.cu` | 440 | rewritten | `mma.sync` and `ldmatrix` (88, 107-115) |
| `src/core/prefill_gemm.cu` | 718 | rewritten | `kind::mxf4nvf4` block-scaled MMA (5, 78), `ldmatrix` (12, 65), `cp.async` |
| `src/core/prefix_kv.cpp` | 370 | unchanged | host C++ with no platform content |
| `src/core/prefix_kv.hpp` | 158 | unchanged | host C++ with no platform content |
| `src/core/safetensors.cpp` | 85 | host, adapted | `open`/`mmap` to `MapViewOfFile` on Windows |
| `src/core/safetensors.hpp` | 54 | unchanged | host C++ with no platform content |
| `src/core/sampling.cpp` | 70 | unchanged | host C++ with no platform content |
| `src/core/sampling.hpp` | 45 | unchanged | host C++ with no platform content |
| `src/core/sparse_head.cu` | 295 | hipify | `cub` becomes hipcub; one `mma.sync` stage is rewritten |
| `src/core/speculate.cpp` | 1,306 | host, adapted | `cudaGraphInstantiateFlagUseNodePriority` (490) has no twin; graph capture to `hipGraph`; NVTX (48, 60-61) |
| `src/core/stream_gemm.cu` | 2,223 | rewritten | FP16 `mma.sync` lane maps, `cvt.rn.f16x2.e2m1x2`/`.e4m3x2` (lines 50, 58), `cp.async.bulk.prefetch` (831, 835, 904, 907), `L1::no_allocate` (119, 137, 162), persistent-block barrier on `ld.acquire.gpu` (859), `red.release.gpu` (863), `%globaltimer` (854) |
| `src/core/topk.cu` | 131 | hipify | `__shfl_*_sync`; wave32 matches |
| `src/core/tree.cpp` | 219 | unchanged | host C++ with no platform content |
| `src/core/tree.hpp` | 65 | unchanged | host C++ with no platform content |
| `src/core/vision.cu` | 559 | rewritten | `mma.sync`, `ldmatrix` (152), `cp.async` |
| `src/core/vision.hpp` | 113 | host, adapted | CUDA runtime types |
| `src/serve/admission.hpp` | 321 | unchanged | host C++ with no platform content |
| `src/serve/api.cpp` | 960 | unchanged | host C++ with no platform content |
| `src/serve/api.hpp` | 276 | unchanged | host C++ with no platform content |
| `src/serve/image_input.cpp` | 514 | host, adapted | a few CUDA runtime calls for the decode buffers |
| `src/serve/image_input.hpp` | 93 | unchanged | host C++ with no platform content |
| `src/serve/kv_policy.hpp` | 95 | unchanged | host C++ with no platform content |
| `src/serve/main.cpp` | 2,012 | replaced | proxygen and folly (48 uses), `SIGINT`/`SIGTERM`: replaced on Windows by a small HTTP server |
| `src/serve/pairing.hpp` | 77 | unchanged | host C++ with no platform content |
| `src/serve/step_trace.hpp` | 108 | unchanged | host C++ with no platform content |
| `src/tokenizer/chat_template.cpp` | 252 | unchanged | host C++ with no platform content |
| `src/tokenizer/chat_template.hpp` | 33 | unchanged | host C++ with no platform content |
| `src/tokenizer/tokenizer.cpp` | 277 | unchanged | host C++ with no platform content |
| `src/tokenizer/tokenizer.hpp` | 73 | unchanged | host C++ with no platform content |
| `tests/admission_tests.cpp` | 143 | unchanged | host test; builds on Windows with the platform shim |
| `tests/api_tests.cpp` | 1,117 | host, adapted | one green-context mention (dropped) |
| `tests/draft_guard_tests.cpp` | 125 | unchanged | host test; builds on Windows with the platform shim |
| `tests/kernel_tests.cu` | 167 | follows the kernels | tests the kernels above; CUDA runtime calls hipified, driver and PTX-specific cases follow the kernel they test |
| `tests/kv_pool_tests.cpp` | 691 | unchanged | host test; builds on Windows with the platform shim |
| `tests/mk_tests.cu` | 470 | follows the kernels | tests the kernels above; CUDA runtime calls hipified, driver and PTX-specific cases follow the kernel they test |
| `tests/paged_tests.cu` | 387 | follows the kernels | tests the kernels above; CUDA runtime calls hipified, driver and PTX-specific cases follow the kernel they test |
| `tests/pairing_tests.cpp` | 182 | unchanged | host test; builds on Windows with the platform shim |
| `tests/prefill_tests.cu` | 484 | follows the kernels | tests the kernels above; CUDA runtime calls hipified, driver and PTX-specific cases follow the kernel they test |
| `tests/prefix_kv_tests.cpp` | 402 | unchanged | host test; builds on Windows with the platform shim |
| `tests/sampling_tests.cpp` | 148 | unchanged | host test; builds on Windows with the platform shim |
| `tests/server/conftest.py` | 131 | unchanged | Python tests of the HTTP API; run against the Windows server |
| `tests/server/test_regressions.py` | 942 | unchanged | Python tests of the HTTP API; run against the Windows server |
| `tests/server/test_sessions.py` | 191 | unchanged | Python tests of the HTTP API; run against the Windows server |
| `tests/server/test_vision.py` | 125 | unchanged | Python tests of the HTTP API; run against the Windows server |
| `tests/spec_tests.cpp` | 372 | unchanged | host test; builds on Windows with the platform shim |
| `tests/stream_tests.cu` | 1,475 | follows the kernels | tests the kernels above; CUDA runtime calls hipified, driver and PTX-specific cases follow the kernel they test |
| `tests/tokenizer_threads_tests.cpp` | 45 | unchanged | host test; builds on Windows with the platform shim |
| `tests/tree_tests.cu` | 301 | follows the kernels | tests the kernels above; CUDA runtime calls hipified, driver and PTX-specific cases follow the kernel they test |
| `tests/vision_input_tests.cpp` | 395 | unchanged | host test; builds on Windows with the platform shim |
| `tests/vision_tests.cu` | 295 | follows the kernels | tests the kernels above; CUDA runtime calls hipified, driver and PTX-specific cases follow the kernel they test |
| `tools/ragon_run.cpp` | 1,164 | host, adapted | CUDA runtime, green-context and PDL options |
| `tools/ragon_template.cpp` | 83 | unchanged | host C++ with no platform content |
| `tools/ragon_tokenize.cpp` | 28 | unchanged | host C++ with no platform content |
| **Total** | **29,450** | | |

## Weeks restated

The study's phases (W0 3–4, W1 6–8, W2 6–9 engineer-weeks, 15–21 in all) priced the engine of `c5ec4ec`. Since then the tree gained `vision.cu` (559 lines, rewritten: `mma.sync`, `ldmatrix`, `cp.async`), `kv_pool` and the paged KV tests, and batches, which touch `engine.cpp` and `speculate.cpp`. The inventory moves the weeks as follows. These are estimates.

| Phase | Study | Now | What moved |
|---|---|---|---|
| W0 bring-up | 3–4 | 3–4 | No change: the portable kernels and host code are as before; `kv_pool` and the new host files are plain C++ |
| W1 decode | 6–8 | 6–8 | No change for the rewritten decode files; paged attention (`attention_body.cuh` is now 609 lines against 582) adds a few days inside it |
| W2 serving | 6–9 | 6–9 | `src/serve/main.cpp` (2,012 lines) is the replaced part, as before; the other serve files are plain C++ |
| W3 parity | not in the study | 2–4 | `vision.cu` (559 lines rewritten), the image decode path, paged KV and batches |

**Total: 17–25 engineer-weeks**, the study's 15–21 plus W3.
