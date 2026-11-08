cmake_minimum_required(VERSION 3.28)
project(ragon_engine LANGUAGES CXX CUDA VERSION 0.1.0)

# ragon-engine v0: one model (Qwen3.8-27B, NVFP4 MLP + FP8 attention), one GPU (GB10, sm_121).
# SPEC.md §8: C++20 and CUDA throughout; separate targets for the core, the tokenizer and the server,
# so a server change never recompiles a kernel.

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# sm_121a, not sm_121: the FP4 conversions and block-scaled MMAs are arch-specific (reports/M0.md §2).
set(CMAKE_CUDA_ARCHITECTURES 121a)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Debug builds compile every kernel with device debug info (SPEC.md §8, "Debuggable from day one").
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -lineinfo")
string(APPEND CMAKE_CUDA_FLAGS " --expt-relaxed-constexpr")

option(RAGON_SANITIZE "Build host code with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
if(RAGON_SANITIZE)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address,undefined>)
  add_link_options(-fsanitize=address,undefined)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(nlohmann_json 3 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(PCRE2 REQUIRED IMPORTED_TARGET libpcre2-8)
find_package(ICU REQUIRED COMPONENTS uc)

# The engine core: weights, kernels, the forward pass and sampling.
add_library(ragon_core STATIC
  src/core/safetensors.cpp
  src/core/kernels.cu
  src/core/stream_gemm.cu
  src/core/prefill_gemm.cu
  src/core/gemm_bf16.cu
  src/core/prefill_gdn.cu
  src/core/topk.cu
  src/core/sparse_head.cu
  src/core/draft_kernels.cu
  src/core/drafter.cpp
  src/core/speculate.cpp
  src/core/lookup.cpp
  src/core/accept.cpp
  src/core/tree.cpp
  src/core/sampling.cpp
  src/core/model.cu
  src/core/engine.cpp
  src/core/engine_lora.cpp
  src/core/engine_fanout.cpp
  src/core/prefix_kv.cpp
  src/core/lora.cpp
  src/core/lora_kernels.cu
  src/core/kv_pool.cpp
  src/core/vision.cu
)
target_include_directories(ragon_core PUBLIC src)
target_link_libraries(ragon_core PUBLIC CUDA::cudart CUDA::cuda_driver CUDA::nvtx3 nlohmann_json::nlohmann_json)
set_target_properties(ragon_core PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

# The tokenizer: byte-level BPE with Qwen's pre-tokenizer pattern (PCRE2) and NFC (ICU).
add_library(ragon_tokenizer STATIC src/tokenizer/tokenizer.cpp src/tokenizer/chat_template.cpp)
target_include_directories(ragon_tokenizer PUBLIC src)
target_link_libraries(ragon_tokenizer PUBLIC PkgConfig::PCRE2 ICU::uc nlohmann_json::nlohmann_json)


# ragon-tokenize: prints token ids for each input line (tested against the reference tokenizer).
add_executable(ragon-tokenize tools/ragon_tokenize.cpp)
target_link_libraries(ragon-tokenize PRIVATE ragon_tokenizer)
# ragon-tokenizer-threads-tests TOKENIZER_JSON TEXT: one tokenizer shared by 16 threads, as ragon-serve's HTTP workers share it.
find_package(Threads REQUIRED)
add_executable(ragon-tokenizer-threads-tests tests/tokenizer_threads_tests.cpp)
target_link_libraries(ragon-tokenizer-threads-tests PRIVATE ragon_tokenizer Threads::Threads)

# The API layer without the HTTP library, so it is testable on its own.
# Image decoding (VISION §20.3 (c)): libpng, libwebp, libjpeg-turbo from the distribution; nvJPEG opened at run time.
pkg_check_modules(IMAGE_LIBS REQUIRED IMPORTED_TARGET libpng libwebp libjpeg)
add_library(ragon_api STATIC src/serve/api.cpp src/serve/image_input.cpp)
target_link_libraries(ragon_api PUBLIC ragon_core ragon_tokenizer PkgConfig::IMAGE_LIBS ${CMAKE_DL_LIBS})

# ragon-run: token ids or text in, tokens out; the correctness and speed harness.
add_executable(ragon-run tools/ragon_run.cpp)
target_link_libraries(ragon-run PRIVATE ragon_api)
# ragon-force: the engine side of the agreement study and of the adapters' bar (bench/lora/bar.sh).
add_executable(ragon-force bench/agreement/ragon_force.cpp)
target_link_libraries(ragon-force PRIVATE ragon_api)

# ragon-template: the chat template, and the Responses-to-prompt conversion, without a model.
add_executable(ragon-template tools/ragon_template.cpp)
target_link_libraries(ragon-template PRIVATE ragon_api)

# ragon-serve: the HTTP server (proxygen), built when proxygen is found.
set(RAGON_PROXYGEN_PREFIX "$ENV{HOME}/ragon-deps/proxygen" CACHE PATH "Where proxygen and its dependencies are installed")
# getdeps installs each project under its own directory of the prefix (folly/, fizz/, proxygen/, ...).
file(GLOB _ragon_dep_prefixes "${RAGON_PROXYGEN_PREFIX}/*")
list(APPEND CMAKE_PREFIX_PATH ${_ragon_dep_prefixes})
# proxygen's dependency configs look for Sodium, Zstd and others through fbcode_builder's Find modules.
set(RAGON_FBCODE_CMAKE "$ENV{HOME}/ragon-deps/src/proxygen/build/fbcode_builder/CMake" CACHE PATH
    "fbcode_builder's CMake modules (from the proxygen checkout used to build the dependencies)")
list(APPEND CMAKE_MODULE_PATH "${RAGON_FBCODE_CMAKE}")
find_package(proxygen CONFIG QUIET)
if(proxygen_FOUND)
  add_executable(ragon-serve src/serve/main.cpp)
  target_link_libraries(ragon-serve PRIVATE ragon_api proxygen::proxygenhttpserver Folly::folly_init_init)
else()
  message(WARNING "proxygen not found under ${RAGON_PROXYGEN_PREFIX}: ragon-serve is not built")
endif()

enable_testing()
add_executable(ragon-kernel-tests tests/kernel_tests.cu)
target_link_libraries(ragon-kernel-tests PRIVATE ragon_core)
add_test(NAME kernels COMMAND ragon-kernel-tests)
add_executable(ragon-tree-tests tests/tree_tests.cu)
target_link_libraries(ragon-tree-tests PRIVATE ragon_core)
add_test(NAME tree COMMAND ragon-tree-tests)
add_executable(ragon-mk-tests tests/mk_tests.cu)
target_link_libraries(ragon-mk-tests PRIVATE ragon_core)
add_test(NAME megakernel COMMAND ragon-mk-tests)
add_executable(ragon-paged-tests tests/paged_tests.cu)
target_link_libraries(ragon-paged-tests PRIVATE ragon_core)

# The paged-attention microbenchmark (bench/micro/paged_attn_bench.cu): flat against paged KV at the engine's geometry.
add_executable(ragon-paged-attn-bench bench/micro/paged_attn_bench.cu)
target_link_libraries(ragon-paged-attn-bench PRIVATE ragon_core)
add_test(NAME paged COMMAND ragon-paged-tests)
add_test(NAME paged_head_major COMMAND ragon-paged-tests head-major)
add_executable(ragon-stream-tests tests/stream_tests.cu)
target_link_libraries(ragon-stream-tests PRIVATE ragon_core CUDA::cuda_driver)
add_test(NAME stream COMMAND ragon-stream-tests)
add_executable(ragon-prefill-tests tests/prefill_tests.cu)
target_link_libraries(ragon-prefill-tests PRIVATE ragon_core)
add_test(NAME prefill COMMAND ragon-prefill-tests)
add_executable(ragon-spec-tests tests/spec_tests.cpp)
target_link_libraries(ragon-spec-tests PRIVATE ragon_core)
add_test(NAME spec COMMAND ragon-spec-tests)
add_executable(ragon-draft-guard-tests tests/draft_guard_tests.cpp)  # host only: the guard's policy (BACKLOG 10)
target_include_directories(ragon-draft-guard-tests PRIVATE src)
add_test(NAME draft_guard COMMAND ragon-draft-guard-tests)
add_executable(ragon-pairing-tests tests/pairing_tests.cpp)  # host only: fair pairing and the step trace (C3)
target_include_directories(ragon-pairing-tests PRIVATE src)
add_test(NAME pairing COMMAND ragon-pairing-tests)
add_executable(ragon-drafter-routes-tests tests/drafter_routes_tests.cpp)  # host only: which drafter a request gets (MULTI_DRAFTER §2)
target_include_directories(ragon-drafter-routes-tests PRIVATE src)
target_link_libraries(ragon-drafter-routes-tests PRIVATE nlohmann_json::nlohmann_json)
add_test(NAME drafter_routes COMMAND ragon-drafter-routes-tests)
find_package(Python3 COMPONENTS Interpreter)  # host only: the --fair-pairing merge gate (bench/agentlong/fair_gate.py)
if(Python3_Interpreter_FOUND)
  add_test(NAME fair_gate COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/bench/agentlong/fair_gate_test.py)
endif()
add_executable(ragon-sampling-tests tests/sampling_tests.cpp)
target_link_libraries(ragon-sampling-tests PRIVATE ragon_core)
add_test(NAME sampling COMMAND ragon-sampling-tests)
add_executable(ragon-vision-tests tests/vision_tests.cu)
target_link_libraries(ragon-vision-tests PRIVATE ragon_core)
add_test(NAME vision COMMAND ragon-vision-tests)
add_executable(ragon-kv-pool-tests tests/kv_pool_tests.cpp)
target_link_libraries(ragon-kv-pool-tests PRIVATE ragon_core)
add_test(NAME kv_pool COMMAND ragon-kv-pool-tests)
add_executable(ragon-prefix-kv-tests tests/prefix_kv_tests.cpp)
target_link_libraries(ragon-prefix-kv-tests PRIVATE ragon_core)
add_executable(ragon-lora-tests tests/lora_tests.cpp)
target_link_libraries(ragon-lora-tests PRIVATE ragon_core)
add_test(NAME lora COMMAND ragon-lora-tests)
add_executable(ragon-lora-kernel-tests tests/lora_kernel_tests.cu)
target_link_libraries(ragon-lora-kernel-tests PRIVATE ragon_core)
add_test(NAME lora_kernels COMMAND ragon-lora-kernel-tests)

add_executable(ragon-api-tests tests/api_tests.cpp)
target_link_libraries(ragon-api-tests PRIVATE ragon_api)
add_executable(ragon-vision-input-tests tests/vision_input_tests.cpp)
target_link_libraries(ragon-vision-input-tests PRIVATE ragon_api)
add_test(NAME vision_input COMMAND ragon-vision-input-tests)
add_test(NAME api COMMAND ragon-api-tests)
add_executable(ragon-admission-tests tests/admission_tests.cpp)
target_link_libraries(ragon-admission-tests PRIVATE ragon_api)
add_test(NAME admission COMMAND ragon-admission-tests)
