# The Metal backend (BACKEND §8, METAL spec, first lane): Objective-C++ host, MSL kernels compiled when the device
# opens (the Mac build needs the Command Line Tools only, no Xcode). Included from the top-level CMakeLists when
# RAGON_BACKEND=metal; none of the CUDA targets are defined in that configuration.
if(NOT APPLE)
  message(FATAL_ERROR "RAGON_BACKEND=metal builds on macOS only")
endif()
enable_language(OBJCXX)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

# kernels.metal becomes a C string in a generated header (a raw literal, split below the compiler's length limit).
set(_metal_src ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/metal/kernels.metal)
set(_metal_hdr ${CMAKE_CURRENT_BINARY_DIR}/generated/metal_kernels_src.h)
add_custom_command(
  OUTPUT ${_metal_hdr}
  COMMAND ${CMAKE_COMMAND} -DIN=${_metal_src} -DOUT=${_metal_hdr}
          -P ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/metal/embed_source.cmake
  DEPENDS ${_metal_src} ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/metal/embed_source.cmake
  COMMENT "Embedding kernels.metal")

add_library(ragon_metal STATIC
  src/backend/metal/device.mm
  src/backend/metal/model.mm
  ${_metal_hdr})
target_include_directories(ragon_metal PUBLIC src ${CMAKE_CURRENT_BINARY_DIR}/generated)
target_compile_options(ragon_metal PRIVATE -fobjc-arc -Wall -Wextra -Wno-unused-parameter)
target_link_libraries(ragon_metal PUBLIC "-framework Metal" "-framework Foundation")

add_executable(ragon-metal-run tools/metal/ragon_metal_run.mm)
target_compile_options(ragon-metal-run PRIVATE -fobjc-arc -Wall -Wextra)
target_link_libraries(ragon-metal-run PRIVATE ragon_metal)

# ragon-serve on Metal (src/serve/metal_serve.mm): the serve's HTTP API layer (serve/api.cpp, built without the CUDA
# engine: RAGON_API_HOST_ONLY), the tokenizer and the sampler, over ragon_metal. Needs nlohmann_json, PCRE2 and ICU
# (Homebrew: nlohmann-json, pcre2, icu4c; put the icu4c prefix on CMAKE_PREFIX_PATH).
find_package(nlohmann_json 3 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(PCRE2 REQUIRED IMPORTED_TARGET libpcre2-8)
find_package(ICU REQUIRED COMPONENTS uc)
add_library(ragon_api_host STATIC
  src/serve/api.cpp
  src/core/sampling.cpp
  src/tokenizer/tokenizer.cpp
  src/tokenizer/chat_template.cpp)
target_compile_features(ragon_api_host PUBLIC cxx_std_20)
target_compile_definitions(ragon_api_host PUBLIC RAGON_API_HOST_ONLY)
target_include_directories(ragon_api_host PUBLIC src)
target_link_libraries(ragon_api_host PUBLIC PkgConfig::PCRE2 ICU::uc nlohmann_json::nlohmann_json)

add_executable(ragon-serve src/serve/metal_serve.mm)
target_compile_options(ragon-serve PRIVATE -fobjc-arc -Wall -Wextra -Wno-unused-parameter)
target_link_libraries(ragon-serve PRIVATE ragon_metal ragon_api_host)

# The tokenizer tool (token ids of a text), to check the Mac's tokenization against the CUDA build's.
add_executable(ragon-tokenize tools/ragon_tokenize.cpp)
target_link_libraries(ragon-tokenize PRIVATE ragon_api_host)
