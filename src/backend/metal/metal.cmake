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
