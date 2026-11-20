# cmake -DIN=kernels.metal -DOUT=metal_kernels_src.h -P embed_source.cmake
# Writes `static const char kMetalKernelsSource[] = R"..."  R"..." ...;` in pieces of at most 8000 bytes.
file(READ ${IN} _src)
string(LENGTH "${_src}" _len)
set(_out "// generated from kernels.metal by embed_source.cmake; do not edit\n#pragma once\nstatic const char kMetalKernelsSource[] =\n")
set(_pos 0)
while(_pos LESS _len)
  string(SUBSTRING "${_src}" ${_pos} 8000 _piece)
  string(APPEND _out "R\"ragonmsl(${_piece})ragonmsl\"\n")
  math(EXPR _pos "${_pos} + 8000")
endwhile()
string(APPEND _out ";\n")
file(WRITE ${OUT} "${_out}")
