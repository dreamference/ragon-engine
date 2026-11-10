#!/usr/bin/env bash
# Builds XGrammar's C++ core (structured output, BACKLOG §19.17) as a prebuilt under ~/ragon-deps/xgrammar,
# the way proxygen sits under ~/ragon-deps/proxygen. CMake finds it through RAGON_XGRAMMAR_PREFIX.
#
#   scripts/build_xgrammar.sh [PREFIX]        (default: $HOME/ragon-deps/xgrammar)
#
# One pinned release, checked by commit sha. C++ only: no Python bindings, no cpptrace, no tests. A static
# library with -fPIC, Release, and without LTO, so the same archive links into GCC builds, sanitizer builds and
# the clang libFuzzer target. Licences: XGrammar Apache-2.0; it bundles picojson (BSD-2-Clause, header only, used
# inside the library) and dlpack (Apache-2.0, headers, installed because matcher.h includes dlpack/dlpack.h).
# The other submodules (cpptrace MIT, googletest BSD-3-Clause) are not fetched.
set -euo pipefail

XGRAMMAR_TAG=v0.2.8
XGRAMMAR_SHA=97787376faee5ed8466cfad57c99855e4ce2f6aa
XGRAMMAR_URL=https://github.com/mlc-ai/xgrammar

PREFIX=${1:-$HOME/ragon-deps/xgrammar}
SRC=${RAGON_DEPS_SRC:-$HOME/ragon-deps/src}/xgrammar
BUILD=$SRC/build-ragon
JOBS=${JOBS:-8}

if [ ! -d "$SRC/.git" ]; then
  mkdir -p "$(dirname "$SRC")"
  git clone --branch "$XGRAMMAR_TAG" --depth 1 "$XGRAMMAR_URL" "$SRC"
fi
git -C "$SRC" fetch --depth 1 origin "refs/tags/$XGRAMMAR_TAG:refs/tags/$XGRAMMAR_TAG" 2>/dev/null || true
git -C "$SRC" checkout --quiet "$XGRAMMAR_TAG"
got=$(git -C "$SRC" rev-parse HEAD)
if [ "$got" != "$XGRAMMAR_SHA" ]; then
  echo "build_xgrammar: $XGRAMMAR_TAG is $got, expected $XGRAMMAR_SHA; refusing to build" >&2
  exit 1
fi
# dlpack only: the headers the public API needs. picojson is in the tree, not a submodule.
git -C "$SRC" submodule update --init --depth 1 3rdparty/dlpack

# XGrammar's cmake/config.cmake sets the Python bindings ON with a plain set(), which shadows -D options; a
# config.cmake in the build directory is read first and replaces it.
mkdir -p "$BUILD"
cat > "$BUILD/config.cmake" <<'EOF'
set(XGRAMMAR_BUILD_PYTHON_BINDINGS OFF)
set(XGRAMMAR_BUILD_CXX_TESTS OFF)
set(XGRAMMAR_ENABLE_CPPTRACE OFF)
set(XGRAMMAR_ENABLE_COVERAGE OFF)
set(XGRAMMAR_ENABLE_INTERNAL_CHECK OFF)
EOF

nice -n 19 cmake -S "$SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_CXX_FLAGS="-fno-lto" \
  -DCMAKE_INSTALL_PREFIX="$PREFIX"
nice -n 19 cmake --build "$BUILD" -j "$JOBS"
cmake --install "$BUILD"
cat > "$PREFIX/XGRAMMAR_VERSION" <<EOF
tag $XGRAMMAR_TAG
sha $XGRAMMAR_SHA
EOF
echo "build_xgrammar: $XGRAMMAR_TAG ($XGRAMMAR_SHA) installed in $PREFIX"
