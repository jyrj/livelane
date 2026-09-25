#!/usr/bin/env bash
# Build kepler-formal, LiveLane's SECOND, independent equivalence backend,
# FROM SOURCE into tools/. No system packages, no root, same prefix as
# scripts/build/toolchain.sh.
#
#     ./scripts/build/kepler.sh                 # everything, in dependency order
#     ./scripts/build/kepler.sh fmt kepler      # just these
#
# WHY this tool exists in LiveLane
# --------------------------------
# The eqy gate does gate-level, partition-based LEC and REQUIRES unchanged
# sequential boundaries (eqy README; confirmed by its "partitions not
# equivalent" methodology). An agent that retimes a pipeline, merges registers,
# or moves a stage boundary produces an edit eqy CANNOT prove, not because the
# edit is wrong but because the partitioning premise no longer holds. Kepler
# additionally does RTL-level Sequential Equivalence Checking, comparing
# sequential behaviour through extracted transition systems, which is exactly
# that class of edit. And two independent checkers let the gate cross-check:
# a disagreement is a reportable result, never an average.
#
# Dependency order (why each is here, all four are ABSENT on this host):
#   fmt          -> slang's external/CMakeLists.txt FetchContent's fmt 12.2.0.
#                   Building the SAME version into the prefix makes slang's
#                   FIND_PACKAGE_ARGS intercept it, so nothing is downloaded at
#                   configure time and the build is offline-reproducible.
#   tomlplusplus -> same story: slang FetchContent's v3.4.0.
#   onetbb       -> naja's find_package(TBB REQUIRED) (cmake/FindTBB.cmake looks
#                   for tbb/tbb.h, libtbb AND libtbbmalloc, all three).
#   capnproto    -> naja-if's find_package(CapnProto REQUIRED); the Naja
#                   interchange format is Cap'n Proto serialised.
#   kepler       -> the checker itself.
#
# boost 1.90, zlib, gmp, bison 3.8.2, flex 2.6.4 and python3.14 (with
# libpython3.14.so and Python.h) are already on this host and are NOT rebuilt;
# slang requires boost >= 1.87 and 1.90 satisfies it.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# CMake 4.x removed compatibility with `cmake_minimum_required(VERSION <3.5)`,
# which several vendored subprojects (glucose, lefdef) still declare. Kepler's
# own CMakeLists sets CMAKE_POLICY_VERSION_MINIMUM for its subprojects; the
# dependency builds below need it too when their own vendored code is old.
KEPLER_CMAKE_COMPAT=(-DCMAKE_POLICY_VERSION_MINIMUM=3.5)

# --------------------------------------------------------------------------
build_fmt() {
  if [[ -f "$PREFIX/lib/cmake/fmt/fmt-config.cmake" ]]; then
    say "fmt: already built"; return 0; fi
  say "fmt $FMT_REF (slang wants EXACTLY 12.2; build it so nothing is fetched)"
  clone_pin fmt "$FMT_URL" "$FMT_REF"
  local B="$BUILD/fmt"; rm -rf "$B"; mkdir -p "$B"
  # PIC: slang links fmt into libraries that end up in a shared naja module.
  ( cd "$B" && cmake "$SRCDIR/fmt" \
      -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_INSTALL_LIBDIR=lib \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DFMT_TEST=OFF -DFMT_DOC=OFF -DBUILD_SHARED_LIBS=OFF && \
    cmake --build . -j"$JOBS" && cmake --install . ) >"$LOGS/fmt.log" 2>&1 \
    || { tail -40 "$LOGS/fmt.log"; die "fmt failed (see $LOGS/fmt.log)"; }
  [[ -f "$PREFIX/include/fmt/format.h" ]] || die "fmt headers not installed"
  info "fmt -> $PREFIX/lib/libfmt.a"
}

# --------------------------------------------------------------------------
build_tomlplusplus() {
  if [[ -f "$PREFIX/include/toml++/toml.hpp" ]]; then
    say "tomlplusplus: already built"; return 0; fi
  say "tomlplusplus $TOMLPLUSPLUS_REF (slang wants EXACTLY v3.4; header-only)"
  clone_pin tomlplusplus "$TOMLPLUSPLUS_URL" "$TOMLPLUSPLUS_REF"
  local B="$BUILD/tomlplusplus"; rm -rf "$B"; mkdir -p "$B"
  ( cd "$B" && cmake "$SRCDIR/tomlplusplus" "${KEPLER_CMAKE_COMPAT[@]}" \
      -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_INSTALL_LIBDIR=lib \
      -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF && \
    cmake --install . ) >"$LOGS/tomlplusplus.log" 2>&1 \
    || { tail -40 "$LOGS/tomlplusplus.log"; die "tomlplusplus failed (see $LOGS/tomlplusplus.log)"; }
  info "toml++ -> $PREFIX/include/toml++"
}

# --------------------------------------------------------------------------
build_onetbb() {
  if [[ -f "$PREFIX/include/tbb/tbb.h" ]] && \
     ls "$PREFIX"/lib/libtbbmalloc.so* >/dev/null 2>&1; then
    say "oneTBB: already built"; return 0; fi
  say "oneTBB $ONETBB_REF (naja's find_package(TBB REQUIRED))"
  clone_pin onetbb "$ONETBB_URL" "$ONETBB_REF"
  local B="$BUILD/onetbb"; rm -rf "$B"; mkdir -p "$B"
  # TBB_STRICT=OFF: -Werror against GCC 16 is a build failure waiting to happen
  # and buys us nothing, we are a consumer, not a TBB developer.
  ( cd "$B" && cmake "$SRCDIR/onetbb" \
      -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_INSTALL_LIBDIR=lib \
      -DCMAKE_BUILD_TYPE=Release \
      -DTBB_TEST=OFF -DTBB_STRICT=OFF -DTBB_EXAMPLES=OFF && \
    cmake --build . -j"$JOBS" && cmake --install . ) >"$LOGS/onetbb.log" 2>&1 \
    || { tail -40 "$LOGS/onetbb.log"; die "oneTBB failed (see $LOGS/onetbb.log)"; }
  # naja's FindTBB REQUIRES all three of these; a missing tbbmalloc is a
  # configure-time failure a hundred lines later, so check it here.
  [[ -f "$PREFIX/include/tbb/tbb.h" ]] || die "tbb/tbb.h not installed"
  ls "$PREFIX"/lib/libtbb.so*      >/dev/null 2>&1 || die "libtbb not installed"
  ls "$PREFIX"/lib/libtbbmalloc.so* >/dev/null 2>&1 || die "libtbbmalloc not installed"
  info "oneTBB -> $PREFIX/lib/libtbb.so, libtbbmalloc.so"
}

# --------------------------------------------------------------------------
build_capnproto() {
  if have capnp; then say "capnproto: already built"; return 0; fi
  say "capnproto $CAPNPROTO_REF (naja-if's find_package(CapnProto REQUIRED))"
  clone_pin capnproto "$CAPNPROTO_URL" "$CAPNPROTO_REF"
  local B="$BUILD/capnproto"; rm -rf "$B"; mkdir -p "$B"
  ( cd "$B" && cmake "$SRCDIR/capnproto" \
      -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_INSTALL_LIBDIR=lib \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DBUILD_TESTING=OFF && \
    cmake --build . -j"$JOBS" && cmake --install . ) >"$LOGS/capnproto.log" 2>&1 \
    || { tail -40 "$LOGS/capnproto.log"; die "capnproto failed (see $LOGS/capnproto.log)"; }
  verify "capnp runs" "$PREFIX/bin/capnp" --version
  report_bin capnp capnpc
}

# --------------------------------------------------------------------------
build_kepler() {
  if have kepler-formal; then say "kepler-formal: already built"; return 0; fi
  say "kepler-formal $KEPLER_REF (RTL-level SEC -- the second equivalence backend)"
  clone_pin kepler-formal "$KEPLER_URL" "$KEPLER_REF"
  local S="$SRCDIR/kepler-formal"
  # naja, slang, naja-if, naja-verilog, yaml-cpp, glucose, kissat and cadical are
  # all submodules. Without --recursive the configure fails on an empty
  # thirdparty/ directory, which reads like a CMake bug and is not one.
  ( cd "$S" && git submodule update --init --recursive --jobs 8 ) \
      >"$LOGS/kepler-submodule.log" 2>&1 \
      || die "kepler submodule update failed (see $LOGS/kepler-submodule.log)"

  local B="$BUILD/kepler-formal"; rm -rf "$B"; mkdir -p "$B"
  # SLANG_USE_MIMALLOC=OFF: mimalloc is the one slang dependency we do NOT
  # provide locally, and leaving it on makes configure reach for the network.
  # It is an allocator swap, not a semantic one.
  # ENABLE_UNIT_TESTS=OFF: kepler's test target pulls GoogleTest; we smoke-test
  # the binary on real designs instead, which is the evidence that matters here.
  #
  # CMAKE_BUILD_WITH_INSTALL_RPATH + CMAKE_INSTALL_RPATH: kepler links ~7 naja
  # shared libraries and has no `install` target for the CLI, so the binary CMake
  # produces carries an rpath into $BUILD. That is a live landmine: `rm -rf var/`
  # silently breaks tools/bin/kepler-formal. Baking the prefix rpath in at build
  # time lets us copy the binary and its libraries into tools/ and have them
  # resolve there, verified below with ldd.
  #
  # Python3_EXECUTABLE: naja embeds Python (SNLPyLoader). Left to itself CMake
  # picks whatever is first on PATH, which under env.sh is the uv-managed venv
  # interpreter under ~/.local/share/uv, a cache directory that uv may prune.
  # Pin the SYSTEM python so the linked libpython lives in /usr/lib64.
  local pyexe="${KEPLER_PYTHON:-/usr/bin/python3}"
  [[ -x "$pyexe" ]] || die "python interpreter not found: $pyexe"
  info "embedding python: $pyexe ($("$pyexe" -V 2>&1))"
  ( cd "$B" && cmake "$S" "${KEPLER_CMAKE_COMPAT[@]}" \
      -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_INSTALL_LIBDIR=lib \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="$PREFIX" \
      -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
      -DCMAKE_INSTALL_RPATH="$PREFIX/lib" \
      -DPython3_EXECUTABLE="$pyexe" \
      -DPython_EXECUTABLE="$pyexe" \
      -DENABLE_UNIT_TESTS=OFF \
      -DBUILD_KEPLER_PYTHON=OFF \
      -DBUILD_NAJA_PYTHON=OFF \
      -DBUILD_BENCHMARKS=OFF \
      -DSLANG_INCLUDE_TESTS=OFF \
      -DSLANG_INCLUDE_TOOLS=OFF \
      -DSLANG_INCLUDE_INSTALL=OFF \
      -DSLANG_USE_MIMALLOC=OFF \
      && cmake --build . -j"$JOBS" ) >"$LOGS/kepler-formal.log" 2>&1 \
    || { tail -60 "$LOGS/kepler-formal.log"; die "kepler-formal failed (see $LOGS/kepler-formal.log)"; }

  # Kepler has no `install` target for the CLI in this revision; place the
  # binary and every naja shared library it links into the prefix by hand.
  local exe="$B/src/bin/kepler-formal"
  [[ -x "$exe" ]] || die "kepler-formal binary not produced at $exe"
  local n=0
  while IFS= read -r so; do
    install -m755 "$so" "$PREFIX/lib/$(basename "$so")"; n=$((n+1))
  done < <(find "$B" -name "libnaja_*.so" -type f)
  [[ $n -gt 0 ]] || die "no libnaja_*.so found under $B; the CLI would not run"
  info "installed $n naja shared libraries -> $PREFIX/lib"
  install -m755 "$exe" "$PREFIX/bin/kepler-formal"
  # naja.so is the embedded-Python module the CLI loads for tech primitives and
  # is looked up beside the executable.
  [[ -f "$B/src/bin/naja.so" ]] && install -m755 "$B/src/bin/naja.so" "$PREFIX/bin/naja.so"

  # THE check that matters: the installed binary must not resolve anything out
  # of the build tree, or `rm -rf var/build` breaks the gate in a way no unit
  # test would catch.
  if ldd "$PREFIX/bin/kepler-formal" | grep -q "$BUILD"; then
    ldd "$PREFIX/bin/kepler-formal" | grep "$BUILD" >&2
    die "installed kepler-formal still links into the build tree"
  fi
  ldd "$PREFIX/bin/kepler-formal" | grep -q "not found" \
    && { ldd "$PREFIX/bin/kepler-formal" | grep "not found" >&2; die "unresolved shared libraries"; }
  info "ldd: no build-tree and no unresolved dependencies"
  verify "kepler-formal runs" "$PREFIX/bin/kepler-formal" --help
  report_bin kepler-formal
}

# --------------------------------------------------------------------------
STAGES=(fmt tomlplusplus onetbb capnproto kepler)

main() {
  local want=("$@")
  [[ ${#want[@]} -eq 0 ]] && want=("${STAGES[@]}")
  say "kepler-formal build -> $PREFIX  (jobs=$JOBS)"
  info "stages: ${want[*]}"
  local t0 failed=()
  for s in "${want[@]}"; do
    t0=$(date +%s)
    if "build_$s"; then
      info "[$s] done in $(( $(date +%s) - t0 ))s"
    else
      failed+=("$s")
      info "[$s] FAILED after $(( $(date +%s) - t0 ))s"
    fi
  done
  say "summary"
  for b in capnp kepler-formal; do
    printf '    %-14s %s\n' "$b" "$([[ -x $PREFIX/bin/$b ]] && echo OK || echo MISSING)"
  done
  if [[ ${#failed[@]} -gt 0 ]]; then
    printf '\n!!! failed stages: %s\n' "${failed[*]}"; return 1
  fi
  say "kepler-formal complete"
}

main "$@"
