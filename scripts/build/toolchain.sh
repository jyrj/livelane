#!/usr/bin/env bash
# Build LiveLane's lane-S evaluator stack FROM SOURCE into tools/.
#
# No system packages are installed and root is never required. Each stage is
# idempotent and independently invocable:
#     ./scripts/build/toolchain.sh            # everything, in dependency order
#     ./scripts/build/toolchain.sh yosys eqy  # just these
#
# Dependency order (why each is here):
#   libffi   -> yosys plugin support (ffi.h is absent on this host, and eqy
#               ships a yosys plugin, so plugins are NOT optional)
#   eigen    -> OpenSTA's find_package(Eigen3 REQUIRED); header-only
#   pcre2    -> swig
#   swig     -> OpenSTA's find_package(SWIG 3.0 REQUIRED), generates TCL bindings
#   yosys    -> lane-S synthesis + the headers Fedora's package omits
#   opensta  -> lane-S timing
#   yices    -> the SMT engine sby/eqy drive
#   sby      -> eqy's engine driver
#   eqy      -> the equivalence gate itself
#   verilator-> the shared functional oracle
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# --------------------------------------------------------------------------
build_libffi() {
  have_lib() { [[ -f "$PREFIX/include/ffi.h" ]]; }
  if have_lib; then say "libffi: already built"; return 0; fi
  say "libffi $LIBFFI_REF (yosys plugin support)"
  clone_pin libffi "$LIBFFI_URL" "$LIBFFI_REF"
  local S="$SRCDIR/libffi"
  ( cd "$S" && ./autogen.sh && \
    ./configure --prefix="$PREFIX" --disable-docs --disable-multi-os-directory && \
    make -j"$JOBS" && make install ) >"$LOGS/libffi.log" 2>&1 \
    || { tail -30 "$LOGS/libffi.log"; die "libffi build failed (see $LOGS/libffi.log)"; }
  # libffi installs ffi.h under lib/libffi-*/include on some versions.
  [[ -f "$PREFIX/include/ffi.h" ]] || {
    local f; f=$(find "$PREFIX" -name ffi.h | head -1)
    [[ -n "$f" ]] && cp "$(dirname "$f")"/*.h "$PREFIX/include/" || die "ffi.h not installed"
  }
  info "ffi.h -> $PREFIX/include/ffi.h"
}

# --------------------------------------------------------------------------
build_eigen() {
  # OpenSTA calls find_package(Eigen3 REQUIRED), which needs Eigen3Config.cmake,
  # copying the headers alone is NOT enough (that was the first attempt and it
  # failed configure). Run Eigen's own CMake install, which generates the config
  # package, then the headers come along with it.
  if [[ -f "$PREFIX/share/eigen3/cmake/Eigen3Config.cmake" ]]; then
    say "eigen: already installed (with CMake config package)"; return 0; fi
  say "eigen $EIGEN_REF (header-only, but OpenSTA needs its CMake config package)"
  clone_pin eigen "$EIGEN_URL" "$EIGEN_REF"
  local B="$BUILD/eigen"; rm -rf "$B"; mkdir -p "$B"
  ( cd "$B" && cmake "$SRCDIR/eigen" -DCMAKE_INSTALL_PREFIX="$PREFIX" \
      -DBUILD_TESTING=OFF -DEIGEN_BUILD_DOC=OFF && \
    cmake --install . ) >"$LOGS/eigen.log" 2>&1 \
    || { tail -30 "$LOGS/eigen.log"; die "eigen failed (see $LOGS/eigen.log)"; }
  local cfg; cfg=$(find "$PREFIX" -name "Eigen3Config.cmake" | head -1)
  [[ -n "$cfg" ]] || die "Eigen3Config.cmake not installed; OpenSTA configure will fail"
  info "Eigen3Config.cmake -> $cfg"
}

# --------------------------------------------------------------------------
build_pcre2() {
  if [[ -f "$PREFIX/lib/libpcre2-8.a" || -f "$PREFIX/lib64/libpcre2-8.a" ]]; then
    say "pcre2: already built"; return 0; fi
  say "pcre2 $PCRE2_REF (swig dependency)"
  clone_pin pcre2 "$PCRE2_URL" "$PCRE2_REF"
  local B="$BUILD/pcre2"; rm -rf "$B"; mkdir -p "$B"
  ( cd "$B" && cmake "$SRCDIR/pcre2" -DCMAKE_INSTALL_PREFIX="$PREFIX" \
      -DCMAKE_INSTALL_LIBDIR=lib -DBUILD_SHARED_LIBS=OFF \
      -DPCRE2_BUILD_TESTS=OFF -DPCRE2_BUILD_PCRE2GREP=OFF \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_BUILD_TYPE=Release && \
    cmake --build . -j"$JOBS" && cmake --install . ) >"$LOGS/pcre2.log" 2>&1 \
    || { tail -30 "$LOGS/pcre2.log"; die "pcre2 failed (see $LOGS/pcre2.log)"; }
  info "libpcre2 -> $PREFIX/lib"
}

# --------------------------------------------------------------------------
build_swig() {
  if have swig; then say "swig: already built ($("$PREFIX/bin/swig" -version 2>&1 | grep -i version | head -1))"; return 0; fi
  say "swig $SWIG_REF (OpenSTA TCL bindings)"
  clone_pin swig "$SWIG_URL" "$SWIG_REF"
  local S="$SRCDIR/swig"
  ( cd "$S" && ./autogen.sh && \
    ./configure --prefix="$PREFIX" --with-pcre2-prefix="$PREFIX" \
                --disable-ccache --without-alllang && \
    make -j"$JOBS" && make install ) >"$LOGS/swig.log" 2>&1 \
    || { tail -40 "$LOGS/swig.log"; die "swig failed (see $LOGS/swig.log)"; }
  verify "swig runs" "$PREFIX/bin/swig" -version
  report_bin swig
}

# --------------------------------------------------------------------------
build_yosys() {
  if have yosys && [[ -f "$PREFIX/share/yosys/include/kernel/yosys.h" ]]; then
    say "yosys: already built with headers"; return 0; fi
  say "yosys $YOSYS_DESCRIBE ($YOSYS_REF)"
  info "this is the EXACT commit Fedora's yosys-0.67+post was built from,"
  info "so lane-S QoR reproduces the 2026-09-02 baseline bit-for-bit"
  clone_pin yosys "$YOSYS_URL" "$YOSYS_REF"
  local S="$SRCDIR/yosys"
  # This yosys generation is CMake-based; the old `make config-gcc` is gone.
  # abc, fmt, cxxopts and frontends/slang/lib are submodules, without them the
  # build either fails or silently produces a yosys with no abc and no read_slang.
  ( cd "$S" && git submodule update --init --recursive --depth 1 ) \
      >"$LOGS/yosys-submodule.log" 2>&1 || info "submodule update reported issues (see log)"
  local B="$BUILD/yosys"; rm -rf "$B"; mkdir -p "$B"
  # readline/editline headers are absent on this host and are optional for yosys;
  # libffi comes from our own prefix via CMAKE_PREFIX_PATH + PKG_CONFIG_PATH.
  ( cd "$B" && cmake "$S" \
      -DCMAKE_INSTALL_PREFIX="$PREFIX" \
      -DCMAKE_INSTALL_LIBDIR=lib \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="$PREFIX" \
      -DYOSYS_WITHOUT_READLINE=ON \
      -DYOSYS_WITHOUT_EDITLINE=ON && \
    cmake --build . -j"$JOBS" && cmake --install . ) >"$LOGS/yosys.log" 2>&1 \
    || { tail -40 "$LOGS/yosys.log"; die "yosys failed (see $LOGS/yosys.log)"; }
  verify "yosys runs" "$PREFIX/bin/yosys" -V
  # NOTE: `yosys-abc -h` prints usage and exits non-zero, do not use it as a
  # smoke test. Run a real command instead.
  verify "yosys-abc runs a command" "$PREFIX/bin/yosys-abc" -q "version"
  # Report the two capabilities that decide lane-S design:
  #  - plugin headers: eqy ships a yosys plugin and cannot build without them
  #  - read_slang: if present, lane S can read XiangShan .sv WITHOUT sv2v, which
  #    removes the sv2v flattening that inflated the medium block to 807k cells
  if [[ -f "$PREFIX/share/yosys/include/kernel/yosys.h" ]]; then
    info "plugin headers: PRESENT ($PREFIX/share/yosys/include)"
  else
    info "plugin headers: ABSENT -- eqy's plugin build will need -DYOSYS_INSTALL_LIBRARY"
  fi
  if "$PREFIX/bin/yosys" -p "help read_slang" >/dev/null 2>&1; then
    info "read_slang: AVAILABLE (lane S can read SystemVerilog natively)"
  else
    info "read_slang: not available (lane S needs sv2v for XiangShan)"
  fi
  report_bin yosys yosys-abc yosys-config
}

# --------------------------------------------------------------------------
build_cudd() {
  if [[ -f "$PREFIX/lib/libcudd.a" ]]; then say "cudd: already built"; return 0; fi
  say "cudd $CUDD_REF (REQUIRED by OpenSTA -- BDDs for conditional timing arcs)"
  clone_pin cudd "$CUDD_URL" "$CUDD_REF"
  local S="$SRCDIR/cudd"
  # CUDD 3.0.0 predates modern autoconf; regenerate rather than trusting the
  # shipped configure, and build PIC because OpenSTA links it into a shared lib.
  ( cd "$S" && (autoreconf -fi || true) && \
    ./configure --prefix="$PREFIX" --enable-shared --enable-obj \
                CFLAGS="-fPIC -O2" CXXFLAGS="-fPIC -O2" && \
    make -j"$JOBS" && make install ) >"$LOGS/cudd.log" 2>&1 \
    || { tail -40 "$LOGS/cudd.log"; die "cudd failed (see $LOGS/cudd.log)"; }
  [[ -f "$PREFIX/include/cudd.h" ]] || {
    # CUDD's install sometimes omits the internal headers OpenSTA's FindCUDD wants.
    for h in "$S"/cudd/cudd.h "$S"/config.h "$S"/util/util.h "$S"/mtr/mtr.h "$S"/epd/epd.h; do
      [[ -f "$h" ]] && cp "$h" "$PREFIX/include/" || true
    done
  }
  info "libcudd -> $PREFIX/lib/libcudd.a"
}

# --------------------------------------------------------------------------
build_opensta() {
  if have sta; then say "OpenSTA: already built"; return 0; fi
  say "OpenSTA $OPENSTA_REF (lane-S timing)"
  local B="$BUILD/opensta"; rm -rf "$B"; mkdir -p "$B"
  ( cd "$B" && cmake "$SRCDIR/OpenSTA" \
      -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_BUILD_TYPE=Release \
      -DSWIG_EXECUTABLE="$PREFIX/bin/swig" \
      -DEIGEN3_INCLUDE_DIR="$PREFIX/include/eigen3" \
      -DCMAKE_PREFIX_PATH="$PREFIX" \
      -DCUDD_DIR="$PREFIX" && \
    cmake --build . -j"$JOBS" && cmake --install . ) >"$LOGS/opensta.log" 2>&1 \
    || { tail -40 "$LOGS/opensta.log"; die "OpenSTA failed (see $LOGS/opensta.log)"; }
  verify "sta runs" sh -c "echo exit | '$PREFIX/bin/sta' -no_splash"
  report_bin sta
}

# --------------------------------------------------------------------------
build_gperf() {
  if have gperf; then say "gperf: already built"; return 0; fi
  say "gperf (REQUIRED by yices2's configure)"
  fetch_tarball gperf "$GPERF_URL" "$GPERF_SHA256"
  ( cd "$SRCDIR/gperf" && ./configure --prefix="$PREFIX" && \
    make -j"$JOBS" && make install ) >"$LOGS/gperf.log" 2>&1 \
    || { tail -30 "$LOGS/gperf.log"; die "gperf failed (see $LOGS/gperf.log)"; }
  verify "gperf runs" "$PREFIX/bin/gperf" --version
  report_bin gperf
}

# --------------------------------------------------------------------------
build_yices() {
  if have yices-smt2 || have yices; then say "yices: already built"; return 0; fi
  say "yices $YICES_REF (SMT engine for the equivalence gate)"
  clone_pin yices2 "$YICES_URL" "$YICES_REF"
  local S="$SRCDIR/yices2"
  ( cd "$S" && export PATH="$PREFIX/bin:$PATH" && autoconf && \
    ./configure --prefix="$PREFIX" && \
    make -j"$JOBS" && make install ) >"$LOGS/yices.log" 2>&1 \
    || { tail -40 "$LOGS/yices.log"; die "yices failed (see $LOGS/yices.log)"; }
  report_bin yices yices-smt2
}

# --------------------------------------------------------------------------
build_sby() {
  if have sby; then say "sby: already built"; return 0; fi
  say "sby $SBY_REF (eqy's engine driver)"
  ( cd "$SRCDIR/sby" && make install PREFIX="$PREFIX" ) >"$LOGS/sby.log" 2>&1 \
    || { tail -30 "$LOGS/sby.log"; die "sby failed (see $LOGS/sby.log)"; }
  report_bin sby
}

# --------------------------------------------------------------------------
build_eqy() {
  if have eqy; then say "eqy: already built"; return 0; fi
  say "eqy $EQY_REF (THE equivalence gate -- runs in every arm)"
  # Recorded patches (patches/eqy-*.patch), applied idempotently.
  # Headers are src/..., so -p0.
  local P
  for P in "$ROOT"/patches/eqy-*.patch; do
    [[ -e "$P" ]] || continue
    if git -C "$SRCDIR/eqy" apply -p0 --reverse --check "$P" >/dev/null 2>&1; then
      info "patch already applied: $(basename "$P")"
    elif git -C "$SRCDIR/eqy" apply -p0 --check "$P" >/dev/null 2>&1; then
      git -C "$SRCDIR/eqy" apply -p0 "$P" || die "patch failed: $P"
      info "applied patch: $(basename "$P")"
    else
      die "patch does not apply cleanly: $P"
    fi
  done
  # eqy builds a yosys plugin, so it needs OUR yosys's yosys-config on PATH.
  ( cd "$SRCDIR/eqy" && \
    make -j"$JOBS" PREFIX="$PREFIX" YOSYS_CONFIG="$PREFIX/bin/yosys-config" && \
    make install PREFIX="$PREFIX" YOSYS_CONFIG="$PREFIX/bin/yosys-config" ) \
    >"$LOGS/eqy.log" 2>&1 \
    || { tail -40 "$LOGS/eqy.log"; die "eqy failed (see $LOGS/eqy.log)"; }
  report_bin eqy
}

# --------------------------------------------------------------------------
build_help2man() {
  if have help2man; then say "help2man: already built"; return 0; fi
  say "help2man (verilator's install step generates man pages with it)"
  fetch_tarball help2man "$HELP2MAN_URL" "$HELP2MAN_SHA256"
  ( cd "$SRCDIR/help2man" && ./configure --prefix="$PREFIX" && \
    make -j"$JOBS" && make install ) >"$LOGS/help2man.log" 2>&1 \
    || { tail -30 "$LOGS/help2man.log"; die "help2man failed (see $LOGS/help2man.log)"; }
  verify "help2man runs" "$PREFIX/bin/help2man" --version
  report_bin help2man
}

# --------------------------------------------------------------------------
build_verilator() {
  if have verilator; then say "verilator: already built"; return 0; fi
  say "verilator $VERILATOR_REF (shared functional oracle, all arms)"
  clone_pin verilator "$VERILATOR_URL" "$VERILATOR_REF"
  local S="$SRCDIR/verilator"
  ( cd "$S" && export PATH="$PREFIX/bin:$PATH" && autoconf && \
    ./configure --prefix="$PREFIX" && \
    make -j"$JOBS" && make install ) >"$LOGS/verilator.log" 2>&1 \
    || { tail -40 "$LOGS/verilator.log"; die "verilator failed (see $LOGS/verilator.log)"; }
  verify "verilator runs" "$PREFIX/bin/verilator" --version
  report_bin verilator
}

# --------------------------------------------------------------------------
STAGES=(libffi eigen pcre2 swig cudd yosys opensta gperf yices sby eqy help2man verilator)

main() {
  local want=("$@")
  [[ ${#want[@]} -eq 0 ]] && want=("${STAGES[@]}")
  say "LiveLane toolchain build -> $PREFIX  (jobs=$JOBS)"
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
  for b in yosys yosys-abc sta eqy sby verilator swig lhd; do
    printf '    %-12s %s\n' "$b" "$([[ -x $PREFIX/bin/$b ]] && echo OK || echo MISSING)"
  done
  if [[ ${#failed[@]} -gt 0 ]]; then
    printf '\n!!! failed stages: %s\n' "${failed[*]}"; return 1
  fi
  say "toolchain complete"
}

main "$@"
