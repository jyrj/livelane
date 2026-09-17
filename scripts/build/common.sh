#!/usr/bin/env bash
# Shared helpers for LiveLane's from-source toolchain builds.
#
# Everything is built into a self-contained prefix at tools/. Nothing is
# installed system-wide and nothing requires root, so the whole evaluator stack
# is reproducible from configs/pins.env on any machine.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/env.sh"
# shellcheck disable=SC1091
source "$ROOT/configs/pins.env"

PREFIX="$LIVELANE_TOOLS"
BUILD="$LIVELANE_VAR/build"
LOGS="$LIVELANE_VAR/build-logs"
SRCDIR="$LIVELANE_TP"
JOBS="${LIVELANE_JOBS:-20}"
mkdir -p "$PREFIX/bin" "$PREFIX/lib" "$PREFIX/include" "$PREFIX/share" "$BUILD" "$LOGS"

# Fedora puts /usr/lib64/ccache ahead of /usr/bin; make every build use the real
# compiler so nothing depends on a cache dir that may not be writable.
export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"
export CCACHE_DISABLE=1
# Let each tool find headers/libs from tools built earlier in the chain.
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:$PREFIX/lib64/pkgconfig:${PKG_CONFIG_PATH:-}"
export LD_LIBRARY_PATH="$PREFIX/lib:$PREFIX/lib64:${LD_LIBRARY_PATH:-}"
export CPPFLAGS="-I$PREFIX/include ${CPPFLAGS:-}"
export LDFLAGS="-L$PREFIX/lib -L$PREFIX/lib64 -Wl,-rpath,$PREFIX/lib -Wl,-rpath,$PREFIX/lib64 ${LDFLAGS:-}"

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
die()  { printf '\n!!! FATAL: %s\n' "$*" >&2; exit 1; }

# clone_pin <name> <url> <ref> -- idempotent clone+checkout into thirdparty/
clone_pin() {
  local name="$1" url="$2" ref="$3" dir="$SRCDIR/$1"
  if [[ -d "$dir/.git" ]]; then
    info "$name: checkout exists"
    git -C "$dir" rev-parse --verify --quiet "$ref^{commit}" >/dev/null 2>&1 \
      || git -C "$dir" fetch --quiet --tags origin || true
  else
    info "$name: cloning $url"
    git clone --quiet --filter=blob:none "$url" "$dir" || die "clone failed: $url"
  fi
  git -C "$dir" checkout --quiet --detach "$ref" || die "checkout failed: $name@$ref"
  info "$name: at $(git -C "$dir" rev-parse --short HEAD) ($ref)"
}

# have <binary> -- is it already built in our prefix?
have() { [[ -x "$PREFIX/bin/$1" ]]; }

# verify <description> <command...> -- run a smoke test, fail loudly.
verify() {
  local what="$1"; shift
  info "verify: $what"
  if "$@" >/dev/null 2>&1; then
    info "  OK"
  else
    printf '    FAILED: %s\n' "$*" >&2
    "$@" 2>&1 | head -20 >&2
    die "$what failed its smoke test"
  fi
}

# Report what a build produced, so a log is auditable at a glance.
report_bin() {
  for b in "$@"; do
    if [[ -x "$PREFIX/bin/$b" ]]; then
      info "built: $PREFIX/bin/$b ($(stat -c%s "$PREFIX/bin/$b" | numfmt --to=iec))"
    else
      info "MISSING: $PREFIX/bin/$b"
    fi
  done
}

# fetch_tarball <name> <url> <sha256> -- download, VERIFY, unpack into thirdparty/.
# A dependency that is not a git project still gets a pin; the pin is its hash.
fetch_tarball() {
  local name="$1" url="$2" want="$3"
  local tgz="$LIVELANE_VAR/cache/$(basename "$url")"
  local dir="$SRCDIR/$name"
  mkdir -p "$LIVELANE_VAR/cache"
  if [[ -d "$dir" ]]; then info "$name: already unpacked"; return 0; fi
  if [[ ! -f "$tgz" ]]; then
    info "$name: downloading $url"
    curl -fsSL -o "$tgz" "$url" || die "download failed: $url"
  fi
  local got; got=$(sha256sum "$tgz" | cut -d" " -f1)
  [[ "$got" == "$want" ]] || die "$name sha256 mismatch: want $want got $got"
  info "$name: sha256 verified"
  mkdir -p "$dir"
  tar -xf "$tgz" -C "$dir" --strip-components=1 || die "unpack failed: $tgz"
}
