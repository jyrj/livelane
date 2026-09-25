#!/usr/bin/env bash
# Install the pinned sky130 PDK with ciel and verify the Liberty BOTH lanes use.
# Version, paths and the Liberty hash all come from configs/pdk.env (via env.sh).
# Idempotent: ciel reuses a version that is already installed. Run inside the venv.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/env.sh"
: "${CIEL_PDK_VERSION:?configs/pdk.env was not loaded}"

if ! command -v ciel >/dev/null 2>&1; then
  echo "==> installing ciel"
  python3 -m pip install --quiet ciel==2.6.1 || { echo "FATAL: pip install ciel failed"; exit 1; }
fi

echo "==> sky130 $CIEL_PDK_VERSION -> $CIEL_ROOT"
ciel enable --pdk-root "$CIEL_ROOT" --pdk-family sky130 "$CIEL_PDK_VERSION" \
  || { echo "FATAL: ciel enable failed"; exit 1; }

[[ -f "$LIVELANE_LIBERTY" ]] || { echo "FATAL: Liberty not found: $LIVELANE_LIBERTY"; exit 1; }
if [[ -n "${LIVELANE_LIBERTY_SHA256:-}" ]]; then
  got=$(sha256sum "$LIVELANE_LIBERTY" | cut -d' ' -f1)
  [[ "$got" == "$LIVELANE_LIBERTY_SHA256" ]] \
    || { echo "FATAL: Liberty sha256 mismatch: want $LIVELANE_LIBERTY_SHA256 got $got"; exit 1; }
  echo "    Liberty sha256 verified"
fi
echo "    OK: $LIVELANE_LIBERTY"
