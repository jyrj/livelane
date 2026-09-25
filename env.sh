# LiveLane environment, single source of truth. `source env.sh` before anything.
# Every path here is absolute and derived; nothing is assumed to be on PATH.
LIVELANE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export LIVELANE_ROOT
export LIVELANE_TP="$LIVELANE_ROOT/thirdparty"
export LIVELANE_TOOLS="$LIVELANE_ROOT/tools"
export LIVELANE_VAR="$LIVELANE_ROOT/var"
export PATH="$LIVELANE_TOOLS/bin:$LIVELANE_ROOT/.venv/bin:$HOME/.local/bin:$PATH"

# Build parallelism (default: all but four hardware threads).
export LIVELANE_JOBS="${LIVELANE_JOBS:-$(( $(nproc 2>/dev/null || echo 8) > 4 ? $(nproc 2>/dev/null || echo 8) - 4 : 1 ))}"

# sky130 Liberty used identically by every lane (installed by scripts/setup/40_pdk.sh)
[ -f "$LIVELANE_ROOT/configs/pdk.env" ] && . "$LIVELANE_ROOT/configs/pdk.env"

# Vertex AI agent seat. Set GOOGLE_CLOUD_PROJECT in the environment, or in
# configs/gcp.local.env (gitignored). Gemini 3.x models answer only on `global`.
export GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-global}"
[ -f "$LIVELANE_ROOT/configs/gcp.local.env" ] && . "$LIVELANE_ROOT/configs/gcp.local.env"

# Python package under development
export PYTHONPATH="$LIVELANE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
