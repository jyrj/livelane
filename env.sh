# Environment for the from-source toolchain. `source env.sh` before running the
# live tests or any node that shells out to a tool.
#
# Every path is absolute and derived; nothing is assumed to be already on PATH.
LIVELANE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export LIVELANE_ROOT
export LIVELANE_TP="$LIVELANE_ROOT/thirdparty"
export LIVELANE_TOOLS="$LIVELANE_ROOT/tools"
export LIVELANE_VAR="$LIVELANE_ROOT/var"
export PATH="$LIVELANE_TOOLS/bin:$LIVELANE_ROOT/.venv/bin:$HOME/.local/bin:$PATH"

# Build parallelism. Leave headroom on an interactive machine.
export LIVELANE_JOBS="${LIVELANE_JOBS:-$(( $(nproc 2>/dev/null || echo 4) > 4 ? $(nproc) - 4 : 1 ))}"

export PYTHONPATH="$LIVELANE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
