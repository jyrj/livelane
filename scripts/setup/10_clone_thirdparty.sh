#!/usr/bin/env bash
# Clone every third-party dependency at its pinned commit into thirdparty/.
# Idempotent: re-running fetches and re-checks-out the pin, never pushes.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/configs/pins.env"
TP="$ROOT/thirdparty"
LOG="$ROOT/var/build-logs"
mkdir -p "$TP" "$LOG"

# clone_pin <name> <url> <ref> [extra git-clone args...]
clone_pin() {
  local name="$1" url="$2" ref="$3"; shift 3
  local dir="$TP/$name"
  echo "==> [$name] pin=$ref"
  if [[ -d "$dir/.git" ]]; then
    echo "    exists; fetching pin"
    git -C "$dir" fetch --quiet origin "$ref" 2>/dev/null || git -C "$dir" fetch --quiet origin
  else
    echo "    cloning $url"
    git clone "$@" "$url" "$dir" || { echo "    CLONE FAILED"; return 1; }
  fi
  git -C "$dir" checkout --quiet --detach "$ref" || { echo "    CHECKOUT FAILED for $ref"; return 1; }
  local got; got=$(git -C "$dir" rev-parse HEAD)
  if [[ "$got" != "$ref" ]]; then echo "    PIN MISMATCH: want $ref got $got"; return 1; fi
  echo "    OK at $got"
}

rc=0
clone_pin lhdsuite    "$LHDSUITE_URL"  "$LHDSUITE_REF"  --filter=blob:none || rc=1
clone_pin chia        "$CHIA_URL"      "$CHIA_REF"                          || rc=1
clone_pin OpenSTA     "$OPENSTA_URL"   "$OPENSTA_REF"                       || rc=1
clone_pin eqy         "$EQY_URL"       "$EQY_REF"                           || rc=1
clone_pin sby         "$SBY_URL"       "$SBY_REF"                           || rc=1
clone_pin sv2v        "$SV2V_URL"      "$SV2V_REF"                          || rc=1
clone_pin picorv32    "$PICORV32_URL"  "$PICORV32_REF"                      || rc=1
# designs under test: full history, because the corpus and the rewrite campaign
# check out other commits of ibex as worktrees
clone_pin ibex        "$IBEX_URL"      "$IBEX_REF"      --filter=blob:none  || rc=1
clone_pin cva6        "$CVA6_URL"      "$CVA6_REF"      --filter=blob:none  || rc=1
# bug-fix corpus: scripts/chia/hwebench.py reads thirdparty/hwe-bench/datasets
clone_pin hwe-bench   "$HWEBENCH_URL"  "$HWEBENCH_REF"                      || rc=1
echo "=== clone summary (rc=$rc) ==="
for d in "$TP"/*/; do
  [[ -d "$d/.git" ]] && printf "%-14s %s\n" "$(basename "$d")" "$(git -C "$d" rev-parse --short HEAD)"
done
exit $rc
