#!/usr/bin/env bash
set -euo pipefail

bypass=0
detect=0
dead_rank=7
hostfile=""
preflight=0
midware_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$(cd "${midware_dir}/.." && pwd)"
ftccl_home="${FTCCL_HOME:-${repo_root}/custom-rccl-dtk-26.04/build}"
hip_path="${HIP_PATH:-/public/software/compiler/dtk-26.04/hip}"
dtk_root="${DTKROOT:-/public/software/compiler/dtk-26.04}"
env_script="${FTCCL_ENV_SCRIPT:-/public/home/scnethpc26107/sdr/env.sh}"
build_dir="${midware_dir}/build"
sync_dir="${midware_dir}/runs/smoke_$(date +%Y%m%d-%H%M%S)/sync"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hostfile) hostfile="${2:?}"; shift 2 ;;
    --bypass) bypass=1; shift ;;
    --detect) detect=1; shift ;;
    --preflight) preflight=1; shift ;;
    --dead-rank) dead_rank="${2:?}"; shift 2 ;;
    --build-dir) build_dir="${2:?}"; shift 2 ;;
    --sync-dir) sync_dir="${2:?}"; shift 2 ;;
    --env-script) env_script="${2:?}"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$hostfile" ]]; then
  echo "Usage: $0 --hostfile <hostfile> [--bypass] [--dead-rank N] [--detect] [--sync-dir DIR]" >&2
  exit 1
fi
if [[ ! -f "$hostfile" ]]; then
  echo "Hostfile not found: $hostfile" >&2
  exit 1
fi

np=$(awk '
  NF && $1 !~ /^#/ {
    slot=0
    for (i=1; i<=NF; i++) {
      if ($i ~ /^slots=/) {
        split($i, a, "=")
        slot=a[2]
      }
    }
    if (slot <= 0) slot=8
    total += slot
  }
  END {print total+0}
' "$hostfile")
slots_per_node=$(awk '
  NF && $1 !~ /^#/ {
    slot=0
    for (i=1; i<=NF; i++) {
      if ($i ~ /^slots=/) {
        split($i, a, "=")
        slot=a[2]
      }
    }
    if (slot <= 0) slot=8
    if (!seen) {
      first=slot
      seen=1
    } else if (slot != first) {
      mismatch=1
    }
  }
  END {
    if (mismatch || !seen) exit 2
    print first+0
  }
' "$hostfile") || {
  echo "Hostfile entries must have the same positive slots value for ppr mapping" >&2
  exit 1
}
if [[ "$np" -le 0 ]]; then
  echo "Hostfile has no usable slots: $hostfile" >&2
  exit 1
fi
host_count=$(awk 'NF && $1 !~ /^#/ {count++} END {print count+0}' "$hostfile")
if [[ "$host_count" -gt 1 && "$sync_dir" == /tmp/* ]]; then
  echo "sync-dir must be on a shared filesystem for multi-node smoke, not local /tmp: $sync_dir" >&2
  echo "Use --sync-dir ${midware_dir}/runs/smoke_manual/sync or omit --sync-dir." >&2
  exit 1
fi

rm -rf "$sync_dir"
mkdir -p "$sync_dir"
if [[ "$bypass" == "1" ]]; then
  "${BASH_SOURCE[0]%/*}/ftccl_bypassctl.py" \
    --sync-dir "$sync_dir" request \
    --rank "$dead_rank" \
    --world-size "$np" \
    --generation 1 \
    --source manual \
    --reason smoke-test
fi

dtk_lib_path="${ftccl_home}:${ftccl_home}/lib:${ftccl_home}/lib64"
dtk_lib_path="${dtk_lib_path}:${hip_path}/lib:${hip_path}/../lib"
dtk_lib_path="${dtk_lib_path}:${dtk_root}/lib:${dtk_root}/lib64"
dtk_lib_path="${dtk_lib_path}:${dtk_root}/dcc/lib:${dtk_root}/dcc/lib64"
dtk_lib_path="${dtk_lib_path}:${dtk_root}/dcc/comgr/lib64:${dtk_root}/dcc/gcvm/lib"
dtk_lib_path="${dtk_lib_path}:${dtk_root}/.hyhal/lib:${dtk_root}/.hyhal/lib64:${dtk_root}/.hyhal/hsa/lib"
dtk_lib_path="${dtk_lib_path}:/opt/hyhal/lib:/opt/hyhal/lib64"

export LD_LIBRARY_PATH="${dtk_lib_path}:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="${build_dir}/libftccl_midware.so"
export NCCL_ALGO=RING
export NCCL_PROTO=SIMPLE
export NCCL_CCLD_ENABLE="${NCCL_CCLD_ENABLE:-1}"
export NCCL_COLLNET_ENABLE=0
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export FTCCL_BYPASS_ENABLE=1
export FTCCL_BYPASS_DIR="$sync_dir"
export FTCCL_BYPASS_WORLD_SIZE="$np"
export FTCCL_BYPASS_DEAD_RANK="$dead_rank"
export FTCCL_BYPASS_VICTIM_MODE=exit0
export FTCCL_LOG_LEVEL="${FTCCL_LOG_LEVEL:-1}"
export FTCCL_DETECT_ENABLE="$detect"
export FTCCL_DETECT_CCLD_ENABLE="${FTCCL_DETECT_CCLD_ENABLE:-1}"
export FTCCL_DETECT_TIMEOUT_MS="${FTCCL_DETECT_TIMEOUT_MS:-5000}"
export FTCCL_DETECT_RUN_ID="smoke-$(date +%s)-$$"

id_file="$sync_dir/nccl_unique_id.bin"
echo "FTCCL_DCU_SMOKE hostfile=$hostfile np=$np map_by=ppr:${slots_per_node}:node sync_dir=$sync_dir"
if [[ "$preflight" == "1" ]]; then
  mpirun --allow-run-as-root \
    --hostfile "$hostfile" \
    --map-by "ppr:${slots_per_node}:node" \
    --bind-to none \
    --nooversubscribe \
    -np "$np" bash -lc \
    'if [[ -f "$0" ]]; then source "$0"; fi
     export LD_LIBRARY_PATH="$1:${LD_LIBRARY_PATH:-}"
     unset LD_PRELOAD
     rank=${OMPI_COMM_WORLD_RANK}
     host=$(hostname)
     echo "PREFLIGHT rank=${rank} host=${host} HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-unset}"
     python - <<'"'"'PY'"'"'
import ctypes, os
for lib in ("libamd_comgr.so", "libamdhip64.so"):
    try:
        ctypes.CDLL(lib)
        print(f"LOAD_OK {lib}", flush=True)
    except OSError as exc:
        print(f"LOAD_FAIL {lib}: {exc}", flush=True)
PY
     if command -v rocm-smi >/dev/null 2>&1; then rocm-smi --showid | sed -n "1,12p"; fi' \
    "$env_script" "$dtk_lib_path"
  exit 0
fi
mpirun --allow-run-as-root \
  --mca orte_abort_on_non_zero_status 0 \
  --mca mpi_abort_on_nonzero_status 0 \
  --hostfile "$hostfile" \
  --map-by "ppr:${slots_per_node}:node" \
  --bind-to none \
  --nooversubscribe \
  -np "$np" bash -lc \
  'if [[ -f "$0" ]]; then source "$0"; fi
   export LD_LIBRARY_PATH="$1:${LD_LIBRARY_PATH:-}"
   export LD_PRELOAD="$2"
   rank=${OMPI_COMM_WORLD_RANK}
   exec "$3" "$rank" "$4" "$5" 1024' \
  "$env_script" "$dtk_lib_path" "${build_dir}/libftccl_midware.so" "${build_dir}/ftccl_allreduce_smoke" "$np" "$id_file"
