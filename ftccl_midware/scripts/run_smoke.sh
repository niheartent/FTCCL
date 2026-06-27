#!/usr/bin/env bash
set -euo pipefail

np=8
bypass=0
detect=0
dead_rank=7
midware_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$(cd "${midware_dir}/.." && pwd)"
ftccl_home="${FTCCL_HOME:-${repo_root}/ftccl_lib/build}"
build_dir="${midware_dir}/build"
sync_dir="/tmp/ftccl_bypass"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --np) np="${2:?}"; shift 2 ;;
    --bypass) bypass=1; shift ;;
    --detect) detect=1; shift ;;
    --dead-rank) dead_rank="${2:?}"; shift 2 ;;
    --build-dir) build_dir="${2:?}"; shift 2 ;;
    --sync-dir) sync_dir="${2:?}"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

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

export LD_LIBRARY_PATH="${ftccl_home}/lib:${ftccl_home}/lib64:${LD_LIBRARY_PATH:-}"
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
mpirun --allow-run-as-root \
  --mca orte_abort_on_non_zero_status 0 \
  --mca mpi_abort_on_nonzero_status 0 \
  -np "$np" bash -lc \
  'rank=${OMPI_COMM_WORLD_RANK}; exec "$0" "$rank" "$1" "$2" 1024' \
  "${build_dir}/ftccl_allreduce_smoke" "$np" "$id_file"
