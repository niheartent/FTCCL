#!/usr/bin/env bash
set -euo pipefail

EXP_DIR=/public/home/scnethpc26107/sdr/ftccl/custom-rccl-dtk-26.04/exp/survivor_collective_ring

source "${EXP_DIR}/env.sh"

HOSTFILE=${EXP_DIR}/hostfile
OUT_DIR=${EXP_DIR}/runs/bypass_matrix_16gpu_ccld
NP=16
COUNT=1024
ITERS=5
TIMEOUT_SEC=60

run_case() {
  local collective="$1"
  local kill_rank="$2"
  local root_rank="${3:-}"
  local case_name="${collective}_kill${kill_rank}"
  local root_arg=()

  if [[ -n "${root_rank}" ]]; then
    case_name="${case_name}_root${root_rank}"
    root_arg=(--root-rank "${root_rank}")
  fi

  local run_dir="${OUT_DIR}/${case_name}"
  local sync_dir="${run_dir}/sync"
  local ccld_dir="${run_dir}/ccld"
  local id_file="${run_dir}/nccl_unique_id.bin"
  local log_file="${run_dir}/run.log"

  mkdir -p "${sync_dir}" "${ccld_dir}"
  rm -f "${id_file}" "${log_file}"
  find "${sync_dir}" -mindepth 1 -maxdepth 1 -type f -delete
  find "${ccld_dir}" -mindepth 1 -maxdepth 1 -type f -name 'ccld.rank*.tsv' -delete

  echo "[matrix] ${case_name}"
  mpirun \
    --hostfile "${HOSTFILE}" \
    --map-by ppr:8:node \
    --bind-to none \
    --mca orte_abort_on_non_zero_status 0 \
    --mca mpi_abort_on_nonzero_status 0 \
    -np "${NP}" \
    bash -lc '
      source /etc/profile
      module load compiler/dtk/26.04
      module load mpi/hpcx/2.18.0/gcc-8.5.0/shca
      module load app/rccl/shca_rdma_plugins/v8

      cd /public/home/scnethpc26107/sdr/ftccl/custom-rccl-dtk-26.04/exp/survivor_collective_ring
      source env.sh

      rank=${OMPI_COMM_WORLD_RANK}
      exec /public/home/scnethpc26107/sdr/ftccl/custom-rccl-dtk-26.04/exp/survivor_collective_ring/bin/survivor_collective_after_kill \
        --mode bypass \
        --collective '"${collective}"' \
        --rank ${rank} \
        --nranks '"${NP}"' \
        --kill-rank '"${kill_rank}"' \
        '"${root_arg[*]}"' \
        --count '"${COUNT}"' \
        --iters '"${ITERS}"' \
        --timeout-sec '"${TIMEOUT_SEC}"' \
        --id-file '"${id_file}"' \
        --sync-dir '"${sync_dir}"' \
        --ccld-dump-dir '"${ccld_dir}"'
    ' 2>&1 | tee "${log_file}"
}

run_root_cases() {
  local collective="$1"
  local kill_rank="$2"
  local root_rank
  for ((root_rank = 0; root_rank < NP; ++root_rank)); do
    if [[ "${root_rank}" == "${kill_rank}" ]]; then
      continue
    fi
    run_case "${collective}" "${kill_rank}" "${root_rank}"
  done
}

for ((kill_rank = 0; kill_rank < NP; ++kill_rank)); do
  run_case allreduce "${kill_rank}"
  run_case reducescatter "${kill_rank}"
  run_case allgather "${kill_rank}"
  run_root_cases broadcast "${kill_rank}"
  run_root_cases reduce "${kill_rank}"
done
