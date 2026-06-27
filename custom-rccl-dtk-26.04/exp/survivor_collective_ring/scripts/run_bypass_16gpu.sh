#!/usr/bin/env bash
set -euo pipefail

EXP_DIR=/public/home/scnethpc26107/sdr/ftccl/custom-rccl-dtk-26.04/exp/survivor_collective_ring

source "${EXP_DIR}/env.sh"

HOSTFILE=${EXP_DIR}/hostfile
RUN_DIR=${EXP_DIR}/runs/bypass_16gpu/allreduce_kill3
SYNC_DIR=${RUN_DIR}/sync

mkdir -p "${SYNC_DIR}"
rm -f "${RUN_DIR}/nccl_unique_id.bin" "${RUN_DIR}/run.log"
find "${SYNC_DIR}" -mindepth 1 -maxdepth 1 -type f -delete

exec > >(tee "${RUN_DIR}/run.log") 2>&1

mpirun \
  --hostfile "${HOSTFILE}" \
  --map-by ppr:8:node \
  --bind-to none \
  --mca orte_abort_on_non_zero_status 0 \
  --mca mpi_abort_on_nonzero_status 0 \
  -np 16 \
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
      --collective allreduce \
      --rank ${rank} \
      --nranks 16 \
      --kill-rank 3 \
      --root-rank 0 \
      --count 1024 \
      --iters 5 \
      --timeout-sec 60 \
      --id-file /public/home/scnethpc26107/sdr/ftccl/custom-rccl-dtk-26.04/exp/survivor_collective_ring/runs/bypass_16gpu/allreduce_kill3/nccl_unique_id.bin \
      --sync-dir /public/home/scnethpc26107/sdr/ftccl/custom-rccl-dtk-26.04/exp/survivor_collective_ring/runs/bypass_16gpu/allreduce_kill3/sync'
