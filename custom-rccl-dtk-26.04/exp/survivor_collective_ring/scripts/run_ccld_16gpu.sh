#!/usr/bin/env bash
set -euo pipefail

EXP_DIR=/public/home/scnethpc26107/sdr/ftccl/custom-rccl-dtk-26.04/exp/survivor_collective_ring

source "${EXP_DIR}/env.sh"

HOSTFILE=${EXP_DIR}/hostfile
RUN_DIR=${EXP_DIR}/runs/ccld_16gpu/allreduce_normal
SYNC_DIR=${RUN_DIR}/sync
CCLD_DUMP_DIR=${RUN_DIR}/ccld

mkdir -p "${SYNC_DIR}" "${CCLD_DUMP_DIR}"
rm -f "${RUN_DIR}/nccl_unique_id.bin" "${RUN_DIR}/run.log"
find "${SYNC_DIR}" -mindepth 1 -maxdepth 1 -type f -delete
find "${CCLD_DUMP_DIR}" -mindepth 1 -maxdepth 1 -type f -name 'ccld.rank*.tsv' -delete

exec > >(tee "${RUN_DIR}/run.log") 2>&1

mpirun \
  --hostfile "${HOSTFILE}" \
  --map-by ppr:8:node \
  --bind-to none \
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
      --mode normal \
      --collective allreduce \
      --rank ${rank} \
      --nranks 16 \
      --kill-rank 3 \
      --root-rank 0 \
      --count 1024 \
      --iters 5 \
      --timeout-sec 60 \
      --id-file /public/home/scnethpc26107/sdr/ftccl/custom-rccl-dtk-26.04/exp/survivor_collective_ring/runs/ccld_16gpu/allreduce_normal/nccl_unique_id.bin \
      --sync-dir /public/home/scnethpc26107/sdr/ftccl/custom-rccl-dtk-26.04/exp/survivor_collective_ring/runs/ccld_16gpu/allreduce_normal/sync \
      --ccld-dump-dir /public/home/scnethpc26107/sdr/ftccl/custom-rccl-dtk-26.04/exp/survivor_collective_ring/runs/ccld_16gpu/allreduce_normal/ccld'
