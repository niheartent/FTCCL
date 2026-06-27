#!/usr/bin/env bash
set -euo pipefail

log_file="${1:-/tmp/ftccl_megatron_patch.log}"

cd /home/ubuntu/sdr/Megatron-LM

TRAIN_ITERS="${TRAIN_ITERS:-3}" \
EVAL_ITERS="${EVAL_ITERS:-0}" \
FTCCL_LOG_LEVEL="${FTCCL_LOG_LEVEL:-1}" \
FTCCL_PATCH_LOG_LEVEL="${FTCCL_PATCH_LOG_LEVEL:-1}" \
FTCCL_BYPASS_MIN_COMM_ORDINAL="${FTCCL_BYPASS_MIN_COMM_ORDINAL:-2}" \
FTCCL_BYPASS_VICTIM_MODE="${FTCCL_BYPASS_VICTIM_MODE:-noop}" \
FTCCL_PATCH_GLOBAL_BATCH_SIZE_AFTER_BYPASS="${FTCCL_PATCH_GLOBAL_BATCH_SIZE_AFTER_BYPASS:-126}" \
FTCCL_PATCH_GRAD_COMPENSATION="${FTCCL_PATCH_GRAD_COMPENSATION:-survivor_global_batch}" \
timeout "${FTCCL_MEGATRON_TIMEOUT_SEC:-300s}" \
  bash examples/llama/train_llama3.2_1b.sh > "${log_file}" 2>&1

required_patterns=(
  "rank 7 entering logical victim mode=noop"
  "bypass complete: survivorRank=0 survivorCount=7"
  "activated survivor training view generation=1"
  "dataloader rebuilt generation=1 dp_rank=0 dp_size=7"
  "iteration        3/       3"
  "[after training is done]"
)

for pattern in "${required_patterns[@]}"; do
  if ! grep -Fq "${pattern}" "${log_file}"; then
    echo "missing expected pattern: ${pattern}" >&2
    echo "log: ${log_file}" >&2
    exit 1
  fi
done

echo "FTCCL_MEGATRON_DEMO_PASS log=${log_file}"
