#!/usr/bin/env bash
set -uo pipefail

usage() {
  cat <<'USAGE'
Build once and run survivor parent-communicator collective tests.

This runs the RING matrix: every failed rank, AllReduce,
ReduceScatter, AllGather, and every surviving Broadcast/Reduce root.

Examples:
  ./exp/survivor_collective_ring/scripts/run_collective_matrix.sh \
    --nccl-home "$(pwd)/build"
  ./exp/survivor_collective_ring/scripts/run_collective_matrix.sh \
    --collective broadcast --kill-rank 1 --root-rank 3 --nccl-home "$(pwd)/build"

Options:
  --algo NAME         ring only. Default: ring.
  --collective NAME   all, allreduce, reducescatter, allgather, broadcast,
                      or reduce. Default: all.
  --np N              Rank count. Default: 4.
  --kill-rank R       Test one failed rank. Overrides --kill-ranks.
  --kill-ranks A,B    Failed ranks to test. Default: all ranks.
  --root-rank R       Test one root for Broadcast/Reduce. Overrides --roots.
  --roots A,B         Root parent ranks for Broadcast/Reduce.
                      Default: all surviving ranks.
  --count N           Number of int elements per rank operation. Default: 1024.
  --iters N           Collective iterations per case. Default: 5.
  --timeout-sec N     Program setup/communication timeout. Default: 30.
  --job-timeout SEC   Whole launcher timeout per case. Default: 90.
  --out-dir DIR       Output root. Default: exp/survivor_collective_ring/runs.
  --build-dir DIR     Binary output directory. Default: exp/survivor_collective_ring/bin.
  --nccl-home DIR     NCCL/FCCL install prefix with include/ and lib/.
  --cuda-home DIR     CUDA prefix. Default: /usr/local/cuda.
  --cxx CXX           C++ compiler. Default: g++.
  --help              Show this help.
USAGE
}

die() {
  echo "error: $*" >&2
  exit 1
}

exp_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src="${exp_dir}/src/survivor_collective_after_kill.cu"

collective="all"
algo="RING"
np="4"
kill_ranks=""
roots=""
count="1024"
iters="5"
timeout_sec="30"
job_timeout="90"
out_dir="${exp_dir}/runs"
build_dir="${exp_dir}/bin"
nccl_home=""
cuda_home="/usr/local/cuda"
cxx="g++"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --collective) collective="${2:?missing value for --collective}"; shift 2 ;;
    --algo) algo="${2:?missing value for --algo}"; shift 2 ;;
    --np) np="${2:?missing value for --np}"; shift 2 ;;
    --kill-rank) kill_ranks="${2:?missing value for --kill-rank}"; shift 2 ;;
    --kill-ranks) kill_ranks="${2:?missing value for --kill-ranks}"; shift 2 ;;
    --root-rank) roots="${2:?missing value for --root-rank}"; shift 2 ;;
    --roots) roots="${2:?missing value for --roots}"; shift 2 ;;
    --count) count="${2:?missing value for --count}"; shift 2 ;;
    --iters) iters="${2:?missing value for --iters}"; shift 2 ;;
    --timeout-sec) timeout_sec="${2:?missing value for --timeout-sec}"; shift 2 ;;
    --job-timeout) job_timeout="${2:?missing value for --job-timeout}"; shift 2 ;;
    --out-dir) out_dir="${2:?missing value for --out-dir}"; shift 2 ;;
    --build-dir) build_dir="${2:?missing value for --build-dir}"; shift 2 ;;
    --nccl-home) nccl_home="${2:?missing value for --nccl-home}"; shift 2 ;;
    --cuda-home) cuda_home="${2:?missing value for --cuda-home}"; shift 2 ;;
    --cxx) cxx="${2:?missing value for --cxx}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

case "$collective" in
  all|allreduce|reducescatter|allgather|broadcast|reduce) ;;
  *) die "unsupported --collective: ${collective}" ;;
esac
case "$algo" in
  ring|RING) nccl_algo="RING" ;;
  *) die "unsupported --algo: ${algo}" ;;
esac
[[ "$np" =~ ^[0-9]+$ && "$np" -ge 2 ]] || die "--np must be at least 2"
[[ -n "$nccl_home" ]] || die "--nccl-home is required"

if [[ -z "$kill_ranks" ]]; then
  for ((rank = 0; rank < np; ++rank)); do
    kill_ranks="${kill_ranks}${kill_ranks:+,}${rank}"
  done
fi
IFS=',' read -r -a kill_rank_list <<< "$kill_ranks"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
algo_tag="$(printf %s "$nccl_algo" | tr '[:upper:]' '[:lower:]')"
matrix_dir="${out_dir%/}/${timestamp}_${algo_tag}_survivor_parent_native_collectives"
bin="${build_dir%/}/survivor_collective_after_kill"
mkdir -p "$matrix_dir" "$build_dir"

build_cmd=(
  "$cxx" -x c++ -std=c++17 -O2
  "-I${cuda_home%/}/include" "-I${nccl_home%/}/include"
  "$src"
  "-L${nccl_home%/}/lib" "-L${nccl_home%/}/lib64"
  "-L${cuda_home%/}/lib64" -lcudart -lnccl
  "-Wl,-rpath,${nccl_home%/}/lib:${nccl_home%/}/lib64:${cuda_home%/}/lib64"
  -o "$bin"
)
printf '%q ' "${build_cmd[@]}" > "${matrix_dir}/build_command.txt"
printf '\n' >> "${matrix_dir}/build_command.txt"
if ! "${build_cmd[@]}" > "${matrix_dir}/build.log" 2>&1; then
  echo "build failed: ${matrix_dir}/build.log" >&2
  exit 1
fi

export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,P2P,GRAPH,COLL,NET,ENV,DESTROY}"
export NCCL_ALGO="$nccl_algo"
export NCCL_PROTO="${NCCL_PROTO:-SIMPLE}"
export NCCL_COLLNET_ENABLE="${NCCL_COLLNET_ENABLE:-0}"

total=0
passed=0
failed=0
summary_file="${matrix_dir}/summary.tsv"
printf "case\talgorithm\tcollective\tkill_rank\troot_rank\tresult\trun_dir\n" > "$summary_file"

run_case() {
  local op="$1"
  local kill_rank="$2"
  local root_rank="${3:-}"
  local case_name="${op}_kill${kill_rank}"
  local root_arg=""
  local run_dir
  local sync_dir
  local id_file
  local cmd
  local launcher_pid
  local survivors=$((np - 1))
  local deadline
  local forced_cleanup=0
  local timeout_cleanup=0
  local launcher_status
  local parent_setup_lines
  local prepare_lines
  local activate_lines
  local kill_lines
  local survivor_done_lines
  local survivor_pass_lines
  local result_pass_lines
  local failed_path_lines
  local result_line
  local verdict="FAIL"
  local normalized_exit=1

  if [[ -n "$root_rank" ]]; then
    case_name="${case_name}_root${root_rank}"
    root_arg="--root-rank ${root_rank}"
  fi
  total=$((total + 1))
  run_dir="${matrix_dir}/$(printf '%03d' "$total")_${case_name}"
  sync_dir="${run_dir}/sync"
  id_file="${run_dir}/nccl_unique_id.bin"
  mkdir -p "$run_dir" "$sync_dir"

  cmd="mpirun --allow-run-as-root --mca orte_abort_on_non_zero_status 0 --mca mpi_abort_on_nonzero_status 0 -np ${np} bash -lc 'rank=\${OMPI_COMM_WORLD_RANK}; export NCCL_DEBUG=${NCCL_DEBUG}; export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS}; export NCCL_ALGO=${NCCL_ALGO}; export NCCL_PROTO=${NCCL_PROTO}; export NCCL_COLLNET_ENABLE=${NCCL_COLLNET_ENABLE}; export NCCL_DEBUG_FILE=${run_dir}/nccl.rank\${rank}.%h.%p.log; exec \"${bin}\" --collective ${op} --rank \${rank} --nranks ${np} --kill-rank ${kill_rank} ${root_arg} --count ${count} --iters ${iters} --timeout-sec ${timeout_sec} --id-file \"${id_file}\" --sync-dir \"${sync_dir}\"'"
  echo "$cmd" > "${run_dir}/command.txt"
  echo "[matrix] start ${case_name}"
  setsid bash -lc "$cmd" > "${run_dir}/stdout.log" 2>&1 &
  launcher_pid=$!
  deadline=$((SECONDS + job_timeout))

  while [[ "$SECONDS" -lt "$deadline" ]]; do
    survivor_pass_lines="$(awk '/^SURVIVOR_NATIVE_DONE .*local_result=0/ {n++} END {print n+0}' "${run_dir}/stdout.log" 2>/dev/null || echo 0)"
    result_pass_lines="$(awk '/^RESULT_FRAGMENT .*result=LOCAL_PASS/ {n++} END {print n+0}' "${run_dir}/stdout.log" 2>/dev/null || echo 0)"
    if [[ "$survivor_pass_lines" == "$survivors" && "$result_pass_lines" -ge 1 ]]; then
      forced_cleanup=1
      kill -TERM "-${launcher_pid}" 2>/dev/null || true
      sleep 2
      kill -KILL "-${launcher_pid}" 2>/dev/null || true
      break
    fi
    if ! kill -0 "$launcher_pid" 2>/dev/null; then
      break
    fi
    sleep 1
  done

  if [[ "$forced_cleanup" == "0" ]] && kill -0 "$launcher_pid" 2>/dev/null; then
    timeout_cleanup=1
    kill -TERM "-${launcher_pid}" 2>/dev/null || true
    sleep 2
    kill -KILL "-${launcher_pid}" 2>/dev/null || true
  fi

  wait "$launcher_pid"
  launcher_status=$?
  parent_setup_lines="$(awk '/^PARENT_SETUP / {n++} END {print n+0}' "${run_dir}/stdout.log")"
  prepare_lines="$(awk '/^SURVIVOR_NATIVE_PREPARE / {n++} END {print n+0}' "${run_dir}/stdout.log")"
  activate_lines="$(awk '/^SURVIVOR_NATIVE_ACTIVATE / {n++} END {print n+0}' "${run_dir}/stdout.log")"
  kill_lines="$(awk '/^FAILED_RANK_SIGKILL / {n++} END {print n+0}' "${run_dir}/stdout.log")"
  survivor_done_lines="$(awk '/^SURVIVOR_NATIVE_DONE / {n++} END {print n+0}' "${run_dir}/stdout.log")"
  survivor_pass_lines="$(awk '/^SURVIVOR_NATIVE_DONE .*local_result=0/ {n++} END {print n+0}' "${run_dir}/stdout.log")"
  failed_path_lines="$(awk -v k="${kill_rank}" '
    /^SURVIVOR_NATIVE_PREPARE / || /^SURVIVOR_NATIVE_ACTIVATE / || /^SURVIVOR_NATIVE_DONE / {
      for (i = 1; i <= NF; i++) {
        if ($i == "prev=" k || $i == "next=" k) n++
      }
    }
    END {print n+0}
  ' "${run_dir}/stdout.log")"
  result_line="$(grep -E '^RESULT_FRAGMENT .*result=' "${run_dir}/stdout.log" | tail -1 || true)"

  if [[ "$forced_cleanup" == "1" && "$parent_setup_lines" == "$np" && \
        "$prepare_lines" == "$survivors" && "$kill_lines" == "1" && \
        "$activate_lines" == "$survivors" && \
        "$survivor_done_lines" == "$survivors" && \
        "$survivor_pass_lines" == "$survivors" && "$failed_path_lines" == "0" && \
        "$result_line" == *"result=LOCAL_PASS"* ]]; then
    verdict="PASS_FORCE_CLEANUP"
    normalized_exit=0
    passed=$((passed + 1))
  else
    failed=$((failed + 1))
  fi

  {
    echo "# Survivor Parent Native Collective After Kill"
    echo
    echo "- algorithm: ${NCCL_ALGO}"
    echo "- collective: ${op}"
    echo "- kill parent rank: ${kill_rank}"
    echo "- root parent rank: ${root_rank:-N/A}"
    echo "- verdict: ${verdict}"
    echo "- normalized exit code: ${normalized_exit}"
    echo "- launcher exit code after cleanup: ${launcher_status}"
    echo "- forced cleanup used: ${forced_cleanup}"
    echo "- timeout cleanup used: ${timeout_cleanup}"
    echo "- parent setup lines: ${parent_setup_lines}/${np}"
    echo "- prepare/activate lines: ${prepare_lines}/${survivors}, ${activate_lines}/${survivors}"
    echo "- survivor done/pass lines: ${survivor_done_lines}/${survivors}, ${survivor_pass_lines}/${survivors}"
    echo "- result line: ${result_line:-<missing>}"
  } > "${run_dir}/result.md"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$total" "$NCCL_ALGO" "$op" "$kill_rank" \
    "${root_rank:-N/A}" "$verdict" "$run_dir" >> "$summary_file"
  echo "[matrix] done ${case_name}: ${verdict}"
}

run_root_cases() {
  local op="$1"
  local kill_rank="$2"
  local root_rank
  local -a root_list
  if [[ -n "$roots" ]]; then
    IFS=',' read -r -a root_list <<< "$roots"
  else
    for ((root_rank = 0; root_rank < np; ++root_rank)); do
      root_list+=("$root_rank")
    done
  fi
  for root_rank in "${root_list[@]}"; do
    if [[ "$root_rank" == "$kill_rank" ]]; then
      continue
    fi
    run_case "$op" "$kill_rank" "$root_rank"
  done
}

for kill_rank in "${kill_rank_list[@]}"; do
  [[ "$kill_rank" =~ ^[0-9]+$ && "$kill_rank" -lt "$np" ]] ||
    die "invalid failed rank: ${kill_rank}"
  if [[ "$collective" == "all" || "$collective" == "allreduce" ]]; then
    run_case allreduce "$kill_rank"
  fi
  if [[ "$collective" == "all" || "$collective" == "reducescatter" ]]; then
    run_case reducescatter "$kill_rank"
  fi
  if [[ "$collective" == "all" || "$collective" == "allgather" ]]; then
    run_case allgather "$kill_rank"
  fi
  if [[ "$collective" == "all" || "$collective" == "broadcast" ]]; then
    run_root_cases broadcast "$kill_rank"
  fi
  if [[ "$collective" == "all" || "$collective" == "reduce" ]]; then
    run_root_cases reduce "$kill_rank"
  fi
done

echo "matrix dir: ${matrix_dir}"
echo "summary: total=${total} passed=${passed} failed=${failed}"
echo "summary file: ${summary_file}"
[[ "$failed" == "0" ]]
