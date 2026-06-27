#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MEGATRON_DIR = Path("/public/home/scnethpc26107/sdr/Megatron-LM")
DEFAULT_LAUNCH_DIR = DEFAULT_MEGATRON_DIR / "examples/llama"
DEFAULT_FTCCL_DIR = Path("/public/home/scnethpc26107/sdr/ftccl/ftccl_midware_dcu")
DEFAULT_NCCL_ENV = Path("/public/home/scnethpc26107/sdr/env.sh")
DEFAULT_DATA_PATH = Path("/public/home/scnethpc26107/sdr/requirements/wikitext_llama3_text_document")
DEFAULT_TOKENIZER_PATH = Path("/public/home/scnethpc26107/sdr/requirements/llama3.1-8B")
DEFAULT_CONDA_BIN = Path("/public/home/scnethpc26107/.conda/envs/ftccl/bin")
DEFAULT_CONDA_LIB = Path("/public/home/scnethpc26107/.conda/envs/ftccl/lib")
DEFAULT_DTK_GCVM_LIB = Path("/public/software/compiler/dtk-26.04/dcc/gcvm/lib")
LOG_LOCK = threading.Lock()


@dataclass(frozen=True)
class RankSpec:
    rank: int
    local_rank: int
    node_rank: int
    host: str


def q(value: object) -> str:
    return shlex.quote(str(value))


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{socket.gethostname()}.{os.getpid()}")
    with tmp.open("w") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    try:
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass


def append_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def parse_hostfile(path: Path, node_num: int) -> tuple[list[str], int]:
    hosts: list[str] = []
    slots_per_node: int | None = None
    with path.open() as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            host = fields[0]
            slots = 8
            for field in fields[1:]:
                if field.startswith("slots="):
                    slots = int(field.split("=", 1)[1])
            if slots <= 0:
                raise ValueError(f"invalid slots in hostfile line: {line}")
            if slots_per_node is None:
                slots_per_node = slots
            elif slots != slots_per_node:
                raise ValueError("all selected hostfile entries must use the same slots value")
            hosts.append(host)
            if len(hosts) >= node_num:
                break
    if len(hosts) != node_num:
        raise ValueError(f"{path} provides {len(hosts)} usable hosts, expected {node_num}")
    return hosts, slots_per_node or 8


def build_rank_specs(hosts: list[str], slots_per_node: int) -> list[RankSpec]:
    specs: list[RankSpec] = []
    for node_rank, host in enumerate(hosts):
        for local_rank in range(slots_per_node):
            specs.append(
                RankSpec(
                    rank=len(specs),
                    local_rank=local_rank,
                    node_rank=node_rank,
                    host=host,
                )
            )
    return specs


def wait_for_all(markers: list[Path], timeout_sec: int) -> None:
    deadline = time.monotonic() + timeout_sec
    remaining = set(markers)
    while remaining and time.monotonic() < deadline:
        for path in list(remaining):
            if path.exists():
                remaining.remove(path)
        time.sleep(0.2)
    if remaining:
        missing = ", ".join(path.name for path in sorted(remaining))
        raise TimeoutError(f"timeout waiting for markers: {missing}")


def marker_ranks(sync_dir: Path, prefix: str, generation: int | None = None, iteration: int | None = None) -> set[int]:
    observed: set[int] = set()
    if generation is None or iteration is None:
        head = f"{prefix}."
        tail = ""
    else:
        head = f"{prefix}."
        tail = f".{generation}.{iteration}"
    try:
        for path in sync_dir.iterdir():
            name = path.name
            if not name.startswith(head):
                continue
            if tail and not name.endswith(tail):
                continue
            rank_text = name[len(head) : len(name) - len(tail) if tail else len(name)]
            if rank_text.isdigit():
                observed.add(int(rank_text))
    except FileNotFoundError:
        pass
    return observed


def expected_markers_present(
    sync_dir: Path,
    prefix: str,
    expected: set[int],
    generation: int | None = None,
    iteration: int | None = None,
) -> tuple[bool, set[int]]:
    observed = marker_ranks(sync_dir, prefix, generation, iteration)
    if expected.issubset(observed):
        return True, observed

    ok_by_stat = True
    for rank in expected:
        if generation is None or iteration is None:
            path = sync_dir / f"{prefix}.{rank}"
        else:
            path = sync_dir / f"{prefix}.{rank}.{generation}.{iteration}"
        if path.exists():
            observed.add(rank)
        else:
            ok_by_stat = False
    return ok_by_stat, observed


def release_all_once(
    sync_dir: Path,
    survivors: list[int],
    generation: int,
    iteration: int,
    supervisor_log: Path,
    released: set[str],
) -> None:
    expected = set(survivors)

    def release_once(key: str, path: Path, observed: set[int]) -> None:
        if key in released:
            return
        if path.exists():
            released.add(key)
            return
        if not expected.issubset(observed):
            return
        write_atomic(path, f"generation={generation} iteration={iteration} survivors={sorted(expected)}\n")
        released.add(key)
        append_log(
            supervisor_log,
            f"DP_FTCCL_RELEASED phase={key} generation={generation} "
            f"iteration={iteration} observed={sorted(observed)} release={path}\n",
        )

    rollback_ready, rollback_observed = expected_markers_present(
        sync_dir, "rollback_ready", expected, generation, iteration
    )
    prepare_ready, prepare_observed = expected_markers_present(sync_dir, "prepare", expected)
    activate_ready, activate_observed = expected_markers_present(sync_dir, "activate", expected)

    release_once(
        "rollback",
        sync_dir / f"rollback_release.{generation}.{iteration}",
        rollback_observed if rollback_ready else set(),
    )
    release_once(
        "prepare",
        sync_dir / "prepare_release",
        prepare_observed if prepare_ready else set(),
    )
    release_once(
        "activate",
        sync_dir / "activate_release",
        activate_observed if activate_ready else set(),
    )


def release_aggregator_loop(
    sync_dir: Path,
    survivors: list[int],
    generation: int,
    iteration: int,
    supervisor_log: Path,
    timeout_sec: int,
) -> int:
    released: set[str] = set()
    append_log(
        supervisor_log,
        "DP_FTCCL_RELEASE_AGGREGATOR_STARTED "
        f"pid={os.getpid()} generation={generation} iteration={iteration} "
        f"survivors={sorted(survivors)} sync_dir={sync_dir}\n",
    )
    deadline = time.monotonic() + timeout_sec
    try:
        while time.monotonic() < deadline:
            release_all_once(sync_dir, survivors, generation, iteration, supervisor_log, released)
            if {"rollback", "prepare", "activate"}.issubset(released):
                return 0
            time.sleep(0.05)
        append_log(
            supervisor_log,
            "DP_FTCCL_RELEASE_AGGREGATOR_TIMEOUT "
            f"released={sorted(released)} generation={generation} iteration={iteration}\n",
        )
        return 1
    except Exception as exc:
        append_log(
            supervisor_log,
            f"DP_FTCCL_RELEASE_AGGREGATOR_FAILED {type(exc).__name__}: {exc}\n",
        )
        return 1
    finally:
        append_log(
            supervisor_log,
            "DP_FTCCL_RELEASE_AGGREGATOR_STOPPED "
            f"released={sorted(released)} generation={generation} iteration={iteration}\n",
        )


def start_release_aggregator_process(
    sync_dir: Path,
    survivors: list[int],
    generation: int,
    iteration: int,
    supervisor_log: Path,
    timeout_sec: int,
) -> subprocess.Popen:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "__release_aggregator",
        "--sync-dir",
        str(sync_dir),
        "--survivors",
        ",".join(str(rank) for rank in survivors),
        "--generation",
        str(generation),
        "--iteration",
        str(iteration),
        "--supervisor-log",
        str(supervisor_log),
        "--timeout-sec",
        str(timeout_sec),
    ]
    return subprocess.Popen(cmd, start_new_session=True, close_fds=True)


def release_process_is_active(proc: subprocess.Popen | None) -> bool:
    return proc is not None and proc.poll() is None


def read_pid_marker(path: Path) -> tuple[str, int]:
    fields: dict[str, str] = {}
    for item in path.read_text().split():
        if "=" in item:
            key, value = item.split("=", 1)
            fields[key] = value
    host = fields.get("host")
    pid_text = fields.get("pid")
    if not host or not pid_text:
        raise ValueError(f"invalid pid marker {path}: {path.read_text()!r}")
    return host, int(pid_text)


def local_hostnames() -> set[str]:
    names = {socket.gethostname(), socket.gethostname().split(".", 1)[0]}
    try:
        names.add(socket.getfqdn())
    except Exception:
        pass
    return {name for name in names if name}


def kill_remote_process(host: str, pid: int) -> None:
    if host in local_hostnames():
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return
    subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=no",
            host,
            f"kill -9 {pid}",
        ],
        check=False,
    )


def terminate_launcher(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    time.sleep(0.5)
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def kill_rank_markers(sync_dir: Path) -> None:
    for marker in sorted(sync_dir.glob("pid.*")):
        try:
            host, pid = read_pid_marker(marker)
            kill_remote_process(host, pid)
        except Exception:
            pass


def build_remote_command(args: argparse.Namespace, spec: RankSpec, world_size: int) -> str:
    llama_dir = args.megatron_dir / "examples/llama"
    launch_with_binding = llama_dir / "launch_with_binding.sh"
    train_script = llama_dir / "train_dp_ftccl.sh"
    checkpoint_path = args.run_root / "ckpt"
    log_dir = args.run_root / "logs"
    sync_dir = args.run_root / "sync"
    lib_path = f"{DEFAULT_CONDA_LIB}:{DEFAULT_DTK_GCVM_LIB}:${{LD_LIBRARY_PATH:-}}"
    pythonpath = f"{args.midware_dir / 'python'}:{args.megatron_dir}:${{PYTHONPATH:-}}"
    extra = " ".join(q(item) for item in args.extra_train_args)
    env_exports = {
        "RANK": spec.rank,
        "LOCAL_RANK": spec.local_rank,
        "NODE_RANK": spec.node_rank,
        "WORLD_SIZE": world_size,
        "OMPI_COMM_WORLD_RANK": spec.rank,
        "OMPI_COMM_WORLD_LOCAL_RANK": spec.local_rank,
        "OMPI_COMM_WORLD_SIZE": world_size,
        "MASTER_ADDR": args.master_addr,
        "MASTER_PORT": args.master_port,
        "FTCCL_SUPERVISOR_LAUNCH": 1,
        "FTCCL_SUPERVISOR_RELEASE_BARRIER": 1 if args.supervisor_release_barrier else 0,
        "FTCCL_BYPASS_DIR": sync_dir,
        "FTCCL_LOG_DIR": log_dir,
        "FTCCL_BYPASS_WORLD_SIZE": world_size,
        "FTCCL_BYPASS_DEAD_RANK": args.kill_rank,
        "FTCCL_BYPASS_TRIGGER_AT_START": 0,
        "FTCCL_BYPASS_TRIGGER_DELAY_SEC": 0,
        "FTCCL_BYPASS_MIN_COMM_ORDINAL": args.min_comm_ordinal,
        "FTCCL_BYPASS_TIMEOUT_SEC": args.bypass_timeout_sec,
        "FTCCL_BYPASS_VICTIM_MODE": "noop",
        "FTCCL_BYPASS_PROPAGATE_ERROR_BEFORE_BYPASS": 0 if args.no_error_propagation else 1,
        "FTCCL_BYPASS_ABORT_ON_PROPAGATED_ERROR": 0,
        "FTCCL_DETECT_ENABLE": 0,
        "FTCCL_DETECT_CCLD_ENABLE": 0,
        "FTCCL_DETECT_MARK_KILLED": 1,
        "FTCCL_DETECT_PROPAGATE_ERROR": 0 if args.no_error_propagation else 1,
        "FTCCL_DETECT_RUN_ID": args.run_name,
        "FTCCL_PATCH_ENABLE": 1,
        "FTCCL_PATCH_PAUSE_AFTER_ITER": args.pause_iter,
        "FTCCL_PATCH_GLOBAL_BATCH_SIZE_AFTER_BYPASS": args.after_bypass_global_batch_size,
        "FTCCL_PATCH_VICTIM_DATA_MODE": "keep",
        "FTCCL_PATCH_DISABLE_EVAL_AFTER_BYPASS": 1,
        "FTCCL_PATCH_DISABLE_GLOO_PROCESS_GROUPS": 1 if args.node_num > 1 else 0,
        "FTCCL_PATCH_LOG_RANK_AFTER_BYPASS": "survivor_last",
        "FTCCL_PATCH_GRAD_COMPENSATION": args.grad_compensation,
        "FTCCL_PATCH_ROLLBACK_ENABLE": 0 if args.no_error_propagation else 1,
        "FTCCL_PATCH_ROLLBACK_MAX_RETRIES": args.rollback_max_retries,
        "FTCCL_PATCH_ROLLBACK_TIMEOUT_SEC": args.bypass_timeout_sec,
        "NCCL_ALGO": "RING",
        "NCCL_PROTO": "SIMPLE",
        "NCCL_COLLNET_ENABLE": 0,
        "NCCL_NVLS_ENABLE": 0,
        "NCCL_CCLD_ENABLE": 1,
        "FTCCL_LOG_LEVEL": args.ftccl_log_level,
        "FTCCL_PATCH_LOG_LEVEL": args.patch_log_level,
        "NCCL_DEBUG": args.nccl_debug,
        "TORCHDYNAMO_DISABLE": 1,
    }
    export_lines = "\n".join(f"export {key}={q(value)}" for key, value in env_exports.items())
    return f"""set -euo pipefail
cd {q(llama_dir)}
eval "$($(command -v conda) shell.bash hook)"
source {q(args.nccl_env)}
export PATH={q(DEFAULT_CONDA_BIN)}:${{PATH}}
export LD_LIBRARY_PATH={lib_path}
export PYTHONPATH={pythonpath}
source {q(args.midware_dir / "scripts/megatron_env.sh")}
export LD_LIBRARY_PATH={q(DEFAULT_CONDA_LIB)}:{q(DEFAULT_DTK_GCVM_LIB)}:${{LD_LIBRARY_PATH:-}}
{export_lines}
bash {q(train_script)} {q(args.master_addr)} {q(args.master_port)} \\
  --data_path={q(args.data_path)} \\
  --tokenizer_path={q(args.tokenizer_path)} \\
  --checkpoint_path={q(checkpoint_path)} \\
  --launch_with_binding={q(launch_with_binding)} \\
  --train-iters={q(args.train_iters)} \\
  --global-batch-size={q(args.global_batch_size)} \\
  --micro-batch-size={q(args.micro_batch_size)} \\
  {extra}
"""


def launch_rank(args: argparse.Namespace, spec: RankSpec, world_size: int) -> subprocess.Popen:
    combined_log_path = args.run_root / "logs" / "combined.live.log"
    remote_cmd = build_remote_command(args, spec, world_size)
    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=no",
        spec.host,
        "bash -lc " + q(remote_cmd),
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        close_fds=True,
    )
    prefix = f"rank{spec.rank}"

    def _pump() -> None:
        assert proc.stdout is not None
        with combined_log_path.open("ab") as combined:
            for raw in proc.stdout:
                line = f"[{prefix}] ".encode() + raw
                with LOG_LOCK:
                    combined.write(line)
                    combined.flush()

    thread = threading.Thread(target=_pump, name=f"ftccl-log-rank{spec.rank}", daemon=True)
    thread.start()
    return proc


def publish_supervisor_request(args: argparse.Namespace, world_size: int, supervisor_log: Path) -> None:
    cmd = [
        sys.executable,
        str(args.midware_dir / "scripts/ftccl_bypassctl.py"),
        "--sync-dir",
        str(args.run_root / "sync"),
        "request",
        "--rank",
        str(args.kill_rank),
        "--world-size",
        str(world_size),
        "--generation",
        "1",
        "--source",
        "supervisor",
        "--reason",
        "planned-rank-kill",
        "--mark-killed",
    ]
    if not args.no_error_propagation:
        cmd.append("--propagate-error")
    with supervisor_log.open("ab") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)


def combine_logs(log_dir: Path, ranks: list[int]) -> Path:
    combined = log_dir / "combined.log"
    live = log_dir / "combined.live.log"
    if live.exists():
        combined.write_text(live.read_text(errors="replace"))
    else:
        combined.write_text("")
    return combined


def verify_success(args: argparse.Namespace, specs: list[RankSpec], exit_codes: dict[int, int | None]) -> None:
    survivors = [spec.rank for spec in specs if spec.rank != args.kill_rank]
    bad_survivors = {rank: exit_codes.get(rank) for rank in survivors if exit_codes.get(rank) != 0}
    if bad_survivors:
        raise RuntimeError(f"survivor process failures: {bad_survivors}")
    victim_code = exit_codes.get(args.kill_rank)
    if victim_code == 0:
        raise RuntimeError("victim rank exited cleanly; expected SIGKILL/nonzero exit")

    sync_dir = args.run_root / "sync"
    log_dir = args.run_root / "logs"
    combined = combine_logs(log_dir, [spec.rank for spec in specs])
    text = combined.read_text(errors="replace")
    final_iteration_pattern = re.compile(
        rf"iteration\s+{re.escape(str(args.train_iters))}/\s*{re.escape(str(args.train_iters))}"
    )
    required = [
        f"continue_after_kill received iteration={args.pause_iter}",
        "bypass complete:",
        "survivorCount=",
        "activated survivor training view generation=1",
        "[after training is done]",
    ]
    if not args.no_error_propagation:
        required.extend(
            [
                "propagating recoverable ncclSystemError",
                "caught recoverable communication error",
                "rollback complete; retrying train_step",
            ]
        )
    missing = [pattern for pattern in required if pattern not in text]
    if final_iteration_pattern.search(text) is None:
        missing.append(f"iteration {args.train_iters}/{args.train_iters}")
    if args.supervisor_release_barrier:
        required_release_files = [
            sync_dir / "prepare_release",
            sync_dir / "activate_release",
        ]
        if not args.no_error_propagation:
            required_release_files.append(sync_dir / f"rollback_release.1.{args.pause_iter}")
        missing.extend(str(path) for path in required_release_files if not path.exists())
    last_iter = args.train_iters - 1
    missing.extend(
        str(sync_dir / f"train_step_done.{rank}.{last_iter}")
        for rank in survivors
        if not (sync_dir / f"train_step_done.{rank}.{last_iter}").exists()
    )
    if not args.no_error_propagation:
        missing.extend(
            str(sync_dir / f"rollback_replay.{rank}.1.{args.pause_iter}")
            for rank in survivors
            if not (sync_dir / f"rollback_replay.{rank}.1.{args.pause_iter}").exists()
        )
    if missing:
        raise RuntimeError(f"missing expected evidence: {missing}; combined={combined}")


def parse_args() -> argparse.Namespace:
    if len(sys.argv) > 1 and sys.argv[1] == "__release_aggregator":
        parser = argparse.ArgumentParser(description="Internal FTCCL release aggregator.")
        parser.add_argument("__release_aggregator")
        parser.add_argument("--sync-dir", type=Path, required=True)
        parser.add_argument("--survivors", required=True)
        parser.add_argument("--generation", type=int, required=True)
        parser.add_argument("--iteration", type=int, required=True)
        parser.add_argument("--supervisor-log", type=Path, required=True)
        parser.add_argument("--timeout-sec", type=int, required=True)
        args = parser.parse_args()
        args.survivors = [int(item) for item in args.survivors.split(",") if item != ""]
        return args

    parser = argparse.ArgumentParser(
        description="Run the DCU Megatron FTCCL real-rank-kill supervisor demo."
    )
    parser.add_argument("--hostfile", type=Path, required=True)
    parser.add_argument("--node-num", type=int, required=True)
    parser.add_argument("--kill-rank", type=int, default=None)
    parser.add_argument("--pause-iter", type=int, default=1)
    parser.add_argument("--train-iters", type=int, default=3)
    parser.add_argument("--master-port", default=os.environ.get("FTCCL_MASTER_PORT", "25910"))
    parser.add_argument("--run-name", default=f"dp_ftccl_{time.strftime('%Y%m%d-%H%M%S')}")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--timeout-sec", type=int, default=900)
    parser.add_argument("--bypass-timeout-sec", type=int, default=300)
    parser.add_argument(
        "--no-error-propagation",
        dest="no_error_propagation",
        action="store_true",
        default=True,
        help="Use the fast C-layer survivor bypass path without forcing a Python rollback.",
    )
    parser.add_argument(
        "--error-propagation",
        dest="no_error_propagation",
        action="store_false",
        help="Force the recoverable-error Python rollback path for debugging.",
    )
    parser.add_argument("--min-comm-ordinal", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--after-bypass-global-batch-size", type=int)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--grad-compensation", default=os.environ.get("FTCCL_PATCH_GRAD_COMPENSATION", "survivor_global_batch"))
    parser.add_argument("--rollback-max-retries", type=int, default=1)
    parser.add_argument("--ftccl-log-level", default=os.environ.get("FTCCL_LOG_LEVEL", "1"))
    parser.add_argument("--patch-log-level", default=os.environ.get("FTCCL_PATCH_LOG_LEVEL", "1"))
    parser.add_argument("--nccl-debug", default=os.environ.get("NCCL_DEBUG", "WARN"))
    parser.add_argument("--megatron-dir", type=Path, default=DEFAULT_MEGATRON_DIR)
    parser.add_argument("--midware-dir", type=Path, default=DEFAULT_FTCCL_DIR)
    parser.add_argument("--nccl-env", type=Path, default=DEFAULT_NCCL_ENV)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    args, extra_train_args = parser.parse_known_args()
    args.extra_train_args = extra_train_args
    if args.extra_train_args and args.extra_train_args[0] == "--":
        args.extra_train_args = args.extra_train_args[1:]
    if args.node_num <= 0:
        raise SystemExit("--node-num must be positive")
    args.hostfile = args.hostfile.resolve()
    args.megatron_dir = args.megatron_dir.resolve()
    args.midware_dir = args.midware_dir.resolve()
    if args.run_root is None:
        args.run_root = args.megatron_dir / "examples/llama/runs" / args.run_name
    else:
        args.run_root = args.run_root.resolve()
    return args


def main() -> int:
    args = parse_args()
    if getattr(args, "__release_aggregator", None) == "__release_aggregator":
        return release_aggregator_loop(
            args.sync_dir,
            args.survivors,
            args.generation,
            args.iteration,
            args.supervisor_log,
            args.timeout_sec,
        )

    hosts, slots_per_node = parse_hostfile(args.hostfile, args.node_num)
    specs = build_rank_specs(hosts, slots_per_node)
    world_size = len(specs)
    if args.kill_rank is None:
        args.kill_rank = world_size - 1
    if args.kill_rank < 0 or args.kill_rank >= world_size:
        raise SystemExit(f"invalid kill rank {args.kill_rank} for world size {world_size}")
    args.master_addr = hosts[0]
    args.supervisor_release_barrier = False
    if args.after_bypass_global_batch_size is None:
        survivor_count = world_size - 1
        args.after_bypass_global_batch_size = (
            args.global_batch_size // survivor_count
        ) * survivor_count
        if args.after_bypass_global_batch_size <= 0:
            args.after_bypass_global_batch_size = survivor_count

    log_dir = args.run_root / "logs"
    sync_dir = args.run_root / "sync"
    log_dir.mkdir(parents=True, exist_ok=True)
    sync_dir.mkdir(parents=True, exist_ok=True)
    hostfile_slots = args.run_root / "hostfile_slots"
    hostfile_slots.write_text("".join(f"{host} slots={slots_per_node}\n" for host in hosts))
    supervisor_log = log_dir / "supervisor.log"
    supervisor_log.write_text(
        "DP_FTCCL_SUPERVISOR "
        f"run_root={args.run_root} hostfile={hostfile_slots} np={world_size} "
        f"map_by=ppr:{slots_per_node}:node kill_rank={args.kill_rank} "
        f"pause_iter={args.pause_iter} train_iters={args.train_iters} "
        f"bypass_timeout_sec={args.bypass_timeout_sec} "
        f"supervisor_release_barrier={int(args.supervisor_release_barrier)} "
        f"error_propagation={int(not args.no_error_propagation)}\n"
    )
    print(supervisor_log.read_text().strip(), flush=True)

    procs: dict[int, subprocess.Popen] = {}
    success = False
    release_proc: subprocess.Popen | None = None
    supervisor_released: set[str] = set()
    try:
        for spec in specs:
            procs[spec.rank] = launch_rank(args, spec, world_size)
            append_log(
                supervisor_log,
                f"DP_FTCCL_LAUNCHED rank={spec.rank} host={spec.host} "
                f"local_rank={spec.local_rank} launcher_pid={procs[spec.rank].pid}\n",
            )

        pause_markers = [sync_dir / f"pause_ready.{spec.rank}.{args.pause_iter}" for spec in specs]
        wait_for_all(pause_markers, args.timeout_sec)
        append_log(supervisor_log, f"DP_FTCCL_ALL_PAUSED iteration={args.pause_iter}\n")

        victim_marker = sync_dir / f"pid.{args.kill_rank}"
        wait_for_all([victim_marker], args.timeout_sec)
        victim_host, victim_pid = read_pid_marker(victim_marker)
        survivors = [spec.rank for spec in specs if spec.rank != args.kill_rank]
        if args.supervisor_release_barrier:
            release_proc = start_release_aggregator_process(
                sync_dir,
                survivors,
                1,
                args.pause_iter,
                supervisor_log,
                args.timeout_sec,
            )
        kill_remote_process(victim_host, victim_pid)
        append_log(
            supervisor_log,
            f"DP_FTCCL_KILLED rank={args.kill_rank} host={victim_host} pid={victim_pid}\n",
        )
        print(f"DP_FTCCL_KILLED rank={args.kill_rank} host={victim_host} pid={victim_pid}", flush=True)

        publish_supervisor_request(args, world_size, supervisor_log)
        write_atomic(sync_dir / f"continue_after_kill.{args.pause_iter}", "1\n")

        deadline = time.monotonic() + args.timeout_sec
        exit_codes: dict[int, int | None] = {}
        try:
            while time.monotonic() < deadline:
                if args.supervisor_release_barrier:
                    release_all_once(
                        sync_dir,
                        survivors,
                        1,
                        args.pause_iter,
                        supervisor_log,
                        supervisor_released,
                    )
                exit_codes = {rank: proc.poll() for rank, proc in procs.items()}
                if all(exit_codes.get(rank) is not None for rank in survivors):
                    break
                time.sleep(1)
            else:
                raise TimeoutError(f"timeout waiting for survivors to finish: {exit_codes}")
        finally:
            if release_proc is not None and release_proc.poll() is None:
                release_proc.terminate()
                try:
                    release_proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    release_proc.kill()

        for rank, proc in procs.items():
            if rank == args.kill_rank and proc.poll() is None:
                terminate_launcher(proc)
        exit_codes = {rank: proc.poll() for rank, proc in procs.items()}
        for rank, proc in procs.items():
            if exit_codes[rank] is None:
                try:
                    exit_codes[rank] = proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    terminate_launcher(proc)
                    exit_codes[rank] = proc.poll()
        verify_success(args, specs, exit_codes)
        combine_logs(log_dir, [spec.rank for spec in specs])
        message = (
            f"FTCCL_REAL_KILL_DEMO_PASS killed_rank={args.kill_rank} "
            f"victim_exit={exit_codes.get(args.kill_rank)} run_root={args.run_root}"
        )
        append_log(supervisor_log, message + "\n")
        print(message, flush=True)
        success = True
        return 0
    except Exception as exc:
        append_log(supervisor_log, f"DP_FTCCL_FAILED {type(exc).__name__}: {exc}\n")
        print(f"DP_FTCCL_FAILED {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        if release_proc is not None and release_proc.poll() is None:
            release_proc.terminate()
            try:
                release_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                release_proc.kill()
        if not success and "sync_dir" in locals():
            kill_rank_markers(sync_dir)
        for proc in procs.values():
            terminate_launcher(proc)


if __name__ == "__main__":
    raise SystemExit(main())
