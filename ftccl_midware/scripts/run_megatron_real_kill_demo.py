#!/usr/bin/env python3
import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


def wait_for(path: Path, timeout_sec: int) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.1)
    return False


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(text)
    tmp.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Megatron FTCCl real-rank-kill demo.")
    parser.add_argument("--np", type=int, default=8)
    parser.add_argument("--kill-rank", type=int, default=7)
    parser.add_argument("--pause-iter", type=int, default=1)
    parser.add_argument("--train-iters", type=int, default=3)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", default=os.environ.get("MASTER_PORT", "6017"))
    parser.add_argument("--timeout-sec", type=int, default=360)
    parser.add_argument("--sync-dir", default="/tmp/ftccl_bypass_real")
    parser.add_argument("--log-dir", default="/tmp/ftccl_real_kill_logs")
    parser.add_argument(
        "--no-error-propagation",
        action="store_true",
        help="Disable the demo path that returns one recoverable NCCL error before survivor bypass.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path("/home/ubuntu/sdr/Megatron-LM")
    train_script = repo / "examples/llama/train_llama3.2_1b.sh"
    ctl_script = Path(__file__).resolve().parent / "ftccl_bypassctl.py"
    sync_dir = Path(args.sync_dir)
    log_dir = Path(args.log_dir)

    if args.kill_rank < 0 or args.kill_rank >= args.np:
        raise SystemExit(f"invalid kill rank {args.kill_rank}")

    shutil.rmtree(sync_dir, ignore_errors=True)
    shutil.rmtree(log_dir, ignore_errors=True)
    sync_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    procs: dict[int, subprocess.Popen] = {}
    log_files = {}

    base_env = os.environ.copy()
    base_env.update(
        {
            "WORLD_SIZE": str(args.np),
            "MASTER_ADDR": args.master_addr,
            "MASTER_PORT": str(args.master_port),
            "TRAIN_ITERS": str(args.train_iters),
            "EVAL_ITERS": "0",
            "FTCCL_SUPERVISOR_LAUNCH": "1",
            "FTCCL_BYPASS_DIR": str(sync_dir),
            "FTCCL_BYPASS_WORLD_SIZE": str(args.np),
            "FTCCL_BYPASS_DEAD_RANK": str(args.kill_rank),
            "FTCCL_BYPASS_TRIGGER_AT_START": "0",
            "FTCCL_BYPASS_TRIGGER_DELAY_SEC": "0",
            "FTCCL_BYPASS_MIN_COMM_ORDINAL": "2",
            "FTCCL_BYPASS_VICTIM_MODE": "noop",
            "FTCCL_BYPASS_PROPAGATE_ERROR_BEFORE_BYPASS": "0" if args.no_error_propagation else "1",
            "FTCCL_BYPASS_ABORT_ON_PROPAGATED_ERROR": "0",
            # The real-kill demo has an explicit supervisor fault decision.
            # Keep detector demos separate so host preflight does not race the planned SIGKILL.
            "FTCCL_DETECT_ENABLE": "0",
            "FTCCL_DETECT_CCLD_ENABLE": "0",
            "FTCCL_PATCH_ENABLE": "1",
            "FTCCL_PATCH_PAUSE_AFTER_ITER": str(args.pause_iter),
            "FTCCL_PATCH_GLOBAL_BATCH_SIZE_AFTER_BYPASS": "126",
            "FTCCL_PATCH_VICTIM_DATA_MODE": "keep",
            "FTCCL_PATCH_GRAD_COMPENSATION": os.environ.get(
                "FTCCL_PATCH_GRAD_COMPENSATION", "survivor_global_batch"
            ),
            "FTCCL_PATCH_ROLLBACK_ENABLE": "1",
            "FTCCL_PATCH_ROLLBACK_MAX_RETRIES": "1",
            "FTCCL_LOG_LEVEL": os.environ.get("FTCCL_LOG_LEVEL", "1"),
            "FTCCL_PATCH_LOG_LEVEL": os.environ.get("FTCCL_PATCH_LOG_LEVEL", "1"),
            "NCCL_DEBUG": os.environ.get("NCCL_DEBUG", "WARN"),
        }
    )

    try:
        for rank in range(args.np):
            env = base_env.copy()
            env.update({"RANK": str(rank), "LOCAL_RANK": str(rank), "NODE_RANK": "0"})
            log_path = log_dir / f"rank{rank}.log"
            log_f = log_path.open("wb")
            log_files[rank] = log_f
            procs[rank] = subprocess.Popen(
                ["bash", str(train_script)],
                cwd=str(repo),
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
            )

        print(f"[supervisor] launched {args.np} ranks log_dir={log_dir} sync_dir={sync_dir}", flush=True)

        for rank in range(args.np):
            marker = sync_dir / f"pause_ready.{rank}.{args.pause_iter}"
            if not wait_for(marker, args.timeout_sec):
                raise TimeoutError(f"timeout waiting for {marker}")
            print(f"[supervisor] rank {rank} reached pause_iter={args.pause_iter}", flush=True)

        victim = procs[args.kill_rank]
        print(f"[supervisor] SIGKILL rank {args.kill_rank} pgid={os.getpgid(victim.pid)}", flush=True)
        os.killpg(os.getpgid(victim.pid), signal.SIGKILL)

        survivors = [rank for rank in range(args.np) if rank != args.kill_rank]
        ctl_cmd = [
            sys.executable,
            str(ctl_script),
            "--sync-dir",
            str(sync_dir),
            "request",
            "--rank",
            str(args.kill_rank),
            "--world-size",
            str(args.np),
            "--generation",
            "1",
            "--source",
            "supervisor",
            "--reason",
            "sigkill-demo",
            "--mark-killed",
        ]
        if not args.no_error_propagation:
            ctl_cmd.append("--propagate-error")
        subprocess.run(ctl_cmd, check=True)
        write_atomic(sync_dir / f"continue_after_kill.{args.pause_iter}", "1\n")
        print(f"[supervisor] published survivors={survivors}", flush=True)

        deadline = time.monotonic() + args.timeout_sec
        exit_codes: dict[int, int | None] = {}
        while time.monotonic() < deadline:
            exit_codes = {rank: proc.poll() for rank, proc in procs.items()}
            survivor_done = all(exit_codes[rank] is not None for rank in survivors)
            if survivor_done:
                break
            time.sleep(1)
        else:
            raise TimeoutError(f"timeout waiting for survivors: {exit_codes}")

        for rank, proc in procs.items():
            if proc.poll() is None:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                time.sleep(1)
                if proc.poll() is None:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)

        exit_codes = {rank: proc.wait() for rank, proc in procs.items()}
        for f in log_files.values():
            f.close()

        bad_survivors = {rank: code for rank, code in exit_codes.items() if rank in survivors and code != 0}
        victim_code = exit_codes.get(args.kill_rank)
        if bad_survivors:
            raise RuntimeError(f"survivor failures: {bad_survivors}; logs={log_dir}")
        if victim_code == 0:
            raise RuntimeError(f"victim rank exited cleanly unexpectedly; logs={log_dir}")

        combined = log_dir / "combined.log"
        with combined.open("w") as out:
            for rank in range(args.np):
                out.write(f"\n===== rank {rank} =====\n")
                out.write((log_dir / f"rank{rank}.log").read_text(errors="replace"))

        required = [
            f"continue_after_kill received iteration={args.pause_iter}",
            "bypass complete: survivorRank=0 survivorCount=7",
            "activated survivor training view generation=1",
            "dataloader rebuilt generation=1 dp_rank=0 dp_size=7",
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
        text = combined.read_text(errors="replace")
        missing = [pattern for pattern in required if pattern not in text]
        last_train_step = args.train_iters - 1
        missing.extend(
            str(sync_dir / f"train_step_done.{rank}.{last_train_step}")
            for rank in survivors
            if not (sync_dir / f"train_step_done.{rank}.{last_train_step}").exists()
        )
        if not args.no_error_propagation:
            missing.extend(
                str(sync_dir / f"rollback_replay.{rank}.1.{args.pause_iter}")
                for rank in survivors
                if not (sync_dir / f"rollback_replay.{rank}.1.{args.pause_iter}").exists()
            )
        if missing:
            raise RuntimeError(f"missing expected patterns {missing}; combined={combined}")

        print(
            f"FTCCL_REAL_KILL_DEMO_PASS killed_rank={args.kill_rank} "
            f"victim_exit={victim_code} log_dir={log_dir} combined={combined}",
            flush=True,
        )
        return 0
    finally:
        for f in log_files.values():
            try:
                f.close()
            except Exception:
                pass
        for proc in procs.values():
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass


if __name__ == "__main__":
    sys.exit(main())
