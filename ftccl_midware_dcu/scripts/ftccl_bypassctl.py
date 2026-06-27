#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(text)
    tmp.replace(path)


def read_int(path: Path, default: int) -> int:
    try:
        return int(path.read_text().strip())
    except Exception:
        return default


def parse_survivors(raw: str | None, world_size: int, failed_rank: int) -> list[int]:
    if raw:
        return [int(item) for item in raw.split(",") if item != ""]
    return [rank for rank in range(world_size) if rank != failed_rank]


def publish_request(args) -> int:
    sync_dir = Path(args.sync_dir)
    sync_dir.mkdir(parents=True, exist_ok=True)

    if args.rank < 0 or args.rank >= args.world_size:
        raise SystemExit(f"invalid failed rank {args.rank} for world size {args.world_size}")

    generation = args.generation
    if generation is None:
        generation = read_int(sync_dir / "generation", 0) + 1
    survivors = parse_survivors(args.survivors, args.world_size, args.rank)
    if args.rank in survivors:
        raise SystemExit(f"failed rank {args.rank} must not be in survivors={survivors}")

    request = {
        "schema": "ftccl.bypass.v1",
        "generation": generation,
        "failed_rank": args.rank,
        "world_size": args.world_size,
        "survivors": survivors,
        "source": args.source,
        "reason": args.reason,
        "created_at_unix": time.time(),
        "mark_killed": args.mark_killed,
        "propagate_error_before_bypass": args.propagate_error,
    }

    write_atomic(sync_dir / "bypass_request.json", json.dumps(request, indent=2, sort_keys=True) + "\n")
    write_atomic(sync_dir / "failed_rank", f"{args.rank}\n")
    write_atomic(sync_dir / "survivors", ",".join(str(rank) for rank in survivors) + "\n")
    write_atomic(sync_dir / "generation", f"{generation}\n")
    write_atomic(sync_dir / "trigger", "1\n")
    if args.mark_killed:
        write_atomic(sync_dir / f"killed.{args.rank}", f"{args.rank}\n")
    if args.propagate_error:
        write_atomic(sync_dir / "propagate_error", "1\n")

    print(
        "FTCCL_BYPASS_REQUEST "
        f"generation={generation} failed_rank={args.rank} survivors={survivors} "
        f"sync_dir={sync_dir} mark_killed={args.mark_killed}",
        flush=True,
    )
    return 0


def show_status(args) -> int:
    sync_dir = Path(args.sync_dir)
    names = [
        "bypass_request.json",
        "trigger",
        "failed_rank",
        "survivors",
        "generation",
        "propagate_error",
    ]
    for name in names:
        path = sync_dir / name
        if path.exists():
            print(f"== {path} ==")
            print(path.read_text().rstrip())

    markers = sorted(
        path.name
        for path in sync_dir.glob("*")
        if path.name.startswith(
            (
                "killed.",
                "prepare.",
                "activate.",
                "rollback_",
                "comm_error.",
                "detected_fault.",
                "probe.",
            )
        )
    )
    if markers:
        print("== markers ==")
        for marker in markers:
            print(marker)
    return 0


def clear_request(args) -> int:
    sync_dir = Path(args.sync_dir)
    for name in (
        "bypass_request.json",
        "trigger",
        "failed_rank",
        "survivors",
        "generation",
        "propagate_error",
    ):
        try:
            (sync_dir / name).unlink()
        except FileNotFoundError:
            pass
    for pattern in (
        "probe.*",
        "detected_fault.*",
        "killed.*",
        "prepare.*",
        "activate.*",
        "comm_error.*",
        "error_propagated.*",
    ):
        for path in sync_dir.glob(pattern):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
    print(f"FTCCL_BYPASS_REQUEST_CLEARED sync_dir={sync_dir}", flush=True)
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Publish or inspect FTCCl bypass requests.")
    parser.add_argument(
        "--sync-dir",
        default=os.environ.get("FTCCL_BYPASS_DIR", "/tmp/ftccl_bypass"),
        help="Shared bypass control directory.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    request = subparsers.add_parser("request", help="Publish a bypass request.")
    request.add_argument("--rank", type=int, required=True, help="Parent DP rank to bypass.")
    request.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", "8")))
    request.add_argument("--generation", type=int)
    request.add_argument("--survivors", help="Comma-separated survivor parent ranks.")
    request.add_argument("--source", default="manual", choices=["manual", "supervisor", "detector"])
    request.add_argument("--reason", default="operator-request")
    request.add_argument(
        "--mark-killed",
        action="store_true",
        help="Also publish killed.<rank>; use for passive detection after a rank is already dead.",
    )
    request.add_argument(
        "--propagate-error",
        action="store_true",
        help="Ask higher layers to exercise recoverable communication error/rollback path.",
    )
    request.set_defaults(func=publish_request)

    status = subparsers.add_parser("status", help="Print the current bypass request and markers.")
    status.set_defaults(func=show_status)

    clear = subparsers.add_parser("clear", help="Remove the current request files.")
    clear.set_defaults(func=clear_request)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
