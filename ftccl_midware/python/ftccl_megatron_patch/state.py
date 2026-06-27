import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def log(message: str, level: int = 1) -> None:
    if _env_int("FTCCL_PATCH_LOG_LEVEL", _env_int("FTCCL_LOG_LEVEL", 1)) >= level:
        rank = os.environ.get("RANK", "?")
        print(f"[ftccl-python][rank {rank}] {message}", file=sys.stderr, flush=True)


@dataclass(frozen=True)
class BypassView:
    active: bool
    generation: int
    parent_world_size: int
    failed_rank: int
    global_rank: int
    survivor_ranks: tuple[int, ...]

    @property
    def survivor_count(self) -> int:
        return len(self.survivor_ranks)

    @property
    def is_victim(self) -> bool:
        return self.global_rank == self.failed_rank

    @property
    def survivor_rank(self) -> Optional[int]:
        if self.global_rank not in self.survivor_ranks:
            return None
        return self.survivor_ranks.index(self.global_rank)


def enabled() -> bool:
    return _env_flag("FTCCL_PATCH_ENABLE", True)


def bypass_dir() -> Path:
    return Path(os.environ.get("FTCCL_BYPASS_DIR", "/tmp/ftccl_bypass"))


def parent_world_size() -> int:
    return _env_int("FTCCL_BYPASS_WORLD_SIZE", _env_int("WORLD_SIZE", 8))


def failed_rank() -> int:
    failed_file = bypass_dir() / "failed_rank"
    if failed_file.exists():
        try:
            return int(failed_file.read_text().strip())
        except Exception:
            pass
    return _env_int("FTCCL_BYPASS_DEAD_RANK", parent_world_size() - 1)


def global_rank() -> int:
    return _env_int("RANK", _env_int("LOCAL_RANK", 0))


def survivor_ranks() -> tuple[int, ...]:
    survivors_file = bypass_dir() / "survivors"
    if survivors_file.exists():
        try:
            raw = survivors_file.read_text().strip()
            if raw:
                return tuple(int(item) for item in raw.split(",") if item != "")
        except Exception:
            pass
    dead = failed_rank()
    return tuple(rank for rank in range(parent_world_size()) if rank != dead)


def _active_marker_exists(rank: int, dead: int) -> bool:
    root = bypass_dir()
    if (root / "survivors").exists():
        return True
    if rank == dead and (root / f"killed.{dead}").exists():
        return True
    if rank != dead and (root / f"activate.{rank}").exists():
        return True
    return False


def current_view() -> BypassView:
    if not enabled():
        return BypassView(False, 0, parent_world_size(), failed_rank(), global_rank(), survivor_ranks())

    world = parent_world_size()
    dead = failed_rank()
    rank = global_rank()
    active = _active_marker_exists(rank, dead)
    gen = 1 if active else 0
    gen_file = bypass_dir() / "generation"
    if gen_file.exists():
        try:
            gen = int(gen_file.read_text().strip())
            active = gen > 0
        except Exception:
            pass
    return BypassView(active, gen, world, dead, rank, survivor_ranks())


def victim_data_mode() -> str:
    return os.environ.get("FTCCL_PATCH_VICTIM_DATA_MODE", "keep").lower()


def after_bypass_global_batch_size() -> Optional[int]:
    value = os.environ.get("FTCCL_PATCH_GLOBAL_BATCH_SIZE_AFTER_BYPASS")
    if not value:
        return None
    return int(value)


def grad_compensation_strategy() -> str:
    return os.environ.get("FTCCL_PATCH_GRAD_COMPENSATION", "survivor_global_batch").lower()


def grad_compensation_scale(view: BypassView) -> float:
    strategy = grad_compensation_strategy()
    if not view.active or view.survivor_count <= 0:
        return 1.0
    if strategy in {"", "0", "off", "none", "survivor_mean", "survivor_global_batch"}:
        return 1.0
    if strategy in {"old_world_size", "old-world-size", "old_world"}:
        return float(view.parent_world_size) / float(view.survivor_count)
    if strategy in {"zero_missing_average", "zero-missing-average", "zero_missing"}:
        return float(view.survivor_count) / float(view.parent_world_size)
    if strategy == "custom":
        return float(os.environ.get("FTCCL_PATCH_GRAD_COMPENSATION_SCALE", "1.0"))
    raise ValueError(
        "unknown FTCCL_PATCH_GRAD_COMPENSATION="
        f"{strategy!r}; expected survivor_global_batch, old_world_size, "
        "zero_missing_average, custom, or off"
    )


def pause_after_iter() -> Optional[int]:
    value = os.environ.get("FTCCL_PATCH_PAUSE_AFTER_ITER")
    if value is None or value == "":
        return None
    return int(value)


def touch_marker(name: str, text: str = "") -> None:
    root = bypass_dir()
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    tmp = root / f"{name}.tmp.{os.getpid()}"
    tmp.write_text(text)
    tmp.replace(path)


def marker_exists(name: str) -> bool:
    return (bypass_dir() / name).exists()


def wait_marker(name: str, timeout_sec: int = 300) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if marker_exists(name):
            return True
        time.sleep(0.05)
    return False


def rollback_enabled() -> bool:
    return _env_flag("FTCCL_PATCH_ROLLBACK_ENABLE", True)


def rollback_max_retries() -> int:
    return _env_int("FTCCL_PATCH_ROLLBACK_MAX_RETRIES", 1)


def rollback_timeout_sec() -> int:
    return _env_int("FTCCL_PATCH_ROLLBACK_TIMEOUT_SEC", 300)


def is_recoverable_comm_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    needles = (
        "nccl",
        "processgroupnccl",
        "disterror",
        "distbackenderror",
        "system error",
        "unhandled cuda error",
        "connection closed",
        "connection reset",
        "collective",
    )
    return any(needle in text for needle in needles)


def wait_survivor_markers(prefix: str, generation: int, iteration: int, timeout_sec: int) -> bool:
    view = current_view()
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if all(marker_exists(f"{prefix}.{rank}.{generation}.{iteration}") for rank in view.survivor_ranks):
            return True
        time.sleep(0.05)
    return False
