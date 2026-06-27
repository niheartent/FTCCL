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
    return _env_int(
        "FTCCL_BYPASS_WORLD_SIZE",
        _env_int("WORLD_SIZE", _env_int("OMPI_COMM_WORLD_SIZE", _env_int("PMI_SIZE", 8))),
    )


def failed_rank() -> int:
    failed_file = bypass_dir() / "failed_rank"
    if failed_file.exists():
        try:
            return int(failed_file.read_text().strip())
        except Exception:
            pass
    return _env_int("FTCCL_BYPASS_DEAD_RANK", parent_world_size() - 1)


def global_rank() -> int:
    return _env_int(
        "RANK",
        _env_int("OMPI_COMM_WORLD_RANK", _env_int("PMI_RANK", _env_int("LOCAL_RANK", 0))),
    )


def local_rank() -> int:
    return _env_int(
        "LOCAL_RANK",
        _env_int("OMPI_COMM_WORLD_LOCAL_RANK", _env_int("MPI_LOCALRANKID", 0)),
    )


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


def recoverable_error_seen(view: Optional[BypassView] = None) -> bool:
    """True after C middleware returned the planned recoverable RCCL error."""
    if view is None:
        world = parent_world_size()
        dead = failed_rank()
        rank = global_rank()
        view = BypassView(False, 0, world, dead, rank, survivor_ranks())
    if view.is_victim:
        return marker_exists(f"killed.{view.failed_rank}")
    if view.survivor_rank is None:
        return False
    return marker_exists(f"comm_error.{view.global_rank}")


def c_layer_bypass_active(view: Optional[BypassView] = None) -> bool:
    """True after the interposed RCCL communicator has activated survivor topo."""
    if view is None:
        world = parent_world_size()
        dead = failed_rank()
        rank = global_rank()
        view = BypassView(False, 0, world, dead, rank, survivor_ranks())
    if view.is_victim:
        return marker_exists(f"killed.{view.failed_rank}")
    if view.survivor_rank is None:
        return False
    return marker_exists(f"activate.{view.global_rank}")


def requested_view() -> BypassView:
    world = parent_world_size()
    dead = failed_rank()
    rank = global_rank()
    survivors = survivor_ranks()
    view = BypassView(False, 0, world, dead, rank, survivors)
    active = recoverable_error_seen(view) or c_layer_bypass_active(view)
    gen = 1 if active else 0
    gen_file = bypass_dir() / "generation"
    if gen_file.exists():
        try:
            requested_gen = int(gen_file.read_text().strip())
            if active:
                gen = requested_gen
        except Exception:
            pass
    return BypassView(active, gen, world, dead, rank, survivors)


def current_view() -> BypassView:
    if not enabled():
        return BypassView(False, 0, parent_world_size(), failed_rank(), global_rank(), survivor_ranks())

    world = parent_world_size()
    dead = failed_rank()
    rank = global_rank()
    survivors = survivor_ranks()
    view = BypassView(False, 0, world, dead, rank, survivors)
    active = c_layer_bypass_active(view)
    gen = 1 if active else 0
    gen_file = bypass_dir() / "generation"
    if gen_file.exists():
        try:
            requested_gen = int(gen_file.read_text().strip())
            if active:
                gen = requested_gen
        except Exception:
            pass
    return BypassView(active, gen, world, dead, rank, survivors)


def victim_data_mode() -> str:
    return os.environ.get("FTCCL_PATCH_VICTIM_DATA_MODE", "keep").lower()


def disable_eval_after_bypass() -> bool:
    return _env_flag("FTCCL_PATCH_DISABLE_EVAL_AFTER_BYPASS", True)


def log_rank_after_bypass() -> str:
    return os.environ.get("FTCCL_PATCH_LOG_RANK_AFTER_BYPASS", "survivor_last").lower()


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
    host = os.uname().nodename.replace("/", "_")
    tmp = root / f"{name}.tmp.{host}.{global_rank()}.{os.getpid()}"
    with tmp.open("w") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    try:
        dir_fd = os.open(root, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass


def marker_exists(name: str) -> bool:
    return (bypass_dir() / name).exists()


def wait_marker(name: str, timeout_sec: int = 300) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if marker_exists(name):
            return True
        time.sleep(0.05)
    return False


def _marker_rank_set(prefix: str, generation: int, iteration: int) -> set[int]:
    root = bypass_dir()
    observed: set[int] = set()
    head = f"{prefix}."
    tail = f".{generation}.{iteration}"
    try:
        with os.scandir(root) as entries:
            for entry in entries:
                name = entry.name
                if not name.startswith(head) or not name.endswith(tail):
                    continue
                rank_text = name[len(head) : -len(tail)]
                if rank_text.isdigit():
                    observed.add(int(rank_text))
    except FileNotFoundError:
        pass
    return observed


def _format_ranks(ranks: tuple[int, ...] | set[int]) -> str:
    return ",".join(str(rank) for rank in sorted(ranks))


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
    if _env_flag("FTCCL_SUPERVISOR_RELEASE_BARRIER", False):
        release = f"{prefix.replace('_ready', '')}_release.{generation}.{iteration}"
        if wait_marker(release, timeout_sec):
            return True
        log(
            "timed out waiting for supervisor marker release "
            f"release={release} timeout_sec={timeout_sec} sync_dir={bypass_dir()}",
            level=0,
        )
        return False

    view = requested_view()
    expected = set(view.survivor_ranks)
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        observed = _marker_rank_set(prefix, generation, iteration)
        if expected.issubset(observed):
            return True

        ok_by_stat = True
        for rank in view.survivor_ranks:
            if not marker_exists(f"{prefix}.{rank}.{generation}.{iteration}"):
                ok_by_stat = False
                break
        if ok_by_stat:
            return True
        time.sleep(0.05)
    observed = _marker_rank_set(prefix, generation, iteration)
    missing = expected - observed
    log(
        "timed out waiting for survivor marker barrier "
        f"prefix={prefix} generation={generation} iteration={iteration} "
        f"timeout_sec={timeout_sec} missing=[{_format_ranks(missing)}] "
        f"observed=[{_format_ranks(observed)}] sync_dir={bypass_dir()}",
        level=0,
    )
    return False
