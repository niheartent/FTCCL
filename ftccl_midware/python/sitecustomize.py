"""Auto-install the FTCCl Megatron runtime patch when PYTHONPATH points here."""

try:
    import multiprocessing
    import os

    if (
        os.environ.get("FTCCL_PATCH_SKIP_NON_MAIN_PROCESS", "1") != "0"
        and multiprocessing.current_process().name != "MainProcess"
    ):
        raise SystemExit

    import ftccl_megatron_patch

    ftccl_megatron_patch.install()
except SystemExit:
    pass
except Exception as exc:  # Keep startup failures explicit and easy to diagnose.
    import sys

    print(f"[ftccl-python] failed to install: {exc!r}", file=sys.stderr, flush=True)
    raise
