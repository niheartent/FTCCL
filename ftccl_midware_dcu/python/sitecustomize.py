"""Auto-install the FTCCL Megatron runtime patch when PYTHONPATH points here."""

try:
    import importlib.abc
    import importlib.machinery
    import multiprocessing
    import os
    import sys

    def _env_flag(name: str, default: bool = False) -> bool:
        value = os.environ.get(name)
        if value is None:
            return default
        return value.strip().lower() not in {"", "0", "false", "no", "off"}

    def _maybe_disable_megatron_gloo_groups() -> None:
        if not _env_flag("FTCCL_PATCH_DISABLE_GLOO_PROCESS_GROUPS"):
            return
        argv0 = os.path.basename(sys.argv[0]) if sys.argv else ""
        if argv0 != "pretrain_gpt.py":
            return
        if "--disable-gloo-process-groups" in sys.argv:
            return
        sys.argv.append("--disable-gloo-process-groups")

    if (
        os.environ.get("FTCCL_PATCH_SKIP_NON_MAIN_PROCESS", "1") != "0"
        and multiprocessing.current_process().name != "MainProcess"
    ):
        raise SystemExit

    _maybe_disable_megatron_gloo_groups()

    import ftccl_megatron_patch

    ftccl_megatron_patch.install()

    class _DcuMegatronAdaptorLoader(importlib.abc.Loader):
        def __init__(self, wrapped: importlib.abc.Loader) -> None:
            self._wrapped = wrapped

        def create_module(self, spec):
            if hasattr(self._wrapped, "create_module"):
                return self._wrapped.create_module(spec)
            return None

        def exec_module(self, module) -> None:
            self._wrapped.exec_module(module)
            ftccl_megatron_patch.reinstall_training_patch("dcu_megatron.adaptor")

    class _DcuMegatronAdaptorFinder(importlib.abc.MetaPathFinder):
        TARGET = "dcu_megatron.adaptor.megatron_adaptor"

        def find_spec(self, fullname, path, target=None):
            if fullname != self.TARGET:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                return spec
            if isinstance(spec.loader, _DcuMegatronAdaptorLoader):
                return spec
            spec.loader = _DcuMegatronAdaptorLoader(spec.loader)
            return spec

    if not any(isinstance(finder, _DcuMegatronAdaptorFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _DcuMegatronAdaptorFinder())
except SystemExit:
    pass
except Exception as exc:  # Keep startup failures explicit and easy to diagnose.
    print(f"[ftccl-python] failed to install: {exc!r}", file=sys.stderr, flush=True)
    raise
