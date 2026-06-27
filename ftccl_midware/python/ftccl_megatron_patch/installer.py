from __future__ import annotations

import functools

from . import state
from .patch_utils import wrap_attr
from .runtime import FaultTolerantDataIterator, RUNTIME


def _install_parallel_state_patch() -> None:
    import megatron.core.parallel_state as ps

    orig_get_dp_world_size = ps.get_data_parallel_world_size
    orig_get_dp_rank = ps.get_data_parallel_rank

    @functools.wraps(orig_get_dp_world_size)
    def get_data_parallel_world_size_patched(*args, **kwargs):
        view = state.current_view()
        if view.active:
            return view.survivor_count
        return orig_get_dp_world_size(*args, **kwargs)

    @functools.wraps(orig_get_dp_rank)
    def get_data_parallel_rank_patched(*args, **kwargs):
        view = state.current_view()
        if view.active:
            mapped = view.survivor_rank
            if mapped is not None:
                return mapped
            return 0
        return orig_get_dp_rank(*args, **kwargs)

    ps.get_data_parallel_world_size = get_data_parallel_world_size_patched
    ps.get_data_parallel_rank = get_data_parallel_rank_patched


def _install_dataloader_patch() -> None:
    def builder_wrapper(original):
        RUNTIME.original_build_pretraining_data_loader = original

        @functools.wraps(original)
        def wrapped_build_pretraining_data_loader(dataset, consumed_samples):
            try:
                if dataset is not None:
                    split = getattr(dataset, "split", getattr(dataset, "index_split", None))
                    split_name = str(split).lower()
                    if "valid" not in split_name and "test" not in split_name:
                        RUNTIME.train_loader.dataset = dataset
                        RUNTIME.train_loader.consumed_samples = consumed_samples
            except Exception:
                pass
            return original(dataset, consumed_samples)

        return wrapped_build_pretraining_data_loader

    wrap_attr(
        "megatron.training.datasets.data_samplers",
        "build_pretraining_data_loader",
        builder_wrapper,
    )
    # training.py imports the function by value at module import time. Patch that
    # binding too if training.py is already imported.
    try:
        wrap_attr("megatron.training.training", "build_pretraining_data_loader", builder_wrapper)
    except Exception:
        pass


def _install_training_patch() -> None:
    def _pause_after_logged_iteration(iteration) -> None:
        if iteration is None:
            return
        pause_iter = state.pause_after_iter()
        if pause_iter is None or int(iteration) != pause_iter:
            return
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as exc:
            state.log(f"cuda synchronize before pause skipped: {exc}")
        state.touch_marker(
            f"pause_ready.{state.global_rank()}.{iteration}",
            f"rank={state.global_rank()} iteration={iteration}\n",
        )
        state.log(f"pause_ready iteration={iteration}; waiting for supervisor")
        if not state.wait_marker(f"continue_after_kill.{iteration}", timeout_sec=300):
            raise TimeoutError(f"timed out waiting for continue_after_kill.{iteration}")
        state.log(f"continue_after_kill received iteration={iteration}")

    def train_wrapper(original):
        @functools.wraps(original)
        def wrapped_train(
            forward_step_func,
            model,
            optimizer,
            opt_param_scheduler,
            train_data_iterator,
            valid_data_iterator,
            process_non_loss_data_func,
            config,
            checkpointing_context,
            non_loss_data_func,
            inference_model=None,
        ):
            if train_data_iterator is not None and not isinstance(
                train_data_iterator, FaultTolerantDataIterator
            ):
                train_data_iterator = FaultTolerantDataIterator(train_data_iterator, RUNTIME)
                state.log("wrapped train_data_iterator for survivor rebuild")
            return original(
                forward_step_func,
                model,
                optimizer,
                opt_param_scheduler,
                train_data_iterator,
                valid_data_iterator,
                process_non_loss_data_func,
                config,
                checkpointing_context,
                non_loss_data_func,
                inference_model=inference_model,
            )

        return wrapped_train

    def train_step_wrapper(original):
        @functools.wraps(original)
        def wrapped_train_step(*args, **kwargs):
            iteration = kwargs.get("iteration")
            if iteration is None and len(args) >= 8:
                iteration = args[7]
            data_iterator = kwargs.get("data_iterator")
            if data_iterator is None and len(args) >= 2:
                data_iterator = args[1]
            model = kwargs.get("model")
            if model is None and len(args) >= 3:
                model = args[2]
            optimizer = kwargs.get("optimizer")
            if optimizer is None and len(args) >= 4:
                optimizer = args[3]

            def begin_iteration_capture(iterator):
                if isinstance(iterator, FaultTolerantDataIterator):
                    iterator.begin_iteration()
                elif isinstance(iterator, (list, tuple)):
                    for item in iterator:
                        begin_iteration_capture(item)

            def finish_iteration_capture(iterator):
                if isinstance(iterator, FaultTolerantDataIterator):
                    iterator.finish_iteration()
                elif isinstance(iterator, (list, tuple)):
                    for item in iterator:
                        finish_iteration_capture(item)

            def rollback_iterator(iterator):
                if isinstance(iterator, FaultTolerantDataIterator):
                    iterator.rollback_for_fault()
                elif isinstance(iterator, (list, tuple)):
                    for item in iterator:
                        rollback_iterator(item)

            def reset_megatron_rerun_state(iter_value: int) -> None:
                try:
                    from megatron.core.rerun_state_machine import (
                        RerunState,
                        get_rerun_state_machine,
                    )

                    machine = get_rerun_state_machine()
                    machine.state = RerunState.NOT_RUNNING_YET
                    machine.current_iteration = iter_value
                    machine.rerun_requested = False
                    machine.checkpoint_requested = False
                    machine.restart_again_requested = False
                    machine.continue_requested = False
                    machine.data_iterator_checkpoints = None
                    machine.saved_results.clear()
                    machine.validation_counts.clear()
                    machine.failed_validation_call = None
                    machine.initial_result = None
                    state.log(
                        f"reset Megatron rerun state before fault replay iteration={iter_value}"
                    )
                except Exception as reset_exc:
                    state.log(f"failed to reset Megatron rerun state: {reset_exc!r}", level=0)
                    raise

            def maybe_wrap_optimizer_step():
                if optimizer is None:
                    return None

                scale = state.grad_compensation_scale(state.current_view())
                if scale == 1.0:
                    return None

                original_step = optimizer.step

                @functools.wraps(original_step)
                def wrapped_step(*step_args, **step_kwargs):
                    RUNTIME.apply_gradient_compensation(model, optimizer, iteration)
                    return original_step(*step_args, **step_kwargs)

                optimizer.step = wrapped_step
                return original_step

            attempts = 0
            while True:
                RUNTIME.apply_effective_training_args()
                begin_iteration_capture(data_iterator)
                restored_step = maybe_wrap_optimizer_step()
                try:
                    result = original(*args, **kwargs)
                    finish_iteration_capture(data_iterator)
                    if iteration is not None:
                        state.touch_marker(
                            f"train_step_done.{state.global_rank()}.{iteration}",
                            f"rank={state.global_rank()} iteration={iteration}\n",
                        )
                    return result
                except Exception as exc:
                    if (
                        not state.rollback_enabled()
                        or attempts >= state.rollback_max_retries()
                        or not state.is_recoverable_comm_error(exc)
                    ):
                        raise
                    attempts += 1
                    view = state.current_view()
                    if not view.active or view.is_victim or view.survivor_rank is None:
                        raise
                    iter_value = int(iteration) if iteration is not None else -1
                    marker = f"rollback_ready.{state.global_rank()}.{view.generation}.{iter_value}"
                    state.touch_marker(
                        marker,
                        (
                            f"rank={state.global_rank()} generation={view.generation} "
                            f"iteration={iter_value} error={type(exc).__name__}: {exc}\n"
                        ),
                    )
                    state.log(
                        "caught recoverable communication error; "
                        f"generation={view.generation} iteration={iter_value} "
                        f"attempt={attempts}/{state.rollback_max_retries()} error={exc!r}"
                    )
                    if not state.wait_survivor_markers(
                        "rollback_ready",
                        view.generation,
                        iter_value,
                        state.rollback_timeout_sec(),
                    ):
                        raise TimeoutError(
                            "timed out waiting for survivor rollback barrier "
                            f"generation={view.generation} iteration={iter_value}"
                        ) from exc
                    RUNTIME.apply_effective_training_args()
                    reset_megatron_rerun_state(iter_value)
                    rollback_iterator(data_iterator)
                    try:
                        import torch

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    state.touch_marker(
                        f"rollback_replay.{state.global_rank()}.{view.generation}.{iter_value}",
                        f"rank={state.global_rank()} generation={view.generation} iteration={iter_value}\n",
                    )
                    state.log(
                        f"rollback complete; retrying train_step generation={view.generation} "
                        f"iteration={iter_value}"
                    )
                finally:
                    if restored_step is not None and optimizer is not None:
                        optimizer.step = restored_step

        return wrapped_train_step

    def checkpoint_and_decide_exit_wrapper(original):
        @functools.wraps(original)
        def wrapped_checkpoint_and_decide_exit(*args, **kwargs):
            result = original(*args, **kwargs)
            iteration = kwargs.get("iteration")
            if iteration is None and len(args) >= 4:
                iteration = args[3]
            _pause_after_logged_iteration(iteration)
            return result

        return wrapped_checkpoint_and_decide_exit

    wrap_attr("megatron.training.training", "train", train_wrapper)
    wrap_attr("megatron.training.training", "train_step", train_step_wrapper)
    wrap_attr(
        "megatron.training.training",
        "checkpoint_and_decide_exit",
        checkpoint_and_decide_exit_wrapper,
    )


def install() -> None:
    if RUNTIME.installed or not state.enabled():
        return
    _install_parallel_state_patch()
    _install_dataloader_patch()
    _install_training_patch()
    RUNTIME.installed = True
    state.log("installed Megatron survivor patch")
