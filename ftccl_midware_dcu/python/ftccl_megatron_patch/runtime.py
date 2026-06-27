from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import state
from megatron.core.rerun_state_machine import RerunDataIterator


@dataclass
class CapturedLoader:
    dataset: Any = None
    consumed_samples: int = 0
    dataloader_type: str = "single"


class RuntimeState:
    def __init__(self) -> None:
        self.installed = False
        self.last_applied_generation = 0
        self.last_logged_grad_compensation_generation = 0
        self.train_loader = CapturedLoader()
        self.original_build_pretraining_data_loader = None

    def apply_effective_training_args(self) -> None:
        view = state.current_view()
        if (
            not view.active
            or not state.c_layer_bypass_active(view)
            or view.generation == self.last_applied_generation
        ):
            return

        try:
            from megatron.training import get_args
            from megatron.core.num_microbatches_calculator import (
                reconfigure_num_microbatches_calculator,
                update_num_microbatches,
            )

            args = get_args()
            old_dp = getattr(args, "data_parallel_size", None)
            old_gbs = getattr(args, "global_batch_size", None)
            old_do_valid = getattr(args, "do_valid", None)
            old_do_test = getattr(args, "do_test", None)
            old_eval_iters = getattr(args, "eval_iters", None)

            args.data_parallel_size = view.survivor_count
            new_gbs = state.after_bypass_global_batch_size()
            if new_gbs is not None:
                args.global_batch_size = new_gbs
            if state.disable_eval_after_bypass():
                args.do_valid = False
                args.do_test = False
                args.eval_iters = 0

            reconfigure_num_microbatches_calculator(
                args.rank,
                args.rampup_batch_size,
                args.global_batch_size,
                args.micro_batch_size,
                args.data_parallel_size,
                args.decrease_batch_size_if_needed,
            )
            update_num_microbatches(args.consumed_train_samples, consistency_check=False)
            state.log(
                "activated survivor training view "
                f"generation={view.generation} failed_rank={view.failed_rank} "
                f"survivors={list(view.survivor_ranks)} "
                f"dp_size {old_dp}->{args.data_parallel_size} "
                f"global_batch_size {old_gbs}->{args.global_batch_size} "
                f"eval do_valid {old_do_valid}->{getattr(args, 'do_valid', None)} "
                f"do_test {old_do_test}->{getattr(args, 'do_test', None)} "
                f"eval_iters {old_eval_iters}->{getattr(args, 'eval_iters', None)}"
            )
            self.last_applied_generation = view.generation
        except Exception as exc:
            state.log(f"failed to apply survivor training args: {exc!r}", level=0)
            raise

    def apply_gradient_compensation(self, model: Any, optimizer: Any, iteration: Any) -> None:
        view = state.current_view()
        if (
            not view.active
            or not state.c_layer_bypass_active(view)
            or view.is_victim
            or view.survivor_rank is None
        ):
            return

        scale = state.grad_compensation_scale(view)
        strategy = state.grad_compensation_strategy()
        if scale == 1.0:
            if self.last_logged_grad_compensation_generation != view.generation:
                state.log(
                    "gradient compensation strategy "
                    f"generation={view.generation} strategy={strategy} scale=1.0"
                )
                self.last_logged_grad_compensation_generation = view.generation
            return

        tensors = []
        seen_tensors = set()
        seen_params = set()

        def add_tensor(tensor: Any) -> None:
            if tensor is None:
                return
            tensor_id = id(tensor)
            if tensor_id in seen_tensors:
                return
            if not hasattr(tensor, "mul_"):
                return
            seen_tensors.add(tensor_id)
            tensors.append(tensor)

        def add_param(param: Any) -> None:
            param_id = id(param)
            if param_id in seen_params:
                return
            seen_params.add(param_id)
            add_tensor(getattr(param, "main_grad", None))
            add_tensor(getattr(param, "grad", None))

        chunks = model if isinstance(model, (list, tuple)) else [model]
        for chunk in chunks:
            if hasattr(chunk, "parameters"):
                for param in chunk.parameters():
                    add_param(param)

        if hasattr(optimizer, "get_parameters"):
            for param in optimizer.get_parameters():
                add_param(param)
        elif hasattr(optimizer, "param_groups"):
            for group in optimizer.param_groups:
                for param in group.get("params", []):
                    add_param(param)

        try:
            import torch

            with torch.no_grad():
                for tensor in tensors:
                    tensor.mul_(scale)
        except Exception as exc:
            state.log(f"failed to apply gradient compensation: {exc!r}", level=0)
            raise

        iter_value = -1 if iteration is None else int(iteration)
        state.touch_marker(
            f"grad_compensation.{state.global_rank()}.{view.generation}.{iter_value}",
            (
                f"rank={state.global_rank()} generation={view.generation} "
                f"iteration={iter_value} strategy={strategy} scale={scale} tensors={len(tensors)}\n"
            ),
        )
        state.log(
            "applied gradient compensation "
            f"generation={view.generation} iteration={iter_value} "
            f"strategy={strategy} scale={scale:.8g} tensors={len(tensors)}"
        )


RUNTIME = RuntimeState()


class FaultTolerantDataIterator(RerunDataIterator):
    """A small iterator proxy that can rebuild its inner dataloader after bypass."""

    def __init__(self, original_iterator: Any, runtime: RuntimeState) -> None:
        super().__init__(original_iterator)
        self._runtime = runtime
        self._rebuilt_generation = 0
        self._capture_microbatches = False

    def _maybe_rebuild(self) -> None:
        view = state.current_view()
        if (
            not view.active
            or not state.c_layer_bypass_active(view)
            or self._rebuilt_generation == view.generation
        ):
            return
        if view.is_victim and state.victim_data_mode() == "keep":
            state.log(
                f"victim keeps original data iterator generation={view.generation} "
                f"failed_rank={view.failed_rank}",
            )
            self._rebuilt_generation = view.generation
            return
        if self._runtime.original_build_pretraining_data_loader is None:
            state.log("cannot rebuild dataloader: original builder missing", level=0)
            self._rebuilt_generation = view.generation
            return
        if self._runtime.train_loader.dataset is None:
            state.log("cannot rebuild dataloader: train dataset missing", level=0)
            self._rebuilt_generation = view.generation
            return

        from megatron.training import get_args

        args = get_args()
        self._runtime.apply_effective_training_args()
        dataloader = self._runtime.original_build_pretraining_data_loader(
            self._runtime.train_loader.dataset,
            args.consumed_train_samples,
        )
        self.iterable = iter(dataloader)
        self.saved_microbatches.clear()
        self.replaying = False
        self.replay_pos = 0
        self._rebuilt_generation = view.generation
        state.log(
            "dataloader rebuilt "
            f"generation={view.generation} dp_rank={view.survivor_rank} "
            f"dp_size={view.survivor_count} consumed_samples={args.consumed_train_samples}"
        )

    def __next__(self):
        self._maybe_rebuild()
        if self.replaying:
            return super().__next__()
        batch = next(self.iterable)
        if self._capture_microbatches:
            self.saved_microbatches.append(batch)
        return batch

    def begin_iteration(self) -> None:
        self.saved_microbatches.clear()
        self.replaying = False
        self.replay_pos = 0
        self._capture_microbatches = True

    def finish_iteration(self) -> None:
        self._capture_microbatches = False
        self.saved_microbatches.clear()
        self.replaying = False
        self.replay_pos = 0

    def rollback_for_fault(self) -> None:
        view = state.current_view()
        if view.active:
            self._rebuilt_generation = 0
            self.saved_microbatches.clear()
            self.replaying = False
            self.replay_pos = 0
            self._maybe_rebuild()
            return
        if self.saved_microbatches:
            self.rewind()
