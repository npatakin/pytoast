from typing import Literal

import torch

from toast._C import (
    trajectory_fwd,
    trajectory_bwd
)

from ..backend import cpp_op
from .ops_utils import last_dim_ctg, apply_grad_mask


class SampleTrajectory(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            query_times: torch.Tensor,
            seq_times: torch.Tensor,
            seq_quats: torch.Tensor,
            seq_t: torch.Tensor,
            seq_mask: torch.Tensor,
            use_only_valid_keyframes: bool,
            extrapolate: bool,
            extrapolation_window: int,
            mask_op_and: bool,
            return_velocities: bool,
            return_indices: bool,
            return_mask: bool,
    ):
        # Indices are always produced: the backward needs them even when the
        # caller did not ask for them.
        out_quat, out_t, out_mask, out_v, out_w, out_indices = trajectory_fwd(
            query_times, seq_times, seq_quats, seq_t, seq_mask,
            use_only_valid_keyframes, extrapolate, extrapolation_window,
            mask_op_and, return_velocities, True, return_mask,
        )

        ctx.save_for_backward(
            query_times, seq_times, seq_quats, seq_t, out_indices)

        return (
            out_quat, out_t, out_mask, out_v, out_w,
            out_indices if return_indices else None,
        )

    @staticmethod
    def backward(
            ctx,
            grad_quat, grad_t,
            grad_mask,
            grad_v, grad_w,
            grad_indices
    ):
        grads = trajectory_bwd(
            *ctx.saved_tensors,
            last_dim_ctg(grad_quat), last_dim_ctg(grad_t),
            last_dim_ctg(grad_v), last_dim_ctg(grad_w),
        )
        return apply_grad_mask(ctx, tuple(grads) + (None,) * 8)


@cpp_op()
def sample_trajectory(
        query_times: torch.Tensor,
        seq_times: torch.Tensor,
        seq_quats: torch.Tensor = None,
        seq_t: torch.Tensor = None,
        seq_mask: torch.Tensor = None,
        use_only_valid_keyframes: bool = True,
        extrapolate: bool = True,
        extrapolation_window: int = 1,
        mask_mode: Literal['and'] | Literal['or'] = 'and',
        return_velocities: bool = False,
        return_indices: bool = False,
        return_mask: bool = False
):
    assert mask_mode in ['and', 'or']

    return SampleTrajectory.apply(
        query_times,
        seq_times,
        seq_quats,
        seq_t,
        seq_mask,
        use_only_valid_keyframes,
        extrapolate,
        extrapolation_window,
        mask_mode == 'and',
        return_velocities,
        return_indices,
        return_mask,
    )
