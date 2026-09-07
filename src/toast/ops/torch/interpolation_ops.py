from typing import Literal

import torch

from ..backend import torch_op
from .quat_ops import quat_slerp
from .motion_ops import angular_velocity


def mask_indices(masks: torch.Tensor, num_valid_frames_clamped: torch.Tensor):
    mask_cumsum = masks.cumsum(dim=-1)
    first_valid = masks.int().argmax(dim=-1)
    kth_valid = (mask_cumsum == num_valid_frames_clamped).int().argmax(dim=-1)
    return first_valid, kth_valid


def extrapolate_indices(masks: torch.Tensor, extrapolate_window: int):
    K = masks.shape[-1]
    num_valid_frames_clamped = masks.sum(dim=-1, keepdim=True)
    num_valid_frames_clamped.clamp_max_(extrapolate_window + 1)
    left_i, left_j = mask_indices(masks, num_valid_frames_clamped)
    right_j, right_i = mask_indices(
        masks.flip(dims=[-1,]), num_valid_frames_clamped)

    result = torch.stack([
        left_i, left_j,
        (K - 1) - right_i,
        (K - 1) - right_j,
        ], dim=-1)
    result[num_valid_frames_clamped.squeeze(-1) == 0] = -1
    return result


def find_bracketing_indices(
        timestamps: torch.Tensor,
        mask: torch.Tensor,
        query_times: torch.Tensor
):
    """
    timestamps: [..., K]
    mask:       [..., K] (bool)
    query_times:[..., Q]

    Returns:
        left_idx:  [..., Q]
            index of closest valid timestamp <= query, or -1 if not exists
        right_idx: [..., Q]
            index of closest valid timestamp > query, or -1 if not exists

    A query landing exactly on a keyframe timestamp is bracketed to the RIGHT of
    it (left_idx == that keyframe), matching the fused kernels.
    """
    timestamps = timestamps.unsqueeze(-2)   # [..., 1, K]
    query_times = query_times.unsqueeze(-1) # [..., Q, 1]
    mask  = mask.unsqueeze(-2)              # [..., 1, K]

    # left_idx: largest timestamp <= query  (exact match counts as left)
    left_valid = (timestamps <= query_times) & mask
    left_vals = torch.where(
        left_valid,
        timestamps,
        torch.full_like(timestamps, -float('inf'))
    )
    left_idx = left_vals.argmax(dim=-1)  # [..., Q]
    left_exists = left_valid.any(dim=-1)
    left_idx = torch.where(
        left_exists,
        left_idx,
        torch.full_like(left_idx, -1)
    )

    # right_idx: smallest timestamp > query
    right_valid = (timestamps > query_times) & mask
    right_vals = torch.where(
        right_valid,
        timestamps,
        torch.full_like(timestamps, float('inf'))
    )
    right_idx = right_vals.argmin(dim=-1)  # [..., Q]
    right_exists = right_valid.any(dim=-1)
    right_idx = torch.where(
        right_exists,
        right_idx,
        torch.full_like(right_idx, -1)
    )

    return left_idx, right_idx


def interpolation_corners_torch(
        times: torch.Tensor,
        masks: torch.Tensor | None,
        extrapolate_window: int,
        use_only_valid: bool,
        extrapolate: bool,
):
    if (masks is not None) and use_only_valid:
        extr_indices = extrapolate_indices(masks, extrapolate_window)
    else:
        num_keyframes = times.shape[-1]
        last_idx = num_keyframes - 1
        cur_k = min(extrapolate_window, last_idx)
        extr_indices = torch.tensor(
            [0, cur_k, last_idx - cur_k, last_idx],
            dtype=torch.int32,
            device=times.device
        ).view(4).broadcast_to(*times.shape[:-1], 4)

    if not extrapolate:
        extr_indices = extr_indices[..., [0,0,3,3]]

    return extr_indices


@torch_op()
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
    # print('query:', query_times)
    # print('seq_times:', seq_times)
    # print('seq_mask:', seq_mask)
    extr_indices = interpolation_corners_torch(
        times=seq_times, masks=seq_mask,
        extrapolate_window=extrapolation_window,
        use_only_valid=use_only_valid_keyframes,
        extrapolate=extrapolate
    )
    # print('extr_indices:', extr_indices.shape)

    if use_only_valid_keyframes and (seq_mask is not None):
        # print('find_bracketing_indices')
        left_idx, right_idx = find_bracketing_indices(
            timestamps=seq_times,
            mask=seq_mask,
            query_times=query_times
        )
    else:
        # print('searchsorted')
        num_keyframes = seq_times.shape[-1]
        # right=True so an exact match lands in the LEFT bracket, matching
        # find_bracketing_indices above and both fused kernels.
        right_idx = torch.searchsorted(seq_times, query_times, right=True)
        left_idx = right_idx - 1
        right_idx[right_idx == num_keyframes] = -1

    # print('left_idx:', left_idx.shape)
    # print('right_idx:', right_idx.shape)

    left_extr_mask = left_idx == -1   # query is before sequence start
    right_extr_mask = right_idx == -1 # query is after sequence end
    # no valid keyframes exist mask
    invalid_mask = torch.logical_and(left_extr_mask, right_extr_mask)
    # print('left_extr_mask:', left_extr_mask)
    # print('right_extr_mask:', right_extr_mask)

    # left extrapolation
    left_idx = torch.where(left_extr_mask, extr_indices[..., 0:1], left_idx)
    right_idx = torch.where(left_extr_mask, extr_indices[..., 1:2], right_idx)
    # right extrapolation
    left_idx = torch.where(right_extr_mask, extr_indices[..., 2:3], left_idx).long()
    right_idx = torch.where(right_extr_mask, extr_indices[..., 3:4], right_idx).long()

    # print('after substituting extr_indices:')
    # print('left_idx:', left_idx, left_idx.shape)
    # print('right_idx:', right_idx, right_idx.shape)
    # print('seq_times:', seq_times.shape)
    left_time = seq_times.take_along_dim(left_idx, dim=-1)
    right_time = seq_times.take_along_dim(right_idx, dim=-1)
    # print('left_time:', left_time.shape, left_time)
    # print('right_time:', right_time.shape, right_time)
    delta_time = right_time - left_time
    delta_bad_mask = (delta_time < torch.finfo(delta_time.dtype).eps) | invalid_mask
    rel_time_enum = query_times - left_time
    rel_time = rel_time_enum / delta_time
    rel_time[delta_bad_mask] = 0.5
    delta_time = delta_time.clamp_min(1e-6)

    out_quats = None
    out_t = None
    out_mask = None
    out_v = None
    out_w = None
    out_indices = None

    left_idx = left_idx.unsqueeze(-1)
    right_idx = right_idx.unsqueeze(-1)

    if seq_quats is not None:
        left_q = seq_quats.take_along_dim(left_idx, dim=-2)
        right_q = seq_quats.take_along_dim(right_idx, dim=-2)
        # print('left_q:', left_q.shape, 'right_q:', right_q.shape)
        out_quats = quat_slerp(left_q, right_q, rel_time.unsqueeze(-1))
        
        # identity quaternion where no valid keyframes exist; assigned through a
        # mask-only index since "tensor[bool_mask, int]" is mis-handled by
        # older torch versions
        out_quats[invalid_mask] = out_quats.new_tensor([1., 0., 0., 0.])

        if return_velocities:
            out_w = angular_velocity(left_q, right_q, delta_time.unsqueeze(-1))
            out_w[delta_bad_mask] = 0

    if seq_t is not None:
        left_t = seq_t.take_along_dim(left_idx, dim=-2)
        right_t = seq_t.take_along_dim(right_idx, dim=-2)
        out_t = torch.lerp(left_t, right_t, rel_time.unsqueeze(-1))
        if return_velocities:
            out_v = (right_t - left_t) / delta_time.unsqueeze(-1)
            out_v[delta_bad_mask] = 0
        out_t[invalid_mask] = 0

    if return_mask:
        out_mask = ~invalid_mask
        if seq_mask is not None:
            left_mask = seq_mask.take_along_dim(left_idx.squeeze(-1), dim=-1)
            right_mask = seq_mask.take_along_dim(right_idx.squeeze(-1), dim=-1)
            if mask_mode == 'and':
                out_mask &= (left_mask & right_mask)
            elif mask_mode == 'or':
                out_mask &= (left_mask | right_mask)
            else:
                raise ValueError('Unsupported mask combination mode: {}'
                                 ' (supported: "and", "or")'.format(mask_mode))

    if return_indices:
        out_indices = torch.cat([left_idx, right_idx], dim=-1).int()

    return out_quats, out_t, out_mask, out_v, out_w, out_indices
