import torch

from ..backend import torch_op
from ..utils import broadcast_args_torch
from .quat_ops import (
    quat_apply,
    quat_apply_inv,
    quat_inverse,
    quat_mul,
    quat_slerp,
    quat_to_matrix,
    quat_from_matrix,
)


@torch_op()
def se3_apply(q, t, p):
    q, t, p = broadcast_args_torch([q, t, p])
    return quat_apply(q, p) + t

@torch_op()
def se3_inverse(q, t):
    q, t = broadcast_args_torch([q, t])
    q_inv = quat_inverse(q)
    return q_inv, -quat_apply(q_inv, t)

@torch_op()
def se3_apply_inv(q, t, p):
    q, t, p = broadcast_args_torch([q, t, p])
    return quat_apply_inv(q, p - t)

@torch_op()
def se3_compose(q1, t1, q2, t2):
    q1, t1, q2, t2 = broadcast_args_torch([q1, t1, q2, t2])
    return quat_mul(q1, q2), se3_apply(q1, t1, t2)

@torch_op()
def se3_slerp(q1, t1, q2, t2, w):
    q1, t1, q2, t2, w = broadcast_args_torch([q1, t1, q2, t2, w])
    return quat_slerp(q1, q2, w), torch.lerp(t1, t2, w)

@torch_op()
def se3_to_matrix_4x4(q, t):
    q, t = broadcast_args_torch([q, t])
    shape = list(t.shape[:-1]) + [4, 4]
    T = torch.eye(4, device=q.device, dtype=q.dtype)
    T = T.broadcast_to(shape).contiguous()
    T[..., :3, :3] = quat_to_matrix(q)
    T[..., :3, 3] = t
    return T

@torch_op()
def se3_to_matrix_3x4(q, t):
    q, t = broadcast_args_torch([q, t])
    R = quat_to_matrix(q)
    return torch.cat([R, t.unsqueeze(-1)], dim=-1)

@torch_op()
def se3_from_matrix(mtx):
    assert mtx.shape[-1] == 4
    assert mtx.shape[-2] == 3 or mtx.shape[-2] == 4
    return quat_from_matrix(mtx[..., :3, :3]), mtx[..., :3, 3]
