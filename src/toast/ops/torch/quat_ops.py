import torch
import torch.nn.functional as F

from ..backend import torch_op
from ..utils import broadcast_args_torch


@torch_op()
def quat_from_matrix(mtx: torch.Tensor) -> torch.Tensor:
    (
        m00, m01, m02,
        m10, m11, m12,
        m20, m21, m22
    ) = mtx.flatten(-2).unbind(dim=-1)

    s0 = 1 + m00 + m11 + m22
    s1 = 1 + m00 - m11 - m22
    s2 = 1 - m00 + m11 - m22
    s3 = 1 - m00 - m11 + m22

    values, indices = torch.stack([s0, s1, s2, s3], dim=-1).max(dim=-1)
    s = 2.0 * values.clamp_min(1e-6).sqrt()
    inv_s = s.reciprocal()
    s_q = 0.25 * s

    case0 = torch.stack([
        s_q, (m21 - m12) * inv_s, (m02 - m20) * inv_s, (m10 - m01)*inv_s], dim=-1)
    case1 = torch.stack([
        (m21 - m12) * inv_s, s_q, (m01 + m10) * inv_s, (m02 + m20) * inv_s], dim=-1)
    case2 = torch.stack([
        (m02 - m20) * inv_s, (m01 + m10) * inv_s, s_q, (m12 + m21) * inv_s], dim=-1)
    case3 = torch.stack([
        (m10 - m01) * inv_s, (m02 + m20) * inv_s, (m12 + m21) * inv_s, 0.25 * s], dim=-1)

    cases = torch.stack([case0, case1, case2, case3], dim=-1)
    return cases.take_along_dim(indices[..., None, None], dim=-1).squeeze(-1)


@torch_op()
def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    mtx_shape = list(q.shape[:-1]) + [3,3]
    mul = 2.0 / (q * q).sum(-1)
    w, x, y, z = q.unbind(dim=-1)

    return torch.stack([
        1 - mul * (y * y + z * z), mul * (x * y - z * w), mul * (x * z + y * w),
        mul * (x * y + z * w), 1 - mul * (x * x + z * z), mul * (y * z - x * w),
        mul * (x * z - y * w), mul * (y * z + x * w), 1 - mul * (x * x + y * y)
    ], dim=-1).reshape(*mtx_shape)


def _sin_scale(theta: torch.Tensor, theta_half: torch.Tensor) -> torch.Tensor:
    # todo: increase this threshold? actually, for fp32 taylor approximation
    # is more precise up to |theta| = 0.08
    mask = theta.abs() > (100 * torch.finfo(theta.dtype).eps)
    sin_scale = torch.zeros_like(theta)
    sin_scale[mask] = theta_half[mask].sin() / theta[mask]
    not_mask = ~mask
    theta_masked = theta[not_mask]
    sin_scale[not_mask] = 0.5 - (1.0 / 48.0) * (theta_masked * theta_masked)
    return sin_scale

@torch_op()
def quat_from_axis_angle(v: torch.Tensor) -> torch.Tensor:
    theta = v.norm(dim=-1).unsqueeze(-1)
    theta_half = 0.5 * theta

    return torch.cat([
        theta_half.cos(), v * _sin_scale(theta, theta_half)
    ], dim=-1)


@torch_op()
def quat_to_axis_angle(q: torch.Tensor) -> torch.Tensor:
    w, xyz = q[..., 0], q[..., 1:]
    theta_half = torch.atan2(xyz.norm(dim=-1), w).unsqueeze(-1)

    return xyz / _sin_scale(2.0 * theta_half, theta_half)


@torch_op()
def quat_std(q: torch.Tensor) -> torch.Tensor:
    return F.normalize(torch.where(q[..., 0:1] >= 0, q, -q), dim=-1)


@torch_op()
def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = broadcast_args_torch([a, b])
    a_real, b_real = a[..., 0:1], b[..., 0:1]
    a_vec, b_vec = a[..., 1:], b[..., 1:]
    new_real = a_real * b_real - (a_vec * b_vec).sum(dim=-1, keepdim=True)
    new_vec = a_real * b_vec + b_real * a_vec + a_vec.cross(b_vec, dim=-1)
    return torch.cat([new_real, new_vec], dim=-1)


@torch_op()
def quat_random(
    *dims,
    dtype: torch.dtype = None,
    device: torch.device | str = None,
    generator: torch.Generator = None
) -> torch.Tensor:
    r1, r2, r3 = torch.rand(
        3, *dims, device=device, dtype=dtype, generator=generator)
    twopi = 2 * torch.pi
    q1 = (1.0 - r1).sqrt() * (twopi * r2).sin()
    q2 = (1.0 - r1).sqrt() * (twopi * r2).cos()
    q3 = r1.sqrt() * (twopi * r3).sin()
    q4 = r1.sqrt() * (twopi * r3).cos()
    return torch.stack([q1, q2, q3, q4], dim=-1)


@torch_op()
def quat_unit(
        *dims,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = 'cpu'
):
    data = torch.zeros(*dims, dtype=dtype, device=device)
    data[..., 0] = 1
    return data


@torch_op()
def quat_apply(
        quat: torch.Tensor,
        _points: torch.Tensor,
        inv_quat: bool = False
):
    assert quat.shape[-1] == 4
    assert _points.shape[-1] == 3
    assert _points.device == quat.device
    assert _points.dtype == quat.dtype
    assert list(quat.shape[:-1]) == list(_points.shape[:-1])

    q_batch_elements = quat.numel() // 4

    w = quat[..., 0:1].view(q_batch_elements, 1, 1)
    xyz = quat[..., 1:].view(q_batch_elements, 1, 3)
    if inv_quat:
        xyz = -xyz

    # explicit 1 rather than -1: the asserts above force _points to share
    # quat's batch shape, so this dimension is always 1, and -1 cannot be
    # inferred when the batch is empty ("cannot reshape tensor of 0 elements")
    points = _points.view(q_batch_elements, 1, 3)

    uv = torch.linalg.cross(xyz, points)
    uuv = torch.linalg.cross(xyz, uv)
    result = points + 2 * (w * uv + uuv)
    return result.view_as(_points)


@torch_op()
def quat_apply_inv(quat: torch.Tensor, _points: torch.Tensor):
    return quat_apply(quat_inverse(quat), _points)


@torch_op()
def quat_slerp(q1, q2, t):
    q1, q2, t = broadcast_args_torch([q1, q2, t])
    eps = 1e-6 if q1.dtype == torch.float32 else 1e-12

    if t.shape[-1] == 1:
        t = t.squeeze(-1)

    z = q2
    cosTheta = (q1 * z).sum(dim=-1)

    z = torch.where(cosTheta.unsqueeze(-1) >= 0, z, -z)
    cosTheta = torch.where(cosTheta >= 0, cosTheta, -cosTheta)

    linear_lerp = torch.lerp(q1, z, t.unsqueeze(-1))

    angle = cosTheta.clamp(eps, 1-eps).acos()
    denom = angle.sin()
    w1 = ((1.0 - t) * angle).sin() / denom
    w2 = (t * angle).sin() / denom

    slerp = w1.unsqueeze(-1) * q1 + w2.unsqueeze(-1) * z
    return quat_std(torch.where(
        cosTheta.unsqueeze(-1) >= (1-eps),
        linear_lerp,
        slerp
    ))


@torch_op()
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., 0:1], -q[..., 1:]], dim=-1)


@torch_op()
def quat_inverse(q: torch.Tensor) -> torch.Tensor:
    return quat_conjugate(q) / (q * q).sum(dim=-1, keepdim=True)


@torch_op()
def quat_from_xyzw(xyzw: torch.Tensor) -> torch.Tensor:
    return xyzw[..., [3,0,1,2]]


@torch_op()
def quat_from_euler_angles(roll_pitch_yaw: torch.Tensor) -> torch.Tensor:
    roll, pitch, yaw = (roll_pitch_yaw * 0.5).unbind(dim=-1)

    cr, sr = roll.cos(), roll.sin()
    cp, sp = pitch.cos(), pitch.sin()
    cy, sy = yaw.cos(), yaw.sin()

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return quat_std(torch.stack([w, x, y, z], dim=-1))
