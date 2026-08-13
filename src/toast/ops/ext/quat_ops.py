import torch

from toast._C import (
    random_quat_from_uniform,
    quat_unit_,
    broadcast_args,

    quat_from_xyzw_fwd,
    quat_from_xyzw_bwd,

    quat_from_axis_angle_fwd,
    quat_from_axis_angle_bwd,

    quat_to_axis_angle_fwd,
    quat_to_axis_angle_bwd,

    quat_to_matrix_fwd,
    quat_to_matrix_bwd,

    quat_from_matrix_fwd,
    quat_from_matrix_bwd,

    quat_std_fwd,
    quat_std_bwd,

    quat_inverse_fwd,
    quat_inverse_bwd,

    quat_conjugate_fwd,
    quat_conjugate_bwd,

    quat_mul_fwd,
    quat_mul_bwd,

    quat_apply_fwd,
    quat_apply_bwd,

    quat_apply_inv_fwd,
    quat_apply_inv_bwd,

    quat_slerp_fwd,
    quat_slerp_bwd,

    quat_from_euler_angles_fwd,
    quat_from_euler_angles_bwd,
)

from ..backend import cpp_op
from .ops_utils import last_dim_ctg, mtx_last_dim_ctg, apply_grad_mask


class QuatFromAxisAngle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, axis_angle: torch.Tensor):
        ctx.save_for_backward(axis_angle)
        return quat_from_axis_angle_fwd(axis_angle)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        axis_angle = ctx.saved_tensors[0]
        return apply_grad_mask(ctx, quat_from_axis_angle_bwd(axis_angle, last_dim_ctg(grad_out)))


@cpp_op()
def quat_from_axis_angle(axis_angle: torch.Tensor) -> torch.Tensor:
    args, req_grad = broadcast_args([axis_angle])
    return (QuatFromAxisAngle.apply if req_grad else quat_from_axis_angle_fwd)(*args)


class QuatFromEulerAngles(torch.autograd.Function):
    @staticmethod
    def forward(ctx, euler_angles: torch.Tensor):
        ctx.save_for_backward(euler_angles)
        return quat_from_euler_angles_fwd(euler_angles)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        euler_angles = ctx.saved_tensors[0]
        return apply_grad_mask(ctx, quat_from_euler_angles_bwd(euler_angles, last_dim_ctg(grad_out)))

@cpp_op()
def quat_from_euler_angles(roll_pitch_yaw: torch.Tensor) -> torch.Tensor:
    args, req_grad = broadcast_args([roll_pitch_yaw])
    return (QuatFromEulerAngles.apply if req_grad else quat_from_euler_angles_fwd)(*args)



class QuatToAxisAngle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, quat: torch.Tensor):
        ctx.save_for_backward(quat)
        return quat_to_axis_angle_fwd(quat)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        quat = ctx.saved_tensors[0]
        return apply_grad_mask(ctx, quat_to_axis_angle_bwd(quat, last_dim_ctg(grad_out)))

@cpp_op()
def quat_to_axis_angle(quat: torch.Tensor) -> torch.Tensor:
    args, req_grad = broadcast_args([quat])
    return (QuatToAxisAngle.apply if req_grad else quat_to_axis_angle_fwd)(*args)


class QuatToMatrix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, quat: torch.Tensor):
        ctx.save_for_backward(quat)
        return quat_to_matrix_fwd(quat)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        quat = ctx.saved_tensors[0]
        return apply_grad_mask(ctx, quat_to_matrix_bwd(quat, mtx_last_dim_ctg(grad_out)))

@cpp_op()
def quat_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    args, req_grad = broadcast_args([quat])
    return (QuatToMatrix.apply if req_grad else quat_to_matrix_fwd)(*args)


class QuatFromMatrix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mtx: torch.Tensor):
        ctx.save_for_backward(mtx)
        return quat_from_matrix_fwd(mtx)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        mtx = ctx.saved_tensors[0]
        return apply_grad_mask(ctx, quat_from_matrix_bwd(mtx, last_dim_ctg(grad_out)))

@cpp_op()
def quat_from_matrix(mtx: torch.Tensor) -> torch.Tensor:
    args, req_grad = broadcast_args([mtx], mtx_args=set([0]))
    return (QuatFromMatrix.apply if req_grad else quat_from_matrix_fwd)(*args)


class QuatStd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, quat: torch.Tensor):
        ctx.save_for_backward(quat)
        return quat_std_fwd(quat)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        quat = ctx.saved_tensors[0]
        return apply_grad_mask(ctx, quat_std_bwd(quat, last_dim_ctg(grad_out)))

@cpp_op()
def quat_std(quat: torch.Tensor) -> torch.Tensor:
    args, req_grad = broadcast_args([quat])
    return (QuatStd.apply if req_grad else quat_std_fwd)(*args)


class QuatInverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, quat: torch.Tensor):
        ctx.save_for_backward(quat)
        return quat_inverse_fwd(quat)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        quat = ctx.saved_tensors[0]
        return apply_grad_mask(ctx, quat_inverse_bwd(quat, last_dim_ctg(grad_out)))


@cpp_op()
def quat_inverse(quat: torch.Tensor) -> torch.Tensor:
    args, req_grad = broadcast_args([quat])
    return (QuatInverse.apply if req_grad else quat_inverse_fwd)(*args)


class QuatConjugate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, quat: torch.Tensor):
        return quat_conjugate_fwd(quat)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        return apply_grad_mask(ctx, quat_conjugate_bwd(last_dim_ctg(grad_out)))

@cpp_op()
def quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    args, req_grad = broadcast_args([quat])
    return (QuatConjugate.apply if req_grad else quat_conjugate_fwd)(*args)


class QuatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor):
        ctx.save_for_backward(a, b)
        return quat_mul_fwd(a, b)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        a, b = ctx.saved_tensors
        return apply_grad_mask(ctx, quat_mul_bwd(a, b, last_dim_ctg(grad_out)))

@cpp_op()
def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    args, req_grad = broadcast_args([a, b])
    return (QuatMul.apply if req_grad else quat_mul_fwd)(*args)


class QuatApply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, quat: torch.Tensor, p: torch.Tensor):
        ctx.save_for_backward(quat, p)
        return quat_apply_fwd(quat, p)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        quat, p = ctx.saved_tensors
        return apply_grad_mask(ctx, quat_apply_bwd(quat, p, last_dim_ctg(grad_out)))

@cpp_op()
def quat_apply(quat: torch.Tensor, p: torch.Tensor):
    args, req_grad = broadcast_args([quat, p])
    return (QuatApply.apply if req_grad else quat_apply_fwd)(*args)


class QuatApplyInv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, quat: torch.Tensor, p: torch.Tensor):
        ctx.save_for_backward(quat, p)
        return quat_apply_inv_fwd(quat, p)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        quat, p = ctx.saved_tensors
        return apply_grad_mask(ctx, quat_apply_inv_bwd(quat, p, last_dim_ctg(grad_out)))

@cpp_op()
def quat_apply_inv(quat: torch.Tensor, p: torch.Tensor):
    args, req_grad = broadcast_args([quat, p])
    return (QuatApplyInv.apply if req_grad else quat_apply_inv_fwd)(*args)


class QuatSlerp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q1: torch.Tensor, q2: torch.Tensor, t: torch.Tensor):
        ctx.save_for_backward(q1, q2, t)
        return quat_slerp_fwd(q1, q2, t)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        q1, q2, t = ctx.saved_tensors
        return apply_grad_mask(ctx, quat_slerp_bwd(q1, q2, t, last_dim_ctg(grad_out)))

@cpp_op()
def quat_slerp(q1: torch.Tensor, q2: torch.Tensor, t: torch.Tensor):
    args, req_grad = broadcast_args([q1, q2, t])
    return (QuatSlerp.apply if req_grad else quat_slerp_fwd)(*args)


@cpp_op()
def quat_random(
        *dims,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = 'cpu',
        generator: torch.Generator = None
) -> torch.Tensor:
    rand_vals = torch.rand(
        *dims, 3,
        dtype=dtype,
        device=device,
        generator=generator
    )
    return random_quat_from_uniform(rand_vals)


class QuatFromXYZW(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyzw: torch.Tensor):
        return quat_from_xyzw_fwd(xyzw)

    @staticmethod
    def backward(ctx, grad_quat: torch.Tensor):
        return apply_grad_mask(ctx, quat_from_xyzw_bwd(last_dim_ctg(grad_quat)))

@cpp_op()
def quat_from_xyzw(xyzw: torch.Tensor):
    args, req_grad = broadcast_args([xyzw])
    return (QuatFromXYZW.apply if req_grad else quat_from_xyzw_fwd)(*args)


@cpp_op()
def quat_unit(*dims, dtype: torch.dtype = torch.float32, device: torch.device | str = 'cpu') -> torch.Tensor:
    q = torch.empty(*dims, 4, dtype=dtype, device=device)
    return quat_unit_(q)
