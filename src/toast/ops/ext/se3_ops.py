import torch

from toast._C import (
    se3_apply_fwd,
    se3_apply_bwd,

    se3_apply_inv_fwd,
    se3_apply_inv_bwd,

    se3_inverse_fwd,
    se3_inverse_bwd,

    se3_compose_fwd,
    se3_compose_bwd,

    se3_slerp_fwd,
    se3_slerp_bwd,

    se3_to_matrix_3x4_fwd,
    se3_to_matrix_3x4_bwd,

    se3_to_matrix_4x4_fwd,
    se3_to_matrix_4x4_bwd,

    se3_from_matrix_fwd,
    se3_from_matrix_bwd,

    broadcast_args,
)

from ..backend import cpp_op


class SE3Apply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, t: torch.Tensor, p: torch.Tensor):
        ctx.save_for_backward(q, t, p)
        return se3_apply_fwd(q, t, p)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        q, t, p = ctx.saved_tensors
        return se3_apply_bwd(q, t, p, grad_out.contiguous())


@cpp_op()
def se3_apply(q: torch.Tensor, t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    args, req_grad = broadcast_args([q, t, p])
    return (SE3Apply.apply if req_grad else se3_apply_fwd)(*args)


class SE3ApplyInv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, t: torch.Tensor, p: torch.Tensor):
        ctx.save_for_backward(q, t, p)
        return se3_apply_inv_fwd(q, t, p)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        q, t, p = ctx.saved_tensors
        return se3_apply_inv_bwd(q, t, p, grad_out.contiguous())

@cpp_op()
def se3_apply_inv(q: torch.Tensor, t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    args, req_grad = broadcast_args([q, t, p])
    return (SE3ApplyInv.apply if req_grad else se3_apply_inv_fwd)(*args)


class SE3Inverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, t: torch.Tensor):
        ctx.save_for_backward(q, t)
        return se3_inverse_fwd(q, t)

    @staticmethod
    def backward(ctx, grad_q: torch.Tensor, grad_t: torch.Tensor):
        q, t = ctx.saved_tensors
        return se3_inverse_bwd(q, t, grad_q.contiguous(), grad_t.contiguous())

@cpp_op()
def se3_inverse(q: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    args, req_grad = broadcast_args([q, t])
    return (SE3Inverse.apply if req_grad else se3_inverse_fwd)(*args)


class SE3Compose(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q1: torch.Tensor, t1: torch.Tensor, q2: torch.Tensor, t2: torch.Tensor):
        ctx.save_for_backward(q1, t1, q2, t2)
        return se3_compose_fwd(q1, t1, q2, t2)

    @staticmethod
    def backward(ctx, grad_q: torch.Tensor, grad_t: torch.Tensor):
        q1, t1, q2, t2 = ctx.saved_tensors
        return se3_compose_bwd(q1, t1, q2, t2, grad_q.contiguous(), grad_t.contiguous())

@cpp_op()
def se3_compose(q1: torch.Tensor, t1: torch.Tensor, q2: torch.Tensor, t2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    args, req_grad = broadcast_args([q1, t1, q2, t2])
    return (SE3Compose.apply if req_grad else se3_compose_fwd)(*args)


class SE3Slerp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q1: torch.Tensor, t1: torch.Tensor, q2: torch.Tensor, t2: torch.Tensor, weight: torch.Tensor):
        ctx.save_for_backward(q1, t1, q2, t2, weight)
        return se3_slerp_fwd(q1, t1, q2, t2, weight)

    @staticmethod
    def backward(ctx, grad_q: torch.Tensor, grad_t: torch.Tensor):
        q1, t1, q2, t2, weight = ctx.saved_tensors
        return se3_slerp_bwd(q1, t1, q2, t2, weight, grad_q.contiguous(), grad_t.contiguous())

@cpp_op()
def se3_slerp(
        q1: torch.Tensor,
        t1: torch.Tensor,
        q2: torch.Tensor,
        t2: torch.Tensor,
        weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    args, req_grad = broadcast_args([q1, t1, q2, t2, weight])
    return (SE3Slerp.apply if req_grad else se3_slerp_fwd)(*args)


class SE3ToMatrix3x4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, t: torch.Tensor):
        ctx.save_for_backward(q, t)
        return se3_to_matrix_3x4_fwd(q, t)

    @staticmethod
    def backward(ctx, grad_mtx: torch.Tensor):
        q, t = ctx.saved_tensors
        return se3_to_matrix_3x4_bwd(q, t, grad_mtx.contiguous())

@cpp_op()
def se3_to_matrix_3x4(q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    args, req_grad = broadcast_args([q, t])
    return (SE3ToMatrix3x4.apply if req_grad else se3_to_matrix_3x4_fwd)(*args)


class SE3ToMatrix4x4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, t: torch.Tensor):
        ctx.save_for_backward(q, t)
        return se3_to_matrix_4x4_fwd(q, t)

    @staticmethod
    def backward(ctx, grad_mtx: torch.Tensor):
        q, t = ctx.saved_tensors
        return se3_to_matrix_4x4_bwd(q, t, grad_mtx.contiguous())

@cpp_op()
def se3_to_matrix_4x4(q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    args, req_grad = broadcast_args([q, t])
    return (SE3ToMatrix4x4.apply if req_grad else se3_to_matrix_4x4_fwd)(*args)


class SE3FromMatrix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mtx: torch.Tensor):
        ctx.save_for_backward(mtx)
        return se3_from_matrix_fwd(mtx)

    @staticmethod
    def backward(ctx, grad_q: torch.Tensor, grad_t: torch.Tensor):
        mtx = ctx.saved_tensors[0]
        return se3_from_matrix_bwd(mtx, grad_q.contiguous(), grad_t.contiguous())

@cpp_op()
def se3_from_matrix(mtx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    args, req_grad = broadcast_args([mtx], mtx_args=set([0]))
    return (SE3FromMatrix.apply if req_grad else se3_from_matrix_fwd)(*args)
