import torch

from toast._C import (
    velocities_fwd,
    velocities_bwd,

    linear_velocity_fwd,
    linear_velocity_bwd,

    angular_velocity_fwd,
    angular_velocity_bwd,

    apply_velocity_fwd,
    apply_velocity_bwd,

    apply_angular_velocity_fwd,
    apply_angular_velocity_bwd,

    linear_motion_transform_fwd,
    linear_motion_transform_bwd,
)

from ..backend import cpp_op
from ..utils import broadcast_args
from .ops_utils import last_dim_ctg, apply_grad_mask


class Velocities(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            q1: torch.Tensor,
            t1: torch.Tensor,
            q2: torch.Tensor,
            t2: torch.Tensor,
            time1: torch.Tensor,
            time2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx.save_for_backward(q1, t1, q2, t2, time1, time2)
        return velocities_fwd(q1, t1, q2, t2, time1, time2)

    @staticmethod
    def backward(ctx, grad_linear_velocity, grad_angular_velocity):
        return apply_grad_mask(ctx, velocities_bwd(*ctx.saved_tensors, last_dim_ctg(grad_linear_velocity), last_dim_ctg(grad_angular_velocity)))


@cpp_op()
def velocities(
        q1: torch.Tensor,
        t1: torch.Tensor,
        q2: torch.Tensor,
        t2: torch.Tensor,
        time1: torch.Tensor,
        time2: torch.Tensor,
):
    args, req_grad = broadcast_args([q1, t1, q2, t2, time1, time2])
    return (Velocities.apply if req_grad else velocities_fwd)(*args)


class LinearVelocity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t1: torch.Tensor, t2: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(t1, t2, dt)
        return linear_velocity_fwd(t1, t2, dt)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        return apply_grad_mask(ctx, linear_velocity_bwd(*ctx.saved_tensors, last_dim_ctg(grad_out)))

@cpp_op()
def linear_velocity(t1: torch.Tensor, t2: torch.Tensor, delta_time: torch.Tensor) -> torch.Tensor:
    args, req_grad = broadcast_args([t1, t2, delta_time])
    return (LinearVelocity.apply if req_grad else linear_velocity_fwd)(*args)


class AngularVelocity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q1: torch.Tensor, q2: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(q1, q2, dt)
        return angular_velocity_fwd(q1, q2, dt)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        return apply_grad_mask(ctx, angular_velocity_bwd(*ctx.saved_tensors, last_dim_ctg(grad_out)))

@cpp_op()
def angular_velocity(q1: torch.Tensor, q2: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    args, req_grad = broadcast_args([q1, q2, dt])
    return (AngularVelocity.apply if req_grad else angular_velocity_fwd)(*args)


class ApplyAngularVelocity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q0: torch.Tensor, w: torch.Tensor, delta_time: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(q0, w, delta_time)
        return apply_angular_velocity_fwd(q0, w, delta_time)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        return apply_grad_mask(ctx, apply_angular_velocity_bwd(*ctx.saved_tensors, last_dim_ctg(grad_out)))

@cpp_op()
def apply_angular_velocity(q0: torch.Tensor, w: torch.Tensor, delta_time: torch.Tensor) -> torch.Tensor:
    args, req_grad = broadcast_args([q0, w, delta_time])
    return (ApplyAngularVelocity.apply if req_grad else apply_angular_velocity_fwd)(*args)


class ApplyVelocity(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            q0: torch.Tensor,
            t0: torch.Tensor,
            v: torch.Tensor,
            w: torch.Tensor,
            delta_time: torch.Tensor,
    ):
        ctx.save_for_backward(q0, t0, v, w, delta_time)
        return apply_velocity_fwd(q0, t0, v, w, delta_time)

    @staticmethod
    def backward(ctx, grad_q: torch.Tensor, grad_t: torch.Tensor):
        return apply_grad_mask(ctx, apply_velocity_bwd(*ctx.saved_tensors, last_dim_ctg(grad_q), last_dim_ctg(grad_t)))

@cpp_op()
def apply_velocity(
        q0: torch.Tensor,
        t0: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        delta_time: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    args, req_grad = broadcast_args([q0, t0, v, w, delta_time])
    return (ApplyVelocity.apply if req_grad else apply_velocity_fwd)(*args)


class LinearMotionTransform(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            q0: torch.Tensor,
            t0: torch.Tensor,
            v: torch.Tensor,
            w: torch.Tensor,
            pose_time: torch.Tensor,
            points: torch.Tensor,
            points_time: torch.Tensor
    ):
        ctx.save_for_backward(q0, t0, v, w, pose_time, points, points_time)
        return linear_motion_transform_fwd(q0, t0, v, w, pose_time, points, points_time)

    @staticmethod
    def backward(
            ctx,
            grad_output
    ):
        return apply_grad_mask(ctx, linear_motion_transform_bwd(
            *ctx.saved_tensors, last_dim_ctg(grad_output)
        ))


@cpp_op()
def linear_motion_transform(
        q0: torch.Tensor,
        t0: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        pose_time: torch.Tensor,
        points: torch.Tensor,
        points_time: torch.Tensor
):
    args, req_grad = broadcast_args([q0, t0, v, w, pose_time, points, points_time])
    return (LinearMotionTransform.apply if req_grad else linear_motion_transform_fwd)(*args)
