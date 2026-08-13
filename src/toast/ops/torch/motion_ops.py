from ..backend import torch_op
from ..utils import broadcast_args_torch
from . import quat_from_axis_angle, se3_apply
from .quat_ops import quat_std, quat_mul, quat_inverse, quat_to_axis_angle


@torch_op()
def linear_velocity(t1, t2, delta_time):
    t1, t2, delta_time = broadcast_args_torch([t1, t2, delta_time])
    return (t2 - t1) / delta_time

@torch_op()
def angular_velocity(q1, q2, delta_time):
    q1, q2, delta_time = broadcast_args_torch([q1, q2, delta_time])
    return quat_to_axis_angle(quat_std(quat_mul(q2, quat_inverse(q1)))) / delta_time


@torch_op()
def velocities(q1, t1, q2, t2, time1, time2):
    q1, t1, q2, t2, time1, time2 = broadcast_args_torch([q1, t1, q2, t2, time1, time2])
    dt = time2 - time1
    return linear_velocity(t1, t2, dt), angular_velocity(q1, q2, dt)


@torch_op()
def apply_angular_velocity(q0, w, delta_time):
    q0, w, delta_time = broadcast_args_torch([q0, w, delta_time])
    return quat_mul(quat_from_axis_angle(w * delta_time), q0)


@torch_op()
def apply_velocity(q0, t0, v, w, delta_time):
    q0, t0, v, w, delta_time = broadcast_args_torch([q0, t0, v, w, delta_time])
    return apply_angular_velocity(q0, w, delta_time), t0 + delta_time * v


@torch_op()
def linear_motion_transform(q0, t0, v, w, pose_time, points, points_time):
    q0, t0, v, w, pose_time, points, points_time = broadcast_args_torch(
        [q0, t0, v, w, pose_time, points, points_time]
    )
    delta_time = points_time - pose_time
    q, t = apply_velocity(q0, t0, v, w, delta_time)
    return se3_apply(q, t, points)
