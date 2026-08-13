import torch

from .. import apply_velocity, linear_motion_transform
from ..utils import DataTensor

from .quat import Quat
from .se3 import SE3


def fit_linear_velocity(p: torch.Tensor, t: torch.Tensor, p0, t0):
    N = p.shape[0]
    assert p.ndim == 2 and p.shape[-1] == 3, "p must be [N, 3]"
    assert t.shape == [N, 1], "t must be [N, 1]"
    assert ((p0.numel() == 3) and (p0.ndim == 1)), "p0 must be [3,]"
    dt = t - t0 # [N, 1]
    dp = p - p0 # [N, 3]
    return (dt * dp).sum(dim=0) / (dt * dt).sum()


def fit_position_and_linear_velocity(p: torch.Tensor, t: torch.Tensor, t0: torch.Tensor):
    assert p.ndim == 2 and p.shape[-1] == 3, "p must be [N, 3]"
    assert t.ndim == 2 and t.shape[-1] == 1 and t.shape[0] == p.shape[0], "t must be [N, 1]"
    dt = t - t0
    mtx = torch.cat([torch.ones_like(dt), dt], dim=1)
    solution = torch.linalg.lstsq(mtx, p).solution  # (2,3)
    return solution[0], solution[1]  # p0, v


def fit_angular_velocity(q: Quat, t: torch.Tensor, q0: Quat, t0: torch.Tensor):
    assert q.dim == 1, "Quat must be [N, ]"
    assert t.ndim == 2 and t.shape[0] == len(q) and t.shape[1] == 1, "t must be [N, 1]"
    assert q0.numel == 1, "q0 must be [1,]"
    assert t0.numel() == 1, "t0 must be [1,]"

    omega_rel = (q @ q0.inv().view(1)).std().to_axis_angle() # [N, 3]
    dt = t - t0 # [N, 1]
    return (omega_rel * dt).sum(dim=0) / (dt * dt).sum()



class LinearMotion(DataTensor):
    r"""A rigid body motion with a constant linear and angular velocity.

    :class:`LinearMotion` pairs an SE(3) reference pose with a timestamp and
    two constant velocity fields.  Given the reference pose
    :math:`(q_0, \mathbf{t}_0)` known at time :math:`t_0`, the pose at any
    other time :math:`t` is obtained by first-order (Euler) integration:

    .. math::

        \Delta t = t - t_0, \qquad
        q(t) = \operatorname{from\_axis\_angle}(\boldsymbol{\omega}\,\Delta t)
               \cdot q_0, \qquad
        \mathbf{t}(t) = \mathbf{t}_0 + \mathbf{v}\,\Delta t

    **Velocity conventions**

    * :attr:`linear_velocity` :math:`\mathbf{v}` is expressed in the
      *target* coordinate frame — it is the rate of change of the target-frame
      translation :math:`\mathbf{t}`.
    * :attr:`angular_velocity` :math:`\boldsymbol{\omega}` is an axis-angle
      vector whose direction is the target-frame rotation axis and whose
      magnitude is the angular speed in radians per time unit.

    Example::

        import torch
        from toast import SE3, LinearMotion
	
	# Transformation from body coordinate system to world
        world_se3_body = SE3.unit(1)
        # Linear velocity in target (world) coordinate system
        v_world = torch.tensor([[1., 0., 0.]])   # 1 m/s along +X
        # Angular velocity
        w_world = torch.zeros(1, 3)
        motion = LinearMotion(pose, torch.tensor([0.0]), v_world, w_world)

        pose_at_2s = motion.pose_at(torch.tensor([2.0]))
        # pose_at_2s.t = tensor([[2., 0., 0.]])
    """

    _buffers = [
        "pose",
        "pose_time",
        "linear_velocity",
        "angular_velocity"
    ]

    #: Reference SE(3) pose :math:`(q_0, \mathbf{t}_0)`, batch shape ``(...,)``.
    pose: SE3
    #: Timestamp :math:`t_0` at which :attr:`pose` is valid, shape ``(..., 1)``.
    pose_time: torch.Tensor
    #: Target-frame (which ``pose`` transforms into) linear velocity :math:`\mathbf{v}`, shape ``(..., 3)``, in world units per time unit.
    linear_velocity: torch.Tensor
    #: Target-frame (which ``pose`` transforms into) angular velocity :math:`\boldsymbol{\omega}` in axis-angle form, shape ``(..., 3)``, in radians per time unit.
    angular_velocity: torch.Tensor

    def __init__(
            self,
            pose: SE3,
            pose_time: torch.Tensor,
            linear_velocity: torch.Tensor,
            angular_velocity: torch.Tensor
    ):
        r"""Construct a :class:`LinearMotion` from its components.

        Args:
            pose: Reference SE(3) pose :math:`(q_0, \mathbf{t}_0)`.
            pose_time: Timestamp :math:`t_0` at which *pose* is valid,
                shape ``(..., 1)``.
            linear_velocity: World-frame linear velocity :math:`\mathbf{v}`,
                shape ``(..., 3)``.
            angular_velocity: World-frame angular velocity
                :math:`\boldsymbol{\omega}` in axis-angle form, shape
                ``(..., 3)``.
        """
        self.pose = pose
        self.pose_time = pose_time
        self.linear_velocity = linear_velocity
        self.angular_velocity = angular_velocity

    @staticmethod
    def fit(poses: SE3, time: torch.Tensor) -> "LinearMotion":
        r"""
        Given [N,] SE3 poses and [N, 1] poses time tensor estimates parameters
        of motion with constant linear and angular velocities to approximate
        given series of poses.

        Args:
            poses: Observed SE3 poses of shape [N, ]
            time: Time at which given poses were observed. Tensor of [N, 1]
                shape
        Returns:
            Estimated LinearMotion. Resulting ``pose_time`` equals to median of
            given ``time`` tensor.
        """
        ref_idx = torch.argsort(time.flatten())[len(time) // 2]
        ref_pose = poses[ref_idx].std()
        ref_time = time[ref_idx]

        p0, v = fit_position_and_linear_velocity(poses.t, time, ref_time)
        ref_pose.t = p0
        w = fit_angular_velocity(poses.q, time, ref_pose.q, ref_time)

        return LinearMotion(
            ref_pose,
            ref_time,
            linear_velocity=v,
            angular_velocity=w
        )

    def lerp(self, other: "LinearMotion", t: torch.Tensor) -> "LinearMotion":
        r"""Linearly interpolate between two :class:`LinearMotion` objects.

        All four fields — :attr:`pose`, :attr:`pose_time`,
        :attr:`linear_velocity`, and :attr:`angular_velocity` — are
        interpolated independently with weight :math:`t`:

        .. math::

            \text{result} = (1 - t)\,\text{self} + t\,\text{other}

        The pose is interpolated via :meth:`SE3.lerp
        <toast.transforms.SE3.lerp>`; the remaining scalar/vector fields
        use :func:`torch.lerp`.

        Args:
            other: Target :class:`LinearMotion`, must be broadcast-compatible
                with ``self``.
            t: Interpolation weight, shape ``(..., 1)`` with values in
                :math:`[0, 1]`.  ``t=0`` returns ``self``; ``t=1`` returns
                ``other``.

        Returns:
            Interpolated :class:`LinearMotion` with the same batch shape as
            ``self`` and ``other`` broadcast together.
        """
        return LinearMotion(
            pose=self.pose.lerp(other.pose, t),
            pose_time=torch.lerp(self.pose_time, other.pose_time, t),
            linear_velocity=torch.lerp(self.linear_velocity, other.linear_velocity, t),
            angular_velocity=torch.lerp(self.angular_velocity, other.angular_velocity, t)
        )

    def pose_at(self, t: torch.Tensor) -> SE3:
        r"""Extrapolate the pose to an arbitrary timestamp.

        Advances the reference pose :math:`(q_0, \mathbf{t}_0)` by
        :math:`\Delta t = t - t_0` using constant-velocity integration:

        .. math::

            q(t) = \operatorname{from\_axis\_angle}(\boldsymbol{\omega}\,\Delta t)
                   \cdot q_0, \qquad
            \mathbf{t}(t) = \mathbf{t}_0 + \mathbf{v}\,\Delta t

        Args:
            t: Query timestamp(s), shape ``(..., 1)``.  Broadcast-compatible
                with the batch shape of ``self``.

        Returns:
            SE(3) pose at time *t*, with batch shape matching the broadcast
            of ``self`` and *t*.
        """
        return SE3(*apply_velocity(
            self.pose.q.std().data,
            self.pose.t,
            self.linear_velocity,
            self.angular_velocity,
            t - self.pose_time
        ))

    def apply_to_points(
            self,
            points: torch.Tensor,
            points_time: torch.Tensor
    ) -> torch.Tensor:
        r"""Transform points into the world frame using per-point timestamps.

        Each point is transformed by the pose extrapolated to that point's
        own timestamp.  For a point :math:`\mathbf{p}` with timestamp
        :math:`t_p`:

        .. math::

            \Delta t = t_p - t_0, \qquad
            (q, \mathbf{t}) = \operatorname{apply\_velocity}(q_0,\,\mathbf{t}_0,\,
                              \mathbf{v},\,\boldsymbol{\omega},\,\Delta t), \qquad
            \mathbf{p}_\text{out} = q \cdot \mathbf{p} + \mathbf{t}

        This is useful for motion-compensating a point cloud where each point
        carries its own acquisition timestamp.

        Args:
            points: Points to transform, shape ``(..., 3)``.
            points_time: Per-point timestamps, shape ``(..., 1)``.  Batch
                dimensions are broadcast against ``points`` and against the
                batch shape of ``self``.  To transform a ``(..., N, 3)`` cloud
                with a ``(...,)`` batch of motions, add the point axis to the
                motion first with ``self.unsqueeze(-1)``.

        Returns:
            Transformed points in the world frame, shape ``(..., 3)``.
        """
        return linear_motion_transform(
            self.pose.q.std().data,
            self.pose.t,
            self.linear_velocity,
            self.angular_velocity,
            self.pose_time,
            points,
            points_time
        )

    def transform(self, trf: SE3) -> "LinearMotion":
        return LinearMotion(
            pose=trf @ self.pose,
            pose_time=self.pose_time,
            linear_velocity=trf.q @ self.linear_velocity,
            angular_velocity=trf.q @ self.angular_velocity
        )
