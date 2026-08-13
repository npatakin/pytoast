r"""Unified motion / velocity operations dispatched to the active backend.

These operations compute velocities from consecutive SE(3) poses and
integrate velocities forward in time.

Conventions:

- Poses are represented as :math:`(q, t)` pairs — see :mod:`toast.ops.se3_ops`.
- Linear velocity :math:`v` has shape ``(..., 3)`` in world units per time unit.
- Angular velocity :math:`\omega` has shape ``(..., 3)`` in axis-angle form:
  the vector direction is the rotation axis and its magnitude is the angular
  speed in radians per time unit.
"""

import torch

from .backend import get_backend


def linear_velocity(
    t1: torch.Tensor,
    t2: torch.Tensor,
    delta_time: torch.Tensor,
) -> torch.Tensor:
    r"""Compute linear velocity from two translations and a time delta.

    Finite-difference estimate:

    .. math::

        v = \frac{t_2 - t_1}{\Delta t}

    Args:
        t1: Starting translation(s), shape ``(..., 3)``.
        t2: Ending translation(s), shape ``(..., 3)``.
        delta_time: Time difference :math:`\Delta t`, shape ``(..., 1)``.

    Returns:
        Linear velocity :math:`v`, shape ``(..., 3)``.

    Example::

        >>> t1 = torch.tensor([0., 0., 0.])
        >>> t2 = torch.tensor([3., 0., 0.])
        >>> linear_velocity(t1, t2, torch.tensor([1.5]))  # tensor([2., 0., 0.])
    """
    return get_backend().linear_velocity(t1, t2, delta_time)


def angular_velocity(
    q1: torch.Tensor,
    q2: torch.Tensor,
    delta_time: torch.Tensor,
) -> torch.Tensor:
    r"""Compute angular velocity from two orientations and a time delta.

    Returns the angular velocity as an axis-angle vector:

    .. math::

        \omega = \frac{\operatorname{axis\_angle}(q_2\, q_1^{-1})}{\Delta t}

    where :math:`\operatorname{axis\_angle}(q)` converts a unit quaternion
    to its axis-angle representation.

    Args:
        q1: Starting quaternion(s), shape ``(..., 4)`` in WXYZ order.
        q2: Ending quaternion(s), shape ``(..., 4)`` in WXYZ order.
        delta_time: Time difference :math:`\Delta t`, shape ``(..., 1)``.

    Returns:
        Angular velocity :math:`\omega`, shape ``(..., 3)``, in radians
        per time unit.

    Example::

        >>> from toast.ops.quat_ops import quat_from_axis_angle
        >>> q1 = quat_from_axis_angle(torch.zeros(3))
        >>> q2 = quat_from_axis_angle(torch.tensor([0., 0., torch.pi / 2]))
        >>> angular_velocity(q1, q2, torch.tensor([1.0]))  # ≈ tensor([0., 0., π/2])
    """
    return get_backend().angular_velocity(q1, q2, delta_time)


def velocities(
    q1: torch.Tensor,
    t1: torch.Tensor,
    q2: torch.Tensor,
    t2: torch.Tensor,
    time1: torch.Tensor,
    time2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Compute linear and angular velocities between two SE(3) poses.

    Finite-difference estimate given two timestamped poses:

    .. math::

        \Delta t = t_2 - t_1, \qquad
        v = \frac{t_2 - t_1}{\Delta t}, \qquad
        \omega = \frac{\operatorname{axis\_angle}(q_2\, q_1^{-1})}{\Delta t}

    Batch dimensions of all inputs are broadcast against each other.

    Args:
        q1: Quaternion of the first pose, shape ``(..., 4)`` in WXYZ order.
        t1: Translation of the first pose, shape ``(..., 3)``.
        q2: Quaternion of the second pose, shape ``(..., 4)`` in WXYZ order.
        t2: Translation of the second pose, shape ``(..., 3)``.
        time1: Timestamp of the first pose, shape ``(..., 1)``.
        time2: Timestamp of the second pose, shape ``(..., 1)``.

    Returns:
        Tuple :math:`(v, \omega)` where:

        - :math:`v` — linear velocity, shape ``(..., 3)``.
        - :math:`\omega` — angular velocity in axis-angle form, shape
          ``(..., 3)``, in radians per time unit.

    Example::

        >>> v, w = velocities(q1, t1, q2, t2,
        ...                   torch.tensor([0.0]), torch.tensor([0.5]))
    """
    return get_backend().velocities(q1, t1, q2, t2, time1, time2)


def apply_angular_velocity(
    q0: torch.Tensor,
    w: torch.Tensor,
    delta_time: torch.Tensor,
) -> torch.Tensor:
    r"""Advance a rotation by an angular velocity over a time step.

    Integrates :math:`\omega` for :math:`\Delta t` seconds and composes the
    result with the initial rotation:

    .. math::

        q_\text{new} = \operatorname{quat\_from\_axis\_angle}(\omega\,\Delta t)\cdot q_0

    Args:
        q0: Initial orientation, shape ``(..., 4)`` in WXYZ order.
        w: Angular velocity :math:`\omega` in axis-angle form,
           shape ``(..., 3)``, radians per time unit.
        delta_time: Integration time step :math:`\Delta t`, shape ``(..., 1)``.

    Returns:
        Updated orientation, shape ``(..., 4)`` in WXYZ order.

    Example::

        >>> q0 = quat_unit(1)
        >>> w = torch.tensor([[0., 0., torch.pi]])   # π rad/s around Z
        >>> q1 = apply_angular_velocity(q0, w, torch.tensor([1.0]))  # 180° around Z
    """
    return get_backend().apply_angular_velocity(q0, w, delta_time)


def apply_velocity(
    q0: torch.Tensor,
    t0: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    delta_time: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Advance an SE(3) pose by linear and angular velocities.

    First-order (Euler) integration of both velocities:

    .. math::

        \begin{aligned}
        q_\text{new} &= \operatorname{quat\_from\_axis\_angle}(\omega\,\Delta t)\cdot q_0 \\
        t_\text{new} &= t_0 + v\,\Delta t
        \end{aligned}

    Accurate when :math:`\Delta t` is small relative to the rate of change
    of the velocities.

    Batch dimensions of all inputs are broadcast against each other.

    Args:
        q0: Initial orientation, shape ``(..., 4)`` in WXYZ order.
        t0: Initial translation, shape ``(..., 3)``.
        v: Linear velocity :math:`v`, shape ``(..., 3)``, world units per
           time unit.
        w: Angular velocity :math:`\omega` in axis-angle form, shape
           ``(..., 3)``, radians per time unit.
        delta_time: Integration time step :math:`\Delta t`, shape ``(..., 1)``.

    Returns:
        Tuple :math:`(q_\text{new}, t_\text{new})` representing the
        advanced pose.

    Example::

        >>> q0 = quat_unit(1)
        >>> t0 = torch.zeros(1, 3)
        >>> v = torch.tensor([[1., 0., 0.]])   # 1 unit/s in +X
        >>> w = torch.zeros(1, 3)
        >>> q1, t1 = apply_velocity(q0, t0, v, w, torch.tensor([2.0]))
        >>> t1  # tensor([[2., 0., 0.]])
    """
    return get_backend().apply_velocity(q0, t0, v, w, delta_time)


def linear_motion_transform(
    q0: torch.Tensor,
    t0: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    pose_time: torch.Tensor,
    points: torch.Tensor,
    points_time: torch.Tensor,
) -> torch.Tensor:
    r"""Transform points assuming constant-velocity motion.
    Initial pose at ``pose_time`` is defined by ``(q0, t0)``.
    Linear velocity ``v`` is assumed to be in resulting coordinate system

    Advances the pose :math:`(q_0, t_0)` from ``pose_time`` to each
    point's ``points_time`` using constant velocities, then applies the
    resulting SE(3) transform to the corresponding point:

    .. math::

        \Delta t = t_\text{point} - t_\text{pose}, \qquad
        (q, t) = \operatorname{apply\_velocity}(q_0, t_0, v, \omega, \Delta t), \qquad
        p_\text{out} = q \cdot p + t

    Batch dimensions of all inputs are broadcast against each other.

    Args:
        q0: Reference orientation, shape ``(..., 4)`` in WXYZ order.
        t0: Reference translation, shape ``(..., 3)``.
        v: Linear velocity :math:`v`, shape ``(..., 3)``, world units per
           time unit.
        w: Angular velocity :math:`\omega` in axis-angle form, shape
           ``(..., 3)``, radians per time unit.
        pose_time: Timestamp at which :math:`(q_0, t_0)` is valid, shape
            ``(..., 1)``.
        points: Points to transform, shape ``(..., 3)``.
        points_time: Timestamp of each point, shape ``(..., 1)``.

    Returns:
        Transformed points, shape ``(..., 3)``.

    Example::

        >>> q0 = quat_unit(1)
        >>> t0 = torch.zeros(1, 3)
        >>> v = torch.tensor([[1., 0., 0.]])
        >>> w = torch.zeros(1, 3)
        >>> p = torch.tensor([[0., 0., 0.]])
        >>> linear_motion_transform(q0, t0, v, w,
        ...     torch.tensor([0.0]), p, torch.tensor([2.0]))
        tensor([[2., 0., 0.]])
    """
    return get_backend().linear_motion_transform(q0, t0, v, w, pose_time, points, points_time)
