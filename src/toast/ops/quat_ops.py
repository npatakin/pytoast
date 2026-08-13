r"""Unified quaternion operations dispatched to the active backend.

All quaternions use the **WXYZ convention**: the scalar (real) part is stored
first, followed by the three imaginary components :math:`(i, j, k)`.
A unit quaternion :math:`q = (w, x, y, z)` with :math:`\|q\| = 1` represents
a 3-D rotation of :math:`2\arccos(w)` radians around the axis
:math:`(x, y, z) / \|(x, y, z)\|`.

Call ``use_backend('cpp')`` or ``use_backend('torch')`` to select which
implementation is used.  The C++/CUDA backend is the default.
"""

from typing import Optional

import torch

from .backend import get_backend


def quat_unit(
    *dims: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    r"""Return a batch of identity (unit) quaternions.

    The identity quaternion represents a zero rotation:
    :math:`q = (1, 0, 0, 0)`.

    Args:
        *dims: Batch dimensions.  The output will have shape ``(*dims, 4)``.
        dtype: Floating-point dtype of the output.
        device: Target device.

    Returns:
        Tensor of shape ``(*dims, 4)`` filled with :math:`[1, 0, 0, 0]` along
        the last dimension.

    Example::

        >>> q = quat_unit(3)          # shape (3, 4)
        >>> q_batch = quat_unit(2, 5) # shape (2, 5, 4)
    """
    return get_backend().quat_unit(*dims, dtype=dtype, device=device)


def quat_random(
    *dims: int,
    dtype: Optional[torch.dtype] = torch.float32,
    device: Optional[torch.device | str] = "cpu",
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    r"""Sample uniformly random unit quaternions (i.e. uniform rotations).

    Uses Shoemake's method: given three uniform samples
    :math:`r_1, r_2, r_3 \in [0, 1)`, the quaternion components are

    .. math::

        q = \bigl(
            \sqrt{1 - r_1}\,\sin(2\pi r_2),\;
            \sqrt{1 - r_1}\,\cos(2\pi r_2),\;
            \sqrt{r_1}\,\sin(2\pi r_3),\;
            \sqrt{r_1}\,\cos(2\pi r_3)
        \bigr)

    which yields a uniform distribution over the 3-sphere.

    Args:
        *dims: Batch dimensions.  The output will have shape ``(*dims, 4)``.
        dtype: Floating-point dtype.  Defaults to ``torch.float32``.
        device: Target device.  Defaults to ``'cpu'``.
        generator: Optional ``torch.Generator`` for reproducibility.

    Returns:
        Tensor of shape ``(*dims, 4)`` containing unit quaternions.

    Example::

        >>> g = torch.Generator().manual_seed(0)
        >>> q = random_quat(100, generator=g)  # shape (100, 4)
        >>> torch.allclose((q * q).sum(-1), torch.ones(100))  # all unit
        True
    """
    return get_backend().quat_random(*dims, dtype=dtype, device=device, generator=generator)


def quat_from_xyzw(xyzw: torch.Tensor) -> torch.Tensor:
    r"""Convert quaternions from XYZW to WXYZ storage order.

    Many external libraries (e.g. SciPy, ROS) store quaternions as
    :math:`(x, y, z, w)`.  This function reorders them to the WXYZ
    convention used throughout toast, i.e. the output is
    :math:`(w, x, y, z)`.

    Args:
        xyzw: Tensor of shape ``(..., 4)`` with components in XYZW order.

    Returns:
        Tensor of shape ``(..., 4)`` with components in WXYZ order.

    Example::

        >>> import torch
        >>> q_xyzw = torch.tensor([0., 0., 0., 1.])  # identity in XYZW
        >>> quat_from_xyzw(q_xyzw)  # tensor([1., 0., 0., 0.])
    """
    return get_backend().quat_from_xyzw(xyzw)


def quat_std(quat: torch.Tensor) -> torch.Tensor:
    r"""Standardize quaternions so that the real (:math:`w`) component is non-negative.

    Because :math:`q` and :math:`-q` represent the same rotation, we pick
    the canonical representative with :math:`w \ge 0`.  The output is also
    :math:`\ell_2`-normalized:

    .. math::

        q_\text{std} = \frac{1}{\|q\|}
        \begin{cases} q & \text{if } w \ge 0 \\ -q & \text{otherwise} \end{cases}

    Args:
        quat: Tensor of shape ``(..., 4)`` in WXYZ order.

    Returns:
        Tensor of same shape with :math:`q_w \ge 0` and :math:`\|q\| = 1`.

    Example::

        >>> import torch
        >>> q = torch.tensor([-1., 0., 0., 0.])   # w < 0
        >>> quat_std(q)  # tensor([1., 0., 0., 0.])
    """
    return get_backend().quat_std(quat)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    r"""Compute the conjugate of a quaternion.

    For :math:`q = (w, x, y, z)` the conjugate is

    .. math::

        q^* = (w,\; -x,\; -y,\; -z).

    For a *unit* quaternion the conjugate equals the inverse, so this is an
    efficient way to invert rotations when :math:`\|q\| = 1`.

    Args:
        q: Tensor of shape ``(..., 4)`` in WXYZ order.

    Returns:
        Tensor of same shape with imaginary components negated.

    Note:
        Only available in the ``'torch'`` backend.

    Example::

        >>> import torch
        >>> q = torch.tensor([0.707, 0.707, 0., 0.])
        >>> quat_conjugate(q)  # tensor([0.707, -0.707, 0., 0.])
    """
    return get_backend().quat_conjugate(q)


def quat_inverse(quat: torch.Tensor) -> torch.Tensor:
    r"""Compute the multiplicative inverse of a quaternion.

    In general

    .. math::

        q^{-1} = \frac{q^*}{\|q\|^2}

    For unit quaternions this simplifies to the conjugate
    :math:`q^{-1} = q^* = (w, -x, -y, -z)`.

    The inverse satisfies :math:`q \cdot q^{-1} = q^{-1} \cdot q = (1, 0, 0, 0)`.

    Args:
        quat: Tensor of shape ``(..., 4)`` in WXYZ order.

    Returns:
        Tensor of same shape containing the inverse quaternion.

    Example::

        >>> q = random_quat(5)
        >>> q_inv = quat_inverse(q)
        >>> quat_mul(q, q_inv)  # close to identity (1, 0, 0, 0)
    """
    return get_backend().quat_inverse(quat)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    r"""Multiply two quaternions (Hamilton product).

    Given :math:`q_1 = (w_1, x_1, y_1, z_1)` and
    :math:`q_2 = (w_2, x_2, y_2, z_2)`:

    .. math::

        \begin{aligned}
        (q_1 q_2)_w &= w_1 w_2 - x_1 x_2 - y_1 y_2 - z_1 z_2 \\
        (q_1 q_2)_x &= w_1 x_2 + x_1 w_2 + y_1 z_2 - z_1 y_2 \\
        (q_1 q_2)_y &= w_1 y_2 - x_1 z_2 + y_1 w_2 + z_1 x_2 \\
        (q_1 q_2)_z &= w_1 z_2 + x_1 y_2 - y_1 x_2 + z_1 w_2
        \end{aligned}

    Composing unit quaternions via multiplication corresponds to composing
    rotations (apply :math:`q_2` first, then :math:`q_1`).

    Batch dimensions of ``a`` and ``b`` are broadcast against each other.

    Args:
        a: Tensor of shape ``(..., 4)`` in WXYZ order.
        b: Tensor of shape ``(..., 4)`` in WXYZ order.

    Returns:
        Tensor of shape ``(..., 4)`` containing the Hamilton product.

    Example::

        >>> import torch
        >>> rx = quat_from_axis_angle(torch.tensor([torch.pi/2, 0., 0.]))
        >>> rz = quat_from_axis_angle(torch.tensor([0., 0., torch.pi/2]))
        >>> q = quat_mul(rx, rz)  # rotate by rz first, then rx
    """
    return get_backend().quat_mul(a, b)


def quat_apply(quat: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    r"""Rotate a 3-D point (or batch of points) by a quaternion.

    Computes the sandwich product

    .. math::

        p' = q \otimes [0,\, p] \otimes q^{-1}

    which expands to

    .. math::

        p' = p + 2w\,(\mathbf{v} \times p) + 2\,(\mathbf{v} \times (\mathbf{v} \times p))

    where :math:`q = (w, \mathbf{v})` and :math:`\mathbf{v} = (x, y, z)`.

    Batch dimensions of ``quat`` and ``p`` are broadcast against each other.

    Args:
        quat: Unit quaternion(s) of shape ``(..., 4)`` in WXYZ order.
        p: 3-D point(s) of shape ``(..., 3)``.

    Returns:
        Rotated point(s) of shape ``(..., 3)``.

    Example::

        >>> import torch
        >>> # 90° rotation around Z rotates X-axis towards Y-axis
        >>> q = quat_from_axis_angle(torch.tensor([0., 0., torch.pi/2]))
        >>> p = torch.tensor([1., 0., 0.])
        >>> quat_apply(q, p)  # ≈ tensor([0., 1., 0.])
    """
    return get_backend().quat_apply(quat, p)


def quat_apply_inv(quat: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    r"""Rotate a 3-D point by the inverse of a quaternion.

    Equivalent to applying the conjugate rotation:

    .. math::

        p' = q^{-1} \otimes [0,\, p] \otimes q

    Batch dimensions of ``quat`` and ``p`` are broadcast against each other.

    Args:
        quat: Unit quaternion(s) of shape ``(..., 4)`` in WXYZ order.
        p: 3-D point(s) of shape ``(..., 3)``.

    Returns:
        Rotated point(s) of shape ``(..., 3)``.

    Example::

        >>> q = random_quat(4)
        >>> p = torch.randn(4, 3)
        >>> torch.allclose(quat_apply_inv(q, quat_apply(q, p)), p, atol=1e-6)
        True
    """
    return get_backend().quat_apply_inv(quat, p)


def quat_slerp(q1: torch.Tensor, q2: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    r"""Spherical linear interpolation (SLERP) between two quaternions.

    Interpolates along the shortest geodesic on the 3-sphere.
    Let :math:`\theta = \arccos\!\bigl(\operatorname{clamp}(q_1 \cdot q_2, -1, 1)\bigr)`.
    Then

    .. math::

        q(t) = \frac{\sin((1-t)\,\theta)}{\sin\theta}\,q_1
               + \frac{\sin(t\,\theta)}{\sin\theta}\,q_2

    The result is standardized (:math:`w \ge 0`) and unit-normalized.
    Near :math:`\theta \approx 0` the function falls back to LERP to avoid
    numerical instability.

    Batch dimensions of ``q1``, ``q2``, and ``t`` are broadcast.

    Args:
        q1: Start quaternion(s), shape ``(..., 4)`` in WXYZ order.
        q2: End quaternion(s), shape ``(..., 4)`` in WXYZ order.
        t: Interpolation weight(s) in :math:`[0, 1]`, shape ``(..., 1)``.
    Returns:
        Interpolated unit quaternion(s) of shape ``(..., 4)``.

    Example::

        >>> q1 = quat_unit(1)
        >>> q2 = random_quat(1)
        >>> t = torch.tensor([[0.5]])
        >>> q_mid = quat_slerp(q1, q2, t)
    """
    return get_backend().quat_slerp(q1, q2, t)


def quat_from_axis_angle(axis_angle: torch.Tensor) -> torch.Tensor:
    r"""Convert an axis-angle rotation vector to a unit quaternion.

    The input encodes the rotation axis and angle in a single vector: its
    direction is the rotation axis and its magnitude :math:`\theta = \|\mathbf{v}\|`
    is the rotation angle in radians.

    .. math::

        q = \left(\cos\frac{\theta}{2},\;
            \frac{\sin(\theta/2)}{\theta}\,\mathbf{v}\right)

    Near :math:`\theta \approx 0` a stable Taylor expansion is used:
    :math:`\sin(\theta/2)/\theta \approx 1/2 - \theta^2/48`.

    Args:
        axis_angle: Tensor of shape ``(..., 3)``.

    Returns:
        Unit quaternion(s) of shape ``(..., 4)`` in WXYZ order.

    Example::

        >>> import torch
        >>> # 180° rotation around the Z axis
        >>> aa = torch.tensor([0., 0., torch.pi])
        >>> q = quat_from_axis_angle(aa)
        >>> quat_apply(q, torch.tensor([1., 0., 0.]))  # ≈ tensor([-1., 0., 0.])
    """
    return get_backend().quat_from_axis_angle(axis_angle)


def quat_to_axis_angle(quat: torch.Tensor) -> torch.Tensor:
    r"""Convert a unit quaternion to an axis-angle rotation vector.

    Inverse of :func:`quat_from_axis_angle`.  For :math:`q = (w, x, y, z)`:

    .. math::

        \theta = 2\arctan2\!\bigl(\|(x, y, z)\|,\; w\bigr), \qquad
        \mathbf{v} = \frac{\theta}{\|(x, y, z)\|}\,(x, y, z)

    Near :math:`\theta \approx 0` a stable Taylor expansion avoids division
    by zero.

    Args:
        quat: Unit quaternion(s) of shape ``(..., 4)`` in WXYZ order.

    Returns:
        Axis-angle vectors of shape ``(..., 3)``.  The magnitude is the
        rotation angle in radians; the direction is the rotation axis.

    Example::

        >>> aa = torch.tensor([0., 0., 1.2])
        >>> torch.allclose(quat_to_axis_angle(quat_from_axis_angle(aa)), aa, atol=1e-6)
        True
    """
    return get_backend().quat_to_axis_angle(quat)


def quat_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    r"""Convert a unit quaternion to a 3×3 rotation matrix.

    For :math:`q = (w, x, y, z)`:

    .. math::

        R = \begin{pmatrix}
            1 - 2(y^2+z^2) & 2(xy - wz) & 2(xz + wy) \\
            2(xy + wz) & 1 - 2(x^2+z^2) & 2(yz - wx) \\
            2(xz - wy) & 2(yz + wx) & 1 - 2(x^2+y^2)
        \end{pmatrix}

    Args:
        quat: Unit quaternion(s) of shape ``(..., 4)`` in WXYZ order.

    Returns:
        Rotation matrix/matrices of shape ``(..., 3, 3)``.

    Example::

        >>> import torch
        >>> q = quat_from_axis_angle(torch.tensor([0., 0., torch.pi / 2]))
        >>> R = quat_to_matrix(q)
        >>> # R ≈ [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    """
    return get_backend().quat_to_matrix(quat)


def quat_from_matrix(mtx: torch.Tensor) -> torch.Tensor:
    r"""Convert a 3×3 rotation matrix to quaternion.

    Uses a numerically stable algorithm
    (Shepperd's method) that selects the largest component to avoid
    cancellation errors near singular cases.

    The returned quaternion may be of non-unit norm if input matrix 
    is not orthonormal.

    Args:
        mtx: Rotation matrix/matrices of shape ``(..., 3, 3)``.  Should be
            orthonormal (:math:`SO(3)`).

    Returns:
        Quaternion(s) of shape ``(..., 4)`` in WXYZ order.

    Example::

        >>> q = random_quat(10)
        >>> R = quat_to_matrix(q)
        >>> q2 = quat_from_matrix(R)
        >>> torch.allclose(q, q2, atol=1e-5)
        True
    """
    return get_backend().quat_from_matrix(mtx)


def quat_from_euler_angles(roll_pitch_yaw: torch.Tensor) -> torch.Tensor:
    return get_backend().quat_from_euler_angles(roll_pitch_yaw)

