r"""Unified SE(3) rigid-transformation operations dispatched to the active backend.

:math:`SE(3)` is the group of rigid-body motions in 3-D space.  Each
transformation is represented by a pair :math:`(q, t)` where:

- :math:`q` — unit quaternion of shape ``(..., 4)`` in WXYZ order,
  encoding the rotational part :math:`R(q)`.
- :math:`t` — translation vector of shape ``(..., 3)``.

The transformation maps a point :math:`p` to

.. math::

    T(p) = R(q)\,p + t.

Call ``use_backend('cpp')`` or ``use_backend('torch')`` to select which
implementation is used.  The C++/CUDA backend is the default.
"""

import torch

from .backend import get_backend


def se3_apply(q: torch.Tensor, t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    r"""Apply an SE(3) transformation to a 3-D point.

    .. math::

        p' = R(q)\,p + t

    where :math:`R(q)` is the :math:`3 \times 3` rotation matrix
    corresponding to quaternion :math:`q`.

    Batch dimensions of ``q``, ``t``, and ``p`` are broadcast against each
    other.

    Args:
        q: Unit quaternion(s), shape ``(..., 4)`` in WXYZ order.
        t: Translation vector(s), shape ``(..., 3)``.
        p: 3-D point(s), shape ``(..., 3)``.

    Returns:
        Transformed point(s), shape ``(..., 3)``.

    Example::

        >>> import torch
        >>> from toast.ops.quat_ops import quat_from_axis_angle
        >>> q = quat_from_axis_angle(torch.tensor([0., 0., torch.pi / 2]))
        >>> t = torch.tensor([1., 0., 0.])
        >>> se3_apply(q, t, torch.tensor([1., 0., 0.]))  # ≈ tensor([1., 1., 0.])
    """
    return get_backend().se3_apply(q, t, p)


def se3_apply_inv(q: torch.Tensor, t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    r"""Apply the inverse of an SE(3) transformation to a 3-D point.

    .. math::

        p' = R(q^{-1}) (p - t)

    which is the inverse of :func:`se3_apply`.

    Batch dimensions of ``q``, ``t``, and ``p`` are broadcast against each
    other.

    Args:
        q: Unit quaternion(s), shape ``(..., 4)`` in WXYZ order.
        t: Translation vector(s), shape ``(..., 3)``.
        p: 3-D point(s), shape ``(..., 3)``.

    Returns:
        Inverse-transformed point(s), shape ``(..., 3)``.

    Example::

        >>> q = random_quat(4)
        >>> t = torch.randn(4, 3)
        >>> p = torch.randn(4, 3)
        >>> torch.allclose(se3_apply_inv(q, t, se3_apply(q, t, p)), p, atol=1e-6)
        True
    """
    return get_backend().se3_apply_inv(q, t, p)


def se3_inverse(
    q: torch.Tensor, t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Compute the inverse of an SE(3) transformation.

    For :math:`T = (q, t)` the inverse is

    .. math::

        T^{-1} = \bigl(q^{-1},\; -R(q^{-1})\,t\bigr)

    where :math:`q^{-1}` is the quaternion inverse (equal to the conjugate for
    unit quaternions).

    Batch dimensions of ``q`` and ``t`` are broadcast against each other.

    Args:
        q: Unit quaternion(s), shape ``(..., 4)`` in WXYZ order.
        t: Translation vector(s), shape ``(..., 3)``.

    Returns:
        Tuple ``(q_inv, t_inv)`` with the same batch shape as the inputs.

    Example::

        >>> q = random_quat(4)
        >>> t = torch.randn(4, 3)
        >>> q_inv, t_inv = se3_inverse(q, t)
        >>> qc, tc = se3_compose(q, t, q_inv, t_inv)
        >>> torch.allclose(tc, torch.zeros_like(tc), atol=1e-6)
        True
    """
    return get_backend().se3_inverse(q, t)


def se3_compose(
    q1: torch.Tensor,
    t1: torch.Tensor,
    q2: torch.Tensor,
    t2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Compose two SE(3) transformations: :math:`T_1 \circ T_2`.

    The composed transformation applies :math:`T_2` first, then :math:`T_1`:

    .. math::

        q_{12} = q_1\,q_2, \qquad
        t_{12} = R(q_1)\,t_2 + t_1

    so that :math:`T_{12}(p) = T_1(T_2(p))`.

    Batch dimensions of all inputs are broadcast against each other.

    Args:
        q1: Quaternion of :math:`T_1`, shape ``(..., 4)`` in WXYZ order.
        t1: Translation of :math:`T_1`, shape ``(..., 3)``.
        q2: Quaternion of :math:`T_2`, shape ``(..., 4)`` in WXYZ order.
        t2: Translation of :math:`T_2`, shape ``(..., 3)``.

    Returns:
        Tuple ``(q_composed, t_composed)`` representing :math:`T_1 \circ T_2`.

    Example::

        >>> p = torch.zeros(3)
        >>> torch.allclose(
        ...     se3_apply(*se3_compose(q1, t1, q2, t2), p),
        ...     se3_apply(q1, t1, se3_apply(q2, t2, p)),
        ...     atol=1e-6
        ... )
        True
    """
    return get_backend().se3_compose(q1, t1, q2, t2)


def se3_slerp(
    q1: torch.Tensor,
    t1: torch.Tensor,
    q2: torch.Tensor,
    t2: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Interpolate between two SE(3) transformations.

    Rotations are interpolated via SLERP and translations via LERP:

    .. math::

        q(w) = \operatorname{slerp}(q_1, q_2, w), \qquad
        t(w) = (1 - w)\,t_1 + w\,t_2

    At :math:`w = 0` the result is :math:`(q_1, t_1)`; at :math:`w = 1`
    it is :math:`(q_2, t_2)`.

    Batch dimensions of all inputs are broadcast against each other.

    Args:
        q1: Start quaternion(s), shape ``(..., 4)`` in WXYZ order.
        t1: Start translation(s), shape ``(..., 3)``.
        q2: End quaternion(s), shape ``(..., 4)`` in WXYZ order.
        t2: End translation(s), shape ``(..., 3)``.
        weight: Interpolation weight(s) :math:`w \in [0, 1]`, shape
            ``(..., 1)``.

    Returns:
        Tuple ``(q_interp, t_interp)`` of the interpolated pose.

    Example::

        >>> w = torch.tensor([[0.5]])
        >>> q_mid, t_mid = se3_slerp(q1, t1, q2, t2, w)
    """
    return get_backend().se3_slerp(q1, t1, q2, t2, weight)


def se3_to_matrix_3x4(q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    r"""Convert an SE(3) pose to a :math:`3 \times 4` transformation matrix.

    .. math::

        M = \bigl[\,R(q)\;\big|\;t\,\bigr] \in \mathbb{R}^{3 \times 4}

    A point :math:`p` is transformed as
    :math:`M\,[p^\top, 1]^\top = R(q)\,p + t`.

    Batch dimensions of ``q`` and ``t`` are broadcast against each other.

    Args:
        q: Unit quaternion(s), shape ``(..., 4)`` in WXYZ order.
        t: Translation vector(s), shape ``(..., 3)``.

    Returns:
        Transformation matrix/matrices of shape ``(..., 3, 4)``.

    Example::

        >>> M = se3_to_matrix_3x4(quat_unit(2), torch.zeros(2, 3))
        >>> M.shape  # (2, 3, 4)
    """
    return get_backend().se3_to_matrix_3x4(q, t)


def se3_to_matrix_4x4(q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    r"""Convert an SE(3) pose to a :math:`4 \times 4` homogeneous matrix.

    .. math::

        M = \begin{bmatrix} R(q) & t \\ \mathbf{0}^\top & 1 \end{bmatrix}
        \in \mathbb{R}^{4 \times 4}

    Batch dimensions of ``q`` and ``t`` are broadcast against each other.

    Args:
        q: Unit quaternion(s), shape ``(..., 4)`` in WXYZ order.
        t: Translation vector(s), shape ``(..., 3)``.

    Returns:
        Homogeneous transformation matrix/matrices of shape ``(..., 4, 4)``.

    Example::

        >>> M = se3_to_matrix_4x4(quat_unit(3), torch.ones(3, 3))
        >>> M[0, 3, :]  # tensor([0., 0., 0., 1.])
    """
    return get_backend().se3_to_matrix_4x4(q, t)


def se3_from_matrix(mtx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Extract an SE(3) pose from a :math:`3 \times 4` or :math:`4 \times 4` matrix.

    Inverse of :func:`se3_to_matrix_3x4` / :func:`se3_to_matrix_4x4`.

    .. math::

        q = \operatorname{quat\_from\_matrix}(M_{:3,:3}), \qquad
        t = M_{:3,\,3}

    Args:
        mtx: Transformation matrix/matrices of shape ``(..., 3, 4)`` or
            ``(..., 4, 4)``.  The upper-left :math:`3 \times 3` block must
            be a valid rotation matrix.

    Returns:
        Tuple ``(q, t)`` where ``q`` has shape ``(..., 4)`` and ``t`` has
        shape ``(..., 3)``.

    Example::

        >>> q = random_quat(5)
        >>> t = torch.randn(5, 3)
        >>> q2, t2 = se3_from_matrix(se3_to_matrix_4x4(q, t))
        >>> torch.allclose(t, t2, atol=1e-6)
        True
    """
    return get_backend().se3_from_matrix(mtx)
