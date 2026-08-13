import torch

from ..ops import (
    se3_inverse,
    se3_apply,
    se3_apply_inv,
    se3_slerp,
    se3_compose,
    se3_from_matrix,
    se3_to_matrix_3x4,
    se3_to_matrix_4x4
)
from ..utils import DataTensor
from .quat import Quat


class SE3(DataTensor):
    """Rigid body transformation (rotation + translation) in SE(3).

    An :class:`SE3` stores a rotation as a :class:`Quat` and a translation
    as a ``torch.Tensor`` of shape ``(..., 3)``.  Applying the transform to a
    point ``p`` is equivalent to ``q.rotate(p) + t``.

    Example::

        import torch
        from toast.transforms import SE3, Quat

        q = Quat.random(4, device="cuda")
        t = torch.randn(4, 3, device="cuda")
        T = SE3(q, t)

        p = torch.randn(4, 3, device="cuda")
        p_out = T @ p               # transform points
        p_back = T.inv() @ p_out    # inverse transform
        assert torch.allclose(p, p_back, atol=1e-5)
    """

    _buffers = ["q", "t"]

    q: Quat
    t: torch.Tensor

    def __init__(self, q, t: torch.Tensor):
        """
        Args:
            q: Rotation, either a :class:`Quat` or a ``torch.Tensor`` of
               shape ``(..., 4)`` in wxyz order.
            t: Translation tensor of shape ``(..., 3)``.
        """
        self.q = Quat(q) if isinstance(q, torch.Tensor) else q
        self.t = t.to(device=q.device, dtype=q.dtype)

    # ------------------------------------------------------------------
    # Static factories
    # ------------------------------------------------------------------

    @staticmethod
    def random(
            *dims,
            device: torch.device | str = 'cpu',
            dtype: torch.dtype = torch.float32
    ):
        """Create random SE3 transformation:
            Quaternion is sampled randomly via ``Quat.random()``
            and translation is generated from :math:`\mathcal{N}(0, 1)` normal distribution

        Args:
            *dims: Batch dimensions
            device: Target device (default: ``cpu``)
            dtype: Floating point dtype (default: ``torch.float32``)
        """
        return SE3(
            Quat.random(*dims, device=device, dtype=dtype),
            torch.randn(*dims, 3, device=device, dtype=dtype)
        )

    @staticmethod
    def unit(
            *dims,
            device: torch.device | str = 'cpu',
            dtype: torch.dtype = torch.float32
    ) -> "SE3":
        """Create identity transform(s): zero translation and unit rotation.

        Args:
            *dims: Batch dimensions.
            device: Target device.
            dtype: Floating-point dtype.

        Returns:
            SE3: Identity transform(s) of shape ``(*dims,)``.
        """
        return SE3(
            Quat.unit(*dims, device=device, dtype=dtype),
            torch.zeros(*dims, 3, device=device, dtype=dtype)
        )

    @staticmethod
    def unit_like(other: "SE3") -> "SE3":
        """Create identity transform(s) matching the shape/device/dtype of *other*."""
        return SE3.unit(*other.shape, device=other.device, dtype=other.dtype)

    @staticmethod
    def from_matrix(mtx: torch.Tensor) -> "SE3":
        """Parse a homogeneous matrix into an :class:`SE3`.

        Args:
            mtx: Matrix of shape ``(..., 3, 4)`` or ``(..., 4, 4)``.  The
                top-left 3×3 block must be a valid rotation matrix.

        Returns:
            SE3: Parsed rigid transform.
        """
        return SE3(*se3_from_matrix(mtx))

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def transform(self, points: torch.Tensor) -> torch.Tensor:
        """Apply the rigid transform to a batch of 3-D points.

        Computes ``q.rotate(p) + t`` for each point.

        Args:
            points: Points of shape ``(..., 3)``.

        Returns:
            torch.Tensor: Transformed points, same shape as *points*.
        """
        return se3_apply(self.q.data, self.t, points)

    def inverse_transform(self, points: torch.Tensor) -> torch.Tensor:
        """Apply the **inverse** transform to points.

        Equivalent to ``self.inv().transform(points)`` but computed in a
        single fused operation.

        Args:
            points: Points of shape ``(..., 3)``.

        Returns:
            torch.Tensor: Transformed points, same shape as *points*.
        """
        return se3_apply_inv(self.q.data, self.t, points)

    def inv(self) -> "SE3":
        """Return the inverse transform ``T⁻¹``.

        If ``T = (q, t)``, then ``T⁻¹ = (q⁻¹, -q⁻¹·t)``.

        Returns:
            SE3: Inverse rigid transform.
        """
        return SE3(*se3_inverse(self.q.data, self.t))

    def compose(self, other: "SE3") -> "SE3":
        """Compose two transforms: ``self ∘ other``.

        Applies *other* first, then *self*.  The result satisfies
        ``(self @ other) @ p == self @ (other @ p)``.

        Args:
            other: Right-hand transform.

        Returns:
            SE3: Composed transform.
        """
        return SE3(*se3_compose(self.q.data, self.t, other.q.data, other.t))

    def __matmul__(self, other):
        """Overloads ``@`` for :meth:`transform` (tensor) and :meth:`compose` (SE3)."""
        if isinstance(other, torch.Tensor):
            return self.transform(other)
        return self.compose(other)

    def lerp(self, other: "SE3", weight: torch.Tensor) -> "SE3":
        """Interpolate between two transforms.

        The rotation is slerp'd and the translation is linearly interpolated.

        Args:
            other: Target transform.
            weight: Interpolation weight, shape ``(..., 1)``.  ``weight=0``
                returns ``self``, ``weight=1`` returns ``other``.

        Returns:
            SE3: Interpolated transform.
        """
        return SE3(*se3_slerp(self.q.data, self.t, other.q.data, other.t, weight))

    # ------------------------------------------------------------------
    # Conversions
    # ------------------------------------------------------------------

    def to_matrix(self, to_3x4: bool = False) -> torch.Tensor:
        """Convert to a homogeneous matrix.

        Args:
            to_3x4: If ``True``, return a 3×4 matrix (last row ``[0,0,0,1]``
                omitted).  Default returns a 4×4 matrix.

        Returns:
            torch.Tensor: Matrix of shape ``(..., 3, 4)`` or ``(..., 4, 4)``.
        """
        if to_3x4:
            return se3_to_matrix_3x4(self.q.data, self.t)
        else:
            return se3_to_matrix_4x4(self.q.data, self.t)

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def normalize(self) -> "SE3":
        """Return a copy with the rotation quaternion L2-normalized."""
        return SE3(self.q.normalize(), self.t)

    def std(self) -> "SE3":
        """Return a copy with the rotation in standard form (unit norm, ``w >= 0``)."""
        return SE3(self.q.std(), self.t)
