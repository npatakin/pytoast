import torch
import torch.nn.functional as F

from ..ops import (
    quat_std,
    quat_inverse,
    quat_conjugate,
    quat_apply,
    quat_apply_inv,
    quat_mul,
    quat_slerp,
    quat_from_axis_angle,
    quat_to_axis_angle,
    quat_from_matrix,
    quat_to_matrix,
    quat_from_xyzw,
    quat_unit,
    quat_random,
    quat_from_euler_angles
)
from ..utils.data_tensor import DataTensor



class safe_acos(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.acos()

    @staticmethod
    def backward(ctx, v_out):
        x, = ctx.saved_tensors
        return -v_out / (1 - x * x).sqrt().clamp_min_(1e-6)


class Quat(DataTensor):
    """Unit quaternion representing a 3D rotation.

    Quaternions are stored in **(w, x, y, z)** order — the real (scalar) part first,
    followed by the three imaginary components.  The underlying storage is a
    ``torch.Tensor`` of shape ``(..., 4)`` accessible via :attr:`data`.

    Example::

        import torch
        from toast.transforms import Quat

        q = Quat.random(8, device="cuda")        # batch of 8 random rotations
        p = torch.randn(8, 3, device="cuda")
        rotated = q @ p                          # rotate points
        q_inv = q.inv()
        assert torch.allclose(q_inv @ (q @ p), p, atol=1e-5)
    """

    _buffers = ["data"]

    data: torch.Tensor

    def __init__(self, data: torch.Tensor):
        self.data = data

    # ------------------------------------------------------------------
    # Component accessors
    # ------------------------------------------------------------------

    @property
    def w(self) -> torch.Tensor:
        """Real (scalar) component - shape ``(...)``. Alias for ``self.data[..., 0]``"""
        return self.data[..., 0]

    @property
    def x(self) -> torch.Tensor:
        """First imaginary component X - shape ``(...)``. Alias for ``self.data[..., 1]``"""
        return self.data[..., 1]

    @property
    def y(self) -> torch.Tensor:
        """Second imaginary component Y - shape ``(...)``. Alias for ``self.data[..., 2]``"""
        return self.data[..., 2]

    @property
    def z(self) -> torch.Tensor:
        """Third imaginary component Z - shape ``(...)``. Alias for ``self.data[..., 3]``"""
        return self.data[..., 3]

    @property
    def real(self) -> torch.Tensor:
        """Alias for :attr:`w`."""
        return self.w

    @property
    def img(self) -> torch.Tensor:
        """Imaginary part (XYZ) as a 3-vector — shape ``(..., 3)``."""
        return self.data[..., 1:]

    @property
    def vec(self) -> torch.Tensor:
        """Alias for :attr:`img`."""
        return self.img

    # ------------------------------------------------------------------
    # Scalar properties
    # ------------------------------------------------------------------

    @property
    def norm(self) -> torch.Tensor:
        """L2 norm of the quaternion — shape ``(...)``."""
        return self.data.norm(dim=-1)

    @property
    def img_norm(self) -> torch.Tensor:
        """L2 norm of the imaginary part — shape ``(...)``."""
        return self.img.norm(dim=-1)

    @property
    def angle(self) -> torch.Tensor:
        """Rotation angle in radians in range :math:`[0, \pi)` — shape ``(...)``."""
        return 2 * safe_acos.apply(self.real / self.norm)

    @property
    def angle_deg(self) -> torch.Tensor:
        """Rotation angle in degrees — shape ``(...)``."""
        return self.angle.rad2deg()

    # ------------------------------------------------------------------
    # Normalization / standardization
    # ------------------------------------------------------------------

    def normalize(self) -> "Quat":
        """Return a unit-norm copy (L2-normalize along last dim).

        Returns:
            Quat: Unit quaternion with the same orientation.
        """
        return Quat(data=F.normalize(self.data, dim=-1))

    def std(self) -> "Quat":
        """Return the *standard form*: unit norm **and** ``w >= 0``.

        Every rotation has two quaternion representations, ``q`` and ``-q``.
        The standard form picks the one with non-negative real part, which
        makes comparisons deterministic.

        Returns:
            Quat: Standardized unit quaternion.
        """
        return Quat(quat_std(self.data))

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def inv(self) -> "Quat":
        """Return the inverse (conjugate divided by squared norm).

        For unit quaternions this equals the conjugate.

        Returns:
            Quat: Quaternion inverse ``q⁻¹`` such that ``q @ q.inv()`` is the identity.
        """
        return Quat(quat_inverse(self.data))

    def conj(self) -> "Quat":
        return Quat(quat_conjugate(self.data))

    def __neg__(self) -> "Quat":
        """Return negative of the quaternion: (-w, -x, -y, -z)"""
        return Quat(-self.data)

    def lerp(self, other: "Quat", t: torch.Tensor) -> "Quat":
        """
        Spherical linear interpolation (slerp)
        between two quaternions: https://en.wikipedia.org/wiki/Spherical_linear_interpolation

        Always returns standardized quaternion (with non-negative real part and unit norm)

        Args:
            other: Target rotation.
            t: Interpolation weight, shape ``(..., 1)``, values in ``[0; 1]`` range.
                ``t=0`` returns ``self``, ``t=1`` returns ``other``.

        Returns:
            Quat: Interpolated rotation.
        """
        return Quat(quat_slerp(self.data, other.data, t))

    def rotate(self, points: torch.Tensor) -> torch.Tensor:
        """Apply the rotation to a batch of 3-D points.

        Args:
            points: Points of shape ``(..., 3)``.

        Returns:
            torch.Tensor: Rotated points, same shape as *points*.
        """
        return quat_apply(self.data, points)

    def rotate_inv(self, points: torch.Tensor) -> torch.Tensor:
        """Apply the *inverse* rotation to a batch of 3-D points.

        Equivalent to ``self.inv().rotate(points)`` but avoids computing the
        inverse explicitly.

        Args:
            points: Points of shape ``(..., 3)``.

        Returns:
            torch.Tensor: Rotated points, same shape as *points*.
        """
        return quat_apply_inv(self.data, points)

    def compose(self, other: "Quat") -> "Quat":
        """Quaternion product (Hamilton product): ``self ⊗ other``.

        Args:
            other: Right-hand quaternion.

        Returns:
            Quat: Composed rotation.
        """
        return Quat(quat_mul(self.data, other.data))

    def __matmul__(self, other):
        """Overloads ``@`` for :meth:`rotate` (tensor) and :meth:`compose` (Quat)."""
        if isinstance(other, torch.Tensor):
            return self.rotate(other)
        return self.compose(other)

    # ------------------------------------------------------------------
    # Conversions
    # ------------------------------------------------------------------

    def to_axis_angle(self) -> torch.Tensor:
        """Convert to axis-angle representation.

        The quaternion is first standardized (:meth:`std`).

        Returns:
            torch.Tensor: Axis-angle vector of shape ``(..., 3)``.  The
            direction encodes the rotation axis and the magnitude encodes the
            angle in radians.
        """
        return quat_to_axis_angle(self.std().data)

    def to_matrix(self) -> torch.Tensor:
        """Convert to a 3×3 rotation matrix.

        Returns:
            torch.Tensor: Rotation matrix of shape ``(..., 3, 3)``.
        """
        return quat_to_matrix(self.data)

    # ------------------------------------------------------------------
    # Static factories
    # ------------------------------------------------------------------

    @staticmethod
    def unit(
            *dims,
            device: torch.device | str = 'cpu',
            dtype: torch.dtype = torch.float32,
    ) -> "Quat":
        """Create identity quaternion(s) ``[1, 0, 0, 0]``.

        Args:
            *dims: Batch dimensions.
            device: Target device.
            dtype: Floating-point dtype.

        Returns:
            Quat: Identity rotation(s) of shape ``(*dims,)``.
        """
        return Quat(quat_unit(*dims, device=device, dtype=dtype))

    @staticmethod
    def unit_like(other: "Quat") -> "Quat":
        """Create identity quaternion(s) matching the shape/device/dtype of *other*."""
        return Quat.unit(*other.shape, device=other.device, dtype=other.dtype)

    @staticmethod
    def random(
            *dims,
            device: torch.device | str = 'cpu',
            dtype: torch.dtype = torch.float32,
            generator: torch.Generator = None
    ) -> "Quat":
        """Sample uniformly random unit quaternions.

        Uses the Shoemake / three-uniform method so the distribution is
        uniform over SO(3).

        Args:
            *dims: Batch dimensions.
            device: Target device.
            dtype: Floating-point dtype.
            generator: Optional :class:`torch.Generator` for reproducibility.

        Returns:
            Quat: Random unit quaternion(s) of shape ``(*dims,)``.
        """
        return Quat(quat_random(
            *dims, device=device, dtype=dtype, generator=generator
        ))

    @staticmethod
    def from_xyzw(xyzw: torch.Tensor) -> "Quat":
        """Create a :class:`Quat` from a tensor in **xyzw** (imaginary-first) order.

        Args:
            xyzw: Tensor of shape ``(..., 4)`` with components in xyzw order.

        Returns:
            Quat: Quaternion with data reordered to wxyz.
        """
        return Quat(quat_from_xyzw(xyzw))

    @staticmethod
    def from_axis_angle(axis_angle: torch.Tensor) -> "Quat":
        """Convert axis-angle vectors to unit quaternions.

        Args:
            axis_angle: Tensor of shape ``(..., 3)``.  The direction is the
                rotation axis and the magnitude is the angle in radians.

        Returns:
            Quat: Corresponding unit quaternion(s).
        """
        assert axis_angle.shape[-1] == 3
        return Quat(quat_from_axis_angle(axis_angle))

    @staticmethod
    def from_matrix(mtx: torch.Tensor) -> "Quat":
        """Convert 3×3 rotation matrices to unit quaternions.

        Args:
            mtx: Rotation matrices of shape ``(..., 3, 3)``.

        Returns:
            Quat: Corresponding unit quaternion(s).
        """
        assert mtx.shape[-1] == 3 and mtx.shape[-2] == 3
        return Quat(quat_from_matrix(mtx))

    @staticmethod
    def from_euler_angles(roll_pitch_yaw: torch.Tensor) -> "Quat":
        assert roll_pitch_yaw.shape[-1] == 3
        return Quat(quat_from_euler_angles(roll_pitch_yaw))
