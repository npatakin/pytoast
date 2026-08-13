from test_utils import BackendCrossTest
import torch

from toast import Quat


class QuatApplyTest(BackendCrossTest):
    fn = lambda q, p: Quat(q) @ p

    def setup_args(self, dtype, device) -> list[torch.Tensor]:
        return [
            Quat.random(self.size, dtype=dtype, device=device).data,
            torch.randn(self.size, 3, dtype=dtype, device=device),
        ]


class QuatApplyEmptyTest(BackendCrossTest):
    """Zero-length batch.

    Regression: the torch reference used to reshape with a -1, which cannot be
    inferred from 0 elements ("cannot reshape tensor of 0 elements into shape
    [0, -1, 3]"), so it raised where the fused kernel returned an empty result.
    Empty batches arise naturally from filtering.
    """
    fn = lambda q, p: Quat(q) @ p

    def setup_args(self, dtype, device) -> list[torch.Tensor]:
        return [
            Quat.random(0, dtype=dtype, device=device).data,
            torch.randn(0, 3, dtype=dtype, device=device),
        ]


class QuatApplyInvTest(BackendCrossTest):
    fn = lambda q, p: Quat(q).rotate_inv(p)

    def setup_args(self, dtype, device) -> list[torch.Tensor]:
        return [
            Quat.random(self.size, dtype=dtype, device=device).data,
            torch.randn(self.size, 3, dtype=dtype, device=device),
        ]


class QuatMulTest(BackendCrossTest):
    fn = lambda q1, q2: (Quat(q1) @ Quat(q2)).data

    def setup_args(self, dtype, device) -> list[torch.Tensor]:
        return [
            Quat.random(self.size, dtype=dtype, device=device).data,
            Quat.random(self.size, dtype=dtype, device=device).data
        ]


class QuatStdTest(BackendCrossTest):
    fn = lambda q: Quat(q).std().data

    def setup_args(self, dtype, device) -> list[torch.Tensor]:
        q = Quat.random(self.size, dtype=dtype, device=device).data
        scale = torch.rand(self.size, 1, device=device, dtype=dtype)
        return [q * (0.5 + scale)]


class QuatInverseTest(BackendCrossTest):
    fn = lambda q: Quat(q).inv().data

    def setup_args(self, dtype, device) -> list[torch.Tensor]:
        return [Quat.random(self.size, dtype=dtype, device=device).data]


class QuatConjugateTest(BackendCrossTest):
    fn = lambda q: Quat(q).conj().data

    def setup_args(self, dtype, device) -> list[torch.Tensor]:
        return [Quat.random(self.size, dtype=dtype, device=device).data]


class QuatToAxisAngle(BackendCrossTest):
    fn = lambda q: Quat(q).to_axis_angle()

    def setup_args(self, dtype, device) -> list[torch.Tensor]:
        return [Quat.random(self.size, dtype=dtype, device=device).std().data]


class QuatFromAxisAngle(BackendCrossTest):
    fn = lambda axis_angle: Quat.from_axis_angle(axis_angle).data

    def setup_args(self, dtype, device) -> list[torch.Tensor]:
        return [torch.randn(self.size, 3, dtype=dtype, device=device)]


class QuatFromEulerAngles(BackendCrossTest):
    fn = lambda euler_angles: Quat.from_axis_angle(euler_angles).data

    def setup_args(self, dtype, device) -> list[torch.Tensor]:
        return [torch.randn(self.size, 3, dtype=dtype, device=device)]



class QuatSlerp(BackendCrossTest):
    fn = lambda q1, q2, weight: Quat(q1).lerp(Quat(q2), weight).data

    def setup_args(self, dtype, device) -> list[torch.Tensor]:
        return [
            Quat.random(self.size, dtype=dtype, device=device).data,
            Quat.random(self.size, dtype=dtype, device=device).data,
            torch.rand(self.size, 1, dtype=dtype, device=device)
        ]


class QuatToMatrix(BackendCrossTest):
    atol = 5e-6

    fn = lambda q: Quat(q).to_matrix()

    def setup_args(self, dtype, device) -> list[torch.Tensor]:
        return [Quat.random(self.size, dtype=dtype, device=device).data]


class QuatFromMatrix(BackendCrossTest):
    atol = 5e-6

    fn = lambda mtx: Quat.from_matrix(mtx).data

    def setup_args(self, dtype, device) -> list[torch.Tensor]:
        return [Quat.random(self.size, dtype=dtype, device=device).to_matrix()]



class QuatFromXYZW(BackendCrossTest):
    atol = 1e-9

    fn  = lambda q: Quat.from_xyzw(q).data

    def setup_args(self, dtype, device) -> list[torch.Tensor]:
        return [Quat.random(self.size, dtype=dtype, device=device).data]


__tests__ = [
    QuatApplyTest,
    QuatApplyEmptyTest,
    QuatApplyInvTest,
    QuatMulTest,
    QuatStdTest,
    QuatInverseTest,
    QuatConjugateTest,
    QuatToAxisAngle,
    QuatFromAxisAngle,
    QuatToMatrix,
    QuatSlerp,
    QuatFromMatrix,
    QuatFromXYZW,
    QuatFromEulerAngles
]
