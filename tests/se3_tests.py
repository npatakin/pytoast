import torch

from toast import SE3, Quat
from test_utils import BackendCrossTest


class SE3ApplyTest(BackendCrossTest):
    fn = lambda q, t, p: SE3(q, t) @ p

    def setup_args(self, dtype, device) -> list[torch.Tensor]:
        return [
            Quat.random(self.size, dtype=dtype, device=device).data,
            torch.randn(self.size, 3, dtype=dtype, device=device),
            torch.randn(self.size, 3, dtype=dtype, device=device),
        ]


class SE3ApplyInvTest(BackendCrossTest):
    fn = lambda q, t, p: SE3(q, t).inverse_transform(p)

    def setup_args(self, dtype, device) -> list[torch.Tensor]:
        return [
            Quat.random(self.size, dtype=dtype, device=device).data,
            torch.randn(self.size, 3, dtype=dtype, device=device),
            torch.randn(self.size, 3, dtype=dtype, device=device),
        ]


class SE3InverseTest(BackendCrossTest):
    atol = 5e-6

    @staticmethod
    def fn(q, t):
        inv = SE3(q, t).inv()
        return inv.q.data, inv.t

    def setup_args(self, dtype, device) -> list[torch.Tensor]:
        return [
            Quat.random(self.size, dtype=dtype, device=device).data,
            torch.randn(self.size, 3, dtype=dtype, device=device),
        ]


class SE3ComposeTest(BackendCrossTest):
    @staticmethod
    def fn(q1, t1, q2, t2):
        res = SE3(q1, t1) @ SE3(q2, t2)
        return res.q.data, res.t

    def setup_args(self, dtype, device) -> list[torch.Tensor]:
        return [
            Quat.random(self.size, dtype=dtype, device=device).data,
            torch.randn(self.size, 3, dtype=dtype, device=device),
            Quat.random(self.size, dtype=dtype, device=device).data,
            torch.randn(self.size, 3, dtype=dtype, device=device),
        ]


class SE3SlerpTest(BackendCrossTest):
    @staticmethod
    def fn(q1, t1, q2, t2, weight):
        res = SE3(q1, t1).lerp(SE3(q2, t2), weight)
        return res.q.data, res.t

    def setup_args(self, dtype, device) -> list[torch.Tensor]:
        return [
            Quat.random(self.size, dtype=dtype, device=device).data,
            torch.randn(self.size, 3, dtype=dtype, device=device),
            Quat.random(self.size, dtype=dtype, device=device).data,
            torch.randn(self.size, 3, dtype=dtype, device=device),
            torch.rand(self.size, 1, dtype=dtype, device=device),
        ]


class SE3ToMatrix4x4Test(BackendCrossTest):
    atol = 1e-5

    fn = lambda q, t: SE3(q, t).to_matrix()

    def setup_args(self, dtype, device) -> list[torch.Tensor]:
        return [
            Quat.random(self.size, dtype=dtype, device=device).data,
            torch.randn(self.size, 3, dtype=dtype, device=device),
        ]


class SE3ToMatrix3x4Test(BackendCrossTest):
    atol = 1e-5

    fn = lambda q, t: SE3(q, t).to_matrix(to_3x4=True)

    def setup_args(self, dtype, device) -> list[torch.Tensor]:
        return [
            Quat.random(self.size, dtype=dtype, device=device).data,
            torch.randn(self.size, 3, dtype=dtype, device=device),
        ]


class SE3FromMatrixTest(BackendCrossTest):
    atol = 5e-6

    @staticmethod
    def fn(mtx):
        res = SE3.from_matrix(mtx)
        return res.q.data, res.t

    def setup_args(self, dtype, device) -> list[torch.Tensor]:
        trf = SE3(
            Quat.random(self.size, dtype=dtype, device=device).data,
            torch.randn(self.size, 3, dtype=dtype, device=device),
        )
        return [trf.to_matrix()]


__tests__ = [
    SE3ApplyTest,
    SE3ApplyInvTest,
    SE3InverseTest,
    SE3ComposeTest,
    SE3SlerpTest,
    SE3ToMatrix4x4Test,
    SE3ToMatrix3x4Test,
    SE3FromMatrixTest,
]
