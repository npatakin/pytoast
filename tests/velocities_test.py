from test_utils import BackendCrossTest
import torch

from toast import Quat, SE3, get_backend


class VelocitiesTest(BackendCrossTest):
    atol = 1e-4

    fn = lambda *args: get_backend().velocities(*args)

    def setup_args(self, dtype, device) -> list[torch.Tensor]:
        t1 = SE3.random(self.size, dtype=dtype, device=device)
        t2 = SE3.random(self.size, dtype=dtype, device=device)

        time1 = torch.rand(len(t1), 1, dtype=dtype, device=device)
        time2 = time1 + torch.rand(len(t2), 1, dtype=dtype, device=device) + 0.1

        return [t1.q.data, t1.t, t2.q.data, t2.t, time1, time2]


class ApplyVelocityTest(BackendCrossTest):
    fn = lambda *args: get_backend().apply_velocity(*args)

    def setup_args(self, dtype, device) -> list[torch.Tensor]:
        # q0, t0, v, w, delta_time
        return [
            Quat.random(self.size, dtype=dtype, device=device).data,
            torch.randn(self.size, 3, dtype=dtype, device=device),
            torch.randn(self.size, 3, dtype=dtype, device=device),
            torch.randn(self.size, 3, dtype=dtype, device=device)*0.1,
            torch.randn(self.size, 1, dtype=dtype, device=device)*0.1
        ]


class LinearMotionTransformTest(BackendCrossTest):
    fn = lambda *args: get_backend().linear_motion_transform(*args)

    def setup_args(self, dtype, device) -> list[torch.Tensor]:
        return [
            Quat.random(self.size, dtype=dtype, device=device).data,
            torch.randn(self.size, 3, dtype=dtype, device=device),
            torch.randn(self.size, 3, dtype=dtype, device=device),
            torch.randn(self.size, 3, dtype=dtype, device=device)*0.1,
            torch.rand(self.size, 1,  dtype=dtype, device=device),
            torch.randn(self.size, 3, dtype=dtype, device=device),
            torch.rand(self.size, 1,  dtype=dtype, device=device),
        ]


class AngularVelocityTest(BackendCrossTest):
    atol = 1e-4

    fn = lambda *args: get_backend().angular_velocity(*args)

    def setup_args(self, dtype, device) -> list[torch.Tensor]:
        t1 = SE3.random(self.size, dtype=dtype, device=device)
        t2 = SE3.random(self.size, dtype=dtype, device=device)
        dt = torch.rand(self.size, 1, dtype=dtype, device=device) + 0.1
        return [t1.q.data, t2.q.data, dt]


class ApplyAngularVelocityTest(BackendCrossTest):
    fn = lambda *args: get_backend().apply_angular_velocity(*args)

    def setup_args(self, dtype, device) -> list[torch.Tensor]:
        return [
            Quat.random(self.size, dtype=dtype, device=device).data,
            torch.randn(self.size, 3, dtype=dtype, device=device) * 0.1,
            torch.randn(self.size, 1, dtype=dtype, device=device) * 0.1,
        ]


class LinearVelocityTest(BackendCrossTest):
    fn = lambda *args: get_backend().linear_velocity(*args)

    def setup_args(self, dtype, device) -> list[torch.Tensor]:
        t1 = torch.randn(self.size, 3, dtype=dtype, device=device)
        t2 = torch.randn(self.size, 3, dtype=dtype, device=device)
        dt = torch.rand(self.size, 1, dtype=dtype, device=device) + 0.1
        return [t1, t2, dt]


__tests__ = [
    VelocitiesTest,
    ApplyVelocityTest,
    LinearMotionTransformTest,
    AngularVelocityTest,
    ApplyAngularVelocityTest,
    LinearVelocityTest,
]
