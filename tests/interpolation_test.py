"""Trajectory interpolation: fused kernels vs the pure-PyTorch reference.

Keyframe timestamps are deliberately WELL SEPARATED here.  The op divides by the
gap between the bracketing keyframes, so a near-degenerate gap makes both
implementations produce enormous, badly conditioned gradients w.r.t. seq_times --
comparing them then measures float noise rather than correctness.  The
KeyframeSpacing test below covers the tight-gap case separately with tolerances
that reflect it.
"""
from test_utils import BackendCrossTest
import torch

from toast import Quat, get_backend


def _sorted_times(*shape, spread=10.0, dtype=None, device=None):
    """Strictly increasing timestamps with a guaranteed minimum gap."""
    steps = torch.rand(*shape, dtype=dtype, device=device) + 0.5
    return steps.cumsum(dim=-1) * (spread / shape[-1])


class TrajectorySampleTest(BackendCrossTest):
    """Quaternions + translations, no mask, queries inside and outside range."""
    num_keyframes = 32
    num_queries = 17
    batch_shape = (64,)
    masked = False

    fn = lambda *args: get_backend().sample_trajectory(
        *args, return_velocities=True, return_indices=True, return_mask=True)

    def setup_args(self, dtype, device) -> list[torch.Tensor]:
        b, k, q = self.batch_shape, self.num_keyframes, self.num_queries
        seq_times = _sorted_times(*b, k, dtype=dtype, device=device)
        # queries span past both ends, so extrapolation is exercised too
        lo = seq_times[..., :1] - 1.0
        hi = seq_times[..., -1:] + 1.0
        query_times = lo + (hi - lo) * torch.rand(*b, q, dtype=dtype, device=device)
        query_times = query_times.sort(dim=-1).values

        args = [
            query_times,
            seq_times,
            Quat.random(*b, k, dtype=dtype, device=device).data,
            torch.randn(*b, k, 3, dtype=dtype, device=device),
        ]
        if self.masked:
            mask = torch.rand(*b, k, device=device) > 0.35
            # keep the ends valid so extrapolation always has something to use
            mask[..., 0] = True
            mask[..., -1] = True
            args.append(mask)
        return args


class TrajectoryMaskedTest(TrajectorySampleTest):
    """Invalid keyframes must drop out of bracketing on both backends."""
    masked = True


class TrajectoryBatchDimsTest(TrajectorySampleTest):
    """Arbitrary batch dimensions -- the old kernel only accepted a flat batch."""
    batch_shape = (4, 5, 3)
    num_keyframes = 20
    num_queries = 9


class TrajectoryUnsortedQueriesTest(TrajectorySampleTest):
    """Binary search means query order does not matter."""
    num_queries = 23

    def setup_args(self, dtype, device) -> list[torch.Tensor]:
        args = super().setup_args(dtype, device)
        perm = torch.randperm(args[0].shape[-1], device=device)
        args[0] = args[0][..., perm].contiguous()
        return args


class TrajectorySingleQueryTest(TrajectorySampleTest):
    """Q=1: the small-batch corner the old scan kernel was tuned for."""
    num_queries = 1
    batch_shape = (256,)


class TrajectoryManyQueriesTest(TrajectorySampleTest):
    """Many queries per element -> heavy contention on the backward atomics."""
    batch_shape = (8,)
    num_queries = 256


class TrajectoryKeyframeSpacingTest(TrajectorySampleTest):
    """Tightly spaced keyframes.

    Interpolation stays well conditioned, but d/d(seq_times) is divided by the
    keyframe gap, so gradients are large and float32 loses several digits.  The
    float32 tolerance is relative and generous for that reason; float64 is held
    to the strict default, which is what actually pins the maths down.
    """
    atol = 1e-2
    rtol = 1e-2

    def setup_args(self, dtype, device) -> list[torch.Tensor]:
        args = super().setup_args(dtype, device)
        args[1] = args[1] * 0.01          # squeeze the keyframes together
        args[0] = args[0] * 0.01
        return args


__tests__ = [
    TrajectorySampleTest,
    TrajectoryMaskedTest,
    TrajectoryBatchDimsTest,
    TrajectoryUnsortedQueriesTest,
    TrajectorySingleQueryTest,
    TrajectoryManyQueriesTest,
    TrajectoryKeyframeSpacingTest,
]
