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


class TrajectoryPaddedTest(TrajectorySampleTest):
    """Sequences of different length, padded to K with masked zero timestamps.

    The padding breaks the sorted order of the row as a whole, so bracketing
    must stay between the first and the last valid keyframe: searching into the
    trailing zeros sends every query to the end of the row.
    """
    masked = True

    def setup_args(self, dtype, device) -> list[torch.Tensor]:
        args = super().setup_args(dtype, device)
        seq_times, mask = args[1], args[4]
        b, k = self.batch_shape, self.num_keyframes

        # each element keeps keyframes [start, stop) and is padded around them
        index = torch.arange(k, device=device).expand(*b, k)
        start = torch.randint(0, k // 4, (*b, 1), device=device)
        stop = torch.randint(k // 2, k + 1, (*b, 1), device=device)
        padding = (index < start) | (index >= stop)

        mask[index == start] = True         # both ends of the kept range valid
        mask[index == stop - 1] = True
        mask[padding] = False
        seq_times[padding] = 0.0
        return args


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


class TrajectoryTranslationsOnlyTest(TrajectorySampleTest):
    """No quaternions: the rotation outputs are stride-0 dummies.

    The rotation output is the first one, and the kernel used to size its
    launch from it: only the first batch element was computed.
    """
    def setup_args(self, dtype, device) -> list[torch.Tensor]:
        args = super().setup_args(dtype, device)
        args[2] = None
        return args


class TrajectoryRotationsOnlyTest(TrajectorySampleTest):
    """No translations: the translation outputs are stride-0 dummies."""
    def setup_args(self, dtype, device) -> list[torch.Tensor]:
        args = super().setup_args(dtype, device)
        args[3] = None
        return args


class TrajectoryDegenerateBracketTest(TrajectorySampleTest):
    """Brackets that collapse onto one keyframe.

    Rows with a single valid keyframe, and queries clamped to the first or last
    keyframe without extrapolation. The gap between the bracketing keyframes is
    zero there, which used to give NaN gradients in the reference.
    """
    masked = True

    fn = lambda *args: get_backend().sample_trajectory(
        *args, extrapolate=False,
        return_velocities=True, return_indices=True, return_mask=True)

    def setup_args(self, dtype, device) -> list[torch.Tensor]:
        args = super().setup_args(dtype, device)
        mask = args[4]
        rows = mask.shape[0] // 4
        mask[:rows] = False                 # a quarter of the rows keep one valid keyframe
        mask[:rows, self.num_keyframes // 2] = True
        return args


class TrajectorySingleKeyframeTest(TrajectorySampleTest):
    """K=1: every bracket is the same keyframe."""
    num_keyframes = 1


class TrajectoryDuplicateTimesTest(TrajectorySampleTest):
    """Pairs of keyframes with equal timestamps, queries on and between them.

    A query at or past a duplicated timestamp is bracketed by the last of the
    duplicates, with or without a mask.
    """
    masked = True

    def setup_args(self, dtype, device) -> list[torch.Tensor]:
        args = super().setup_args(dtype, device)
        query, times = args[0], args[1]
        times[..., 5] = times[..., 4]
        times[..., 12] = times[..., 11]
        times[..., 13] = times[..., 11]
        query[..., :3] = times[..., [4, 11, 13]]
        args[0] = query.sort(dim=-1).values
        return args


class TrajectorySmallGapTest(TrajectorySampleTest):
    """Keyframes a fraction of a millisecond apart (float32) or of a microsecond
    (float64): still well above the degenerate gap.

    The kernel backward used to clamp the squared gap to eps, which distorted the
    gradients w.r.t. keyframe times for gaps below sqrt(eps). float32 gradients
    are large and lose digits, hence the relative float32 tolerance.
    """
    atol = 1e-2
    rtol = 1e-2

    def setup_args(self, dtype, device) -> list[torch.Tensor]:
        args = super().setup_args(dtype, device)
        # mean keyframe gap ~0.3: about 1e-4 in float32, 1e-8 in float64
        scale = 3e-4 if dtype == torch.float32 else 3e-8
        args[0] = args[0] * scale
        args[1] = args[1] * scale
        return args


class TrajectorySharedSequenceTest(TrajectorySampleTest):
    """One keyframe sequence and mask for the whole batch of queries."""
    masked = True

    def setup_args(self, dtype, device) -> list[torch.Tensor]:
        query, times, quats, t, mask = super().setup_args(dtype, device)
        times, quats, t, mask = times[0], quats[0], t[0], mask[0]
        query = times[:1] - 1.0 + (times[-1:] - times[:1] + 2.0) * torch.rand_like(query)
        return [query, times, quats, t, mask]


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
    TrajectoryPaddedTest,
    TrajectoryTranslationsOnlyTest,
    TrajectoryRotationsOnlyTest,
    TrajectoryDegenerateBracketTest,
    TrajectorySingleKeyframeTest,
    TrajectoryDuplicateTimesTest,
    TrajectorySmallGapTest,
    TrajectorySharedSequenceTest,
]
