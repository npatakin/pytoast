"""Trajectory sampling with inputs and gradients in unusual memory layouts.

The kernels read each operand row by row through typed pointers, so anything
other than plain contiguous rows has to be handled explicitly: slices along
the keyframes, broadcast masks, transposed or misaligned storage, gradients
that are slices of a concatenation.

The layout tests compare the cpp backend on such a layout with the cpp backend
on contiguous copies broadcast to the full batch, so the reference shares none
of the layout handling under test.
"""
import torch

from test_utils import TestCase, BackendCrossTest, Failure, _fmt_dtype, _fmt_device
from interpolation_test import TrajectorySampleTest
from toast import get_backend, use_backend


def sample(*args):
    return get_backend().sample_trajectory(
        *args, return_velocities=True, return_indices=True, return_mask=True)


def canonical(args):
    """Every sequence operand broadcast to the full batch and made contiguous.

    clone() keeps the autograd graph, so gradients still reach the original
    tensors and can be compared with the ones of the layout under test.
    """
    query, times, quats, t = args[:4]
    mask = args[4] if len(args) > 4 else None
    batch = torch.broadcast_shapes(query.shape[:-1], times.shape[:-1])
    k = times.shape[-1]

    def full(x, row):
        if x is None:
            return None
        return x.expand(*batch, *row).clone(memory_format=torch.contiguous_format)

    return [full(query, (query.shape[-1],)), full(times, (k,)), full(quats, (k, 4)),
            full(t, (k, 3)), full(mask, (k,))]


class TrajectoryLayoutTest(TestCase):
    """Base: contiguous inputs, compared with their canonical copies."""
    num_keyframes = 24
    num_queries = 9
    batch_shape = (16,)
    masked = True

    def base_args(self, dtype, device, num_keyframes=None):
        case = TrajectorySampleTest()
        case.num_keyframes = num_keyframes or self.num_keyframes
        case.num_queries = self.num_queries
        case.batch_shape = self.batch_shape
        case.masked = self.masked
        return case.setup_args(dtype, device)

    def setup_args(self, dtype, device):
        return self.base_args(dtype, device)

    def test_fn(self, *args):
        with use_backend('cpp'):
            return sample(*args)

    def ref_fn(self, *args):
        with use_backend('cpp'):
            return sample(*canonical(args))


class TrajectoryRowSlicedTest(TrajectoryLayoutTest):
    """Keyframes sliced out of longer rows: row strides are not multiples of K."""
    def setup_args(self, dtype, device):
        query, times, quats, t, mask = self.base_args(dtype, device, self.num_keyframes + 1)
        mask[..., 1] = True     # the new first keyframe stays valid
        return [query, times[..., 1:], quats[..., 1:, :], t[..., 1:, :], mask[..., 1:]]


class TrajectoryBroadcastMaskTest(TrajectoryLayoutTest):
    """One mask for the whole batch, and keyframe times shared by the batch."""
    def setup_args(self, dtype, device):
        query, times, quats, t, mask = self.base_args(dtype, device)
        times = times[:1].clone()
        mask = mask[0].clone()
        # queries must lie around the shared keyframes
        query = times[..., :1] - 1.0 + (times[..., -1:] - times[..., :1] + 2.0) * torch.rand_like(query)
        return [query, times, quats, t, mask]


class TrajectoryTransposedTest(TrajectoryLayoutTest):
    """Batch-major storage: the keyframe dimension has a stride other than 1."""
    def setup_args(self, dtype, device):
        query, times, quats, t, mask = self.base_args(dtype, device)
        times = times.t().contiguous().t()
        mask = mask.t().contiguous().t()
        quats = quats.permute(2, 1, 0).contiguous().permute(2, 1, 0)
        t = t.permute(1, 0, 2).contiguous().permute(1, 0, 2)
        return [query, times, quats, t, mask]


class TrajectoryMisalignedTest(TrajectoryLayoutTest):
    """Storage starting one element off a vector boundary."""
    def setup_args(self, dtype, device):
        args = self.base_args(dtype, device)

        def shifted(x):
            store = torch.empty(x.numel() + 1, dtype=x.dtype, device=x.device)
            return store[1:].view(x.shape).copy_(x)

        args[2] = shifted(args[2])      # quaternions, loaded as one vec4 each
        args[3] = shifted(args[3])
        return args


class TrajectoryGradientLayoutTest(BackendCrossTest):
    """Gradients that are slices of a concatenation: strided and misaligned."""
    num_keyframes = 24
    num_queries = 9
    batch_shape = (16,)

    @staticmethod
    def fn(*args):
        out_q, out_t, _, out_v, out_w, _ = sample(*args)
        return torch.cat([out_t, out_q, out_v, out_w], dim=-1)

    def setup_args(self, dtype, device):
        case = TrajectorySampleTest()
        case.num_keyframes, case.num_queries, case.batch_shape = (
            self.num_keyframes, self.num_queries, self.batch_shape)
        return case.setup_args(dtype, device)


class TrajectoryInputValidationTest(TestCase):
    """Invalid inputs raise instead of being misread; edge sizes work."""

    def expect_raise(self, name, dtype, device, fn):
        try:
            fn()
        except RuntimeError:
            return
        self.fail(Failure(test=self.__class__.__name__, phase='forward',
                          dtype=_fmt_dtype(dtype), device=_fmt_device(device),
                          summary='{}: expected an error, the call succeeded'.format(name)))

    def run_forward_test(self, dtype, device):
        opts = dict(dtype=dtype, device=device)
        times = torch.arange(6, **opts).view(1, 6).expand(3, 6).contiguous()
        query = torch.full((3, 2), 2.5, **opts)
        t = torch.randn(3, 6, 3, **opts)
        mask = torch.ones(3, 6, dtype=torch.bool, device=device)
        other = torch.float64 if dtype == torch.float32 else torch.float32

        with use_backend('cpp'):
            self.expect_raise('no keyframes', dtype, device, lambda: get_backend().sample_trajectory(
                query, times[:, :0], seq_t=t[:, :0]))
            self.expect_raise('extrapolation window 0', dtype, device, lambda: get_backend().sample_trajectory(
                query, times, seq_t=t, extrapolation_window=0))
            self.expect_raise('negative extrapolation window', dtype, device, lambda: get_backend().sample_trajectory(
                query, times, seq_t=t, extrapolation_window=-1))
            self.expect_raise('query dtype', dtype, device, lambda: get_backend().sample_trajectory(
                query.to(other), times, seq_t=t))
            self.expect_raise('translation dtype', dtype, device, lambda: get_backend().sample_trajectory(
                query, times, seq_t=t.to(other)))
            self.expect_raise('uint8 mask', dtype, device, lambda: get_backend().sample_trajectory(
                query, times, seq_t=t, seq_mask=mask.to(torch.uint8) * 255))

            # empty batch and no queries, with a mask
            for q, tm, tt, mk in [(query[:0], times[:0], t[:0], mask[:0]),
                                  (query[:, :0], times, t, mask)]:
                out = get_backend().sample_trajectory(q, tm, seq_t=tt, seq_mask=mk, return_indices=True)
                assert out[1].numel() == 0 and out[5].numel() == 0

            # a single keyframe gives its own pose everywhere
            out = get_backend().sample_trajectory(query, times[:, :1], seq_t=t[:, :1])
            assert torch.equal(out[1], t[:, :1].expand(3, 2, 3))

    def run_backward_test(self, dtype, device):
        opts = dict(dtype=dtype, device=device)
        query = torch.full((3, 2), 2.5, **opts)
        with use_backend('cpp'):
            # a single keyframe receives the gradient of every query
            t = torch.randn(3, 1, 3, **opts, requires_grad=True)
            out = get_backend().sample_trajectory(query, torch.zeros(3, 1, **opts), seq_t=t)[1]
            grad, = torch.autograd.grad(out.sum(), t)
            assert torch.equal(grad, torch.full_like(t, 2.0))

            # an empty batch gives an empty gradient
            t = torch.randn(0, 6, 3, **opts, requires_grad=True)
            times = torch.arange(6, **opts).expand(0, 6)
            out = get_backend().sample_trajectory(query[:0], times, seq_t=t)[1]
            grad, = torch.autograd.grad(out.sum(), t, allow_unused=True)
            assert grad is None or grad.shape == t.shape


__tests__ = [
    TrajectoryLayoutTest,
    TrajectoryRowSlicedTest,
    TrajectoryBroadcastMaskTest,
    TrajectoryTransposedTest,
    TrajectoryMisalignedTest,
    TrajectoryGradientLayoutTest,
    TrajectoryInputValidationTest,
]
