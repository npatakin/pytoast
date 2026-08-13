"""Cross-backend correctness harness.

Every test case runs the same op through the `cpp` backend and through the
`torch` reference implementation and compares the results (and their gradients)
element-wise.  A mismatch, a shape/dtype disagreement or an exception is turned
into a `Failure` record: it is printed loudly as soon as it happens *and*
returned to the runner, which aggregates everything, prints a summary and sets
the process exit code.  Nothing is ever swallowed.
"""

import os
import sys
import traceback
from dataclasses import dataclass, field

import torch
from termcolor import colored

from toast import use_backend


# --------------------------------------------------------------------------
# output helpers
# --------------------------------------------------------------------------

_USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def paint(text: str, color: str, attrs=None) -> str:
    """`termcolor.colored`, but a no-op when the output is not a terminal."""
    if not _USE_COLOR:
        return text
    return colored(text, color, attrs=attrs)


# --------------------------------------------------------------------------
# tolerances
# --------------------------------------------------------------------------
#
# float32 has ~1.19e-07 of machine epsilon, float64 ~2.22e-16 -- nine orders of
# magnitude apart.  A single threshold shared by both dtypes is therefore
# meaningless: 1e-5 is a reasonable bound for a float32 kernel but it accepts a
# float64 result that is wrong by ~1e11 machine epsilons, i.e. it does not test
# the float64 kernels at all.  Tolerances are consequently per dtype.
#
#   float32: `atol` / `rtol` class attributes below.  These are the historical
#            values and are deliberately left untouched, as are the per-test
#            overrides in quat_tests.py / se3_tests.py / velocities_test.py.
#
#   float64: `atol_f64` / `rtol_f64` if a test declares them, otherwise
#            `min(<the float32 value>, FLOAT64_ATOL/RTOL)` -- i.e. never looser
#            than the global float64 default and never looser than whatever the
#            test already declared.
#
# The float64 defaults are calibrated from the error actually observed across
# the whole suite (worst case ~1.1e-14 absolute, on 1000-element batches, for
# the multi-step ops such as `velocities` and `angular_velocity`) with roughly
# four orders of magnitude of headroom.  That is tight enough to catch a real
# numerical regression and loose enough to tolerate the benign reassociation
# differences between a fused CUDA kernel and a chain of torch ops.
FLOAT64_ATOL = 1e-10
FLOAT64_RTOL = 1e-9


# --------------------------------------------------------------------------
# failure records
# --------------------------------------------------------------------------

@dataclass
class Failure:
    """One failed check.  Knows how to print itself."""

    test: str
    phase: str          # 'forward' | 'backward' | 'setup'
    dtype: str
    device: str
    summary: str
    details: list = field(default_factory=list)
    tb: str = None
    expected_reason: str = None     # set when this is a declared known failure

    @property
    def tag(self) -> str:
        return "XFAIL" if self.expected_reason else "FAIL"

    @property
    def header(self) -> str:
        return "[{}][{}][{}] dtype={} device={}".format(
            self.test, self.phase, self.tag, self.dtype, self.device)

    def render(self) -> str:
        lines = ["{} {}".format(self.header, self.summary)]
        lines += ["    " + d for d in self.details]
        if self.expected_reason:
            lines.append("    KNOWN ISSUE: {}".format(self.expected_reason))
        if self.tb:
            lines += ["    " + line for line in self.tb.rstrip().splitlines()]
        return "\n".join(lines)


@dataclass
class TestReport:
    """Outcome of running a single `TestCase`."""

    name: str
    checks_run: int = 0
    checks_passed: int = 0
    checks_skipped: int = 0
    failures: list = field(default_factory=list)
    known_failures: list = field(default_factory=list)

    @property
    def checks_failed(self) -> int:
        return len(self.failures)


def _fmt_dtype(dtype) -> str:
    return str(dtype)


def _fmt_device(device) -> str:
    return str(device)


class TestCase:
    test_fn = None
    ref_fn = None

    # float32 tolerances (also the fallback for any non-float64 dtype).
    atol = 1e-5
    rtol = 1e-4

    # float64 tolerances; `None` means "derive from the float32 values and the
    # global float64 defaults" -- see the comment block above.
    atol_f64 = None
    rtol_f64 = None

    # Declared known-bad checks: {(phase, dtype): "why"}.  Such a check is
    # still executed and still printed in full, but as `[XFAIL]` instead of
    # `[FAIL]`, and it does not turn the suite red.  This exists so that a
    # known defect in the C++ side stays *visible* on every run instead of
    # being papered over by a loosened tolerance.  If a declared known failure
    # starts passing, the suite fails with an `XPASS` so the waiver gets
    # removed.  Use it only with a comment naming the root cause.
    known_failures = {}

    def __init__(
            self,
            size: int = 1000,
            dtypes: list[torch.dtype] = (torch.float32, torch.double),
            devices: list[torch.device] = (torch.device("cpu"), torch.device("cuda")),
    ):
        self.dtypes = dtypes
        self.devices = devices
        self.size = size
        self.report = TestReport(self.__class__.__name__)
        self._expected_reason = None

    def setup_args(self, dtype, device) -> list[torch.Tensor]:
        return []

    # ------------------------------------------------------------------
    # tolerances
    # ------------------------------------------------------------------

    def tolerances(self, dtype) -> tuple[float, float]:
        """(atol, rtol) for `dtype`.  Never looser than the class attributes."""
        if dtype == torch.float64:
            atol = self.atol_f64 if self.atol_f64 is not None else min(self.atol, FLOAT64_ATOL)
            rtol = self.rtol_f64 if self.rtol_f64 is not None else min(self.rtol, FLOAT64_RTOL)
            return atol, rtol
        return self.atol, self.rtol

    # ------------------------------------------------------------------
    # driving
    # ------------------------------------------------------------------

    def run(self) -> TestReport:
        self.report = TestReport(self.__class__.__name__)

        for device in self.devices:
            for dtype in self.dtypes:
                if not self._device_available(device):
                    self.report.checks_skipped += 2
                    print(paint(
                        "[{}][skipped] dtype={} device={} (device unavailable)".format(
                            self.__class__.__name__, _fmt_dtype(dtype), _fmt_device(device)),
                        "yellow"))
                    continue
                self._run_phase("forward", self.run_forward_test, dtype, device)
                self._run_phase("backward", self.run_backward_test, dtype, device)

        return self.report

    @staticmethod
    def _device_available(device) -> bool:
        if torch.device(device).type == "cuda":
            return torch.cuda.is_available()
        return True

    def _run_phase(self, phase, fn, dtype, device):
        """Run one phase, converting any exception into a reported failure."""
        self.report.checks_run += 1
        failures_before = len(self.report.failures)
        known_before = len(self.report.known_failures)

        self._expected_reason = self.known_failures.get((phase, dtype))
        try:
            fn(dtype, device)
        except Exception as exc:                      # noqa: BLE001 - deliberate
            self.fail(Failure(
                test=self.__class__.__name__,
                phase=phase,
                dtype=_fmt_dtype(dtype),
                device=_fmt_device(device),
                summary="raised {}: {}".format(type(exc).__name__, exc),
                tb=traceback.format_exc(),
            ))
        finally:
            expected_reason, self._expected_reason = self._expected_reason, None

        failed = len(self.report.failures) > failures_before
        xfailed = len(self.report.known_failures) > known_before

        if expected_reason and not failed and not xfailed:
            # The waiver is stale: delete it rather than leave it lying around.
            self.fail(Failure(
                test=self.__class__.__name__, phase=phase,
                dtype=_fmt_dtype(dtype), device=_fmt_device(device),
                summary="XPASS: declared known failure now passes -- remove the "
                        "`known_failures` entry ({})".format(expected_reason)))
            failed = True

        if not failed and not xfailed:
            self.report.checks_passed += 1

    def fail(self, failure) -> None:
        """Record a failure and print it immediately, loudly."""
        if isinstance(failure, str):        # backwards-compatible free-form call
            failure = Failure(
                test=self.__class__.__name__, phase="-", dtype="-", device="-",
                summary=failure)

        expected_reason = getattr(self, "_expected_reason", None)
        if expected_reason and failure.expected_reason is None:
            failure.expected_reason = expected_reason
            self.report.known_failures.append(failure)
            print(paint(failure.render(), "yellow"), flush=True)
            return

        self.report.failures.append(failure)
        print(paint(failure.render(), "red"), flush=True)

    def as_list(self, outs):
        if not (isinstance(outs, tuple) or isinstance(outs, list)):
            return [outs]
        return list(outs)

    # ------------------------------------------------------------------
    # comparison
    # ------------------------------------------------------------------

    def check_results(self, test_out, ref_out, phase, dtype, device, what="output"):
        """Compare backend output against the reference.

        Returns `(ok, diffs)` where `diffs` is one `(max, mean)` pair per
        output.  On mismatch a `Failure` is recorded (and printed) describing
        the actual error against the tolerated one.
        """
        ref_out = self.as_list(ref_out)
        test_out = self.as_list(test_out)
        atol, rtol = self.tolerances(dtype)

        details = []
        diffs = []

        if len(test_out) != len(ref_out):
            self.fail(Failure(
                test=self.__class__.__name__, phase=phase,
                dtype=_fmt_dtype(dtype), device=_fmt_device(device),
                summary="backend returned {} {}s, reference returned {}".format(
                    len(test_out), what, len(ref_out))))
            return False, diffs

        for i, (cur_test_out, cur_ref_out) in enumerate(zip(test_out, ref_out)):
            if cur_test_out is None or cur_ref_out is None:
                if cur_test_out is not cur_ref_out:
                    details.append(
                        "{} #{}: backend={} reference={} (one side is missing)".format(
                            what, i,
                            "None" if cur_test_out is None else "tensor",
                            "None" if cur_ref_out is None else "tensor"))
                diffs.append(None)
                continue

            if cur_test_out.shape != cur_ref_out.shape:
                details.append("{} #{}: shape mismatch backend={} reference={}".format(
                    what, i, tuple(cur_test_out.shape), tuple(cur_ref_out.shape)))
                diffs.append(None)
                continue

            if cur_test_out.dtype != cur_ref_out.dtype:
                details.append("{} #{}: dtype mismatch backend={} reference={}".format(
                    what, i, cur_test_out.dtype, cur_ref_out.dtype))
                diffs.append(None)
                continue

            # Integer / boolean outputs (indices, masks) must match EXACTLY --
            # a tolerance is meaningless for them, and `-` is not even defined
            # for bool tensors.
            if not cur_ref_out.is_floating_point():
                mismatched = (cur_test_out != cur_ref_out)
                num_bad = int(mismatched.sum())
                if num_bad:
                    flat = mismatched.flatten().nonzero()[0].item()
                    details.append(
                        "{} #{}: {} of {} exact-match elements differ "
                        "(first at flat index {}: backend={} reference={})".format(
                            what, i, num_bad, cur_ref_out.numel(), flat,
                            cur_test_out.flatten()[flat].item(),
                            cur_ref_out.flatten()[flat].item()))
                diffs.append(None)
                continue

            diff = (cur_test_out - cur_ref_out).abs()
            allowed = atol + rtol * cur_ref_out.abs()
            # `x > y` is False whenever either side is NaN, so NaNs have to be
            # caught separately; `==` keeps matching infinities (inf - inf = NaN).
            bad = torch.gt(diff, allowed) | (torch.isnan(diff) & ~(cur_test_out == cur_ref_out))

            if diff.numel() == 0:
                # an empty batch: shapes and dtypes already matched above, and
                # amax()/mean() are undefined on zero elements
                diffs.append((0.0, 0.0))
                continue

            diff_max = diff.amax().item()
            diff_mean = diff.mean().item()
            diffs.append((diff_max, diff_mean))

            n_bad = int(bad.sum().item())
            if n_bad == 0:
                continue

            n_total = bad.numel()
            # Worst offender = largest excess over its own allowance; NaNs win.
            excess = torch.nan_to_num(diff - allowed, nan=float("inf"))
            flat_idx = int(excess.flatten().argmax().item())
            worst_err = diff.flatten()[flat_idx].item()
            worst_allowed = allowed.flatten()[flat_idx].item()
            worst_ref = cur_ref_out.flatten()[flat_idx].item()
            worst_test = cur_test_out.flatten()[flat_idx].item()

            details.append(
                "{} #{}: {}/{} elements outside tolerance "
                "(max_abs_err={:.6E}, mean_abs_err={:.6E})".format(
                    what, i, n_bad, n_total, diff_max, diff_mean))
            details.append(
                "    worst element (flat index {}): backend={:.12G} reference={:.12G} "
                "-> abs_err={:.6E}, tolerated={:.6E} (atol={:.3E} + rtol={:.3E} * |ref|={:.6E})"
                .format(flat_idx, worst_test, worst_ref, worst_err, worst_allowed,
                        atol, rtol, abs(worst_ref)))

            n_nonfinite = int((~torch.isfinite(cur_test_out)).sum().item())
            if n_nonfinite:
                details.append("    backend produced {} non-finite value(s)".format(n_nonfinite))

        if details:
            self.fail(Failure(
                test=self.__class__.__name__, phase=phase,
                dtype=_fmt_dtype(dtype), device=_fmt_device(device),
                summary="{} of {} {}(s) mismatched (atol={:.3E}, rtol={:.3E})".format(
                    sum(1 for d in details if d.startswith(what)), len(ref_out), what,
                    atol, rtol),
                details=details))
            return False, diffs

        return True, diffs

    @staticmethod
    def _format_diffs(diffs) -> str:
        return ', '.join(
            '[max={:.4E}, mean={:.4E}]'.format(d[0], d[1]) if d is not None else '[n/a]'
            for d in diffs)

    # ------------------------------------------------------------------
    # phases
    # ------------------------------------------------------------------

    def run_forward_test(self, dtype: torch.dtype, device: torch.device):
        args = self.setup_args(dtype, device)
        test_out = self.test_fn(*args)
        ref_out = self.ref_fn(*args)

        ok, diffs = self.check_results(test_out, ref_out, 'forward', dtype, device)
        if ok:
            print(paint(
                "[{}][forward][OK] dtype={} device={} diffs=[{}]".format(
                    self.__class__.__name__, _fmt_dtype(dtype), _fmt_device(device),
                    self._format_diffs(diffs)), "green"))

    def run_backward_test(self, dtype: torch.dtype, device: torch.device):
        args = self.setup_args(dtype, device)

        grad_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.is_floating_point():
                arg.requires_grad_(True)
                grad_args.append(arg)

        if not grad_args:
            print(paint(
                "[{}][backward][skipped] dtype={} device={} (no differentiable inputs)".format(
                    self.__class__.__name__, _fmt_dtype(dtype), _fmt_device(device)),
                "yellow"))
            return

        ref_out = self.as_list(self.ref_fn(*args))
        test_out = self.as_list(self.test_fn(*args))

        # Compare the output structure first: `torch.autograd.backward` would
        # otherwise blow up with a much less informative message.  An operation
        # may legitimately return None for an output the caller did not request,
        # so compare those positionally rather than dereferencing them.
        def structure(outs):
            return [None if o is None else tuple(o.shape) for o in outs]

        if structure(test_out) != structure(ref_out):
            self.check_results(test_out, ref_out, 'backward', dtype, device)
            return

        # Skip anything that carries no gradient: outputs the op did not produce
        # (None), and integer / boolean outputs such as bracketing indices and
        # validity masks -- those are checked exactly by the forward test.
        # randn_like fails on them and autograd.backward rejects them outright.
        differentiable = [i for i, r in enumerate(ref_out)
                          if r is not None and r.is_floating_point()
                          and r.requires_grad]
        if not differentiable:
            print(paint(
                "[{}][backward][skipped] dtype={} device={} (no differentiable outputs)".format(
                    self.__class__.__name__, _fmt_dtype(dtype), _fmt_device(device)),
                "yellow"))
            return
        ref_out = [ref_out[i] for i in differentiable]
        test_out = [test_out[i] for i in differentiable]

        grads_out = [torch.randn_like(x) for x in ref_out]

        torch.autograd.backward(ref_out, grads_out)
        grads_ref = [a.grad.clone() if a.grad is not None else None for a in grad_args]
        for arg in grad_args:
            arg.grad = None

        torch.autograd.backward(test_out, grads_out)
        grads_test = [a.grad.clone() if a.grad is not None else None for a in grad_args]
        for arg in grad_args:
            arg.grad = None

        # Inputs that receive no gradient from *either* side carry no signal.
        keep = [i for i in range(len(grad_args))
                if not (grads_ref[i] is None and grads_test[i] is None)]
        grads_ref = [grads_ref[i] for i in keep]
        grads_test = [grads_test[i] for i in keep]

        ok, diffs = self.check_results(
            grads_test, grads_ref, 'backward', dtype, device, what='grad')
        if ok:
            print(paint(
                "[{}][backward][OK] dtype={} device={} diffs=[{}]".format(
                    self.__class__.__name__, _fmt_dtype(dtype), _fmt_device(device),
                    self._format_diffs(diffs)), "green"))


class BackendCrossTest(TestCase):
    fn = None

    def __init__(
            self,
            size: int = 1000,
            test_backend: str = 'cpp',
            ref_backend: str = 'torch',
            dtypes: list[torch.dtype] = (torch.float32, torch.float64),
            devices: list[torch.device] = ('cpu', 'cuda')
    ):
        super().__init__(size=size, dtypes=dtypes, devices=devices)
        self.test_backend = test_backend
        self.ref_backend = ref_backend

    def ref_fn(self, *args):
        with use_backend(self.ref_backend):
            return self.__class__.fn(*args)

    def test_fn(self, *args):
        with use_backend(self.test_backend):
            return self.__class__.fn(*args)
