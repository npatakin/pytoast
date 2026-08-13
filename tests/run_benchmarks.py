"""Benchmark the fused kernels against the PyTorch reference backend.

Times every operation on both backends, on CPU and CUDA, forward and
forward+backward, and renders the charts used in the README.

    python tests/run_benchmarks.py                # both devices, write charts
    python tests/run_benchmarks.py --device cpu   # one device only
    python tests/run_benchmarks.py --quick        # few iterations, for a smoke test
    python tests/run_benchmarks.py --rerun        # ignore cached timings

Timings are cached in ``assets/benchmark-timings.json`` so the charts can be
restyled without re-measuring; pass ``--rerun`` to force fresh measurements.
"""

import argparse
import json
import os
import platform
import re
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import toast
from toast import Quat, use_backend

from benchmark_utils import plot, time_to_str

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(REPO, 'assets')
CACHE = os.path.join(ASSETS, 'benchmark-timings.json')

CPP_LABEL = 'pytoast (fused)'
TORCH_LABEL = 'PyTorch ops'

# sample_trajectory is charted separately: its problem size is (batch, keyframes,
# queries) rather than a flat element count, and its cpp path is CUDA-only.
# Pass --with-trajectory to measure and plot it anyway.
TRAJECTORY = 'sample_\ntrajectory'

ALL_OPS = ('quat_apply', 'quat_mul', 'quat_slerp', 'se3_apply',
           'se3_compose', 'se3_slerp', 'se3_inverse', TRAJECTORY)


# ----------------------------------------------------------------------
# Measurement
# ----------------------------------------------------------------------

def sync(device):
    if device == 'cuda':
        torch.cuda.synchronize()


def bench(fn, device, num_iters, warmup):
    """Wall-clock seconds per call, synchronising once per batch.

    Syncing after every call would charge each iteration a full device
    round-trip (~10-18us on CUDA) and stop launches from pipelining, which at
    small sizes costs more than the kernel itself. One sync per batch amortises
    that to 1/num_iters and measures steady-state throughput.
    """
    for _ in range(warmup):
        fn()
    sync(device)
    start = time.time()
    for _ in range(num_iters):
        fn()
    sync(device)
    return (time.time() - start) / num_iters


def peak_memory(fn, device, warmup=2):
    """Peak MiB allocated by one call, over the memory already resident."""
    if device != 'cuda':
        return None
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2 ** 20


def build_cases(device, n, b, k, q):
    """Build the input tensors and the op closures that consume them."""
    g = dict(
        q1=Quat.random(n, device=device).data.contiguous(),
        q2=Quat.random(n, device=device).data.contiguous(),
        t1=torch.randn(n, 3, device=device),
        t2=torch.randn(n, 3, device=device),
        p=torch.randn(n, 3, device=device),
        w=torch.rand(n, 1, device=device),
        seq_times=torch.sort(torch.rand(b, k, device=device) * 10, dim=-1).values,
        seq_quats=Quat.random(b, k, device=device).data.contiguous(),
        seq_t=torch.randn(b, k, 3, device=device),
        query_times=torch.sort(torch.rand(b, q, device=device) * 10, dim=-1).values,
    )
    ops = {
        'quat_apply': (
            lambda: toast.quat_apply(g['q1'], g['p']),
            ['q1', 'p']),
        'quat_mul': (
            lambda: toast.quat_mul(g['q1'], g['q2']),
            ['q1', 'q2']),
        'quat_slerp': (
            lambda: toast.quat_slerp(g['q1'], g['q2'], g['w']),
            ['q1', 'q2', 'w']),
        'se3_apply': (
            lambda: toast.se3_apply(g['q1'], g['t1'], g['p']),
            ['q1', 't1', 'p']),
        'se3_compose': (
            lambda: toast.se3_compose(g['q1'], g['t1'], g['q2'], g['t2']),
            ['q1', 't1', 'q2', 't2']),
        'se3_slerp': (
            lambda: toast.se3_slerp(g['q1'], g['t1'], g['q2'], g['t2'], g['w']),
            ['q1', 't1', 'q2', 't2', 'w']),
        'se3_inverse': (
            lambda: toast.se3_inverse(g['q1'], g['t1']),
            ['q1', 't1']),
        # newline so the label wraps on the chart
        'sample_\ntrajectory': (
            lambda: toast.sample_trajectory(
                g['query_times'], g['seq_times'], g['seq_quats'], g['seq_t']),
            ['query_times', 'seq_times', 'seq_quats', 'seq_t']),
    }
    return g, ops


def output_tensors(out):
    if isinstance(out, (tuple, list)):
        return [o for o in out if isinstance(o, torch.Tensor)]
    return [out]


def measure(device, n, b, k, q, iters_fwd, iters_bwd, skip=()):
    g, ops = build_cases(device, n, b, k, q)
    fwd, fwdbwd, memory = {}, {}, {}

    for name, (op, in_names) in ops.items():
        if name in skip:
            continue
        fwd[name], fwdbwd[name], memory[name] = {}, {}, {}

        for label, backend in [(CPP_LABEL, 'cpp'), (TORCH_LABEL, 'torch')]:
            # Everything that is not the operation itself is hoisted out of the
            # timed region: the backend switch, no_grad, and requires_grad
            # bookkeeping all cost ~2us per call and belong to neither backend.
            for key in in_names:
                g[key].requires_grad_(False)
                g[key].grad = None

            with use_backend(backend), torch.no_grad():
                fwd[name][label] = bench(
                    op, device, iters_fwd, max(3, iters_fwd // 10))

            for key in in_names:
                g[key].requires_grad_(True)

            with use_backend(backend):
                # The gradient tensors are built once, up front. Driving the
                # backward with a .sum() loss instead would add a reduction to
                # the forward and an expand to the backward, over the whole
                # tensor -- at 1M elements that costs as much as the operation
                # under test and lands in every bar equally.
                outs = [o for o in output_tensors(op()) if o.requires_grad]
                grads = [torch.ones_like(o) for o in outs]

                def run_fwd_bwd(op=op, grads=grads):
                    produced = [o for o in output_tensors(op()) if o.requires_grad]
                    if len(produced) == 1:
                        produced[0].backward(grads[0])
                    else:
                        torch.autograd.backward(produced, grads)

                fwdbwd[name][label] = bench(
                    run_fwd_bwd, device, iters_bwd, max(3, iters_bwd // 10))
                memory[name][label] = peak_memory(run_fwd_bwd, device)

            for key in in_names:
                g[key].requires_grad_(False)
                g[key].grad = None

        print('  [{}] {:<20} fwd {:>9} vs {:>9}   fwd+bwd {:>9} vs {:>9}'.format(
            device, name.replace('\n', ''),
            time_to_str(fwd[name][CPP_LABEL]), time_to_str(fwd[name][TORCH_LABEL]),
            time_to_str(fwdbwd[name][CPP_LABEL]), time_to_str(fwdbwd[name][TORCH_LABEL])),
            flush=True)

    return {'fwd': fwd, 'fwdbwd': fwdbwd, 'memory_mib': memory}


# ----------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------

ORANGE = '#E94C23'
THEMES = {
    'dark': dict(style='dark_background', bg='#162129', fg='#E8EDF2', other='#4E6376'),
    'light': dict(style='default', bg='#FFFFFF', fg='#162129', other='#9FB1BF'),
}


BAR_WIDTH = 0.34
BAR_XMUL = 1.15


def annotate_speedups(ax, timings):
    """Print the reference/fused ratio above each operation's bar group."""
    import numpy as np

    num_series = len(next(iter(timings.values())))
    centers = np.arange(len(timings)) * BAR_XMUL + 0.5 * (num_series - 1) * BAR_WIDTH
    for center, op in zip(centers, timings):
        fused, reference = timings[op][CPP_LABEL], timings[op][TORCH_LABEL]
        ax.annotate(
            '×{:.1f}'.format(reference / fused),
            xy=(center, max(fused, reference)),
            xytext=(0, 26), textcoords='offset points',
            ha='center', va='bottom',
            color=ORANGE, fontsize=13, fontweight='bold',
        )


# Fewer operations for the side-by-side figure, so two panels keep roughly the
# aspect ratio of a single full-width chart.
COMBINED_OPS = ('quat_apply', 'quat_slerp', 'se3_apply', 'se3_slerp')


def theme_rc(theme):
    from cycler import cycler

    return {
        'axes.prop_cycle': cycler(color=[ORANGE, theme['other']]),
        'figure.facecolor': theme['bg'],
        'axes.facecolor': theme['bg'],
        'text.color': theme['fg'],
        'axes.labelcolor': theme['fg'],
        'xtick.color': theme['fg'],
        'ytick.color': theme['fg'],
        'font.size': 11,
    }


def style_axes(ax, theme, ylabel=None, legend=True):
    ax.margins(y=0.42)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color(theme['fg'])
    if ylabel:
        ax.set_ylabel(ylabel, color=theme['fg'])
    if legend:
        legend_obj = ax.legend(frameon=False, loc='upper left', ncol=2)
        for text in legend_obj.get_texts():
            text.set_color(theme['fg'])
    elif ax.get_legend():
        ax.get_legend().remove()


def render_combined(data, kind, suptitle, out_name, theme, ops=COMBINED_OPS):
    """One figure, CPU on the left panel and CUDA on the right."""
    import matplotlib.pyplot as plt

    with plt.style.context(theme['style']), plt.rc_context(theme_rc(theme)):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=160)
        for ax, device in zip(axes, ('cpu', 'cuda')):
            subset = {op: data[device][kind][op] for op in ops if op in data[device][kind]}
            plt.sca(ax)
            plot(subset, '', width=BAR_WIDTH, xmul=BAR_XMUL)
            annotate_speedups(ax, subset)
            hardware = data[device].get('label', device)
            hardware = hardware.split(', ', 1)[1].rsplit(', fp32', 1)[0]
            ax.set_title('{} — {}'.format(device.upper(), hardware),
                         color=theme['fg'], pad=12, fontsize=12)
            style_axes(ax, theme,
                       ylabel='time per call (log scale)' if device == 'cpu' else None,
                       legend=(device == 'cpu'))
        fig.suptitle(suptitle, color=theme['fg'], fontsize=13)
        plt.tight_layout()
        fig.savefig(os.path.join(ASSETS, out_name), facecolor=theme['bg'])
        plt.close(fig)
    print('  wrote assets/{}'.format(out_name))


def render(timings, title, out_name, theme):
    import matplotlib.pyplot as plt
    from cycler import cycler

    rc = {
        'axes.prop_cycle': cycler(color=[ORANGE, theme['other']]),
        'figure.facecolor': theme['bg'],
        'axes.facecolor': theme['bg'],
        'text.color': theme['fg'],
        'axes.labelcolor': theme['fg'],
        'xtick.color': theme['fg'],
        'ytick.color': theme['fg'],
        'font.size': 11,
    }
    with plt.style.context(theme['style']), plt.rc_context(rc):
        fig = plt.figure(figsize=(11, 4.6), dpi=160)
        plot(timings, title, width=BAR_WIDTH, xmul=BAR_XMUL)
        ax = plt.gca()
        annotate_speedups(ax, timings)
        ax.set_ylabel('time per call (log scale)', color=theme['fg'])
        ax.margins(y=0.42)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.spines['bottom'].set_color(theme['fg'])
        legend = ax.legend(frameon=False, loc='upper left', ncol=2)
        for text in legend.get_texts():
            text.set_color(theme['fg'])
        ax.set_title(title, color=theme['fg'], pad=14)
        plt.tight_layout()
        fig.savefig(os.path.join(ASSETS, out_name), facecolor=theme['bg'])
        plt.close(fig)
    print('  wrote assets/{}'.format(out_name))


# ----------------------------------------------------------------------

def cpu_name():
    name = None
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    name = line.split(':', 1)[1].strip()
                    break
    except OSError:
        pass
    name = name or platform.processor() or 'CPU'
    # "AMD Ryzen 9 5950X 16-Core Processor" -> "AMD Ryzen 9 5950X"
    name = re.sub(r'\s*\d+-Core Processor', '', name)
    name = re.sub(r'\s*(CPU)?\s*@.*$', '', name)
    return re.sub(r'\((R|TM)\)', '', name).strip()


def describe(device, n):
    elements = '{}M elements'.format(n // 10 ** 6) if n >= 10 ** 6 else '{} elements'.format(n)
    if device == 'cuda':
        return '{}, {}, fp32'.format(elements, torch.cuda.get_device_name(0))
    return '{}, {} ({} threads), fp32'.format(
        elements, cpu_name(), torch.get_num_threads())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'both'], default='both')
    parser.add_argument('--elements', type=int, default=1 << 20)
    parser.add_argument('--tag', default=None,
                        help='suffix for chart/cache filenames, e.g. "1k"; '
                             'use it so runs at different sizes do not overwrite '
                             'each other')
    parser.add_argument('--traj-batch', type=int, default=4096,
                        help='batch size for sample_trajectory (--with-trajectory)')
    parser.add_argument('--with-trajectory', action='store_true',
                        help='also measure sample_trajectory (CUDA only)')
    parser.add_argument('--combined-only', action='store_true',
                        help='render only the side-by-side CPU|CUDA figure, and '
                             'measure only the operations it shows')
    parser.add_argument('--iters', type=int, default=None,
                        help='forward iterations per measurement (overrides default)')
    parser.add_argument('--quick', action='store_true',
                        help='few iterations, for checking the script runs')
    parser.add_argument('--rerun', action='store_true',
                        help='ignore cached timings and measure again')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()

    suffix = '-{}'.format(args.tag) if args.tag else ''
    cache = CACHE if not args.tag else CACHE.replace('.json', '{}.json'.format(suffix))

    devices = ['cpu', 'cuda'] if args.device == 'both' else [args.device]
    if 'cuda' in devices and not torch.cuda.is_available():
        print('CUDA is not available, benchmarking CPU only')
        devices = [d for d in devices if d != 'cuda']

    b, k, q = args.traj_batch, 64, 64
    iters = {
        'cuda': (20, 10) if args.quick else (200, 100),
        'cpu': (5, 3) if args.quick else (30, 15),
    }
    if args.iters:
        iters = {d: (args.iters, max(1, args.iters // 2)) for d in iters}

    # sample_trajectory is off by default, and has no cpp path on CPU at all
    def skip_for(device):
        skipped = set() if (args.with_trajectory and device == 'cuda') else {TRAJECTORY}
        if args.combined_only:
            skipped |= {op for op in ALL_OPS if op not in COMBINED_OPS}
        return tuple(skipped)

    data = {}
    if os.path.exists(cache) and not args.rerun:
        data = json.load(open(cache))
        print('loaded cached timings from {}'.format(os.path.basename(cache)))

    torch.manual_seed(0)
    for device in devices:
        if device in data and not args.rerun:
            continue
        print('measuring on {}...'.format(device))
        iters_fwd, iters_bwd = iters[device]
        data[device] = measure(
            device, args.elements, b, k, q, iters_fwd, iters_bwd,
            skip=skip_for(device))
        data[device]['label'] = describe(device, args.elements)

    # refresh cached labels so restyling picks up any formatting changes
    for device in data:
        if device == 'cuda' and not torch.cuda.is_available():
            continue
        data[device]['label'] = describe(device, args.elements)

    os.makedirs(ASSETS, exist_ok=True)
    json.dump(data, open(cache, 'w'), indent=2)

    def charted(device, kind):
        """Cached timings minus anything excluded from this run."""
        return {op: v for op, v in data[device][kind].items()
                if op not in skip_for(device)}

    for device in data:
        for kind in ('fwd', 'fwdbwd'):
            ratios = [v[TORCH_LABEL] / v[CPP_LABEL] for v in charted(device, kind).values()]
            print('{} {}: speedup {:.1f}x - {:.1f}x'.format(
                device, kind, min(ratios), max(ratios)))

    if args.no_plots:
        return

    titles = {
        'fwd': '{} — forward pass ({})',
        'fwdbwd': '{} — forward + backward ({})',
    }
    for device in (() if args.combined_only else data):
        label = data[device].get('label', device)
        for kind, template in titles.items():
            for theme_name, theme in THEMES.items():
                render(
                    charted(device, kind),
                    template.format(device.upper(), label),
                    'bench-{}-{}{}-{}.png'.format(
                        device, 'forward' if kind == 'fwd' else 'forward-backward',
                        suffix, theme_name),
                    theme)

    if 'cpu' in data and 'cuda' in data:
        size = describe('cpu', args.elements).split(',')[0]
        for kind, what in [('fwd', 'forward pass'), ('fwdbwd', 'forward + backward')]:
            for theme_name, theme in THEMES.items():
                render_combined(
                    data, kind,
                    '{} — {}, fp32'.format(what.capitalize(), size),
                    'bench-combined-{}{}-{}.png'.format(
                        'forward' if kind == 'fwd' else 'forward-backward',
                        suffix, theme_name),
                    theme)


if __name__ == '__main__':
    main()
