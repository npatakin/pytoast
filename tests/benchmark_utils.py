import time
import numpy as np
import torch


def benchmark(fn, args, num_iters: int = 1000, warmup: int = 5):
    def iter():
        torch.cuda.synchronize()
        result = fn(*args)
        torch.cuda.synchronize()

    for _ in range(warmup):
        iter()

    s = time.time()
    for _ in range(num_iters):
        iter()
    t = time.time()
    return (t-s) / num_iters


def benchmark_fns(fns, args, num_iters: int = 10000, warmup: int = 5):
    results = {}
    for fn_name, fn in fns.items():
        results[fn_name] = benchmark(fn, args, num_iters=num_iters, warmup=warmup)
    return results


def run_benchmark(cases, fns, num_iters: int = 1000, warmup: int = 5):
    timings = {}
    for case_name in cases:
        timings[case_name] = benchmark_fns(
            fns, cases[case_name],
            num_iters=num_iters,
            warmup=warmup
        )
    return timings


def time_to_str(t):
    if t < 1e-3:
        return '{:.0f} us'.format(t/1e-6)
    if t < 1:
        return '{:.1f} ms'.format(t/1e-3)
    return '{:.2f} s'.format(t)


def plot(timings, title: str, width = 0.25, xmul = 1):
    import matplotlib.pyplot as plt
    fig, ax = plt.gcf(), plt.gca()
    x = np.arange(len(timings)) * xmul

    fns = list(next(iter(timings.values())).keys())
    for i, fn_name in enumerate(fns):
        rects = ax.bar(
            x + i * width,
            [timings[case][fn_name] for case in timings],
            width, label=fn_name
        )
        ax.bar_label(rects, labels=[
            time_to_str(timings[case][fn_name]) for case in timings
        ], padding=3)

    plt.legend()
    plt.yticks([])
    plt.xticks(x + 0.5*(len(fns)-1)*width, list(timings.keys()))
    plt.yscale('log')
    plt.title(title)
