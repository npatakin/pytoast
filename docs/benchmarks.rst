Benchmarks
==========

Every chart below compares the fused kernels (the ``cpp`` backend) against the
same operation expressed in stock PyTorch ops — the ``torch`` backend shipped in
this repository. Both sides run identical inputs, the only thing that changes is which backend is active.

Reproduce any of it with::

    python tests/run_benchmarks.py

CPU
---

16 threads on an AMD Ryzen 9 5950X. Forward is **5.1–38×** faster, forward and backward together **3.6–7.8×**.

.. image:: /_static/benchmarks/bench-cpu-forward-light.png
   :class: only-light
   :alt: CPU forward pass benchmark
   :width: 100%

.. image:: /_static/benchmarks/bench-cpu-forward-dark.png
   :class: only-dark
   :alt: CPU forward pass benchmark
   :width: 100%

.. image:: /_static/benchmarks/bench-cpu-forward-backward-light.png
   :class: only-light
   :alt: CPU forward and backward benchmark
   :width: 100%

.. image:: /_static/benchmarks/bench-cpu-forward-backward-dark.png
   :class: only-dark
   :alt: CPU forward and backward benchmark
   :width: 100%

CUDA
----

NVIDIA RTX 4090. Forward is **3.1–24×** faster, forward and backward together **2.1–6.1×**.

.. image:: /_static/benchmarks/bench-cuda-forward-light.png
   :class: only-light
   :alt: CUDA forward pass benchmark
   :width: 100%

.. image:: /_static/benchmarks/bench-cuda-forward-dark.png
   :class: only-dark
   :alt: CUDA forward pass benchmark
   :width: 100%

.. image:: /_static/benchmarks/bench-cuda-forward-backward-light.png
   :class: only-light
   :alt: CUDA forward and backward benchmark
   :width: 100%

.. image:: /_static/benchmarks/bench-cuda-forward-backward-dark.png
   :class: only-dark
   :alt: CUDA forward and backward benchmark
   :width: 100%


Side by side, small batches
---------------------------

The same four operations on both devices at **1024 elements** -- the regime where
per-call cost, not arithmetic, decides the result. It is the least flattering
size for a fused kernel, since there is barely enough work to hide the dispatch,
and it is also the size most likely to appear inside a training loop that calls
the op many times on small tensors.

Note the panels carry independent log scales: a 5950X and a 4090 are not
directly comparable, only each backend against the other on the same hardware.

.. image:: /_static/benchmarks/bench-combined-forward-1k-light.png
   :class: only-light
   :alt: Forward pass on CPU and CUDA, 1024 elements
   :width: 100%

.. image:: /_static/benchmarks/bench-combined-forward-1k-dark.png
   :class: only-dark
   :alt: Forward pass on CPU and CUDA, 1024 elements
   :width: 100%

.. image:: /_static/benchmarks/bench-combined-forward-backward-1k-light.png
   :class: only-light
   :alt: Forward and backward on CPU and CUDA, 1024 elements
   :width: 100%

.. image:: /_static/benchmarks/bench-combined-forward-backward-1k-dark.png
   :class: only-dark
   :alt: Forward and backward on CPU and CUDA, 1024 elements
   :width: 100%

Forward holds up well even here -- 5.1--8.3× on CPU and 5.0--27.8× on CUDA.
Forward and backward together narrow to 2.7--3.7× and 2.5--8.1×, because both
backends then pay PyTorch's fixed autograd-engine cost per call, which at this
size is a large share of the total and is identical for either implementation.

Reproduce with::

    python tests/run_benchmarks.py --elements 1024 --tag 1k --iters 2000 --combined-only


Generate the other sizes with::

    python tests/run_benchmarks.py --elements 10000000 --tag 10m --combined-only

