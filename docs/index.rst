pytoast
=======

.. image:: /_static/pytoast-header.svg
   :alt: pytoast
   :width: 100%

**py**\ Torch **A**\ ccelerated **S**\ patial **T**\ ransforms

**pytoast** is an efficient PyTorch C++/CUDA extension for spatial transforms —
quaternions, rigid transforms, velocities and trajectories. Every operation is
one fused pre-compiled kernel, not a chain of elementwise ops.

Features
--------

**Fused on both CPU and CUDA.**
    CPU operations are multi-threaded; CUDA operations are a single kernel
    launch. Neither is a fallback for the other. See :doc:`benchmarks`.

**Differentiable.**
    Every operation produces gradients w.r.t. all floating-point arguments,
    from hand-written analytic backward passes rather than a recorded graph.

**Multiple dtypes.**
    ``float32`` and ``float64`` kernels are compiled by default.

**Batch dimensions with broadcasting.**
    An arbitrary number of batch dimensions, following PyTorch broadcasting
    semantics.

**Native support for non-contiguous tensors.**
    No ``.contiguous()`` copies when batch dimensions are non-contiguous or
    broadcast, which saves both memory and time. [#contig]_

**Object-oriented wrappers.**
    :class:`~toast.transforms.Quat`, :class:`~toast.transforms.SE3` and
    :class:`~toast.transforms.LinearMotion` support the usual PyTorch
    operations — indexing, slicing, ``cat``, ``stack``, broadcasting, and moves
    across devices and dtypes. Functional-style operations remain available if
    you prefer them.

**A pure-PyTorch reference for every op.**
    Switchable at runtime through the :doc:`backend system <api/backend>`, for
    readable stack traces or for checking the kernels against it.

**Header-only C++/CUDA math.**
    Usable from your own extensions, backward passes included, via
    ``toast.get_include_path()``.

.. [#contig] A tensor whose *last* (fixed-size) dimension is non-contiguous is
   still copied. Operations on a quaternion tensor of shape ``[B, K, 4]``
   natively handle non-contiguous ``B`` and ``K``, but a ``4`` dimension with
   ``stride != 1`` is made contiguous first. This is required for efficiency.

Benchmarks
----------

Against the same operations written with stock PyTorch ops, at 1M elements:

.. image:: /_static/benchmarks/bench-combined-forward-light.png
   :class: only-light
   :alt: Forward pass on CPU and CUDA
   :width: 100%

.. image:: /_static/benchmarks/bench-combined-forward-dark.png
   :class: only-dark
   :alt: Forward pass on CPU and CUDA
   :width: 100%

Full methodology and more sizes are on the :doc:`benchmarks` page.

.. toctree::
    :caption: User guide
    :titlesonly:

    installation
    getting_started
    custom_containers
    benchmarks

.. toctree::
    :caption: API Reference
    :titlesonly:

    api/data_tensor
    api/transforms
    api/ops
    api/backend

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
