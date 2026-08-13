Functional Operations
=====================

The functional API exposes every operation as a plain Python function that
dispatches to the currently active backend (C++/CUDA ``'cpp'`` by default,
or pure-PyTorch ``'torch'``).  Switch backends with
:func:`~toast.ops.backend.use_backend`.

All quaternions use the **WXYZ convention** (real part first, shape
``(..., 4)``).  SE(3) poses are represented as ``(q, t)`` pairs.
Gradients flow through every operation in both backends.


.. toctree::
    :caption: Operations
    :hidden:

    quat_ops
    se3_ops
    motion_ops
    interpolation_ops

