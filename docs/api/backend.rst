
Backend Control
===============

toast supports two backends that share the same Python API:

- **cpp** (default) — fused C++/CUDA kernels, the fastest backend.
- **torch** — reference pure PyTorch implementations, 
  useful for debugging or benchmarking against it.

Switch backends with a context manager so the change is automatically
reverted on exit:

.. code-block:: python

   from toast import use_backend

   with use_backend("torch"):
       q = random_quat(1000)  # runs through PyTorch path

Or switch globally for the lifetime of the process:

.. code-block:: python

   from toast import use_backend, set_backend
   set_backend("torch")

.. currentmodule:: toast.ops.backend

.. autofunction:: use_backend

.. autofunction:: get_backend

.. autofunction:: set_backend